import logging
from typing import TypeVar
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs

from models.vit import _create_vision_transformer

logger = logging.getLogger()

T = TypeVar('T', bound='nn.Module')

default_cfgs['vit_base_patch16_224_flyprompt_lsh'] = _cfg(
    url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
    num_classes=21843)

def stable_cholesky(matrix, reg=1e-4):
    reg_matrix = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.cholesky(matrix + reg_matrix)

def transform_to_target_covariance(Fi, target_cor, reg=1e-4):
    Fi_centered = Fi - Fi.mean(dim=0)
    n_samples = Fi_centered.size(0)
    C = (Fi_centered.T @ Fi_centered) / (n_samples - 1)
    
    L = stable_cholesky(C, reg)
    L_cor = stable_cholesky(target_cor, reg)
    
    A = torch.linalg.solve(L, L_cor)
    
    Fj = Fi_centered @ A
    return Fj

@register_model
def vit_base_patch16_224_flyprompt_lsh(pretrained=False, **kwargs):
    """ ViT-Base model with LSH-based prompt selection """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_flyprompt_lsh', pretrained=pretrained, **model_kwargs)
    return model

class LSHPromptSelector(nn.Module):
    def __init__(self, 
                 input_dim=768, 
                 expansion_dim=4096, 
                 pool_size=30, 
                 selection_size=5, 
                 keep_ratio=0.05,  # 5% sparsity
                 winner_type='topk',  # 'topk' or 'threshold'
                 return_binary=True,  # True for binary, False for float
                 hash_type='chunked',  # 'chunked' or 'overlapping'
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.pool_size = pool_size
        self.selection_size = selection_size
        self.keep_ratio = keep_ratio
        self.winner_type = winner_type
        self.return_binary = return_binary
        self.hash_type = hash_type
        
        # Initialize frozen random projection matrix
        self._init_projection_matrix(input_dim, expansion_dim)
        
        # For locality-sensitive hashing
        self.chunk_size = expansion_dim // selection_size  # Divide into chunks for multiple hashes
        
    def _init_projection_matrix(self, input_dim, expansion_dim):
        """Initialize frozen random projection matrix"""
        # Simple random Gaussian matrix as requested
        projection = torch.randn(expansion_dim, input_dim) / math.sqrt(input_dim)
        self.register_buffer('projection', projection)
        
    def winner_take_all_topk(self, features, keep_ratio):
        """Winner-take-all using top-k strategy"""
        B, D = features.shape
        k = int(D * keep_ratio)
        
        # Get top-k values and indices
        topk_values, topk_indices = features.topk(k, dim=1)
        
        # Create sparse representation
        sparse_features = torch.zeros_like(features)
        if self.return_binary:
            # Binary version: set top-k positions to 1
            sparse_features.scatter_(1, topk_indices, 1.0)
        else:
            # Float version: keep original top-k values
            sparse_features.scatter_(1, topk_indices, topk_values)
            
        return sparse_features
    
    def winner_take_all_threshold(self, features, keep_ratio):
        """Winner-take-all using threshold strategy"""
        # Calculate threshold to keep approximately keep_ratio of neurons
        threshold = features.quantile(1.0 - keep_ratio, dim=1, keepdim=True)
        
        if self.return_binary:
            # Binary version: 1 if above threshold, 0 otherwise
            sparse_features = (features > threshold).float()
        else:
            # Float version: keep values above threshold, zero others
            sparse_features = torch.where(features > threshold, features, torch.zeros_like(features))
            
        return sparse_features
    
    def hash_to_indices_chunked(self, sparse_code):
        """
        Locality-sensitive hash-to-index mapping using chunked hashing
        Similar sparse patterns will produce similar hash values
        """
        B, D = sparse_code.shape
        
        # Divide sparse code into chunks for multiple hash functions
        chunk_size = D // self.selection_size
        indices = []
        
        for i in range(self.selection_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, D)
            chunk = sparse_code[:, start_idx:end_idx]
            
            # Simple hash function: weighted sum with prime numbers
            weights = torch.arange(1, chunk.size(1) + 1, device=chunk.device, dtype=torch.float)
            weights = (weights * 2654435761) % self.pool_size  # Use prime for better distribution
            
            # Hash each chunk to a prompt index
            hash_values = (chunk * weights.unsqueeze(0)).sum(dim=1)
            prompt_idx = (hash_values % self.pool_size).long()
            indices.append(prompt_idx)
        
        return torch.stack(indices, dim=1)  # (B, selection_size)
    
    def hash_to_indices_overlapping(self, sparse_code):
        """
        Alternative: Use overlapping windows for better locality sensitivity
        """
        B, D = sparse_code.shape
        indices = []
        
        # Create overlapping windows
        window_size = D // (self.selection_size - 1) if self.selection_size > 1 else D
        
        for i in range(self.selection_size):
            # Overlapping windows with 50% overlap
            start_idx = (i * window_size // 2) % (D - window_size + 1)
            end_idx = start_idx + window_size
            window = sparse_code[:, start_idx:end_idx]
            
            # Create position-weighted hash
            positions = torch.arange(window.size(1), device=window.device, dtype=torch.float)
            weights = torch.exp(-positions / (window_size / 4))  # Exponential decay weights
            
            # Compute hash
            hash_values = (window * weights.unsqueeze(0)).sum(dim=1)
            # Add offset to reduce collisions between different windows
            hash_values += i * 17  # Prime offset
            prompt_idx = (hash_values % self.pool_size).long()
            indices.append(prompt_idx)
        
        return torch.stack(indices, dim=1)  # (B, selection_size)
    
    def forward(self, query):
        """
        Forward pass: query (B, 768) -> prompt_indices (B, selection_size)
        """
        # 1. Project to high-dimensional space
        expanded = F.linear(query, self.projection)  # (B, expansion_dim)
        
        # 2. Winner-take-all sparsification
        if self.winner_type == 'topk':
            sparse_code = self.winner_take_all_topk(expanded, self.keep_ratio)
        elif self.winner_type == 'threshold':
            sparse_code = self.winner_take_all_threshold(expanded, self.keep_ratio)
        else:
            raise ValueError(f"Unknown winner_type: {self.winner_type}")
        
        # 3. Hash to prompt indices (user-selectable hash function)
        if self.hash_type == 'chunked':
            prompt_indices = self.hash_to_indices_chunked(sparse_code)
        elif self.hash_type == 'overlapping':
            prompt_indices = self.hash_to_indices_overlapping(sparse_code)
        else:
            raise ValueError(f"Unknown hash_type: {self.hash_type}")
        
        return prompt_indices, sparse_code


class PromptLSH(nn.Module):
    def __init__(self,
                 pool_size=30,
                 selection_size=5,
                 prompt_len=10,
                 dimension=768,
                 expansion_dim=4096,
                 keep_ratio=0.05,
                 winner_type='topk',
                 return_binary=True,
                 hash_type='chunked',
                 **kwargs):
        super().__init__()

        self.pool_size = pool_size
        self.selection_size = selection_size
        self.prompt_len = prompt_len
        self.dimension = dimension

        # LSH-based prompt selector (replaces similarity-based selection)
        self.lsh_selector = LSHPromptSelector(
            input_dim=dimension,
            expansion_dim=expansion_dim,
            pool_size=pool_size,
            selection_size=selection_size,
            keep_ratio=keep_ratio,
            winner_type=winner_type,
            return_binary=return_binary,
            hash_type=hash_type
        )

        # Learnable prompt embeddings
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimension, requires_grad=True))
        torch.nn.init.uniform_(self.prompts, -1, 1)

        # For tracking (optional, for analysis)
        self.register_buffer('usage_counter', torch.zeros(pool_size))
    
    def forward(self, query: torch.Tensor, **kwargs):
        B, D = query.shape
        assert D == self.dimension, f'Query dimension {D} does not match prompt dimension {self.dimension}'
        
        # LSH-based prompt selection (deterministic, no training needed)
        prompt_indices, sparse_code = self.lsh_selector(query)  # (B, selection_size)
        
        # Retrieve selected prompts
        # prompt_indices: (B, selection_size), prompts: (pool_size, prompt_len, D)
        selected_prompts = self.prompts[prompt_indices]  # (B, selection_size, prompt_len, D)
        
        # Track usage for analysis
        unique_indices = prompt_indices.flatten()
        self.usage_counter += torch.bincount(unique_indices, minlength=self.pool_size)
        
        # Return dummy similarity for compatibility (always 0 since no similarity computed)
        similarity = torch.zeros(B, self.selection_size, device=query.device)
        
        return similarity, selected_prompts


class FlyPromptLSH(nn.Module):
    def __init__(self,
                 pool_size=30,
                 selection_size=5,
                 prompt_len=5,
                 class_num=100,
                 backbone_name=None,
                 lambd=0.5,
                 expansion_dim=4096,
                 keep_ratio=0.05,
                 winner_type='topk',
                 return_binary=True,
                 hash_type='chunked',
                 **kwargs):

        super().__init__()
        
        self.features = torch.empty(0)
        self.keys = torch.empty(0)  # Keep for compatibility, but won't be used

        # Load additional parameters (same as original FlyPrompt)
        self.load_pt = kwargs.get("load_pt")
        self.cor_path = kwargs.get("cor_path")
        self.update_cor = kwargs.get("update_cor")
        self.pretrain_cor = kwargs.get("pretrain_cor")
        self.cor_coef = kwargs.get("cor_coef")

        if self.cor_path is not None:
            self.cov_matrix_tensor = np.load(self.cor_path)
            self.cov_matrix_tensor = torch.from_numpy(self.cov_matrix_tensor).to(device='cuda', dtype=torch.float32)

            if self.update_cor:
                self.cov_matrix_tensor.requires_grad = True
            else:
                self.cov_matrix_tensor.requires_grad = False
            
            if self.pretrain_cor:
                print("Load covariance from:", self.cor_path)
            else:
                self.cov_matrix_tensor = torch.randn(768, 768, device='cuda', dtype=torch.float32)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len = prompt_len
        self.selection_size = selection_size
        self.lambd = lambd
        self.class_num = class_num

        # Initialize backbone (same as original)
        self.add_module('backbone', timm.models.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True
        
        # LSH-based prompt module (replaces original Prompt)
        self.prompt = PromptLSH(
            pool_size=pool_size,
            selection_size=selection_size,
            prompt_len=prompt_len,
            dimension=self.backbone.num_features,
            expansion_dim=expansion_dim,
            keep_ratio=keep_ratio,
            winner_type=winner_type,
            return_binary=return_binary,
            hash_type=hash_type
        )

        self.register_buffer('similarity', torch.zeros(1), persistent=False)

        # print all the network hyperparameters and module size:
        print(f"pool_size: {pool_size}")
        print(f"selection_size: {selection_size}")
        print(f"prompt_len: {prompt_len}")
        print(f"class_num: {class_num}")
        print(f"backbone_name: {backbone_name}")
        print(f"lambd: {lambd}")
        print(f"expansion_dim: {expansion_dim}")
        print(f"keep_ratio: {keep_ratio}")
        print(f"winner_type: {winner_type}")
        print(f"return_binary: {return_binary}")
        print(f"hash_type: {hash_type}")

        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Total number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"Total number of non-trainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad)}")
        print(f"Total number of parameters in the backbone: {sum(p.numel() for p in self.backbone.parameters())}")
        print(f"Total number of trainable parameters in the backbone: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")
        print(f"Total number of non-trainable parameters in the backbone: {sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)}")
        print(f"Total number of parameters in the prompt: {sum(p.numel() for p in self.prompt.parameters())}")
        print(f"Total number of trainable parameters in the prompt: {sum(p.numel() for p in self.prompt.parameters() if p.requires_grad)}")
        print(f"Total number of non-trainable parameters in the prompt: {sum(p.numel() for p in self.prompt.parameters() if not p.requires_grad)}")

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        
        with torch.no_grad():
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()
        
        if self.training:
            self.features = torch.cat((self.features, query.detach().cpu()), dim=0)
        
        # LSH-based prompt selection (deterministic, no similarity computation)
        similarity, prompts = self.prompt(query)
        
        # Process prompts (same as original FlyPrompt)
        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        
        # Optional covariance transformation (same as original)
        if self.cor_path is not None:
            Fj = transform_to_target_covariance(x, self.cov_matrix_tensor)
            Fj = Fj / torch.norm(Fj, dim=1, keepdim=True)
            x = (1-self.cor_coef)*x + self.cor_coef*Fj
            
        x = self.backbone.fc_norm(x)  
        x = self.backbone.fc(x)
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        # Note: no similarity loss since LSH selection is deterministic
        return F.cross_entropy(output, target)

    def get_count(self):
        """Return prompt usage statistics for analysis"""
        return self.prompt.usage_counter

    def get_lsh_stats(self):
        """Get LSH-specific statistics for analysis"""
        return {
            'usage_counter': self.prompt.usage_counter.cpu().numpy(),
            'expansion_dim': self.prompt.lsh_selector.expansion_dim,
            'keep_ratio': self.prompt.lsh_selector.keep_ratio,
            'winner_type': self.prompt.lsh_selector.winner_type
        } 