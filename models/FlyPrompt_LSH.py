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

# ======================== Bio-Plausible Modules ========================

class FeatureNormalizer(nn.Module):
    """Divisive normalization - implements ORN->PN concentration independence"""
    def __init__(self, input_dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        if self.training:
            # Update running statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean.to(x.device) + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var.to(x.device) + self.momentum * batch_var
                self.num_batches_tracked += 1
        
        # Apply normalization to maintain exponential distribution with consistent mean
        running_mean = self.running_mean.to(x.device)
        running_var = self.running_var.to(x.device)
        normalized = (x - running_mean) / torch.sqrt(running_var + self.eps)
        return normalized

class SparseBinaryProjection(nn.Module):
    """Sparse binary projection matrix - implements PN->KC expansion with sparse connections"""
    def __init__(self, input_dim, expansion_dim, connectivity=6):
        super().__init__()
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.connectivity = connectivity
        
        # Create frozen sparse binary projection matrix
        self._init_sparse_binary_matrix()
        
    def _init_sparse_binary_matrix(self):
        """Each KC (row) connects to exactly 'connectivity' random PNs (columns)"""
        projection = torch.zeros(self.expansion_dim, self.input_dim)
        
        for i in range(self.expansion_dim):
            # Randomly select 'connectivity' input connections for each KC
            indices = torch.randperm(self.input_dim)[:self.connectivity]
            projection[i, indices] = 1.0
            
        # Register as buffer (frozen, no gradients)
        self.register_buffer('projection', projection)
        
    def forward(self, x):
        # Sparse binary projection: each KC sums exactly 'connectivity' PN inputs
        return F.linear(x, self.projection)

class WinnerTakeAll(nn.Module):
    """APL-like global inhibition - implements strong feedback inhibition"""
    def __init__(self, keep_ratio=0.05, winner_type='topk'):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.winner_type = winner_type
        
    def forward(self, expanded_features):
        """Apply global inhibition to keep only top keep_ratio neurons active"""
        B, D = expanded_features.shape
        k = max(1, int(D * self.keep_ratio))
        
        if self.winner_type == 'topk':
            # Get top-k active neurons
            topk_values, topk_indices = expanded_features.topk(k, dim=1)
            
            # Apply strong inhibition: set non-winners to zero
            sparse_features = torch.zeros_like(expanded_features)
            sparse_features.scatter_(1, topk_indices, topk_values)
            
        elif self.winner_type == 'threshold':
            # Threshold-based inhibition
            threshold = expanded_features.quantile(1.0 - self.keep_ratio, dim=1, keepdim=True)
            sparse_features = torch.where(expanded_features > threshold, 
                                        expanded_features, 
                                        torch.zeros_like(expanded_features))
        else:
            raise ValueError(f"Unknown winner_type: {self.winner_type}")
            
        return sparse_features

class BioLSH(nn.Module):
    """Bio-plausible locality-sensitive hashing from sparse KC activations to prompt indices"""
    def __init__(self, expansion_dim, pool_size, selection_size):
        super().__init__()
        self.expansion_dim = expansion_dim
        self.pool_size = pool_size
        self.selection_size = selection_size
        
        # Validate parameters
        if selection_size > expansion_dim:
            raise ValueError(f"selection_size ({selection_size}) cannot be larger than expansion_dim ({expansion_dim})")
        
        self.chunk_size = max(1, expansion_dim // selection_size)
        
    def forward(self, sparse_code):
        """Convert sparse KC activations to synchronized prompt indices"""
        B, D = sparse_code.shape
        indices = []
        
        for i in range(self.selection_size):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, D)
            chunk = sparse_code[:, start_idx:end_idx]
            
            # Locality-sensitive hash function
            positions = torch.arange(chunk.size(1), device=chunk.device, dtype=torch.float)
            weights = (positions * 2654435761) % 1000000007  # Prime-based weighting
            
            # Compute hash values
            hash_values = (chunk * weights.unsqueeze(0)).sum(dim=1)
            prompt_idx = (hash_values % self.pool_size).long()
            indices.append(prompt_idx)
            
        return torch.stack(indices, dim=1)  # (B, selection_size)

class TriPartitePromptPool(nn.Module):
    """Three-part hierarchical prompt pool with different temporal scales"""
    def __init__(self, pool_size, selection_size, prompt_len, dimension, ema_alpha=0.05, ema_beta=0.1):
        super().__init__()
        
        self.pool_size = pool_size
        self.selection_size = selection_size
        self.prompt_len = prompt_len
        self.dimension = dimension
        self.ema_alpha = ema_alpha  # EMA rate for part B from A
        self.ema_beta = ema_beta    # EMA rate for part C from B
        
        # Three parallel prompt pools with different time scales
        self.part_A = nn.Parameter(torch.randn(pool_size, prompt_len, dimension))  # Fast - trainable
        self.part_B = nn.Parameter(torch.randn(pool_size, prompt_len, dimension))  # Medium - EMA from A
        self.part_C = nn.Parameter(torch.randn(pool_size, prompt_len, dimension))  # Slow - EMA from B
        
        # Initialize all parts uniformly
        torch.nn.init.uniform_(self.part_A, -1, 1)
        torch.nn.init.uniform_(self.part_B, -1, 1)
        torch.nn.init.uniform_(self.part_C, -1, 1)
        
        # Freeze parts B and C from gradient updates
        self.part_B.requires_grad = False
        self.part_C.requires_grad = False
        
        # Task counter
        self.register_buffer('task_count', torch.tensor(0, dtype=torch.long))
        
    def update_part_B(self):
        """EMA update for part B from A - called during online_step after task 1"""
        if self.task_count > 0:
            with torch.no_grad():
                self.part_B.data = (1 - self.ema_alpha) * self.part_B.data + self.ema_alpha * self.part_A.data
                
    def update_part_C(self):
        """EMA update for part C from B - called during online_after_task after task 1"""
        if self.task_count > 0:
            with torch.no_grad():
                self.part_C.data = (1 - self.ema_beta) * self.part_C.data + self.ema_beta * self.part_B.data
                
    def initialize_parts_after_first_task(self):
        """Initialize B and C from A after first task"""
        if self.task_count == 0:
            with torch.no_grad():
                self.part_B.data.copy_(self.part_A.data)
                self.part_C.data.copy_(self.part_A.data)
            print(f"Initialized tripartite prompt pool: B and C copied from A")
            return True
        return False
    
    def increment_task_count(self):
        """Increment task count - should be called after each task"""
        self.task_count += 1
        print(f"Task count incremented to: {self.task_count.item()}")
        
    def forward(self, indices):
        """Retrieve prompts from all three parts using same indices"""
        # indices: (B, selection_size)
        prompts_A = self.part_A[indices]  # (B, selection_size, prompt_len, dimension)
        prompts_B = self.part_B[indices]
        prompts_C = self.part_C[indices]
        
        # Concatenate prompts from all three parts along prompt length dimension
        # Result: (B, selection_size, 3*prompt_len, dimension)
        selected_prompts = torch.cat([prompts_A, prompts_B, prompts_C], dim=2)
        
        return selected_prompts

# ======================== Modified Existing Classes ========================

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
                 projection_type='dense_gaussian',  # 'dense_gaussian' or 'sparse_binary'
                 connectivity=6,  # For sparse binary projection
                 bio_plausible=False,  # Enable full bio-plausible pipeline
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
        self.projection_type = projection_type
        self.connectivity = connectivity
        self.bio_plausible = bio_plausible
        
        # Bio-plausible components
        if self.bio_plausible:
            # Full bio-plausible pipeline
            self.normalizer = FeatureNormalizer(input_dim)
            self.projector = SparseBinaryProjection(input_dim, expansion_dim, connectivity)
            self.wta = WinnerTakeAll(keep_ratio, winner_type)
            self.bio_lsh = BioLSH(expansion_dim, pool_size, selection_size)
        else:
            # Original or hybrid pipeline
            if projection_type == 'sparse_binary':
                self.projector = SparseBinaryProjection(input_dim, expansion_dim, connectivity)
            else:
                # Original dense Gaussian projection
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
        # Ensure features is float/double for quantile operation
        features_float = features.float()
        threshold = features_float.quantile(1.0 - keep_ratio, dim=1, keepdim=True)
        
        if self.return_binary:
            # Binary version: 1 if above threshold, 0 otherwise
            sparse_features = (features_float > threshold).float()
        else:
            # Float version: keep values above threshold, zero others
            sparse_features = torch.where(features_float > threshold, features_float, torch.zeros_like(features_float))
            
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
        if self.bio_plausible:
            # Full bio-plausible pipeline
            # 1. Divisive normalization (ORN->PN)
            normalized = self.normalizer(query)
            
            # 2. Sparse binary projection (PN->KC) 
            expanded = self.projector(normalized)
            
            # 3. Winner-take-all with APL-like inhibition
            sparse_code = self.wta(expanded)
            
            # 4. Bio-plausible LSH (KC->prompt indices)
            prompt_indices = self.bio_lsh(sparse_code)
            
        else:
            # Original or hybrid pipeline
            # 1. Project to high-dimensional space
            if self.projection_type == 'sparse_binary':
                expanded = self.projector(query)
            else:
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
                 projection_type='dense_gaussian',
                 connectivity=6,
                 bio_plausible=False,
                 ema_alpha=0.05,
                 ema_beta=0.1,
                 **kwargs):
        super().__init__()

        self.pool_size = pool_size
        self.selection_size = selection_size
        self.prompt_len = prompt_len
        self.dimension = dimension
        self.bio_plausible = bio_plausible

        # LSH-based prompt selector (replaces similarity-based selection)
        self.lsh_selector = LSHPromptSelector(
            input_dim=dimension,
            expansion_dim=expansion_dim,
            pool_size=pool_size,
            selection_size=selection_size,
            keep_ratio=keep_ratio,
            winner_type=winner_type,
            return_binary=return_binary,
            hash_type=hash_type,
            projection_type=projection_type,
            connectivity=connectivity,
            bio_plausible=bio_plausible
        )

        if self.bio_plausible:
            # Use tripartite prompt pool for bio-plausible architecture
            self.prompt_pool = TriPartitePromptPool(
                pool_size=pool_size,
                selection_size=selection_size,
                prompt_len=prompt_len,
                dimension=dimension,
                ema_alpha=ema_alpha,
                ema_beta=ema_beta
            )
        else:
            # Original single prompt pool
            self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimension, requires_grad=True))
            torch.nn.init.uniform_(self.prompts, -1, 1)

        # For tracking (optional, for analysis)
        self.register_buffer('usage_counter', torch.zeros(pool_size))
    
    def forward(self, query: torch.Tensor, **kwargs):
        B, D = query.shape
        assert D == self.dimension, f'Query dimension {D} does not match prompt dimension {self.dimension}'
        
        # LSH-based prompt selection (deterministic, no training needed)
        prompt_indices, sparse_code = self.lsh_selector(query)  # (B, selection_size)
        
        if self.bio_plausible:
            # Use tripartite prompt pool
            selected_prompts = self.prompt_pool(prompt_indices)  # (B, selection_size, 3*prompt_len, D)
        else:
            # Use original single prompt pool
            # Retrieve selected prompts
            # prompt_indices: (B, selection_size), prompts: (pool_size, prompt_len, D)
            selected_prompts = self.prompts[prompt_indices]  # (B, selection_size, prompt_len, D)
        
        # Track usage for analysis
        unique_indices = prompt_indices.flatten()
        self.usage_counter += torch.bincount(unique_indices, minlength=self.pool_size)
        
        # Return dummy similarity for compatibility (always 0 since no similarity computed)
        similarity = torch.zeros(B, self.selection_size, device=query.device)
        
        return similarity, selected_prompts

    def update_part_B(self):
        """Update part B from A - to be called during online_step"""
        if self.bio_plausible:
            self.prompt_pool.update_part_B()
    
    def update_part_C(self):
        """Update part C from B - to be called during online_after_task"""
        if self.bio_plausible:
            self.prompt_pool.update_part_C()
    
    def initialize_parts_after_first_task(self):
        """Initialize parts B and C from A after first task"""
        if self.bio_plausible:
            return self.prompt_pool.initialize_parts_after_first_task()
        return False
    
    def increment_task_count(self):
        """Increment task count - should be called after each task"""
        if self.bio_plausible:
            self.prompt_pool.increment_task_count()


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
                 projection_type='dense_gaussian',
                 connectivity=6,
                 bio_plausible=False,
                 ema_alpha=0.05,
                 ema_beta=0.1,
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
        self.bio_plausible = bio_plausible
        self.ema_alpha = ema_alpha
        self.ema_beta = ema_beta

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
            hash_type=hash_type,
            projection_type=projection_type,
            connectivity=connectivity,
            bio_plausible=bio_plausible,
            ema_alpha=ema_alpha,
            ema_beta=ema_beta
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
        print(f"projection_type: {projection_type}")
        print(f"connectivity: {connectivity}")
        print(f"bio_plausible: {bio_plausible}")
        print(f"ema_alpha: {ema_alpha}")
        print(f"ema_beta: {ema_beta}")

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
        
        # Process prompts (handle both original and tripartite pools)
        if hasattr(self.prompt, 'bio_plausible') and self.prompt.bio_plausible:
            # Tripartite pool: prompts shape is (B, selection_size, 3*prompt_len, D)
            effective_prompt_len = self.selection_size * 3 * self.prompt_len
            prompts = prompts.contiguous().view(B, effective_prompt_len, D)
        else:
            # Original pool: prompts shape is (B, selection_size, prompt_len, D)
            effective_prompt_len = self.selection_size * self.prompt_len
            prompts = prompts.contiguous().view(B, effective_prompt_len, D)
            
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(effective_prompt_len, -1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:effective_prompt_len + 1].clone()
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
        stats = {
            'usage_counter': self.prompt.usage_counter.cpu().numpy(),
            'expansion_dim': self.prompt.lsh_selector.expansion_dim,
            'keep_ratio': self.prompt.lsh_selector.keep_ratio,
            'winner_type': self.prompt.lsh_selector.winner_type
        }

        
        if hasattr(self.prompt, 'bio_plausible') and self.prompt.bio_plausible:
            stats['bio_plausible'] = True
            stats['projection_type'] = self.prompt.lsh_selector.projection_type
            stats['connectivity'] = self.prompt.lsh_selector.connectivity
            stats['task_count'] = self.prompt.prompt_pool.task_count.item()
        else:
            stats['bio_plausible'] = False
            
        return stats

    def update_part_B(self):
        """Update part B from A - to be called during online_step"""
        self.prompt.update_part_B()
    
    def update_part_C(self):
        """Update part C from B - to be called during online_after_task"""
        self.prompt.update_part_C()
    
    def initialize_parts_after_first_task(self):
        """Initialize parts B and C from A after first task"""
        return self.prompt.initialize_parts_after_first_task()
    
    def increment_task_count(self):
        """Increment task count - should be called after each task"""
        self.prompt.increment_task_count() 