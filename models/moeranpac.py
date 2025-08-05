import logging
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.ranpac import Adapter

logger = logging.getLogger()


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for linear transformations"""
    def __init__(self, in_features, out_features, rank=64, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # LoRA parameters: W = W_0 + (alpha/rank) * B @ A
        # Initialize A with normal distribution, B with zeros (standard LoRA initialization)
        # This ensures initial LoRA contribution is exactly zero
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        # return: [batch_size, seq_len, out_features]
        return (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)

    def get_merged_weight(self):
        """Get the weight matrix to be added to the original weight"""
        return (self.alpha / self.rank) * (self.lora_B @ self.lora_A)


class LoRAAttention(nn.Module):
    """Attention module with LoRA applied to K and V projections"""
    def __init__(self, original_attention, rank=64, alpha=1.0):
        super().__init__()
        self.original_attention = original_attention
        self.dim = original_attention.qkv.in_features
        self.num_heads = original_attention.num_heads
        self.lora_merged = False  # Track if LoRA weights have been merged

        # LoRA for K and V projections (each takes dim -> dim of the qkv projection)
        self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        # If LoRA weights have been merged, just use original attention
        if self.lora_merged:
            return self.original_attention(x)

        B, N, C = x.shape

        # Original QKV projection
        qkv = self.original_attention.qkv(x)  # [B, N, 3*dim]

        # Add LoRA contributions to K and V
        k_lora = self.lora_k(x)  # [B, N, dim]
        v_lora = self.lora_v(x)  # [B, N, dim]

        # Reshape and split QKV
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Add LoRA to K and V
        k_lora = k_lora.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_lora = v_lora.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = k + k_lora
        v = v + v_lora

        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.original_attention.scale
        attn = attn.softmax(dim=-1)
        attn = self.original_attention.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.original_attention.proj(x)
        x = self.original_attention.proj_drop(x)
        return x

    def merge_lora_weights(self):
        """Merge LoRA weights into the original QKV projection"""
        if self.lora_merged:
            logger.warning("LoRA weights already merged for this attention layer")
            return

        with torch.no_grad():
            # Get the merged weights for K and V
            k_weight = self.lora_k.get_merged_weight()  # [dim, dim]
            v_weight = self.lora_v.get_merged_weight()  # [dim, dim]

            # The original qkv weight is [3*dim, dim], structured as [Q_weight; K_weight; V_weight]
            # We need to add our LoRA weights to the K and V portions
            qkv_weight = self.original_attention.qkv.weight.data  # [3*dim, dim]

            # Add LoRA weights to K and V portions
            qkv_weight[self.dim:2*self.dim, :] += k_weight  # K portion
            qkv_weight[2*self.dim:3*self.dim, :] += v_weight  # V portion

            # Mark as merged
            self.lora_merged = True


class MoERanPACClassifier(nn.Module):
    def __init__(self,
                 feature_dim  : int,
                 num_classes  : int,
                 use_RP       : bool,
                 M            : int,
                 **kwargs):

        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_RP = use_RP
        self.M = M

        self.rp_initialized = False

        # Initialize with standard linear layer
        self.fc = nn.Linear(feature_dim, num_classes, bias=False)

        # Random projection matrix (will be initialized after first task)
        self.register_buffer('W_rand', torch.empty(0))

        # Statistics matrices for RanPAC
        if self.use_RP and self.M > 0:
            self.register_buffer('Q', torch.zeros(self.M, num_classes))
            self.register_buffer('G', torch.zeros(self.M, self.M))
        else:
            self.register_buffer('Q', torch.zeros(feature_dim, num_classes))
            self.register_buffer('G', torch.zeros(feature_dim, feature_dim))

        # Buffers for collecting features and labels during each task
        self.register_buffer('collected_features', torch.empty(0))
        self.register_buffer('collected_labels', torch.empty(0))

    def setup_rp(self, device):
        if self.use_RP and self.M > 0 and not self.rp_initialized:
            self.W_rand = torch.randn(self.feature_dim, self.M, device=device)
            self.fc = nn.Linear(self.M, self.num_classes, bias=False).to(device)
            self.rp_initialized = True
            logger.info(f"Random projection initialized: {self.feature_dim} -> {self.M}")

    def collect_features_labels(self, features, labels):
        features = features.detach().cpu()
        labels = labels.detach().cpu()

        if self.collected_features.numel() == 0:
            self.collected_features = features
            self.collected_labels = labels
        else:
            self.collected_features = torch.cat([self.collected_features, features], dim=0)
            self.collected_labels = torch.cat([self.collected_labels, labels], dim=0)

    def clear_collected_data(self):
        self.collected_features = torch.empty(0)
        self.collected_labels = torch.empty(0)

    def target2onehot(self, targets, num_classes):
        onehot = torch.zeros(targets.size(0), num_classes)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        return onehot

    def optimise_ridge_parameter(self, Features, Y):
        """Optimize ridge parameter using cross-validation"""
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []

        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]

        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))

        ridge = ridges[np.argmin(np.array(losses))]
        return ridge

    def update_statistics_and_classifier(self):
        """Update Q, G matrices and classifier weights using collected data"""
        if self.collected_features.numel() == 0:
            logger.warning("No collected features to update statistics")
            return

        features = self.collected_features
        labels = self.collected_labels

        # Convert labels to one-hot
        Y = self.target2onehot(labels, self.num_classes)

        if self.use_RP and self.rp_initialized:
            # Apply random projection with ReLU
            features_h = F.relu(features @ self.W_rand.cpu())
        else:
            features_h = features

        # Move Q, G to CPU for computation
        Q_cpu = self.Q.cpu()
        G_cpu = self.G.cpu()

        # Update statistics matrices (all on CPU)
        Q_cpu = Q_cpu + features_h.T @ Y
        G_cpu = G_cpu + features_h.T @ features_h

        # Move updated matrices back to original device
        self.Q = Q_cpu.to(self.Q.device)
        self.G = G_cpu.to(self.G.device)

        # Optimize ridge parameter and compute classifier weights
        if features_h.size(0) > 1:  # Need at least 2 samples for cross-validation
            # ridge = self.optimise_ridge_parameter(features_h, Y)
            ridge = 1e4
            Wo = torch.linalg.solve(G_cpu + ridge * torch.eye(G_cpu.size(dim=0)), Q_cpu).T

            # Update classifier weights
            device = self.fc.weight.device
            self.fc.weight.data = Wo.to(device)
            logger.info(f"Classifier weights updated using optimal ridge {ridge}")

        # Clear collected data for next task
        self.clear_collected_data()

    def save_classifier_state(self):
        saved_state = {
            'Q': self.Q.clone(),
            'G': self.G.clone(),
            'collected_features': self.collected_features.clone(),
            'collected_labels': self.collected_labels.clone(),
            'fc_weight': self.fc.weight.data.clone(),
        }
        return saved_state

    def restore_classifier_state(self, saved_state):
        self.Q = saved_state['Q']
        self.G = saved_state['G']
        self.collected_features = saved_state['collected_features']
        self.collected_labels = saved_state['collected_labels']
        self.fc.weight.data = saved_state['fc_weight']

    def forward_rp_features(self, x):
        if self.use_RP and self.rp_initialized:
            x = F.relu(x @ self.W_rand)
        return x

    def forward_rp_head(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.forward_rp_features(x)
        x = self.forward_rp_head(x)
        return x


class MoERanPAC(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 adapter_dim    : int   = 64,
                 ranpac_M       : int   = 10000,
                 ranpac_use_RP  : bool  = True,
                 backbone_name  : str   = None,
                 use_lora       : bool  = True,
                 lora_rank      : int   = 64,
                 lora_alpha     : float = 1.0,
                 merge_lora     : bool  = True,
                 **kwargs):

        super().__init__()

        self.M              = ranpac_M
        self.kwargs         = kwargs
        self.use_RP         = ranpac_use_RP
        self.task_num       = task_num
        self.num_classes    = num_classes
        self.adapter_dim    = adapter_dim
        self.use_lora       = use_lora
        self.lora_rank      = lora_rank
        self.lora_alpha     = lora_alpha
        self.merge_lora     = merge_lora

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        if self.use_lora:
            # Apply LoRA to attention layers
            self.lora_attentions = []
            for name, module in self.backbone.named_modules():
                if isinstance(module, vit.Block):
                    # Replace attention with LoRA attention
                    original_attn = module.attn
                    lora_attn = LoRAAttention(original_attn, self.lora_rank, self.lora_alpha)
                    module.attn = lora_attn
                    self.lora_attentions.append(lora_attn)

            logger.info(f"LoRA applied to {len(self.lora_attentions)} attention layers with rank={self.lora_rank}, alpha={self.lora_alpha}")
        else:
            # Insert adapter with mlp to each block
            for name, module in self.backbone.named_modules():
                if isinstance(module, vit.Block):
                    module.adapter = Adapter(
                        down_size=self.adapter_dim,
                        n_embd=module.mlp.fc1.in_features,
                        dropout=0.1,
                    )
                    # Create a closure to capture the current module
                    def create_forward_with_adapter(block):
                        def forward_with_adapter(x):
                            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
                            residual = x
                            adapt_x = block.adapter(x)
                            mlp_x = block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
                            x = adapt_x + mlp_x + residual
                            return x
                        return forward_with_adapter

                    module.forward = create_forward_with_adapter(module)

            logger.info("Adapters initialized in all transformer blocks")

        self.classifier = MoERanPACClassifier(
            feature_dim=self.backbone.num_features,
            num_classes=num_classes,
            use_RP=ranpac_use_RP,
            M=ranpac_M,
        )

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(inputs)
        x = self.forward_head(x)
        return x
    
    def forward_features(self, x):
        # Forward pass
        x = self.backbone.forward_features(x)
        x = x[:, 0] # CLS token
        return x

    def forward_head(self, x):
        x = self.classifier(x)
        return x

    def collect_features_labels(self, x, labels):
        with torch.no_grad():
            features = self.forward_features(x)
            # Ensure labels are on same device as features before collection
            if labels.device != features.device:
                labels = labels.to(features.device)
            self.classifier.collect_features_labels(features, labels)

    def setup_rp(self):
        """Setup random projection after first task"""
        device = next(self.parameters()).device
        self.classifier.setup_rp(device)

    def update_statistics_and_classifier(self):
        """Update statistics and classifier weights"""
        self.classifier.update_statistics_and_classifier()

    def freeze_backbone_except_adapters(self):
        """Freeze backbone except adapters (for adapter mode)"""
        if not self.use_lora:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            for name, module in self.backbone.named_modules():
                if isinstance(module, Adapter):
                    for param in module.parameters():
                        param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_backbone_except_lora(self):
        """Freeze backbone except LoRA parameters (for LoRA mode)"""
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            # Enable LoRA parameters
            for lora_attn in self.lora_attentions:
                for param in lora_attn.lora_k.parameters():
                    param.requires_grad = True
                for param in lora_attn.lora_v.parameters():
                    param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def merge_lora_weights(self):
        """Merge LoRA weights into the original model weights"""
        if not self.use_lora:
            logger.warning("merge_lora_weights called but use_lora=False")
            return

        if not self.merge_lora:
            logger.info("LoRA merging disabled by merge_lora=False, keeping LoRA parameters separate")
            return

        for lora_attn in self.lora_attentions:
            lora_attn.merge_lora_weights()
        logger.info("All LoRA weights merged into backbone")

    def freeze_all_except_classifier(self):
        """Freeze all parameters except classifier parameters"""
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def save_classifier_state(self):
        return self.classifier.save_classifier_state()
    
    def restore_classifier_state(self, saved_state):
        self.classifier.restore_classifier_state(saved_state)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1