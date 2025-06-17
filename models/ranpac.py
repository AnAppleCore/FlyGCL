import logging
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.l2p import Prompt

logger = logging.getLogger()


class Adapter(nn.Module):
    def __init__(
            self,
            down_size    : int,
            n_embd       : int,
            dropout      : float = 0.0,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.down_size = down_size

        self.scale = nn.Parameter(torch.ones(1))
        self.dropout = dropout

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        return up
    
class RanPACClassifier(nn.Module):
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
        logger.info(f"Optimal ridge parameter: {ridge}")
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
            ridge = self.optimise_ridge_parameter(features_h, Y)
            Wo = torch.linalg.solve(G_cpu + ridge * torch.eye(G_cpu.size(dim=0)), Q_cpu).T

            # Update classifier weights
            device = self.fc.weight.device
            self.fc.weight.data = Wo.to(device)
            logger.info("Classifier weights updated using RanPAC statistics")

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

class RanPAC(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 adapter_dim    : int   = 64,
                 ranpac_M       : int   = 10000,
                 ranpac_use_RP  : bool  = True,
                 backbone_name  : str   = None,
                 use_g_prompt   : bool  = False,
                 pos_g_prompt   : list  = [0, 1, 2, 3, 4],
                 len_g_prompt   : int   = 5,
                 g_pool         : int   = 1,
                 **kwargs):

        super().__init__()

        self.M              = ranpac_M
        self.kwargs         = kwargs
        self.use_RP         = ranpac_use_RP
        self.task_num       = task_num
        self.num_classes    = num_classes
        self.adapter_dim    = adapter_dim
        self.use_g_prompt   = use_g_prompt
        self.len_g_prompt   = len_g_prompt
        self.g_length       = len(pos_g_prompt) if pos_g_prompt else 0

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        if self.use_g_prompt:
            # G-prompt setup
            self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
            
            self.g_prompt = Prompt(
                g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, 
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
            )
            self.g_prompt.key = None  # No key selection for g-prompt
            
            logger.info(f"G-prompt initialized at positions {pos_g_prompt} with length {len_g_prompt}")
        else:
            # Insert adapter with mlp to each block (existing logic)
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

        self.classifier = RanPACClassifier(
            feature_dim=self.backbone.num_features,
            num_classes=num_classes,
            use_RP=ranpac_use_RP,
            M=ranpac_M,
        )

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      **kwargs):
        """G-prompt tuning similar to DualPrompt"""
        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(inputs)
        x = self.forward_head(x)
        return x
    
    def forward_features(self, x):
        if self.use_g_prompt:
            # G-prompt forward pass
            x = self.backbone.patch_embed(x)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)

            # Get g_prompt (no query needed for task-agnostic prompts)
            g_p = self.g_prompt.prompts[0]
            g_p = g_p.expand(B, -1, -1)

            # Apply prompt tuning
            x = self.prompt_tuning(x, g_p)
            x = self.backbone.norm(x)
            cls_token = x[:, 0]
            return cls_token
        else:
            # Adapter forward pass (existing logic)
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
        if not self.use_g_prompt:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            for name, module in self.backbone.named_modules():
                if isinstance(module, Adapter):
                    for param in module.parameters():
                        param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_backbone_except_prompts(self):
        """Freeze backbone except g-prompts (for g-prompt mode)"""
        if self.use_g_prompt:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            for param in self.g_prompt.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_all_except_classifier(self):
        for name, param in self.named_parameters():
            if 'classifier.fc' not in name:
                param.requires_grad = False
        for param in self.classifier.fc.parameters():
            param.requires_grad = False

    def save_classifier_state(self):
        return self.classifier.save_classifier_state()
    
    def restore_classifier_state(self, saved_state):
        self.classifier.restore_classifier_state(saved_state)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1