import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseExpert(nn.Module):
    """Abstract base class for per-task experts.
    Must implement forward(backbone, inputs, expert_ids) -> raw CLS features (no RP).
    """
    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_new_expert(self, task_id: int):
        """Initialize new expert's parameters."""
        raise NotImplementedError


# ---------------------------- PromptExpert (per-task) ---------------------------
class PromptExpert(BaseExpert):
    """Per-task fixed prompt expert.

    Each expert owns a fixed set of prompt tokens per specified layer position.
    No dynamic selection, no key/query mechanism.
    """
    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim
        self.prompts = nn.Parameter(torch.randn(num_experts, len_prompt, embed_dim, requires_grad=True))
        nn.init.uniform_(self.prompts, -1, 1)

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)

        # Create mask for valid expert_ids (>= 0)
        valid_mask = expert_ids >= 0

        if valid_mask.all():
            # All samples have valid expert_ids, use original logic
            prompts = self.prompts[expert_ids.long()]
            prompts = prompts + backbone.pos_embed[:,0].unsqueeze(0).expand(B, self.len_prompt, D)
            x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        elif not valid_mask.any():
            # No samples have valid expert_ids, skip prompt concatenation entirely
            pass
        else:
            # Mixed batch: some valid, some invalid expert_ids
            # Need to process separately to maintain correct sequence lengths
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            invalid_indices = (~valid_mask).nonzero(as_tuple=True)[0]

            # Process valid samples with prompts
            if len(valid_indices) > 0:
                x_valid = x[valid_indices]
                valid_expert_ids = expert_ids[valid_indices]
                prompts = self.prompts[valid_expert_ids.long()]
                prompts = prompts + backbone.pos_embed[:,0].unsqueeze(0).expand(len(valid_indices), self.len_prompt, D)
                x_valid = torch.cat((x_valid[:,0].unsqueeze(1), prompts, x_valid[:,1:]), dim=1)

            # Process invalid samples without prompts
            if len(invalid_indices) > 0:
                x_invalid = x[invalid_indices]

            # Process through transformer blocks separately
            if len(valid_indices) > 0:
                x_valid = backbone.blocks(x_valid)
                x_valid = backbone.norm(x_valid)
                cls_valid = x_valid[:, 0]

            if len(invalid_indices) > 0:
                x_invalid = backbone.blocks(x_invalid)
                x_invalid = backbone.norm(x_invalid)
                cls_invalid = x_invalid[:, 0]

            # Reconstruct output in original order
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            if len(valid_indices) > 0:
                output[valid_indices] = cls_valid
            if len(invalid_indices) > 0:
                output[invalid_indices] = cls_invalid

            return output

        x = backbone.blocks(x)
        x = backbone.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0:
            return
        assert expert_id >= 1
        assert expert_id < self.num_experts

        # mean of previous experts
        prev_experts = self.prompts[:expert_id].clone()
        prev_experts_mean = prev_experts.mean(dim=0)
        self.prompts.data[expert_id] = prev_experts_mean


# ------------------------------ LoRAExpert (per-task) ------------------------------

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
        self.qkv = original_attention.qkv
        self.dim = original_attention.qkv.in_features
        self.num_heads = original_attention.num_heads
        self.scale = original_attention.scale
        self.attn_drop = original_attention.attn_drop
        self.proj = original_attention.proj
        self.proj_drop = original_attention.proj_drop
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
        qkv = self.qkv(x)  # [B, N, 3*dim]

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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
            qkv_weight = self.qkv.weight.data  # [3*dim, dim]

            # Add LoRA weights to K and V portions
            qkv_weight[self.dim:2*self.dim, :] += k_weight  # K portion
            qkv_weight[2*self.dim:3*self.dim, :] += v_weight  # V portion

            # Mark as merged
            self.lora_merged = True


class LoRAExpert(BaseExpert):
    """Per-task LoRA experts: only add LoRA to the first K blocks' attention K/V.
    Other behavior and API match PromptExpert (output CLS features; expert_ids>=0 valid).
    """
    def __init__(self,
                 num_experts: int,
                 embed_dim: int = 768,
                 num_lora_layers: int = 5,
                 lora_rank: int = 64,
                 lora_alpha: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_lora_layers = num_lora_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Per (layer, expert) LoRA for K and V projections
        self.lora_k = nn.ModuleList([
            nn.ModuleList([
                LoRALayer(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha)
                for _ in range(num_experts)
            ]) for _ in range(num_lora_layers)
        ])
        self.lora_v = nn.ModuleList([
            nn.ModuleList([
                LoRALayer(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha)
                for _ in range(num_experts)
            ]) for _ in range(num_lora_layers)
        ])

    def _attn_with_lora(self, attn: nn.Module, x: torch.Tensor, lora_k: LoRALayer, lora_v: LoRALayer) -> torch.Tensor:
        """Attention forward with LoRA added to K and V projections."""
        B, N, C = x.shape
        qkv = attn.qkv(x)
        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        k_l = lora_k(x).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        v_l = lora_v(x).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        k = k + k_l
        v = v + v_l

        a = (q @ k.transpose(-2, -1)) * attn.scale
        a = a.softmax(dim=-1)
        a = attn.attn_drop(a)
        out = (a @ v).transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    def _forward_with_lora_layers(self, backbone: nn.Module, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        # Apply expert-specific LoRA to first K blocks only; others unchanged
        for idx, block in enumerate(backbone.blocks):
            if idx < self.num_lora_layers:
                x_norm = block.norm1(x)
                attn_out = self._attn_with_lora(block.attn, x_norm, self.lora_k[idx][expert_id], self.lora_v[idx][expert_id])
                attn_out = block.ls1(attn_out)
                attn_out = block.drop_path1(attn_out)
                x = x + attn_out
                mlp_out = block.ls2(block.mlp(block.norm2(x)))
                mlp_out = block.drop_path2(mlp_out)
                x = x + mlp_out
            else:
                x = block(x)
        x = backbone.norm(x)
        return x[:, 0]

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        # Standard ViT embedding (no prompts)
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(x + backbone.pos_embed)

        valid_mask = expert_ids >= 0
        if valid_mask.all():
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            for eid in expert_ids.unique().tolist():
                idxs = (expert_ids == eid).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                cls_feats = self._forward_with_lora_layers(backbone, x[idxs], int(eid))
                output[idxs] = cls_feats
            return output
        elif not valid_mask.any():
            x = backbone.blocks(x)
            x = backbone.norm(x)
            return x[:, 0]
        else:
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            invalid_idx = (~valid_mask).nonzero(as_tuple=True)[0]
            if invalid_idx.numel() > 0:
                xi = x[invalid_idx]
                xi = backbone.blocks(xi)
                xi = backbone.norm(xi)
                output[invalid_idx] = xi[:, 0]

            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            valid_ids = expert_ids[valid_idx]
            for eid in valid_ids.unique().tolist():
                idxs = valid_idx[valid_ids == eid]
                if idxs.numel() == 0:
                    continue
                cls_feats = self._forward_with_lora_layers(backbone, x[idxs], int(eid))
                output[idxs] = cls_feats
            return output

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        # Mean init from previous experts (matching PromptExpert policy)
        if expert_id == 0 or expert_id >= self.num_experts:
            return
        assert expert_id >= 1
        for layer in range(self.num_lora_layers):
            prev_As = torch.stack([self.lora_k[layer][e].lora_A.data for e in range(expert_id)], dim=0)
            prev_Bs = torch.stack([self.lora_k[layer][e].lora_B.data for e in range(expert_id)], dim=0)
            self.lora_k[layer][expert_id].lora_A.data.copy_(prev_As.mean(dim=0))
            self.lora_k[layer][expert_id].lora_B.data.copy_(prev_Bs.mean(dim=0))

            prev_As = torch.stack([self.lora_v[layer][e].lora_A.data for e in range(expert_id)], dim=0)
            prev_Bs = torch.stack([self.lora_v[layer][e].lora_B.data for e in range(expert_id)], dim=0)
            self.lora_v[layer][expert_id].lora_A.data.copy_(prev_As.mean(dim=0))
            self.lora_v[layer][expert_id].lora_B.data.copy_(prev_Bs.mean(dim=0))
