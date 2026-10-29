"""SinglePrompt model adapter for FlyGCL, isolated from shared ViT/Trainers.

Reference: efficient-learning-lab/SinglePrompt, upstream models/singlePrompt.py.
The existing FlyGCL ViT provides the pretrained weights and frozen blocks.
Only prompted attention calls use the upstream-equivalent K/V prefix path.
No routing, task-id-specific prompts, or implicit expert expansion.
"""
import torch
from torch import nn
from torch.nn import functional as F



class SinglePrompt(nn.Module):
    """Official buffer-free SinglePrompt model (Stage 2: model only)."""

    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=100,
        task_num=None,
        pos_prompt=None,
        len_prompt=20,
        singleprompt_logit_type="cos_sim",
        **kwargs,
    ):
        super().__init__()
        if backbone_name != "vit_base_patch16_224":
            raise ValueError("Official SinglePrompt requires vit_base_patch16_224")
        positions = [0, 1, 2, 3, 4] if pos_prompt is None else list(pos_prompt)
        if positions != list(range(len(positions))) or not positions or len(positions) > 12:
            raise ValueError("Only upstream-compatible prefix layers [0, ..., N-1] are supported")
        if not isinstance(len_prompt, int) or len_prompt <= 0:
            raise ValueError("len_prompt must be a positive integer")
        if singleprompt_logit_type not in ("linear", "cos_sim"):
            raise ValueError("singleprompt_logit_type must be linear or cos_sim")

        # Import after model registry initialization to avoid a circular import.
        from models import vit as shared_vit

        # Reuse FlyGCL's existing pretrained ViT weight loader, not its
        # task-aware prompt-pool wrappers. All backbone parameters are frozen.
        self.backbone = shared_vit.vit_base_patch16_224(
            pretrained=pretrained, num_classes=num_classes
        )
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        dim = self.backbone.embed_dim
        self.prompt_length = len_prompt
        self.logit_type = singleprompt_logit_type
        self.register_buffer("pos_prompt", torch.tensor(positions, dtype=torch.long))
        self.prompt = nn.Parameter(torch.empty(len(positions) * 2 * len_prompt, dim))
        nn.init.uniform_(self.prompt, -1.0, 1.0)

        if self.logit_type == "cos_sim":
            self.fc = nn.Linear(dim, num_classes, bias=False)
        else:
            self.fc = nn.Linear(dim, num_classes)

    @staticmethod
    def _prefix_attention(attn, x, prompt):
        """Exact arithmetic/order of upstream PreAttention.forward."""
        batch, num_tokens, dim = x.shape
        heads = attn.num_heads
        qkv = attn.qkv(x).reshape(
            batch, num_tokens, 3, heads, dim // heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        length = prompt.size(2)
        prefix = prompt.view(batch, 2, length, heads, dim // heads)
        prefix = prefix.permute(1, 0, 3, 2, 4).contiguous()
        k = torch.cat([prefix[0], k], dim=2)
        v = torch.cat([prefix[1], v], dim=2)
        attention = (q @ k.transpose(-2, -1)) * attn.scale
        attention = attn.attn_drop(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(
            batch, num_tokens, dim
        )
        return attn.proj_drop(attn.proj(output))

    def forward_features(self, inputs):
        bb = self.backbone
        x = bb.patch_embed(inputs)
        x = torch.cat((bb.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = bb.pos_drop(x + bb.pos_embed)

        batch, _, dim = x.shape
        count = len(self.pos_prompt)
        prompt = self.prompt.expand(batch, -1, -1)
        prompt = prompt.contiguous().view(
            batch, count, 2, self.prompt_length, dim
        )
        prompt = prompt + bb.pos_embed[:, :1, :].unsqueeze(1).unsqueeze(1).expand(
            batch, count, 2, self.prompt_length, dim
        )

        for index, block in enumerate(bb.blocks):
            if index < count:
                attended = self._prefix_attention(
                    block.attn, block.norm1(x), prompt[:, index]
                )
                x = x + block.drop_path1(block.ls1(attended))
                x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            else:
                x = block(x)
        return bb.norm(x)[:, 0]

    def forward_head(self, features):
        if self.logit_type == "cos_sim":
            return (
                F.normalize(features, dim=1)
                @ F.normalize(self.fc.weight, dim=1).T
                / 0.1
            )
        return self.fc(features)

    def forward(self, inputs):
        return self.forward_head(self.forward_features(inputs))
