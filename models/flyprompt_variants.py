import logging
from typing import Optional

import torch
import torch.nn as nn

from .experts import LoRAExpert
from .flyprompt import FlyPrompt
from .ranpac import Adapter

logger = logging.getLogger()


class FlyAdapterExpert(nn.Module):
    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        num_adapter_layers: int = 5,
        adapter_down_dim: int = 10,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_adapter_layers = num_adapter_layers
        self.adapter_down_dim = adapter_down_dim
        self.adapters = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Adapter(adapter_down_dim, embed_dim, dropout=adapter_dropout)
                        for _ in range(num_experts)
                    ]
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def _forward_with_adapter_layers(self, backbone: nn.Module, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        for idx, block in enumerate(backbone.blocks):
            if idx < self.num_adapter_layers:
                x_norm = block.norm1(x)
                attn_out = block.attn(x_norm)
                attn_out = block.ls1(attn_out)
                attn_out = block.drop_path1(attn_out)
                x = x + attn_out

                residual = x
                adapt_x = self.adapters[idx][expert_id](x)
                mlp_out = block.mlp(block.norm2(x))
                mlp_out = block.ls2(mlp_out)
                mlp_out = block.drop_path2(mlp_out)
                x = residual + adapt_x + mlp_out
            else:
                x = block(x)
        x = backbone.norm(x)
        return x[:, 0]

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, _, D = x.size()
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
                output[idxs] = self._forward_with_adapter_layers(backbone, x[idxs], int(eid))
            return output

        if not valid_mask.any():
            x = backbone.blocks(x)
            x = backbone.norm(x)
            return x[:, 0]

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
            output[idxs] = self._forward_with_adapter_layers(backbone, x[idxs], int(eid))
        return output

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0 or expert_id >= self.num_experts:
            return
        for layer in range(self.num_adapter_layers):
            target_adapter = self.adapters[layer][expert_id]
            prev_adapters = self.adapters[layer][:expert_id]
            target_state = target_adapter.state_dict()
            for key in target_state:
                values = torch.stack([adapter.state_dict()[key] for adapter in prev_adapters], dim=0)
                target_state[key].copy_(values.mean(dim=0))
            target_adapter.load_state_dict(target_state)


class _FlyExpertMixin:
    expert_type: str

    def _load_expert_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected {self.expert_type} checkpoint dict, got raw tensor/module: {checkpoint_path}")
        if "prompts" in checkpoint and "expert_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is a FlyPrompt prompt checkpoint, not a {self.expert_type} expert checkpoint."
            )
        ckpt_type = checkpoint.get("expert_type", None)
        if ckpt_type != self.expert_type:
            raise ValueError(f"Expert checkpoint type mismatch: checkpoint={ckpt_type}, model={self.expert_type}")
        if checkpoint.get("embed_dim", self.embed_dim) != self.embed_dim:
            raise ValueError(f"Expert checkpoint embed_dim mismatch: checkpoint={checkpoint.get('embed_dim')}, model={self.embed_dim}")

        state = checkpoint.get("expert_state_dict", None)
        if state is None:
            raise ValueError(f"No expert_state_dict found in checkpoint: {checkpoint_path}")
        ckpt_experts = int(checkpoint.get("num_experts", 1))
        if ckpt_experts == self.task_num:
            self.experts.load_state_dict(state, strict=True)
        elif ckpt_experts == 1:
            self._load_single_expert_state(state)
        elif ckpt_experts < self.task_num:
            self._load_partial_expert_state(state, ckpt_experts)
            for expert_id in range(ckpt_experts, self.task_num):
                self.experts.init_new_expert(expert_id)
        else:
            self._load_partial_expert_state(state, self.task_num)
        logger.info("Loaded %s expert checkpoint from %s", self.expert_type, checkpoint_path)

    def _load_single_expert_state(self, state: dict) -> None:
        self._load_partial_expert_state(state, 1)
        for expert_id in range(1, self.task_num):
            self._copy_expert_parameters(src_expert=0, dst_expert=expert_id)

    def _load_partial_expert_state(self, state: dict, num_experts_to_load: int) -> None:
        own_state = self.experts.state_dict()
        for key, value in state.items():
            if self._get_expert_index(key) is None or self._get_expert_index(key) >= num_experts_to_load:
                continue
            if key not in own_state:
                continue
            if tuple(value.shape) != tuple(own_state[key].shape):
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} vs model {tuple(own_state[key].shape)}"
                )
            own_state[key].copy_(value)
        self.experts.load_state_dict(own_state, strict=True)

    def _copy_expert_parameters(self, src_expert: int, dst_expert: int) -> None:
        own_state = self.experts.state_dict()
        for key in list(own_state.keys()):
            if self._get_expert_index(key) != dst_expert:
                continue
            src_key = self._replace_expert_index(key, src_expert)
            if src_key in own_state:
                own_state[key].copy_(own_state[src_key])
        self.experts.load_state_dict(own_state, strict=True)

    def _get_expert_index(self, key: str) -> Optional[int]:
        raise NotImplementedError

    def _replace_expert_index(self, key: str, expert_id: int) -> str:
        raise NotImplementedError

    def load_prompt(self, load_pt: bool = False, prompt_path: str = None):
        if not load_pt:
            return
        if prompt_path is None:
            raise ValueError(f"prompt_path must be specified when load_pt=True for {self.expert_type}.")
        self._load_expert_checkpoint(prompt_path)


class FlyAdapter(_FlyExpertMixin, FlyPrompt):
    expert_type = "adapter"

    def __init__(self, *args, load_pt: bool = False, flyprompt_pt_path: str = "./checkpoints/flyprompt_misa_prompt.pt", **kwargs):
        self.fly_adapter_layers = int(kwargs.get("fly_adapter_layers", 5))
        self.fly_adapter_down_dim = int(kwargs.get("fly_adapter_down_dim", 10))
        super().__init__(*args, load_pt=False, flyprompt_pt_path=flyprompt_pt_path, **kwargs)
        self.experts = FlyAdapterExpert(
            num_experts=self.task_num,
            embed_dim=self.embed_dim,
            num_adapter_layers=self.fly_adapter_layers,
            adapter_down_dim=self.fly_adapter_down_dim,
        ).to(self.backbone.fc.weight.device)
        self.load_prompt(load_pt=load_pt, prompt_path=flyprompt_pt_path)

    def _get_expert_index(self, key: str) -> Optional[int]:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "adapters" and parts[2].isdigit():
            return int(parts[2])
        return None

    def _replace_expert_index(self, key: str, expert_id: int) -> str:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "adapters" and parts[2].isdigit():
            parts[2] = str(expert_id)
            return ".".join(parts)
        return key


class FlyLoRA(_FlyExpertMixin, FlyPrompt):
    expert_type = "lora"

    def __init__(self, *args, load_pt: bool = False, flyprompt_pt_path: str = "./checkpoints/flyprompt_misa_prompt.pt", **kwargs):
        self.fly_lora_layers = int(kwargs.get("fly_lora_layers", 5))
        self.fly_lora_rank = int(kwargs.get("fly_lora_rank", 5))
        self.fly_lora_alpha = float(kwargs.get("fly_lora_alpha", 1.0))
        super().__init__(*args, load_pt=False, flyprompt_pt_path=flyprompt_pt_path, **kwargs)
        self.experts = LoRAExpert(
            num_experts=self.task_num,
            embed_dim=self.embed_dim,
            num_lora_layers=self.fly_lora_layers,
            lora_rank=self.fly_lora_rank,
            lora_alpha=self.fly_lora_alpha,
        ).to(self.backbone.fc.weight.device)
        self.load_prompt(load_pt=load_pt, prompt_path=flyprompt_pt_path)

    def _get_expert_index(self, key: str) -> Optional[int]:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] in {"lora_k", "lora_v"} and parts[2].isdigit():
            return int(parts[2])
        return None

    def _replace_expert_index(self, key: str, expert_id: int) -> str:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] in {"lora_k", "lora_v"} and parts[2].isdigit():
            parts[2] = str(expert_id)
            return ".".join(parts)
        return key
