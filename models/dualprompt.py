import logging
from typing import Iterable

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.l2p import Prompt



def _stable_cholesky(matrix: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.cholesky(matrix + reg * eye)


def _transform_to_target_covariance(features: torch.Tensor,
                                     target_cov: torch.Tensor,
                                     reg: float = 1e-4) -> torch.Tensor:
    """Align feature covariance to target_cov using a linear transform.

    features: [B, D]
    target_cov: [D, D]
    """
    if features.size(0) <= 1:
        # Not enough samples to estimate covariance; skip calibration.
        return features

    orig_dtype = features.dtype
    # Compute covariance and transform in float32 for numerical stability
    features_f = features.to(dtype=torch.float32)
    target_cov_f = target_cov.to(device=features.device, dtype=torch.float32)

    centered = features_f - features_f.mean(dim=0, keepdim=True)
    n = centered.size(0)
    C = centered.T @ centered / (n - 1)
    L = _stable_cholesky(C, reg)
    L_target = _stable_cholesky(target_cov_f, reg)
    A = torch.linalg.solve(L, L_target)
    Fj = centered @ A
    return Fj.to(dtype=orig_dtype)


logger = logging.getLogger()


class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 g_pool         : int   = 1,
                 e_pool         : int   = 10,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 load_pt        : bool  = False,
                 **kwargs):
        super().__init__()

        self.lambd = lambd
        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes

        # MEPO configuration
        self.mepo_backbone_path = self.kwargs.get("mepo_backbone_path", None)
        self.cov_path = self.kwargs.get("cov_path", None)
        self.cov_coef = float(self.kwargs.get("cov_coef", 0.7))
        # Enforce cov_coef in [0, 1]
        self.cov_coef = max(0.0, min(1.0, self.cov_coef))

        # Require both MEPO paths to be specified together, or neither
        if (self.mepo_backbone_path is None) != (self.cov_path is None):
            raise ValueError(
                "For MEPO, both mepo_backbone_path and cov_path must be provided; "
                "set both or leave both as None for plain DualPrompt."
            )

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))

        # Optionally override backbone weights with MEPO checkpoint (without loading fc/head)
        if self.mepo_backbone_path is not None:
            logger.info(f"Loading MEPO backbone from {self.mepo_backbone_path}")
            self._load_mepo_backbone(self.mepo_backbone_path)

        # Freeze backbone except the final classifier head
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True


        # Optional EMA head bank for ensembling (online + EMA heads)
        self.use_ema_head = bool(self.kwargs.get("use_ema_head", False))
        # Accept ema_ratio from kwargs if provided; otherwise default to (0.9, 0.99)
        ema_ratio_cfg = self.kwargs.get("ema_ratio", (0.9, 0.99))
        try:
            self.ema_ratio = tuple(float(r) for r in ema_ratio_cfg)
        except Exception:
            self.ema_ratio = (0.9, 0.99)
        self.num_ema = len(self.ema_ratio)
        if self.use_ema_head and self.num_ema > 0:
            self.ema_heads = nn.ModuleList([
                nn.Linear(self.backbone.num_features, self.num_classes, bias=True)
                for _ in range(self.num_ema)
            ])
            for head in self.ema_heads:
                for p in head.parameters():
                    p.requires_grad = False
            self._init_ema_heads()
        else:
            self.ema_heads = nn.ModuleList()

        # Optionally load covariance matrix for MEPO calibration
        if self.cov_path is not None:
            self._load_mepo_covariance(self.cov_path)

        # Slice the eprompt
        # We fix the base expert-prompt pool size to 10 for up to 10 tasks,
        # and use 20 when there are 20 tasks. This keeps compatibility with
        # existing checkpoints trained with e_pool=10 while allowing a larger
        # pool when task_num=20.
        if self.task_num <= 10:
            self.e_pool = 10
        elif self.task_num == 20:
            self.e_pool = 20
        else:
            raise ValueError(f"Unsupported task_num={self.task_num} for DualPrompt; only <=10 or 20 are supported.")

        assert self.e_pool >= self.task_num, "e_pool must be at least as large as task_num"
        self.num_pt_per_task = int(self.e_pool / self.task_num)
        assert self.num_pt_per_task > 0, "Each task must get at least one prompt slot"

        self.len_g_prompt = len_g_prompt if not load_pt else 10
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.ones(1).view(1))

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                self.e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                self.e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        self.g_prompt.key = None

        self.load_prompt(load_pt)

    def load_prompt(self, load_pt: bool = False,):
        g_path = "./checkpoints/g_prompt.pt"
        e_path = "./checkpoints/e_prompt.pt"
        if load_pt:
            logger.info(f"load prompt from {g_path} and {e_path}")
            g_prompt = torch.load(g_path)
            e_prompt = torch.load(e_path)

            # Load global prompts as-is
            self.g_prompt.prompts = nn.Parameter(g_prompt.detach().clone())

            # e_prompt checkpoints are assumed to be saved with pool_size=10.
            # If the current model uses a larger pool (e.g., e_pool=20 for
            # task_num=20), we tile the loaded prompts along the pool
            # dimension to match self.e_pool.
            e_prompt = e_prompt.detach().clone()
            orig_pool = e_prompt.size(0)
            if self.e_pool == orig_pool:
                expanded = e_prompt
            elif self.e_pool > orig_pool:
                repeat_factor = self.e_pool // orig_pool
                if self.e_pool % orig_pool != 0:
                    raise ValueError(
                        f"Cannot expand e_prompt from pool_size={orig_pool} to {self.e_pool} (non-integer repeat)."
                    )
                expanded = e_prompt.repeat(repeat_factor, 1, 1)
            else:
                # If desired pool is smaller than checkpoint pool, truncate.
                expanded = e_prompt[: self.e_pool]

            self.e_prompt.prompts = nn.Parameter(expanded)

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x

    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat(xk, (g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat(xk, (e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (e_prompt[:, pos_e * 2 + 1]), dim = 1)

            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias

            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x

    def forward(self, inputs: torch.Tensor, return_feat: bool = False):
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0]
            g_p = g_p.expand(B, -1, -1)
        else:
            g_p = None

        if self.e_prompt is not None:
            start_id = self.task_count * self.num_pt_per_task
            end_id = (self.task_count + 1) * self.num_pt_per_task
            if self.training and start_id < self.e_pool:
                res_e = self.e_prompt(query, s=start_id, e=end_id)
            else:
                res_e = self.e_prompt(query)
            e_s, e_p = res_e

        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]

        # Apply MEPO covariance calibration only after prompts and transformer
        cls_token = self._apply_mepo_cov_calibration(cls_token)

        x = self.backbone.fc(cls_token)

        # keep similarity for compatibility
        if isinstance(e_s, torch.Tensor):
            self.similarity = e_s.mean()
        else:
            self.similarity = torch.tensor(0., device=x.device)

        if return_feat:
            return x, cls_token
        else:
            return x

    @torch.no_grad()
    def _init_ema_heads(self) -> None:
        """Initialize EMA heads to match the online classifier head."""
        if not getattr(self, "use_ema_head", False):
            return
        if not hasattr(self, "ema_heads") or len(self.ema_heads) == 0:
            return
        w = self.backbone.fc.weight.data
        b = self.backbone.fc.bias.data
        for head in self.ema_heads:
            head.weight.data.copy_(w)
            head.bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self) -> None:
        """Momentum-update EMA heads from the online classifier head."""
        if not getattr(self, "use_ema_head", False):
            return
        if not hasattr(self, "ema_heads") or len(self.ema_heads) == 0:
            return
        online_w = self.backbone.fc.weight.data
        online_b = self.backbone.fc.bias.data
        for i, head in enumerate(self.ema_heads):
            m = float(self.ema_ratio[i])
            head.weight.data.mul_(m).add_(online_w, alpha=1.0 - m)
            head.bias.data.mul_(m).add_(online_b, alpha=1.0 - m)

    def forward_with_ema(self, inputs: torch.Tensor, **kwargs):
        """Forward pass returning a list of logits from [online, *EMA heads]."""
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, _, _ = x.size()
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0].expand(B, -1, -1)
        else:
            g_p = None

        if self.e_prompt is not None:
            start_id = self.task_count * self.num_pt_per_task
            end_id = (self.task_count + 1) * self.num_pt_per_task
            if self.training and start_id < self.e_pool:
                res_e = self.e_prompt(query, s=start_id, e=end_id)
            else:
                res_e = self.e_prompt(query)
            e_s, e_p = res_e
        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        cls_token = self._apply_mepo_cov_calibration(cls_token)

        outputs_ls = [self.backbone.fc(cls_token)]
        if getattr(self, "use_ema_head", False) and hasattr(self, "ema_heads") and len(self.ema_heads) > 0:
            for head in self.ema_heads:
                outputs_ls.append(head(cls_token))

        # keep similarity for compatibility
        if isinstance(e_s, torch.Tensor):
            self.similarity = e_s.mean()
        else:
            self.similarity = torch.tensor(0., device=cls_token.device)

        return outputs_ls

    def _apply_mepo_cov_calibration(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Apply MEPO covariance calibration to CLS token if enabled.

        This uses the batch CLS features to estimate current covariance and
        aligns it to the target covariance matrix loaded from cov_path.
        """
        if getattr(self, "cov_matrix", None) is None:
            return cls_token

        # Run MEPO calibration in full precision regardless of outer AMP context
        with torch.cuda.amp.autocast(enabled=False):
            cls_fp32 = cls_token.to(dtype=torch.float32)
            cov = self.cov_matrix.to(device=cls_fp32.device, dtype=torch.float32)
            Fj = _transform_to_target_covariance(cls_fp32, cov)
            # Normalize to unit norm to avoid scale explosion
            norm = Fj.norm(dim=1, keepdim=True).clamp_min(1e-6)
            Fj = Fj / norm

            out = (1.0 - float(self.cov_coef)) * cls_fp32 + float(self.cov_coef) * Fj

        return out.to(dtype=cls_token.dtype)

    def _load_mepo_backbone(self, ckpt_path: str) -> None:
        """Load MEPO backbone weights from a checkpoint without loading fc/head.

        The provided meta_epoch_*.pth checkpoints are plain state_dicts with
        ViT backbone weights (cls_token, pos_embed, patch_embed, blocks, norm).
        We also defensively drop any keys that look like classifier heads.
        """
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            state_dict = state
        else:
            state_dict = state

        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip common wrappers if ever present
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            # Do not load classifier heads
            if k.startswith("fc.") or k.startswith("head."):
                continue
            new_state_dict[k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"[MEPO] Missing keys when loading backbone from {ckpt_path}: {missing}")
        if unexpected:
            logger.warning(f"[MEPO] Unexpected keys when loading backbone from {ckpt_path}: {unexpected}")

    def _load_mepo_covariance(self, cov_path: str) -> None:
        """Load covariance matrix from .npy and register as buffer."""
        cov = np.load(cov_path)
        cov = torch.from_numpy(cov).float()
        if cov.dim() != 2 or cov.size(0) != cov.size(1):
            raise ValueError(f"Covariance matrix at {cov_path} must be square, got {cov.shape}")
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            # Fallback: infer from cls_token dimension at runtime
            feat_dim = cov.size(0)
        if cov.size(0) != feat_dim:
            raise ValueError(
                f"Covariance dim {cov.size(0)} does not match backbone features {feat_dim}"
            )
        self.register_buffer("cov_matrix", cov)
        logger.info(f"[MEPO] Loaded covariance matrix from {cov_path} with shape {cov.shape}.")




    def forward_with_task(self, inputs: torch.Tensor, task_id: int, return_feat: bool = False) -> torch.Tensor:
        """Forward pass using prompts corresponding to a specific task id.

        This is used for oracle evaluation where we explicitly choose the
        prompt segment for task `task_id` instead of routing.
        """
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0].expand(B, -1, -1)
        else:
            g_p = None

        if self.e_prompt is not None:
            start_id = task_id * self.num_pt_per_task
            end_id = (task_id + 1) * self.num_pt_per_task
            start_id = max(0, min(start_id, self.e_pool))
            end_id = max(start_id, min(end_id, self.e_pool))
            if start_id < end_id:
                e_s, e_p = self.e_prompt(query, s=start_id, e=end_id)
            else:
                e_p = None
                e_s = 0
        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]

        # Apply MEPO covariance calibration in oracle forward as well
        cls_token = self._apply_mepo_cov_calibration(cls_token)

        logits = self.backbone.fc(cls_token)

        # keep similarity for compatibility, although gradients are not used here
        if isinstance(e_s, torch.Tensor):
            self.similarity = e_s.mean()
        else:
            self.similarity = torch.tensor(0., device=logits.device)

        if return_feat:
            return logits, cls_token
        else:
            return logits


    def get_e_prompt_count(self):
        return self.e_prompt.update()

    def process_task_count(self):
        self.task_count += 1

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.lambd * self.similarity