import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class Prompt(nn.Module):
    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768,
                 pos_prompt: Iterable[int] = (0, 1, 2, 3, 4)):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim

        self.register_buffer('pos_prompt', torch.tensor(list(pos_prompt), dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())

        self.prompts = nn.Parameter(
            torch.empty(self.num_layers, num_experts, len_prompt, embed_dim)
        )
        nn.init.uniform_(self.prompts)

    def _build_batched_prompts(self, backbone: nn.Module, expert_ids: torch.Tensor) -> torch.Tensor:
        B = expert_ids.size(0)
        prompts = []
        for l_idx in range(self.num_layers):
            p_l = self.prompts[l_idx][expert_ids.long()]  # [B, len_prompt, D]
            prompts.append(p_l)
        prompts = torch.stack(prompts, dim=1)  # [B, num_layers, len_prompt, D]

        D = prompts.size(-1)
        pos_bias = backbone.pos_embed[:, :1, :].unsqueeze(1).expand(B, self.num_layers, self.len_prompt, D)
        prompts = prompts + pos_bias
        return prompts

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)
        orig_N = x.size(1)

        prompts = self._build_batched_prompts(backbone, expert_ids)  # [B, num_layers, len_prompt, D]

        for n, block in enumerate(backbone.blocks):
            pos_n = (self.pos_prompt.eq(n)).nonzero(as_tuple=False).squeeze()
            if pos_n.numel() != 0:
                x = torch.cat((x, prompts[:, pos_n]), dim=1)
            x = block(x)
            x = x[:, :orig_N, :]

        x = backbone.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0 or expert_id >= self.num_experts:
            return
        prev_experts = self.prompts[:, :expert_id].clone()  # [num_layers, expert_id, L, D]
        prev_experts_mean = prev_experts.mean(dim=1)        # [num_layers, L, D]
        self.prompts.data[:, expert_id] = prev_experts_mean


class RPFC(nn.Module):
    def __init__(self,
                 M            : int,
                 ridge        : float = 1e4,
                 embed_dim    : int = 768,
                 num_classes  : int = 100,
                 **kwargs):

        super().__init__()
        
        self.ridge = ridge
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if M == 0:
            self.M = embed_dim
            self.use_rp = False
            self.register_buffer('W_rand', torch.empty(0))
            self.register_buffer('Q', torch.zeros(embed_dim, num_classes))
            self.register_buffer('G', torch.zeros(embed_dim, embed_dim))
        else:
            self.M = M
            self.use_rp = True
            self.register_buffer('W_rand', torch.randn(embed_dim, M))
            self.register_buffer('Q', torch.zeros(M, num_classes))
            self.register_buffer('G', torch.zeros(M, M))

        self.fc = nn.Linear(self.M, num_classes, bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def target2onehot(self, targets):
        device = targets.device
        onehot = torch.zeros(targets.size(0), self.num_classes, device=device)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        return onehot

    def collect(self, features, labels):
        features = features.detach()
        labels = labels.detach()

        if self.use_rp:
            features_h = F.relu(features @ self.W_rand)
        else:
            features_h = features
        Y = self.target2onehot(labels)
        self.Q = self.Q + features_h.T @ Y
        self.G = self.G + features_h.T @ features_h

    def update(self):
        device = self.fc.weight.device
        Wo = torch.linalg.solve(self.G + self.ridge * torch.eye(self.M, device=device), self.Q).T
        self.fc.weight.data = Wo.to(device)

    def forward(self, x):
        if self.use_rp:
            x = F.relu(x @ self.W_rand)
        x = self.fc(x)
        return x


class WhitenedSubspaceHead(nn.Module):
    """Whitened-subspace routing head — drop-in replacement for RPFC.

    For each task t we store per-task statistics (mu, var, Bw) fitted from
    buffered CLS features.  At inference the augmented residual score
        e_t(r) = 1 - ||Bw^T (r * w)||^2 / (||r * w||^2 + eps)
    is computed for every fitted task; lower e means better match.
    ``forward`` returns ``-e`` so that the interface matches RPFC (higher = better).

    Reference: TaskWhitenedSubspaceRouter in CLEGO/skill_benchmark/task_router.py
    """

    def __init__(self,
                 embed_dim    : int = 768,
                 num_classes  : int = 10,
                 k            : int = 32,
                 eps          : float = 1e-6,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.k = k
        self.eps = eps

        self._feat_buffers: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._stats: Dict[int, dict] = {}

        for param in self.parameters():
            param.requires_grad = False

    # ----- collect / update / forward (same signature as RPFC) -----

    @torch.no_grad()
    def collect(self, features: torch.Tensor, labels: torch.Tensor):
        features = features.detach().cpu()
        labels = labels.detach().cpu()
        for feat, lab in zip(features, labels):
            tid = int(lab.item())
            self._feat_buffers[tid].append(feat)

    @torch.no_grad()
    def update(self):
        for tid, feat_list in self._feat_buffers.items():
            if len(feat_list) < 2:
                continue
            self._fit_task(tid, feat_list)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        out = torch.full((B, self.num_classes), float("-inf"), device=device)
        eps = self.eps

        for tid, st in self._stats.items():
            if tid >= self.num_classes:
                continue
            mu = st["mu"].to(device)
            var = st["var"].to(device)
            Bw = st["Bw"].to(device)

            w = 1.0 / torch.sqrt(var.clamp(min=0.0) + eps)
            xw = x * w.unsqueeze(0)
            xw_sq = (xw * xw).sum(dim=1)
            proj = xw @ Bw
            proj_sq = (proj * proj).sum(dim=1)
            residual = 1.0 - proj_sq / (xw_sq + eps)
            out[:, tid] = -residual

        return out

    # ----- internal fitting -----

    def _fit_task(self, tid: int, feat_list: List[torch.Tensor]):
        R = torch.stack(feat_list, dim=0)  # [N, d]
        N, d = R.shape
        k = min(self.k, d - 1)
        eps = self.eps

        mu = R.mean(dim=0)
        var = (R * R).mean(dim=0) - mu * mu
        var = var.clamp(min=0.0)

        w = 1.0 / torch.sqrt(var + eps)
        Z = (R - mu.unsqueeze(0)) * w.unsqueeze(0)
        Z64 = Z.to(torch.float64)
        cov = (Z64.T @ Z64) / float(max(N - 1, 1))
        cov_np = cov.to(torch.float32).numpy()

        eig, V = np.linalg.eigh(cov_np)
        order = np.argsort(eig)[::-1][:k]
        Uw = V[:, order].astype(np.float32, copy=False)

        mw = mu * w
        mw_norm = mw.norm(p=2).clamp(min=eps)
        mw = mw / mw_norm
        mw_np = mw.numpy().astype(np.float32, copy=False)

        A = np.concatenate([mw_np[:, None], Uw], axis=1)  # [d, 1+k]
        Bw, _ = np.linalg.qr(A)
        Bw = Bw.astype(np.float32, copy=False)

        self._stats[tid] = {
            "mu": torch.from_numpy(mu.numpy().copy()),
            "var": torch.from_numpy(var.numpy().copy()),
            "Bw": torch.from_numpy(Bw.copy()),
        }


class FlyPrompt(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 len_prompt     : int   = 20,
                 pos_prompt     : Iterable[int] = (0, 1, 2, 3, 4),
                 rp_dim         : int   = 10000,
                 rp_ridge       : float = 1e4,
                 ema_ratio      : Iterable[float] = (0.9, 0.99),
                 router_type    : str   = "rpfc",
                 ws_k           : int   = 32,
                 **kwargs):

        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt
        self.rp_dim = rp_dim
        self.rp_ridge = rp_ridge
        self.ema_ratio = ema_ratio
        self.num_ema = len(ema_ratio)
        self.router_type = router_type

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Expert prompts
        self.experts = Prompt(
            num_experts = self.task_num,
            len_prompt = self.len_prompt,
            embed_dim = self.embed_dim,
            pos_prompt = self.pos_prompt,
        )

        # Expert FCs
        self.experts_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.num_ema)
            ]) for _ in range(self.task_num)
        ])
        for expert_fc in self.experts_fc:
            for fc in expert_fc:
                for param in fc.parameters():
                    param.requires_grad = False
        self.init_fc(expert_id = 0)

        # Routing head (RPFC or Whitened Subspace)
        if self.router_type == "ws":
            logger.info(f'Using WhitenedSubspaceHead router (k={ws_k})')
            self.rp_head = WhitenedSubspaceHead(
                embed_dim = self.embed_dim,
                num_classes = self.task_num,
                k = ws_k,
            )
        else:
            logger.info(f'Using RPFC router (dim={rp_dim}, ridge={rp_ridge})')
            self.rp_head = RPFC(
                M = self.rp_dim,
                ridge = self.rp_ridge,
                embed_dim = self.embed_dim,
                num_classes = self.task_num,
            )

    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x
    
    def forward_with_rp(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.forward_features(inputs)
        x = x[:, 0]
        x = self.rp_head(x)
        return x
    
    def forward_with_ema(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        outputs_ls = []

        # online head
        outputs_ls.append(self.backbone.fc(x))
        
        # ema head
        for i in range(self.num_ema):
            outputs = []
            for x_i, e_i in zip(x, expert_ids):
                outputs.append(self.experts_fc[e_i.item()][i](x_i))
            outputs = torch.stack(outputs, dim=0)
            outputs_ls.append(outputs)

        return outputs_ls
    
    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        features = self.backbone.forward_features(inputs)
        features = features[:, 0]
        labels = torch.full((labels.size(0),), self.task_count, device=labels.device, dtype=torch.long)
        self.rp_head.collect(features, labels)

    def update(self):
        self.rp_head.update()

    @torch.no_grad()
    def init_fc(self, expert_id: int = None):
        if expert_id is None:
            expert_id = self.task_count
        if expert_id >= self.task_num:
            return
        w, b = self.backbone.fc.weight.data, self.backbone.fc.bias.data
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int = None):
        if expert_id is None:
            expert_id = self.task_count
        for i in range(self.num_ema):
            ema_ratio = self.ema_ratio[i]
            online_w = self.backbone.fc.weight.data
            online_b = self.backbone.fc.bias.data
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data
            ema_w.mul_(ema_ratio).add_(online_w, alpha=1.0 - ema_ratio)
            ema_b.mul_(ema_ratio).add_(online_b, alpha=1.0 - ema_ratio)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1
        self.rp_head.update()
        self.experts.init_new_expert(self.task_count)
        self.init_fc(self.task_count)