import logging
from typing import Iterable

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

        self.task_count = 0

        # Routing configuration
        self.routing_mode = self.kwargs.get("routing_mode", "rpfc")
        self.routing_mlp_hidden_dim = self.kwargs.get("routing_mlp_hidden_dim", 512)
        self.routing_mlp_dropout = self.kwargs.get("routing_mlp_dropout", 0.1)

        # KNN routing buffers
        self.knn_max_samples = 1000
        self.knn_num_centers = 5
        self.knn_max_iters = 10
        self.knn_current_features = []
        self.knn_current_count = 0
        self.knn_task_prototypes = {}

        # Gaussian Naive Bayes statistics (lazy initialization)
        self.nb_class_count = None
        self.nb_sum = None
        self.nb_sum_sq = None

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
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

        # Random projection head
        self.rp_head = RPFC(
            M = self.rp_dim,
            ridge = self.rp_ridge,
            embed_dim = self.embed_dim,
            num_classes = self.task_num,
        )

        # Two-layer MLP router for task inference (independent branch)
        if self.routing_mode == "mlp":
            rp_feature_dim = self.rp_head.M
            self.router_mlp = nn.Sequential(
                nn.Linear(rp_feature_dim, self.routing_mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.routing_mlp_dropout),
                nn.Linear(self.routing_mlp_hidden_dim, self.task_num),
            )
        else:
            self.router_mlp = None

    def _forward_backbone_cls(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(inputs)
        return x[:, 0]

    def _get_rp_features(self, cls_features: torch.Tensor) -> torch.Tensor:
        if getattr(self.rp_head, "use_rp", False):
            return F.relu(cls_features @ self.rp_head.W_rand)
        return cls_features

    def _update_knn_buffer(self, rp_features: torch.Tensor):
        if self.routing_mode != "knn" or self.knn_max_samples <= 0:
            return
        with torch.no_grad():
            feats = rp_features.detach().cpu()
            num_new = feats.size(0)
            if num_new == 0:
                return
            remaining = self.knn_max_samples - self.knn_current_count
            if remaining <= 0:
                return
            if num_new > remaining:
                idx = torch.randperm(num_new)[:remaining]
                feats = feats[idx]
                num_new = remaining
            self.knn_current_features.append(feats)
            self.knn_current_count += num_new

    @torch.no_grad()
    def _compute_knn_prototypes_from_current(self) -> torch.Tensor:
        if len(self.knn_current_features) == 0:
            return None
        feats = torch.cat(self.knn_current_features, dim=0)
        if feats.numel() == 0:
            return None
        # L2-normalize features
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
        if feats.size(0) <= self.knn_num_centers:
            centers = feats.clone()
        else:
            k = min(self.knn_num_centers, feats.size(0))
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=0).fit(feats.numpy())
                centers = torch.from_numpy(kmeans.cluster_centers_).float()
            except Exception:
                centers = self._torch_kmeans(feats, k, self.knn_max_iters)
        centers = centers / (centers.norm(dim=1, keepdim=True) + 1e-6)
        return centers

    def _torch_kmeans(self, feats: torch.Tensor, k: int, num_iters: int) -> torch.Tensor:
        # Simple PyTorch KMeans on CPU
        N, D = feats.shape
        device = feats.device
        indices = torch.randperm(N, device=device)[:k]
        centers = feats[indices].clone()
        for _ in range(num_iters):
            x_norm2 = (feats ** 2).sum(dim=1, keepdim=True)
            c_norm2 = (centers ** 2).sum(dim=1)
            dist2 = x_norm2 + c_norm2.unsqueeze(0) - 2 * feats @ centers.t()
            labels = dist2.argmin(dim=1)
            new_centers = []
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers.append(feats[mask].mean(dim=0))
                else:
                    new_centers.append(centers[j])
            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        return centers

    def _ensure_nb_stats_initialized(self, device: torch.device = None):
        if self.nb_sum is not None:
            return
        if device is None:
            device = self.rp_head.fc.weight.device
        rp_dim = self.rp_head.M
        self.nb_class_count = torch.zeros(self.task_num, device=device)
        self.nb_sum = torch.zeros(self.task_num, rp_dim, device=device)
        self.nb_sum_sq = torch.zeros(self.task_num, rp_dim, device=device)

    def _update_nb_stats(self, rp_features: torch.Tensor):
        if self.routing_mode != "nb":
            return
        with torch.no_grad():
            self._ensure_nb_stats_initialized(device=rp_features.device)
            x = rp_features.detach()
            t = self.task_count
            if t >= self.task_num:
                t = self.task_num - 1
            self.nb_class_count[t] += x.size(0)
            self.nb_sum[t] += x.sum(dim=0)
            self.nb_sum_sq[t] += (x * x).sum(dim=0)



    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x

    def forward_with_rp(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._forward_backbone_cls(inputs)
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

    def _route_knn(self, rp_features: torch.Tensor) -> torch.Tensor:
        # L2-normalize features
        if rp_features.dim() != 2:
            rp_features = rp_features.view(rp_features.size(0), -1)
        device = rp_features.device
        z = rp_features / (rp_features.norm(dim=1, keepdim=True) + 1e-6)

        num_seen = min(self.task_count + 1, self.task_num)
        task_ids = []
        centers_list = []

        for t in range(num_seen):
            centers = self.knn_task_prototypes.get(t, None)
            if centers is None and t == self.task_count:
                centers = self._compute_knn_prototypes_from_current()
            if centers is None:
                continue
            centers = centers.to(device)
            centers = centers / (centers.norm(dim=1, keepdim=True) + 1e-6)
            task_ids.append(t)
            centers_list.append(centers)

        if len(centers_list) == 0:
            # Fall back to random routing over seen tasks
            return torch.randint(0, num_seen, (rp_features.size(0),), device=device)

        dists = []
        for centers in centers_list:
            x_norm2 = (z ** 2).sum(dim=1, keepdim=True)
            c_norm2 = (centers ** 2).sum(dim=1)
            dist2 = x_norm2 + c_norm2.unsqueeze(0) - 2 * z @ centers.t()
            mean_dist = dist2.mean(dim=1, keepdim=True)
            dists.append(mean_dist)
        dists = torch.cat(dists, dim=1)  # [B, T_valid]

        min_idx = dists.argmin(dim=1)
        task_ids_tensor = torch.tensor(task_ids, device=device, dtype=torch.long)
        expert_ids = task_ids_tensor[min_idx]
        return expert_ids

    def _route_nb(self, rp_features: torch.Tensor) -> torch.Tensor:
        self._ensure_nb_stats_initialized(device=rp_features.device)
        x = rp_features
        device = self.nb_sum.device
        if x.device != device:
            x = x.to(device)

        num_seen = min(self.task_count + 1, self.task_num)
        counts = self.nb_class_count[:num_seen]
        valid_mask = counts > 0
        if not valid_mask.any():
            # Fall back to random routing over seen tasks
            return torch.randint(0, num_seen, (rp_features.size(0),), device=rp_features.device)

        counts_valid = counts[valid_mask]
        sum_valid = self.nb_sum[:num_seen][valid_mask]
        sum_sq_valid = self.nb_sum_sq[:num_seen][valid_mask]
        n = counts_valid.unsqueeze(1)  # [T_valid, 1]

        mean = sum_valid / (n + 1e-6)
        var = sum_sq_valid / (n + 1e-6) - mean * mean
        var = torch.clamp(var, min=1e-6)

        x2 = x.unsqueeze(1)  # [B, 1, D]
        mean2 = mean.unsqueeze(0)  # [1, T_valid, D]
        var2 = var.unsqueeze(0)    # [1, T_valid, D]

        log_prob = -0.5 * (torch.log(2 * torch.pi * var2) + (x2 - mean2) ** 2 / var2)
        log_prob = log_prob.sum(dim=-1)  # [B, T_valid]

        log_prior = torch.log(counts_valid / counts_valid.sum())
        log_post = log_prob + log_prior.unsqueeze(0)
        idx = torch.argmax(log_post, dim=1)

        valid_task_ids = torch.arange(num_seen, device=device)[valid_mask]
        routed_task_ids = valid_task_ids[idx]
        return routed_task_ids.to(rp_features.device)

    def _route_mlp(self, rp_features: torch.Tensor) -> torch.Tensor:
        if self.router_mlp is None:
            # Fall back to random routing over seen tasks
            num_seen = min(self.task_count + 1, self.task_num)
            return torch.randint(0, num_seen, (rp_features.size(0),), device=rp_features.device)
        logits = self.router_mlp(rp_features)
        expert_ids = torch.argmax(logits, dim=-1)
        return expert_ids

    def route_experts(self, inputs: torch.Tensor, end: bool = False) -> torch.Tensor:
        """Route each sample to a task expert according to routing_mode."""
        mode = getattr(self, "routing_mode", "rpfc")
        batch_size = inputs.size(0)

        # Uniform random routing over seen tasks
        if mode == "random":
            num_seen = min(self.task_count + 1, self.task_num)
            return torch.randint(0, num_seen, (batch_size,), device=inputs.device)

        # Extract CLS features once
        cls_features = self._forward_backbone_cls(inputs)

        if mode == "rpfc":
            logits = self.rp_head(cls_features)
            return torch.argmax(logits, dim=-1)

        # For other routers we reuse RP features
        rp_features = self._get_rp_features(cls_features)

        if mode == "knn":
            return self._route_knn(rp_features)
        if mode == "nb":
            return self._route_nb(rp_features)
        if mode == "mlp":
            return self._route_mlp(rp_features)

        # Default fallback
        logits = self.rp_head(cls_features)
        return torch.argmax(logits, dim=-1)


    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        features = self._forward_backbone_cls(inputs)
        labels = torch.full((labels.size(0),), self.task_count, device=labels.device, dtype=torch.long)
        self.rp_head.collect(features, labels)

        # Additional statistics for alternative routers
        if self.routing_mode in ("knn", "nb"):
            rp_features = self._get_rp_features(features)
            if self.routing_mode == "knn":
                self._update_knn_buffer(rp_features)
            if self.routing_mode == "nb":
                self._update_nb_stats(rp_features)

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
        # Finalize routing statistics for the just-finished task
        prev_task = self.task_count
        if self.routing_mode == "knn":
            centers = self._compute_knn_prototypes_from_current()
            if centers is not None:
                self.knn_task_prototypes[prev_task] = centers.cpu()
            self.knn_current_features = []
            self.knn_current_count = 0

        self.task_count += 1
        self.rp_head.update()
        self.experts.init_new_expert(self.task_count)
        self.init_fc(self.task_count)