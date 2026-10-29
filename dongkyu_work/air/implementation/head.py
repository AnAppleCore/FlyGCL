"""AIR GeneralizedARM adapted from ZHUANGHP/Analytic-continual-learning (MIT).

See ../original/SOURCE_MANIFEST.json and ../original/air/LICENSE.
Keep full per-class covariance statistics; no diagonal approximation.
"""
import math
import torch
from torch import nn


class AIRHead(nn.Module):
    def __init__(self, in_features, num_classes, gamma=0.1, device='cpu', max_stats_gib=64.0):
        super().__init__()
        if min(in_features, num_classes) <= 0 or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('AIR dimensions and gamma must be positive and finite')
        if not math.isfinite(max_stats_gib) or max_stats_gib <= 0:
            raise ValueError('air_max_stats_gib must be positive and finite')
        gib = num_classes * in_features * in_features * 8 / 2**30
        if gib > max_stats_gib:
            raise ValueError(f'AIR class statistics require {gib:.2f} GiB, above air_max_stats_gib={max_stats_gib}; reduce air_expansion_size or explicitly raise the limit')
        # Dense official AIR needs 100 * 8192^2 * 8 = 50 GiB just for A.
        # Fail early on machines with insufficient free RAM/VRAM instead of OOM-killing.
        work_gib = 4 * in_features * in_features * 8 / 2**30 + 2
        device = torch.device(device)
        available = self._host_available_bytes() if device.type == 'cpu' else (
            torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else None)
        if available is not None and gib + work_gib > available / 2**30:
            raise MemoryError(
                f'AIR {in_features}-D / {num_classes}-class statistics need approximately '
                f'{gib + work_gib:.1f} GiB free {device.type} memory '
                f'(currently {available / 2**30:.1f} GiB). '
                'Use a server with enough memory, or explicitly set '
                '--air_expansion_size 1024 for a non-official lightweight run.')
        self.gamma = gamma
        kw = dict(dtype=torch.float64, device=device)
        self.register_buffer('A', torch.zeros(num_classes, in_features, in_features, **kw))
        self.register_buffer('C', torch.zeros(in_features, num_classes, **kw))
        self.register_buffer('weight', torch.zeros(in_features, num_classes, **kw))
        self.register_buffer('cnt', torch.zeros(num_classes, dtype=torch.long, device=device))
        self.register_buffer('num_seen', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('dirty', torch.ones((), dtype=torch.bool, device=device))

    @torch.no_grad()
    def fit(self, x, y):
        x, y = x.to(self.weight), y.to(device=self.weight.device, dtype=torch.long)
        self.num_seen.copy_(torch.maximum(self.num_seen, y.max() + 1))
        for c in y.unique():
            features = x[y == c]
            self.A[c].add_(features.T @ features)
            self.C[:, c].add_(features.sum(0))
            self.cnt[c].add_(len(features))
        self.num_updates.add_(len(x))
        self.dirty.fill_(True)

    @torch.no_grad()
    def update(self):
        n = int(self.num_seen)
        if not self.dirty or n == 0:
            return
        counts = self.cnt[:n].to(self.weight)
        factors = torch.where(counts > 0, counts.reciprocal(), 0)
        factors.mul_(n / factors.sum())
        a = torch.eye(self.weight.shape[0], dtype=self.weight.dtype, device=self.weight.device) * self.gamma
        # Avoid a second C x D x D tensor while preserving the upstream formula.
        for c in range(n):
            # Avoid allocating a 512 MiB temporary at 8192 dimensions.
            a.add_(self.A[c], alpha=float(factors[c]))
        self.weight[:, :n].copy_(torch.linalg.solve(a, self.C[:, :n] * factors))
        if not torch.isfinite(self.weight).all():
            raise FloatingPointError('Nonfinite AIR weights; check gamma and input features')
        self.dirty.fill_(False)

    @staticmethod
    def _host_available_bytes():
        """Available host RAM, conservatively bounded by cgroup v2 when present."""
        available = []
        try:
            with open('/proc/meminfo', encoding='utf-8') as info:
                for line in info:
                    if line.startswith('MemAvailable:'):
                        available.append(int(line.split()[1]) * 1024)
                        break
        except (OSError, ValueError):
            pass
        try:
            from pathlib import Path
            limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
            if limit != 'max':
                used = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
                available.append(max(0, int(limit) - used))
        except (OSError, ValueError):
            pass
        return min(available) if available else None

    @torch.no_grad()
    def forward_cached(self, x):
        """No solve: diagnostic logits from last evaluation (initially zero)."""
        return x.to(self.weight) @ self.weight

    def forward(self, x):
        self.update()
        return self.forward_cached(x)
