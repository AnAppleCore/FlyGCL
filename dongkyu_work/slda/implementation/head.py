"""StreamingLDA from tyler-hayes/Deep_SLDA (MIT), with registered state.

See ../original/SOURCE_MANIFEST.json and ../original/slda/LICENSE.
Starts from the author's constructor state; no separate base-session OAS fit.
"""
import math
import torch
from torch import nn


class SLDAHead(nn.Module):
    def __init__(self, in_features, num_classes, shrinkage=1e-4, device='cpu'):
        super().__init__()
        if min(in_features, num_classes) <= 0 or not math.isfinite(shrinkage) or not 0 < shrinkage <= 1:
            raise ValueError('SLDA dimensions must be positive and shrinkage must be in (0, 1]')
        self.shrinkage = shrinkage
        kw = dict(dtype=torch.float32, device=device)
        self.register_buffer('muK', torch.zeros(num_classes, in_features, **kw))
        self.register_buffer('cK', torch.zeros(num_classes, **kw))
        self.register_buffer('Sigma', torch.ones(in_features, in_features, **kw))
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('Lambda', torch.zeros(in_features, in_features, **kw), persistent=False)
        self.register_buffer('precision_updates', torch.full((), -1, dtype=torch.long, device=device), persistent=False)

    @torch.no_grad()
    def fit(self, x, y):
        x, y = x.to(self.muK), y.to(device=self.muK.device, dtype=torch.long)
        # Covariance must use the previous class mean, then update that mean.
        for features, label in zip(x, y):
            n = self.num_updates.to(self.Sigma)
            delta = features - self.muK[label]
            outer = torch.outer(delta, delta) * n / (n + 1)
            self.Sigma.mul_(n).add_(outer).div_(n + 1)
            self.muK[label].add_(delta / (self.cK[label] + 1))
            self.cK[label].add_(1)
            self.num_updates.add_(1)

    @torch.no_grad()
    def forward(self, x):
        if self.precision_updates != self.num_updates:
            covariance = (1 - self.shrinkage) * self.Sigma
            covariance.diagonal().add_(self.shrinkage)
            self.Lambda.copy_(torch.pinverse(covariance))
            self.precision_updates.copy_(self.num_updates)
        means = self.muK.T
        weight = self.Lambda @ means
        return x.to(self.muK) @ weight - 0.5 * (means * weight).sum(0)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.precision_updates.fill_(-1)
