"""GACL recursive ridge update, adapted from CHEN-YIZHU/GACL (GPL-3.0).

See ../original/SOURCE_MANIFEST.json and ../original/gacl/LICENSE.
The solve form is algebraically equivalent to the upstream inverse update.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GACLHead(nn.Module):
    def __init__(self, in_features, num_classes, gamma=100.0, device='cpu'):
        super().__init__()
        if min(in_features, num_classes) <= 0 or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('GACL dimensions and gamma must be positive and finite')
        self.register_buffer('R', torch.eye(in_features, dtype=torch.float64, device=device) / gamma)
        self.register_buffer('weight', torch.zeros(in_features, num_classes, dtype=torch.float64, device=device))
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def fit(self, x, y):
        x, y = x.to(self.weight), y.to(device=self.weight.device, dtype=torch.long)
        targets = F.one_hot(y, self.weight.shape[1]).to(self.weight)
        rx = self.R @ x.T
        k = torch.eye(len(x), dtype=x.dtype, device=x.device) + x @ rx
        self.R.sub_(rx @ torch.linalg.solve(k, x @ self.R))
        self.weight.add_(self.R @ x.T @ (targets - x @ self.weight))
        self.num_updates.add_(len(x))
        if not torch.isfinite(self.R).all() or not torch.isfinite(self.weight).all():
            raise FloatingPointError('Nonfinite GACL statistics; check gamma and input features')

    def forward(self, x):
        return x.to(self.weight) @ self.weight
