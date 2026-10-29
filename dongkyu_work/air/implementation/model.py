import torch
from torch import nn
from dongkyu_work.analytic_common.model import FrozenAnalyticModel
from .head import AIRHead


class AIR(FrozenAnalyticModel):
    def __init__(self, *args, air_expansion_size=8192, air_gamma=0.1, air_max_stats_gib=64.0, **kwargs):
        self.expansion_size, self.gamma, self.max_stats_gib = air_expansion_size, air_gamma, air_max_stats_gib
        # Store the official-size per-class covariance on host RAM by default.
        # Explicit --analytic_stats_device model still allows large-VRAM devices.
        if kwargs.get('analytic_stats_device', 'auto') == 'auto':
            kwargs['analytic_stats_device'] = 'cpu'
        super().__init__(*args, **kwargs)

    def _build_statistics(self, device):
        self.head = AIRHead(self.expansion_size, self.num_classes, self.gamma, device, self.max_stats_gib)
        self.projection = nn.Linear(self.feature_dim, self.expansion_size, bias=False, dtype=torch.float64, device=device)
