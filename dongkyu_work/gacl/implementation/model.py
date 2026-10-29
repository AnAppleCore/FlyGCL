import torch
from torch import nn
from dongkyu_work.analytic_common.model import FrozenAnalyticModel
from .head import GACLHead


class GACL(FrozenAnalyticModel):
    def __init__(self, *args, gacl_expansion_size=5000, gacl_gamma=100.0, **kwargs):
        self.expansion_size, self.gamma = gacl_expansion_size, gacl_gamma
        super().__init__(*args, **kwargs)

    def _build_statistics(self, device):
        self.head = GACLHead(self.expansion_size, self.num_classes, self.gamma, device)
        self.projection = nn.Linear(self.feature_dim, self.expansion_size, bias=False, dtype=torch.float32, device=device)
