from dongkyu_work.analytic_common.model import FrozenAnalyticModel
from .head import SLDAHead


class SLDA(FrozenAnalyticModel):
    def __init__(self, *args, slda_shrinkage=1e-4, **kwargs):
        self.shrinkage = slda_shrinkage
        super().__init__(*args, **kwargs)

    def _build_statistics(self, device):
        self.head = SLDAHead(self.feature_dim, self.num_classes, self.shrinkage, device)
