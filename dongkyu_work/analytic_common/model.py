"""Frozen FlyGCL backbone and lazy analytic statistics shared by AIR/GACL/SLDA."""
import torch
from torch import nn


class FrozenAnalyticModel(nn.Module):
    def __init__(self, backbone_name, num_classes, analytic_stats_device='auto', **kwargs):
        super().__init__()
        # Local imports avoid importing the model registry while it initializes.
        from models import vit
        import timm
        factory = getattr(vit, backbone_name, None)
        opts = dict(pretrained=True, num_classes=num_classes, drop_rate=0., drop_path_rate=0.)
        self.backbone = factory(**opts) if factory else timm.create_model(backbone_name, **opts)
        self.feature_dim = self.backbone.num_features
        self.num_classes = num_classes
        if analytic_stats_device not in ('auto', 'model', 'cpu'):
            raise ValueError('analytic_stats_device must be auto, model or cpu')
        self.stats_device_policy = 'model' if analytic_stats_device == 'auto' else analytic_stats_device
        self.head = None
        self.projection = None
        self.requires_grad_(False)
        self.backbone.eval()

    def initialize_statistics(self):
        if self.head is None:
            device = next(self.backbone.parameters()).device if self.stats_device_policy == 'model' else torch.device('cpu')
            self._build_statistics(device)
            self.requires_grad_(False)

    def _build_statistics(self, device):
        raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, images):
        features = self.backbone.forward_features(images)
        if not isinstance(features, torch.Tensor) or features.ndim not in (2, 3):
            raise ValueError('Analytic adapters require a ViT feature tensor')
        if features.ndim == 3:
            if getattr(self.backbone, 'cls_token', None) is None:
                raise ValueError('Token features require a CLS token')
            features = features[:, 0]
        if hasattr(self.backbone, 'fc_norm'):
            features = self.backbone.fc_norm(features)
        if features.shape[-1] != self.feature_dim:
            raise ValueError('Unexpected backbone feature dimension')
        return features.float()

    @torch.no_grad()
    def expand(self, features):
        self.initialize_statistics()
        if self.projection is None:
            return features
        return self.projection(features.to(self.projection.weight)).relu()

    @torch.no_grad()
    def fit_features(self, features, labels):
        self.initialize_statistics()
        self.head.fit(features, labels)

    @torch.no_grad()
    def scores(self, features, device, refresh=True):
        # AIR diagnostic metrics can use cached weights without solving per batch.
        if not refresh and hasattr(self.head, 'forward_cached'):
            return self.head.forward_cached(features).to(device=device)
        return self.head(features).to(device=device)

    @torch.no_grad()
    def forward(self, images):
        return self.scores(self.expand(self.features(images)), images.device)

    def load_state_dict(self, *args, **kwargs):
        self.initialize_statistics()
        return super().load_state_dict(*args, **kwargs)
