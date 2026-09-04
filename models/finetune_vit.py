import logging

import timm
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class FinetuneViT(nn.Module):
    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.task_num = task_num
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.task_count = 0
        self.kwargs = kwargs

        assert backbone_name is not None, "backbone_name must be specified"
        if hasattr(vit, backbone_name):
            logger.info(f"Using custom ViT model for full fine-tuning: {backbone_name}")
            self.backbone = getattr(vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        else:
            logger.info(f"Using timm model for full fine-tuning: {backbone_name}")
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1
