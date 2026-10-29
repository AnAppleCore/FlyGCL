import torch
import torch.nn as nn
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, _create_vision_transformer
from utils.train_utils import load_pretrain



@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model



class ModifiedViT(nn.Module):
    def __init__(self, n_classes):
        super(ModifiedViT, self).__init__()
        self.numclass=n_classes
        self.vit = timm.create_model("deit_small_patch16_224", pretrained=False)
        self.vit = load_pretrain(self.vit)
        self.feature_dim = 384
        for name, param in self.vit.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
       
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(self.feature_dim, n_classes)

    def features(self, x):
        x = self.vit(x)
        return x

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x




class VITACIL(nn.Module):
    def __init__(self, n_classes, hidden):
        super(VITACIL, self).__init__()
        self.numclass = n_classes
        self.vit = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes = n_classes)
        self.vit = load_pretrain(self.vit)
        self.feature_dim = 384

        for name, param in self.vit.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    
        self.vit.head = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(self.feature_dim,hidden,bias=False),
                                nn.ReLU(),
                                nn.Linear(hidden,n_classes,bias=False))
        self.exp = self.fc[:2]

        self.eval()
        
    @torch.inference_mode()
    def features(self, x):
        x = self.vit(x)
        return x

    @torch.inference_mode()
    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x

    @torch.inference_mode()
    def expansion(self, x):
        x = self.vit(x)
        x = self.exp(x)
        return x
