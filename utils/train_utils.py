# import torch_optimizer
# from easydict import EasyDict as edict
from torch import optim
import torch.nn as nn
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, _create_vision_transformer, default_cfgs
from timm.models import create_model
from models.vit import _create_vision_transformer
from models.L2P import L2P
from models.dualprompt import DualPrompt
from models.mvp import MVP
from optim.sam import SAM
from optim.fam import FAM

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)
@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def select_optimizer(opt_name, lr, model):

    if opt_name == "adam":
        # print("opt_name: adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == 'adam_adapt':
        fc_params = []
        other_params = []
        fc_params_name = []
        other_params_name = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc.' in name:  # If the parameter is from a fully-connected layer
                    fc_params.append(param)
                    fc_params_name.append(name)
                else:  # All other layers
                    other_params.append(param)
                    other_params_name.append(name)
        opt = optim.Adam([
                        {'params': fc_params, 'lr': lr},       # Learning rate lr1 for fully-connected layers
                        {'params': other_params, 'lr': lr*5}     # Learning rate lr2 for all other layers
                    ], weight_decay=0)
        
    # elif opt_name == "radam":
    #     opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    elif opt_name == "sam":
        base_optimizer = optim.Adam
        opt = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0)
    elif opt_name == "fam":
        base_optimizer = optim.Adam
        opt = FAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    elif sched_name == "sam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    elif sched_name == "fam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

def select_model(model_name, dataset, num_classes=None,selection_size=None, kwargs=None):
    
    opt = dict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

#! cifar and imageNet --> ViT model 
    model_class = None

    #* vit method(L2p) --> cifar_vit,vision_transformer

    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    elif model_name == "vit":
        opt["depth"] = 12
    elif model_name == "vit_finetune":
        opt["depth"] = 12
    elif model_name == "vit_finetune_last":
        opt["depth"] = 12
    elif model_name == "vit_init_last":
        opt["depth"] = 12
    elif model_name == "L2P":
        opt["depth"] = 12
    elif model_name == "DualPrompt":
        opt["depth"] = 12
    elif model_name == "mvp":
        opt["depth"] = 12
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    if model_name == "vit":
        model = timm.create_model(
                            "vit_small_patch16_224",pretrained=True,num_classes=num_classes, # vit_small_patch16_224
                            drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        for n, p in model.named_parameters():
            if "fc." in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif model_name == "vit_finetune_last":
        model = timm.create_model(
                            "vit_base_patch16_224",pretrained=True,num_classes=num_classes,
                            drop_rate=0.,drop_path_rate=0.,drop_block_rate=None,)
        for n, p in model.named_parameters():
            if "fc." in n or 'blocks.11' in n or 'blocks.10' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    elif model_name == "vit_init_last":
        model = timm.create_model(
                            "vit_base_patch16_224",pretrained=True,num_classes=num_classes,
                            drop_rate=0.,drop_path_rate=0.,drop_block_rate=None,)
        for n, p in model.named_parameters():
            if "fc." in n:
                p.requires_grad = True
                
            elif 'blocks.11' in n or 'blocks.10' in n: #blocks.11.mlp
                p.requires_grad = True
                # nn.init.uniform_(p)
                nn.init.normal_(p, mean=0, std=1)
                
            else:
                p.requires_grad = False

    elif model_name == "vit_finetune":
        model = timm.create_model(
                            "vit_base_patch16_224",pretrained=True,num_classes=num_classes,
                            drop_rate=0.,drop_path_rate=0.,drop_block_rate=None,) 
    elif model_name == "L2P":
        model = L2P(backbone_name="vit_base_patch16_224", class_num=num_classes)
    elif model_name == "DualPrompt":
        model = DualPrompt(backbone_name="vit_base_patch16_224", class_num=num_classes, **kwargs)
    elif model_name == "mvp":
        model = MVP(backbone_name="vit_small_patch16_224", class_num=num_classes, selection_size = selection_size)
    elif model_name == "resnet18":
        model = timm.create_model('resnet18', num_classes=num_classes,pretrained=True)
        for n, p in model.named_parameters():
            if "fc." in n:
                p.requires_grad = True    
            else:
                p.requires_grad = False
    else:
        raise NotImplementedError(
            "Please select the appropriate model"
        )

    print("[Selected Model]:", model_name )
    return model