# import torch_optimizer
# from easydict import EasyDict as edict
import timm
from torch import optim

from models import MODELS
from optim.fam import FAM
from optim.sam import SAM


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def select_optimizer(opt_name, lr, model):

    if opt_name == "adam":
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
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    elif opt_name == 'sgd_sl':
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
        opt = optim.SGD([
                        {'params': other_params, 'lr': lr},       # Learning rate lr1 for fully-connected layers
                        {'params': fc_params, 'lr': 0.005}     # Learning rate lr2 for all other layers
                    ], weight_decay=5e-4)
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

def select_model(method, backbone, num_classes=None, n_tasks=None, kwargs=None):

    if method=="slca":
        import models.vit as vit
        model = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None
        )
    elif method in MODELS.keys():
        model = MODELS[method](
            backbone_name=backbone,
            pretrained=True,
            num_classes=num_classes,
            task_num=n_tasks,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    return model