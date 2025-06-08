import copy
import datetime
import gc
import logging
import time
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from methods._trainer import _Trainer
from methods.er_baseline import ER
from models.vit import _create_vision_transformer
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from utils.data_loader import (ImageDataset, MemoryDataset, StreamDataset,
                               cutmix_data, get_statistics)
from utils.memory import MemoryBatchSampler, MemoryOrderedSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

# from randaugment.randaugment import RandAugment

# logger = logging.getLogger()

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
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

# class L2P(ER):
class L2P(_Trainer):
    def __init__(self, *args, **kwargs):
        super(L2P, self).__init__(*args, **kwargs)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.labels = torch.empty(0)
        
        # self.class_mask = None
        # self.class_mask_dict={}
    
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
            self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=4)
            self.memory_provider     = iter(self.memory_dataloader)

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        if self.memory_size > 0:
            self.update_memory(idx, labels)
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
            
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            # for i in range(len(memory_labels)):
            #     memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        self.labels = torch.cat((self.labels, y), 0)

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y, mymask=logit_mask)
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mymask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mymask is not None:
                logit += mymask
            else:
                logit += self.mask
            loss = self.criterion(logit, y)
        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        if end:
            self.disjoint_classes
            print(self.disjoint_classes)
            print(self.exposed_classes)
            if task_id is not None:
                for ii in range(task_id+1):
                    cls_ii = self.disjoint_classes[ii]
                    cls_mask = [i for i in cls_ii] 
                    for j in range(len(cls_mask)):
                        cls_mask[j] = self.exposed_classes.index(cls_mask[j])
                    num_data = num_data_l[cls_mask].sum()
                    num_correct = correct_l[cls_mask].sum()
                    print('Per_Task: {}: {}, seed:{}'.format(ii, num_correct/num_data, self.rnd_seed))
            
            
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()
            
    def online_before_task(self,train_loader):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # self.class_mask_dict[cur_iter]=self.class_mask
        self.model_without_ddp.keys = torch.cat([self.model_without_ddp.keys, self.model_without_ddp.prompt.key.clone().detach().cpu()], dim=0)
        pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)