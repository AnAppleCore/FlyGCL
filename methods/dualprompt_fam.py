from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.distributed as dist
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
# from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

import logging
import copy
import time
import datetime

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from methods._trainer import _Trainer

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler
from datasets import *
from utils.onlinesampler import OnlineSampler, OnlineTestSampler

from utils.memory import MemoryBatchSampler, Memory, DummyMemory
from torch.utils.data import DataLoader
import timm
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from models.vit import _create_vision_transformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random



logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

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
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
    return model

class DualPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(DualPrompt, self).__init__(*args, **kwargs)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999

        self.labels = torch.empty(0)
        self.class_mask = None
        self.class_mask_dict={}
        self.task_id = 1
        self.sessionmask = kwargs.get("sessionmask")
    
    def create_dataloader_ood(self, dataset, cls_lst=None):
        if cls_lst is None:
            cls_lst = [i for i in range(900, 910)]
        # Additional dataset for OOD
        _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        if 'imagenet' in dataset:
            self.ood_dataset   = self.datasets["imagenetrandom"](root=self.data_dir, train=True,  download=True, transform=self.load_transform, cls_lst=cls_lst)
        elif 'cub' in dataset:
            self.ood_dataset   = self.datasets["cubrandom"](root=self.data_dir, train=True,  download=True, transform=self.load_transform, cls_lst=cls_lst)
        self.online_iter_dataset_ood = OnlineIterDataset(self.ood_dataset, 1)

        self.train_sampler_ood   = OnlineSampler(self.online_iter_dataset_ood, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        
        self.ood_dataloader    = DataLoader(self.online_iter_dataset_ood, batch_size=self.temp_batchsize, sampler=self.train_sampler_ood, pin_memory=False, num_workers=0)
        self.ood_iter = iter(self.ood_dataloader) 

    def setup_distributed_dataset(self):

        self.datasets = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "tinyimagenet": TinyImageNet,
        "cub200": CUB200,
        "cub175": CUB175,
        "cubrandom": CUBRandom,
        "imagenet": ImageNet,
        "imagenet100": ImageNet100,
        "imagenet900": ImageNet900,
        "imagenetrandom": ImageNetRandom,
        "imagenetsub": ImageNetSub,
        "imagenet-r": Imagenet_R,
        "places365": Places365,
        "gtsrb": GTSRB
        }

        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        if self.model_name in ['vit', 'vit_finetune', 'L2P', 'mvp', 'DualPrompt', 'LinearProbe', 'vit_init_last']:
            print(self.model_name)
            inp_size = 224    
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
        self.cutmix = "cutmix" in self.transforms 
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
            if self.gpu_transform:
                self.gpu_transform = False
        
        if "autoaug" in self.transforms:
            if 'cifar' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            elif 'svhn' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('svhn')))
                
        self.train_transform = transforms.Compose([
                lambda x: (x * 255).to(torch.uint8),
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                lambda x: x.float() / 255,
                # transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        print(f"Using train-transforms {self.train_transform}")
        # applying more aggresive data aug to ood data
        self.ood_transform = transforms.Compose([
                lambda x: (x * 255).to(torch.uint8),
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                *train_transform,
                lambda x: x.float() / 255,
                # transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        self.inp_size = inp_size

        _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        print('=================_r{}================='.format(_r))
        print('=================_r{}================='.format(_r))

        if 'imagenet' in self.dataset or 'cub' in self.dataset:
            self.load_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()])
        else:
            self.load_transform = transforms.ToTensor()

        self.train_dataset   = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=self.load_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)

        self.test_dataset    = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)

        if self.distributed:
            self.train_sampler   = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        else:
            self.train_sampler   = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.test_sampler    = OnlineTestSampler(self.test_dataset, [], _w, _r)
        # self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler,pin_memory=False, num_workers=self.n_worker)
        self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler,pin_memory=False, num_workers=0)
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        self.seen = 0
        if not hasattr(self, 'memory'):
            self.memory = Memory()

        # Randomly select 10 numbers between 100 and 200
        if 'imagenet' in self.dataset:
            random_cls = random.sample(range(900, 1000), 10)
        elif 'cub' in self.dataset:
            random_cls = random.sample(range(175, 200), 10)
        self.create_dataloader_ood(self.dataset, cls_lst=random_cls)


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
        self.update_memory(idx, labels)
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def add_new_class(self, class_name):
        # print('using DP mask')
        exposed_classes = []
        new = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
                new.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for clas in exposed_classes:
                if clas not in self.exposed_classes:
                    self.exposed_classes.append(clas)
        self.memory.add_new_class(cls_list=self.exposed_classes)

        cls_lst = torch.unique(class_name)
        for cc in cls_lst:
           idx = self.exposed_classes.index(cc.item())  
           if self.mask[idx] != 0:
               self.mask[idx] = 0
        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data


        try:
            memory_images, memory_labels, _ = next(self.ood_iter)
            
        except StopIteration:
            prob = random.random()
            # Initial random setting, change at the prob of 0.5
            if prob > 0.67:
                self.ood_iter = iter(self.ood_dataloader) 
            else:
                if 'imagenet' in self.dataset:
                    random_cls = random.sample(range(900, 1000), 10)
                elif 'cub' in self.dataset:
                    random_cls = random.sample(range(175, 200), 10)
                self.create_dataloader_ood(self.dataset, cls_lst=random_cls)

            # # Random setting 2, chang at the prob of 1
            # random_cls = random.sample(range(100, 201), 10)
            # self.create_dataloader_ood(cls_lst=random_cls)

            memory_images, memory_labels, _ = next(self.ood_iter)

        self.add_new_class(memory_labels)
        for i in range(len(memory_labels)):
            memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())

        memory_images = memory_images.to(self.device)
        memory_labels = memory_labels.to(self.device)
        
        memory_images = self.ood_transform(memory_images)

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.train_transform(x)

        logit, loss = self.model_forward(memory_images,memory_labels)

        ### Sam update
        loss.backward()

        self.optimizer.first_step(zero_grad=True)

        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        loss.backward()
        self.optimizer.second_step(zero_grad=True)
 

                
        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # logit, dist_loss = self.model(x)
            res = self.model(x)
            if isinstance(res, tuple):
                logit, dist_loss = res
            else:
                logit = res
            logit += self.mask
            # print(dist_loss)
            loss = self.criterion(logit, y) #+ 0.3 * dist_loss
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

                # logit, _ = self.model(x)
                logit = self.model(x)
                logit = logit # + self.mask
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

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        # per task acc
        num_per_task = int(self.n_classes/self.n_tasks)

        if end:
            if self.dataset != 'imagenetsub':
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

        if task_id is not None and self.dataset != 'imagenetsub':
            for ii in range(task_id+1):
                num_data = num_data_l[ii*num_per_task:(ii+1)*num_per_task].sum()
                num_correct = correct_l[ii*num_per_task:(ii+1)*num_per_task].sum()
                print('Task: {}: {}'.format(ii, num_correct/num_data))

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
        if not self.distributed:
            self.model.task_id += 1
        else:
            self.model.module.task_id += 1
        self.task_id += 1
        # self.model.freeze_eg_proj()
        if self.model_name == 'CodaPrompt':
            self.model.process_task_count()
        if self.sessionmask:
            self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        
        # self.model_without_ddp.keys = torch.cat([self.model_without_ddp.keys, self.model_without_ddp.e_prompt.key.detach().cpu()], dim=0)
        # pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1,)).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier() # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]], index)
            else:
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])