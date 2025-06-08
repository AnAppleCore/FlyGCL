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
import torch.distributed as dist
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
from utils.memory import MemoryBatchSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

# from randaugment.randaugment import RandAugment

logger = logging.getLogger()

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

class CodaPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(CodaPrompt, self).__init__(*args, **kwargs)
        self.nobatchmask = kwargs.get("nobatchmask")
        self.sessionmask = kwargs.get("sessionmask")
        self.rnd_seed = kwargs.get("rnd_seed")
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999

        self.labels = torch.empty(0)
        self.class_mask = None
        self.class_mask_dict={}
        # self.disjoint_classes_lst = [[45, 130, 128, 38, 91, 123, 149, 96, 151, 80, 78, 185, 168, 172], [93, 105, 198, 6, 190, 10, 170, 18, 150, 162, 179, 51, 30, 31], [121, 33, 98, 174, 167, 49, 169, 188, 187, 183, 40, 43, 85, 112, 36, 83, 47, 21, 50, 107, 129, 117, 164, 61, 64, 191, 113, 63, 8, 13, 5, 54, 126, 26, 65, 141, 101, 59, 73, 1, 7, 29, 34, 97], [148, 15, 60, 173, 89, 152, 193, 92, 53, 46, 122, 2, 143, 11], [153, 116, 25, 119, 100, 88, 14, 99, 66, 137, 175, 180, 0, 19]]
        # self.disjoint_classes_lst = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 72, 73, 74, 75, 76, 77, 78, 79], [80, 81, 82, 83, 84, 85, 86, 87, 88, 89], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
        self.disjoint_classes_lst = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
        self.task_id = 0

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
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)

        self.memory.add_new_class(cls_list=self.exposed_classes)
        if self.sessionmask:            
            cls_lst = torch.unique(class_name)
            for cc in cls_lst:
                idx = self.exposed_classes.index(cc.item())  
                if self.mask[idx] != 0:
                    self.mask[idx] = 0
        else:
            self.memory.add_new_class(cls_list=self.exposed_classes)
            self.mask[:len(self.exposed_classes)] = 0
        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        if len(self.memory) > 0 and self.memory_batchsize > 0:
        # if self.memory_size > 0:
            # if self.memory_size > len(self.memory):
            #     memory_bs = len(self.memory)
            # else:
            #     memory_bs = self.memory_size
            memory_bs = self.memory_batchsize
            self.memory_sampler  = MemoryBatchSampler(self.memory, memory_bs, self.temp_batchsize * self.online_iter * self.world_size)
            self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=memory_bs, sampler=self.memory_sampler, num_workers=4)
            self.memory_provider     = iter(self.memory_dataloader)

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        # if len(self.memory) > 0 and self.memory_batchsize > 0:
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
        # if self.memory_size > 0 and len(self.memory) > 0:
            memory_images, memory_labels = next(self.memory_provider)
            # for i in range(len(memory_labels)):
            #     memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

            if self.sessionmask:            
                cls_lst = torch.unique(memory_labels)
                for cc in cls_lst:
                    idx = self.exposed_classes.index(cc.item())  
                    if self.mask[idx] != 0:
                        self.mask[idx] = 0

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if self.nobatchmask:
            logit, loss = self.model_forward(x,y)
        else:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
            
        _, preds = logit.topk(self.topk, 1, True, True)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        # do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        # if do_cutmix:
        #     x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
        #     with torch.cuda.amp.autocast(enabled=self.use_amp):
        #         logit = self.model(x)
        #         logit += self.mask
        #         loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        # else:
        ortho_loss = None
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # logit, dist_loss = self.model(x)
            res = self.model(x)
            if isinstance(res, tuple):
                logit, ortho_loss = res
            else:
                logit = res
            # logit = self.model(x)
            if mask is not None:
                
                # randomly active some logits
                # inf_indices = torch.where(torch.logical_and(torch.isinf(mask),self.mask==0))[0]
                # inf_indices = torch.where(torch.isinf(mask)[:len(self.exposed_classes)])[0]
                # if len(inf_indices) > 0:
                #     num_inf_to_replace = min(3, len(inf_indices))
                #     indices_to_replace = torch.randperm(len(inf_indices))[:num_inf_to_replace]
                #     mask[inf_indices[indices_to_replace]] = 0

                logit += mask
            else:
                logit += self.mask
            
            # print(dist_loss)
            # print(self.mask)
            # print(y)
            loss = self.criterion(logit, y)
            if ortho_loss is not None:
                loss += ortho_loss
            
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
                res = self.model(x)
                if isinstance(res, tuple):
                    logit, _ = res
                else:
                    logit = res                
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
        # per task acc
        num_per_task = int(self.n_classes/self.n_tasks)
        
        # if end:
        #     self.disjoint_classes
        #     print(self.disjoint_classes)
        #     print(self.exposed_classes)
        #     if task_id is not None:
        #         for ii in range(task_id+1):
        #             cls_ii = self.disjoint_classes[ii]
        #             cls_mask = [i for i in cls_ii] 
        #             for j in range(len(cls_mask)):
        #                 cls_mask[j] = self.exposed_classes.index(cls_mask[j])
        #             num_data = num_data_l[cls_mask].sum()
        #             num_correct = correct_l[cls_mask].sum()
        #             print('Per_Task: {}: {}, seed:{}'.format(ii, num_correct/num_data, self.rnd_seed))
            
            
            # # disjoint acc
            # cur_dis = self.disjoint_classes_lst[task_id]
            # cur_count = 0
            # cur_correct = 0
            # for cc in cur_dis:
            #     cur_count += num_data_l[cc]
            #     cur_correct += correct_l[cc]
            # pre_dix = []
            # for ii in range(task_id):
            #     pre_dix += self.disjoint_classes_lst[ii]
            # pre_count = 0
            # pre_correct = 0
            # for cc in pre_dix:
            #     pre_count += num_data_l[cc]
            #     pre_correct += correct_l[cc]
            # if cur_count > 0:
            #     print('current disjoint acc: {}'.format(cur_correct/cur_count))
            # if pre_count > 0:
            #     print('previous disjoint acc: {}'.format(pre_correct/pre_count))
                
            

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
            
    def online_before_task(self, task_id):
        # self.model.convert_train_task(task_id)
        pass

    def online_after_task(self, cur_iter):
        # self.model_without_ddp.keys = torch.cat([self.model_without_ddp.keys, self.model_without_ddp.e_prompt.key.detach().cpu()], dim=0)
        # self.model.reload_pt()
        # self.model.freeze_ss()
        # self.model.freeze_eg_proj()
        # self.model.freeze_prompt()
        if not self.distributed:
            self.model.task_id += 1
        else:
            self.model.module.task_id += 1
        self.model.process_task_count()
        if self.sessionmask:
            self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        self.task_id += 1
        # pass

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    # def main_worker(self, gpu) -> None:
    #     super(DualPrompt, self).main_worker(gpu)
        
        # idx = torch.randperm(self.model_without_ddp.features.shape[0])
        # print(self.labels.size())
        # print(self.model_without_ddp.features.shape)
        # labels = self.labels[idx[:10000]]

        # self.model_without_ddp.features = torch.cat([self.model_without_ddp.features[idx[:10000]], self.model_without_ddp.keys], dim=0)
        # self.model_without_ddp.features = F.normalize(self.model_without_ddp.features, dim=1)

        # tsne = TSNE(n_components=2, random_state=0)
        # X_2d = tsne.fit_transform(self.model_without_ddp.features.detach().cpu().numpy())
        
        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-50:-40, 0], X_2d[-50:-40, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-50:-40, 0][i] + 0.1, X_2d[-50:-40, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'DP_tsne{self.rnd_seed}_Task1.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-40:-30, 0], X_2d[-40:-30, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-40:-30, 0][i] + 0.1, X_2d[-40:-30, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'DP_tsne{self.rnd_seed}_Task2.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-30:-20, 0], X_2d[-30:-20:, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-30:-20, 0][i] + 0.1, X_2d[-30:-20:, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'DP_tsne{self.rnd_seed}_Task3.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-20:-10, 0], X_2d[-20:-10, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-20:-10, 0][i] + 0.1, X_2d[-20:-10, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'DP_tsne{self.rnd_seed}_Task4.png')
        # plt.clf()

        # for i in range(100):
        #     plt.scatter(X_2d[:10000][labels==i, 0], X_2d[:10000][labels==i, 1], s = 1, alpha=0.2)
        # plt.scatter(X_2d[-10:, 0], X_2d[-10:, 1], s = 50, marker='^', c='black')
        # for i in range(10):
        #     plt.text(X_2d[-10:, 0][i] + 0.1, X_2d[-10:, 1][i], "{}".format(i), fontsize=10)
        # plt.savefig(f'DP_tsne{self.rnd_seed}_Task5.png')
        # plt.clf()

    def update_memory(self, sample, label):
        for j in range(len(label)):
            label[j] = self.exposed_classes.index(label[j].item())
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
                    # self.memory.replace_data([sample[i], self.exposed_classes.index(label[i].item())], index)
            else:
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])
                # self.memory.replace_data([sample[i], self.exposed_classes.index(label[i].item())], index)