# When we make a new one, we should inherit the Finetune class.
import logging
import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import cutmix_data
from utils.train_utils import select_scheduler
from utils.buffer import Buffer

import torchvision.transforms as transforms
from methods._trainer import _Trainer
from torch.nn import functional as F

import torch.distributed as dist
from utils.memory import MemoryBatchSampler

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class DERPP(_Trainer):
    def __init__(self, *args, **kwargs):
        super(DERPP, self).__init__(*args, **kwargs)
        self.buffer_size = kwargs.get("memory_size")
        self.ngpus_per_nodes = torch.cuda.device_count()
        self.gpu    = 0 % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        self.buffer = Buffer(self.buffer_size, device=self.device)
        self.alpha = 0.2
        self.beta = 0.5

    def online_step(self, images, labels, idx):
        # image, label = sample
        self.add_new_class(labels)
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)): # * self.temp_batchsize * self.world_size):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter
    

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data

        x_init = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x_init)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        

        if len(self.buffer) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels, memory_logits = self.buffer.get_data(
                self.memory_batchsize)

            memory_images = self.train_transform(memory_images)
            memory_outputs, _ = self.model_forward(memory_images, memory_labels)

            unmask = memory_logits != -torch.inf
            memory_logits = memory_logits[unmask]
            memory_outputs = memory_outputs[unmask]

            loss_1 = F.mse_loss(memory_outputs, memory_logits)

            loss += self.alpha * loss_1

            memory_images, memory_labels, memory_logits = self.buffer.get_data(
                self.memory_batchsize)
            memory_images = self.train_transform(memory_images)
            _, memory_loss = self.model_forward(memory_images, memory_labels)
            loss += self.beta * memory_loss
            
        self.buffer.add_data(examples=x_init.detach().clone(),
                             labels=y.detach().clone(),
                             logits=logit.detach().clone())

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit = logit + self.mask
                loss = lam * self.criterion(logit, labels_a.to(torch.int64)) + (1 - lam) * self.criterion(logit, labels_b.to(torch.int64))
        else:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y.to(torch.int64))
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
            # per task acc
            num_per_task = int(self.n_classes/self.n_tasks)
            if task_id is not None:
                for ii in range(task_id+1):
                    num_data = num_data_l[ii*num_per_task:(ii+1)*num_per_task].sum()
                    num_correct = correct_l[ii*num_per_task:(ii+1)*num_per_task].sum()
                    print('Task: {}: {}'.format(ii, num_correct/num_data))
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