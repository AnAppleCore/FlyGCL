# When we make a new one, we should inherit the Finetune class.
import logging
import time
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import cutmix_data
from utils.train_utils import select_scheduler

import torchvision.transforms as transforms
from methods._trainer import _Trainer

import torch.distributed as dist
from utils.memory import MemoryBatchSampler

# logger = logging.getLogger()


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class ER(_Trainer):
    def __init__(self, *args, **kwargs):
        super(ER, self).__init__(*args, **kwargs)
        self.nobatchmask = kwargs.get("nobatchmask")
        self.sessionmask = kwargs.get("sessionmask")
        self.model_name  = kwargs.get("model_name")

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
        # image, label = sample
        self.add_new_class(labels)
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())
        self.memory_sampler  = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader   = DataLoader(self.train_dataset, batch_size=self.memory_batchsize, sampler=self.memory_sampler, num_workers=4)
        self.memory_provider     = iter(self.memory_dataloader)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)): # * self.temp_batchsize * self.world_size):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        if self.model_name == 'resnet34':
            # print('update using ewc')
            self.update_memory_ewc(images, labels)
        else:
            self.update_memory(idx, labels)
        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter
    
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
            else:
                # print(self.exposed_classes, label[i].item())
                self.memory.replace_data([sample[i], self.exposed_classes[label[i].item()]])
    def update_memory_ewc(self, sample, label):
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

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        if self.sessionmask:
            self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
    
    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in range(len(memory_labels)):
                memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        x = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x,y)
        _, preds = logit.topk(self.topk, 1, True, True)
        
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
        # per task acc
        num_per_task = int(self.n_classes/self.n_tasks)
        
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

        # if task_id is not None:
        #     for ii in range(task_id+1):
        #         num_data = num_data_l[ii*num_per_task:(ii+1)*num_per_task].sum()
        #         num_correct = correct_l[ii*num_per_task:(ii+1)*num_per_task].sum()
        #         print('Task: {}: {}'.format(ii, num_correct/num_data))

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