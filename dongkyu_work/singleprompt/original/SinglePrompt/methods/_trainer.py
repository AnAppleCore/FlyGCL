import os
import sys
import random
import time
import datetime
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.augment import Cutout
from utils.data_loader import get_statistics
from datasets import *
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.memory import Memory

import matplotlib.pyplot as plt
import numpy as np


########################################################################################################################
# Based on the following tutorial:                                                                                     #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py                                                       #
# And Deit by FaceBook                                                                                                 #
# https://github.com/facebookresearch/deit                                                                             #
########################################################################################################################

class _Trainer():
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.mode    = kwargs.get("mode")
        self.dataset = kwargs.get("dataset")
        
        self.n_tasks = kwargs.get("n_tasks")
        self.n   = kwargs.get("n")
        self.m   = kwargs.get("m")
        self.rnd_NM  = kwargs.get("rnd_NM")
        self.rnd_seed    = kwargs.get("rnd_seed")

        self.log_path    = kwargs.get("log_path")
        self.model_name  = kwargs.get("model_name")
        self.opt_name    = kwargs.get("opt_name")
        self.sched_name  = kwargs.get("sched_name")
        self.batchsize  = kwargs.get("batchsize")
        self.n_worker    = kwargs.get("n_worker")
        self.lr  = kwargs.get("lr")

        self.topk    = kwargs.get("topk")
        self.use_amp = kwargs.get("use_amp")
        self.transforms  = kwargs.get("transforms")

        self.data_dir    = kwargs.get("data_dir")
        self.note    = kwargs.get("note")
        self.selection_size = kwargs.get("selection_size")      
        
        self.eval_period     = kwargs.get("eval_period")
        self.temp_batchsize  = kwargs.get("temp_batchsize")
        self.online_iter     = kwargs.get("online_iter")
        self.num_epochs     = kwargs.get("num_epochs")

        self.start_time = time.time()

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.temp_batchsize = self.batchsize // 2
        self.exposed_classes = []
        
        self.pos_prompt = kwargs.get("pos_prompt")
        self.prompt_length = kwargs.get("prompt_length")
        self.logit_type = kwargs.get("logit_type")

        # for memory
        self.memory_size = kwargs.get("memory_size")
        self.memory_batchsize = self.batchsize - self.temp_batchsize
        if not hasattr(self, 'memory'):
            print(f'memory size: {self.memory_size}')
            self.memory = Memory()

        os.makedirs(f"{self.log_path}/{self.note}", exist_ok=True)
        return

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
        "imagenetsub": ImageNetSub,
        "imagenet-r": Imagenet_R,
        'nch': NCH,
        'places365': Places365,
        "gtsrb": GTSRB,
        "wikiart": WIKIART
        }

        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        if self.model_name in ['vit', 'vit_base', 'DualPrompt', 'model_only']:
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
            elif 'wikiart' in self.dataset:
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
        print(f"Using train-transforms {train_transform}")
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        self.inp_size = inp_size

        if 'imagenet' in self.dataset or 'cub' in self.dataset:
            self.load_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()])
        else:
            self.load_transform = transforms.ToTensor()

        self.train_dataset   = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=self.load_transform)
        self.test_dataset    = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        
        self.train_sampler   = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM)
        self.test_sampler    = OnlineTestSampler(self.test_dataset, [])
        
        self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=self.n_worker) # num_workers=0
        
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        self.seen = 0

    def setup_distributed_model(self):

        print("Building model...")
        self.model = select_model(self.model_name, self.dataset, self.n_classes,self.selection_size, self.kwargs).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.model_without_ddp = self.model

        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n,p in self.model_without_ddp.named_parameters() if p.requires_grad]
        print(learnables)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def run(self):
        self.main_worker(0)
    
    def main_worker(self, gpu) -> None:
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)

        print(f"[1] Select a CIL method ({self.mode})")
        self.setup_distributed_model() 

        print(f"[2] Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period
        num_report = 2000
        
        start_time = time.time()
        for task_id in range(self.n_tasks):

            print("\n" + "#" * 50)
            print(f"# Task {task_id} iteration")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")
            
            self.train_sampler.set_task(task_id)

            for i, (images, labels, idx) in enumerate(self.train_dataloader):
                samples_cnt += images.size(0)
                loss, acc = self.online_step(images, labels, idx)
                if samples_cnt + images.size(0) > num_report:
                    self.report_training(samples_cnt, loss, acc)
                    num_report += 500
                
                if samples_cnt + images.size(0) > num_eval:
                    with torch.no_grad():
                        test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                        eval_dict = self.online_evaluate(test_dataloader)
                        eval_results["test_acc"].append(eval_dict['avg_acc'])
                        eval_results["avg_acc"].append(eval_dict['cls_acc'])
                        eval_results["data_cnt"].append(num_eval)
                        self.report_test(num_eval, eval_dict["avg_loss"], eval_dict['avg_acc'])
                        num_eval += self.eval_period

                sys.stdout.flush()

            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
            eval_dict = self.online_evaluate(test_dataloader)
            task_acc = eval_dict['avg_acc']

            print("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            print("[2-5] Report task result")
            print(task_records['task_acc'])

        # time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        # Accuracy (A)
        A_auc = np.mean(eval_results["test_acc"])
        A_avg = np.mean(task_records["task_acc"])
        A_last = task_records["task_acc"][self.n_tasks - 1]

        # Forgetting (F)
        cls_acc = np.array(task_records["cls_acc"]) # shape: (num_tasks, num_classes)
        acc_diff = []
        if self.n_tasks >1:
            for j in range(self.n_classes):
                if np.max(cls_acc[:-1, j]) > 0: 
                    acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
            F_last = np.mean(acc_diff)
        else:
            F_last = -999

        # ----------------------------- save log file -----------------------------
        log_dir = os.path.join(self.log_path, self.note)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'seed{str(self.rnd_seed)}_summary.txt')

        lines = []
        lines.append("======== Summary =======")
        lines.append(self.note)
        lines.append(f"random seed: {self.rnd_seed}")
        lines.append(f"logit type: {self.logit_type}")
        lines.append(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}\n")
        lines.append("A_auc after each task:")
        lines.append(str(task_records['task_acc']))
        lines.append("Accs after 1000 training samples:")
        lines.append(str(eval_results["test_acc"]))
        lines.append(f"Exposed class: {self.exposed_classes}")
        lines.append(f"Disjoint class idx: {self.train_sampler.disjoint_classes}")
        lines.append(f"Blurry class idx: {self.train_sampler.blurry_classes}")
        lines.append(f"Total training time: {total_time:.2f} seconds")
        lines.append("=" * 24)
        # ----------------------------- save log file -----------------------------

        with open(log_file, 'w') as f:
            for line in lines:
                print(line)
                f.write(line + '\n')

    def add_new_class(self, class_name):
        new = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
                new.append(label.item())

        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_evaluate(self, test_loader):
        raise NotImplementedError()

    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples*self.num_epochs-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        print(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects