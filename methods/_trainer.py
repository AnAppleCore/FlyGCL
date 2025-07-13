import datetime
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import *
from utils.augment import Cutout
from utils.data_loader import get_statistics
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
mp.set_sharing_strategy('file_system')


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:

        self.kwargs = kwargs
        self.__dict__.update(kwargs)

        self.start_time = time.time()
        self.eval_period = np.inf if self.eval_period < 0 else self.eval_period

        # Distributed training setup
        self.world_size = 1
        self.ngpus_per_nodes = torch.cuda.device_count()
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size  = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes

        self.distributed = self.world_size > 1
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        if self.distributed:
            self.batchsize = self.batchsize // self.world_size
        
        os.makedirs(f"{self.log_path}/logs/{self.dataset}/{self.note}", exist_ok=True)

        return

    def setup_distributed_dataset(self):

        self.datasets = DATASETS

        mean, std, n_classes, inp_size, in_channels = get_statistics(dataset=self.dataset)
        inp_size = 224 # override for ViT
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
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
        logger.info(f"Using train-transforms {train_transform}")
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        # Create tensor-compatible test transform for cases where input is already a tensor
        self.test_transform_tensor = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                # No ToTensor() since input is already a tensor
                transforms.Normalize(mean, std),])

        if 'imagenet' in self.dataset or 'cub' in self.dataset or 'car' in self.dataset:
            self.load_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()])
        else:
            self.load_transform = transforms.ToTensor()

        self.train_dataset = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=self.load_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        self.test_dataset = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)

        _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        self.train_sampler = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        self.train_dataloader = DataLoader(self.online_iter_dataset, batch_size=self.batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=0)
        self.test_sampler = OnlineTestSampler(self.test_dataset, [], _w, _r)

        self.seen = 0
        self.exposed_classes = []
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def setup_distributed_model(self):

        logger.info(f"Building model: {self.method}")
        self.model = select_model(self.method, self.backbone, self.n_classes, self.n_tasks, self.kwargs).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.model_without_ddp = self.model

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = getattr(self.model_without_ddp, "loss_fn", nn.CrossEntropyLoss(reduction="mean"))
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.lr_gamma = 0.99995 if 'imagenet' in self.dataset else 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logger.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n, p in self.model_without_ddp.named_parameters() if p.requires_grad]
        logger.info(f"Learnable Parameters :\t{n_params}")
        logger.info(learnables)
        logger.info("")

    def run(self):
        if self.profile:
            self.profile_worker(0)
        else:
            # Distributed Launch
            if self.ngpus_per_nodes > 1:
                mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
            else:
                self.main_worker(0)
    
    def main_worker(self, gpu) -> None:
        # ========= Distributed training setup =========
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
            logger.info(
                'You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.'
            )
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)

        logger.info(f"[1] Select a GCL method ({self.method})")
        self.setup_distributed_model()

        # =========== Incrementally training ===========
        logger.info(f"[2] Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period
        num_report = 2000
        report_period = 500

        for task_id in range(self.n_tasks):

            logger.info("\n")
            logger.info("#" * 50)
            logger.info(f"# Task {task_id} iteration")
            logger.info("#" * 50 + "\n")
            logger.info("[2-1] Prepare a datalist for the current task")

            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
                for i, (images, labels, idx) in enumerate(self.train_dataloader):
                    samples_cnt += images.size(0) * self.world_size

                    loss, acc = self.online_step(images, labels, idx)

                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += report_period
                        
                    if samples_cnt + images.size(0) * self.world_size > num_eval:
                        with torch.no_grad():
                            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                            eval_dict = self.online_evaluate(test_dataloader)
                            if self.distributed:
                                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                                eval_dict = eval_dict.cpu().numpy()
                                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
                            if self.is_main_process():
                                eval_results["test_acc"].append(eval_dict['avg_acc'])
                                eval_results["avg_acc"].append(eval_dict['cls_acc'])
                                eval_results["data_cnt"].append(num_eval)
                                self.report_test(num_eval, eval_dict["avg_loss"], eval_dict['avg_acc'])
                            num_eval += self.eval_period

                    sys.stdout.flush()
                
                test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                eval_dict = self.online_evaluate(test_dataloader, task_id=task_id, end=True)

            self.online_after_task(task_id)

            if self.distributed:
                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                eval_dict = eval_dict.cpu().numpy()
                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
            task_acc = eval_dict['avg_acc']

            logger.info("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            logger.info("[2-5] Report task result")
            logger.info(task_records['task_acc'])

        # ================== Summary ===================
        if self.is_main_process():

            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][self.n_tasks - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            if self.n_tasks >1:
                for j in range(self.n_classes):
                    if np.max(cls_acc[:-1, j]) > 0:
                        acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
                F_last = np.mean(acc_diff)
            else:
                F_last = -999
            logger.info(f"======== Summary =======")
            logger.info(self.note)
            logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            logger.info(f"="*24)
            logger.info(eval_results['test_acc'])
        
            np.save(f"{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

            if self.eval_period != np.inf:
                np.save(f'{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_eval.npy', eval_results['test_acc'])
                np.save(f'{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    def profile_worker(self, gpu) -> None:
        # ============ Toy experiment setup ============
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass
        
        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)

        self.setup_distributed_model()

        samples_cnt = 0
        self.train_sampler.set_task(0)
        self.online_before_task(0)
        for i, (images, labels, idx) in enumerate(self.train_dataloader):
            samples_cnt += images.size(0) * self.world_size
            loss, acc = self.online_step(images, labels, idx)
            self.report_training(samples_cnt, loss, acc)
            break
        self.online_after_task(0)

    def add_new_class(self, class_name):
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
        self.mask[:len(self.exposed_classes)] = 0

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
        raise NotImplementedError()
    
    def online_evaluate(self, test_loader, samples_cnt, task_id=None, end=False):
        raise NotImplementedError()

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()
            
    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables print, logging when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

        class MasterOnlyFilter(logging.Filter):
            def __init__(self, is_master):
                super().__init__()
                self.is_master = is_master
            
            def filter(self, record):
                return self.is_master or record.levelno < logging.INFO
        
        for h in logging.getLogger().handlers:
            h.addFilter(MasterOnlyFilter(is_master))

    def report_training(self, sample_num, train_loss, train_acc):
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples*self.num_epochs-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        logger.info(
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

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=self.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    
    def train_data_config(self, n_task, train_dataset, train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,batch_size=self.batchsize,sampler=train_sampler,num_workers=4)
            data_info={}
            for i,data in enumerate(train_dataloader):
                _,label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_'+str(label[b].item()) in data_info.keys():
                        data_info['Class_'+str(label[b].item())] += 1
                    else:
                        data_info['Class_'+str(label[b].item())] = 1
            logger.info(f"[Train] Task{t_i} Data Info")
            logger.info(data_info)
            convert_data_info = self.convert_class_label(data_info)
            np.save(f"{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_task{t_i}_train_data.npy", convert_data_info)
            logger.info(f"[Train] Task{t_i} Converted Data Info")
            logger.info(convert_data_info)
            logger.info("")
            
    def test_data_config(self, test_dataloader, task_id):
        data_info={}
        for i,data in enumerate(test_dataloader):
            _,label = data
            label = label.to(self.device)
            for b in range(len(label)):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())]+=1
                else:
                    data_info['Class_'+str(label[b].item())]=1
        logger.info("[Test] Exposed Classes:")
        logger.info(self.exposed_classes)
        logger.info(f"[Test] Task {task_id} Data Info")
        logger.info(data_info)
        logger.info(f"[Test] Task{task_id} Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")
        
    def convert_class_label(self,data_info):
        #* self.class_list => original class label
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key= int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)
        return data_info
    
    def current_task_data(self,train_loader):
        data_info={}
        for i,data in enumerate(train_loader):
            _,label = data
            for b in range(label.shape[0]):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())] +=1
                else:
                    data_info['Class_'+str(label[b].item())] =1
        logger.info("[Current Task] Data Info")
        logger.info(data_info)
        logger.info("[Current Task] Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")