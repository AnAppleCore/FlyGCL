import os
import sys
import random
import time
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random
from collections import defaultdict
import numpy as np
import torch
# from randaugment import RandAugment
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.augment import Cutout
from utils.data_loader import get_statistics
from datasets import *
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.memory import Memory, DummyMemory
import torch.cuda.profiler as profiler
from torch.utils.data import Subset

# import pyprof
# pyprof.init()

########################################################################################################################
# This is trainer with a DistributedDataParallel                                                                       #
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

        self.memory_size = kwargs.get("memory_size")
        self.log_path    = kwargs.get("log_path")
        self.model_name  = kwargs.get("model_name")
        self.opt_name    = kwargs.get("opt_name")
        self.sched_name  = kwargs.get("sched_name")
        self.batchsize  = kwargs.get("batchsize")
        self.n_worker    = kwargs.get("n_worker")
        self.lr  = kwargs.get("lr")

        self.init_model  = kwargs.get("init_model")
        self.init_opt    = kwargs.get("init_opt")
        self.topk    = kwargs.get("topk")
        self.use_amp = kwargs.get("use_amp")
        self.transforms  = kwargs.get("transforms")

        self.reg_coef    = kwargs.get("reg_coef")
        self.data_dir    = kwargs.get("data_dir")
        self.debug   = kwargs.get("debug")
        self.note    = kwargs.get("note")
        #* for Prompt Based
        self.selection_size = kwargs.get("selection_size")
        # self.alpha = kwargs.get("alpha")
        # self.gamma = kwargs.get("gamma")
        # self.beta = kwargs.get("beta")
        # self.charlie = kwargs.get("charlie")
        # self.use_baseline = kwargs.get("use_baseline")

        self.profile = kwargs.get("profile")        
        
        self.eval_period     = kwargs.get("eval_period")
        if self.eval_period < 0:
            self.eval_period = np.inf
        self.temp_batchsize  = kwargs.get("temp_batchsize")
        self.online_iter     = kwargs.get("online_iter")
        self.num_epochs     = kwargs.get("num_epochs")
        self.num_gpus    = kwargs.get("num_gpus")
        self.workers_per_gpu     = kwargs.get("workers_per_gpu")
        self.imp_update_period   = kwargs.get("imp_update_period")

        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        # self.dist_url = 'tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT']

        self.lr_step     = kwargs.get("lr_step")    # for adaptive LR
        self.lr_length   = kwargs.get("lr_length")  # for adaptive LR
        self.lr_period   = kwargs.get("lr_period")  # for adaptive LR

        self.memory_epoch    = kwargs.get("memory_epoch")    # for RM
        self.distilling  = kwargs.get("distilling") # for BiC
        self.agem_batch  = kwargs.get("agem_batch") # for A-GEM
        self.mir_cands   = kwargs.get("mir_cands")  # for MIR
        self.loadweakvit = kwargs.get("loadweakvit") 
        self.isa = kwargs.get("isa") 

        self.start_time = time.time()
        self.num_updates = 0
        self.train_count = 0

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size  = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes
        self.distributed     = self.world_size > 1
        if self.distributed:
            self.batchsize = self.batchsize // self.world_size
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batchsize // 2
        if self.temp_batchsize > self.batchsize:
            self.temp_batchsize = self.batchsize
        # self.memory_batchsize = 32 # = self.batchsize - self.temp_batchsize
        self.memory_batchsize  = self.batchsize - self.temp_batchsize

        self.exposed_classes = []
        
        os.makedirs(f"{self.log_path}/logs/{self.dataset}/{self.note}", exist_ok=True)
        os.makedirs(f"{self.log_path}/tensorboard/{self.dataset}/{self.note}", exist_ok=True)
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
        # if "randaug" in self.transforms:
        #     train_transform.append(RandAugment())
        #     if self.gpu_transform:
        #         self.gpu_transform = False
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

    def setup_distributed_model(self):

        print("Building model...")
        self.model = select_model(self.model_name, self.dataset, self.n_classes,self.selection_size, self.kwargs).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.loadweakvit:
            print('loading weaker checkpoint...')
            self.model.head = nn.Linear(384, 1000)
            checkpoint = torch.load('./checkpoints/pytorch_model.bin')
            
            
            if self.model_name == "vit":
                self.model.load_state_dict(checkpoint['model'], strict=False)
                for n, p in self.model.named_parameters():
                    if "fc." in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.model_name == "vit_finetune":
                # self.model.load_state_dict(checkpoint['model'], strict=False) # Imgn1k_excllude
                self.model.load_state_dict(checkpoint, strict=False)
                for name, param in self.model.named_parameters():
                    param.requires_grad = True
                self.model.head.weight.requires_grad = False
                self.model.head.bias.requires_grad = False
            elif self.model_name == "DualPrompt":
                self.model.backbone.load_state_dict(checkpoint, strict=False)
                for name, param in self.model.backbone.named_parameters():
                    param.requires_grad = False
                self.model.backbone.fc.weight.requires_grad = True
                self.model.backbone.fc.bias.requires_grad   = True
                self.model.head.weight.requires_grad = False
                self.model.head.bias.requires_grad = False
            elif self.model_name == "mvp":
                # self.model.backbone.load_state_dict(checkpoint['model'], strict=False)# Imgn1k_excllude
                self.model.backbone.load_state_dict(checkpoint, strict=False)
                for name, param in self.model.backbone.named_parameters():
                    param.requires_grad = False
                self.model.backbone.fc.weight.requires_grad = True
                self.model.backbone.fc.bias.requires_grad   = True
                self.model.head.weight.requires_grad = False
                self.model.head.bias.requires_grad = False

        
        self.model.to(self.device)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
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

        # self.memory = DummyMemory(datasize=self.memory_size, shape=(3,224,224))

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
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                print(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                print(f"| Init Process group 0 : {self.local_rank}")
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

        # macs, params = ptflops.get_model_complexity_info(self.model, (3, 224, 224), as_strings=True,
        #                                     print_per_layer_stat=True, verbose=True, flops_units='flops')
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


        num_eval = self.eval_period
        num_report = 2000
        num_save = 300000
        eval_period = 300000
        


        for task_id in range(self.n_tasks):

            # if task_id > 2:
            #     print('stopping at task 2 here')
            #     print(self.note )
            #     break
                

            if self.mode == "joint" and task_id > 0:
                return
            
            # if task_id ==0 and not self.debug:
            #     print()
            #     self.train_data_config(self.n_tasks,self.train_dataset,self.train_sampler)
            
            print("\n" + "#" * 50)
            print(f"# Task {task_id} iteration")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")
            
            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)     
            hessian_eig_db = {}    
            Flag=True
            for epoch in range(self.num_epochs):
                for i, (images, labels, idx) in enumerate(self.train_dataloader):
                    if self.debug and (i+1) * self.temp_batchsize >= 500:
                        break
                    samples_cnt += images.size(0) * self.world_size

                    loss, acc = self.online_step(images, labels, idx)
                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += 500
                        
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
                        
                        # Log eigenvalue
                        # for name, param in self.model.backbone.named_parameters():
                        #         param.requires_grad = True
                    
                        # for name, param in self.model.backbone.named_parameters():
                        #         param.requires_grad = False
                        # self.model.backbone.fc.weight.requires_grad = True
                        # self.model.backbone.fc.bias.requires_grad   = True
                            
                    if samples_cnt + images.size(0) * self.world_size > num_save and epoch<3:

                        num_save += eval_period

                        if self.isa:
                            # Save prompt
                            prompt_save_root = 'pretrained_prompt/'
                            save_folder = 'MISA'
                            if not os.path.exists(os.path.join(prompt_save_root, save_folder)):
                                # Create the directory if it does not exist
                                os.makedirs(os.path.join(prompt_save_root, save_folder))
                            print('saving to {}'.format(save_folder))
                            if self.model_name != 'CodaPrompt':
                                prompt_name = 'prompt'
                                if not prompt_name.endswith('.pt'):
                                    prompt_name += '.pt'
                                e_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_e_'.format(task_id,num_save) + prompt_name)
                                eproj_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_eproj_'.format(task_id,num_save) + prompt_name)
                                g_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_g_'.format(task_id,num_save) + prompt_name)
                                gproj_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_gproj_'.format(task_id,num_save) + prompt_name)
                                if self.distributed:
                                    torch.save(self.model.module.e_prompt, e_prompt_save_path)
                                    torch.save(self.model.module.g_prompt, g_prompt_save_path)
                                else:
                                    torch.save(self.model.e_prompt, e_prompt_save_path)
                                    torch.save(self.model.g_prompt, g_prompt_save_path)
                                    if self.model.proj_g_pt is not None:
                                        g_p_proj = self.model.proj_g_pt(self.model.g_prompt.prompts)+self.model.g_prompt.prompts
                                        # # g_p_proj = self.model.g_ratio * self.model.proj_g_pt(self.model.g_prompt.prompts)+self.model.g_prompt.prompts
                                        torch.save(g_p_proj, gproj_prompt_save_path)
                                    if self.model.proj_e_pt is not None:
                                        e_p_proj = self.model.proj_e_pt(self.model.e_prompt.prompts)+self.model.e_prompt.prompts
                                        # # e_p_proj = self.model.e_ratio * self.model.proj_e_pt(self.model.e_prompt.prompts)+self.model.e_prompt.prompts
                                        torch.save(e_p_proj, eproj_prompt_save_path)
                            else:
                                prompt_name = 'prompt.pt'
                                e_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_e_'.format(task_id,num_save) + prompt_name)
                                eproj_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_eproj_'.format(task_id,num_save) + prompt_name)

                                if self.distributed:
                                    for e in range(5):
                                        K = getattr(self.model.module,f'e_k_{e}')
                                        torch.save(K, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_K_{}_'.format(task_id,num_save,e) + prompt_name))
                                        A = getattr(self.model.module,f'e_a_{e}')
                                        torch.save(A, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_A_{}_'.format(task_id,num_save,e) + prompt_name))
                                        p = getattr(self.model.module,f'e_p_{e}')
                                        torch.save(p, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_p_{}_'.format(task_id,num_save,e) + prompt_name))

                                else:
                                    for e in range(5):
                                        K = getattr(self.model,f'e_k_{e}')
                                        torch.save(K, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_K_{}_'.format(task_id,num_save,e) + prompt_name))
                                        A = getattr(self.model,f'e_a_{e}')
                                        torch.save(A, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_A_{}_'.format(task_id,num_save,e) + prompt_name))
                                        p = getattr(self.model,f'e_p_{e}')
                                        torch.save(p, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_p_{}_'.format(task_id,num_save,e) + prompt_name))
                                    if self.model.proj_e_pt is not None:
                                        for e in range(5):
                                            p = getattr(self.model,f'e_p_{e}')
                                            p_proj = self.model.proj_e_pt(p)+p
                                            torch.save(p_proj, os.path.join(prompt_save_root, save_folder, 'T{}_sample{}_p_proj_{}_'.format(task_id,num_save,e) + prompt_name))


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

            print("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            print("[2-5] Report task result")
            print(task_records['task_acc'])

            

            if self.isa:
                # Save prompt
                prompt_save_root = 'pretrained_prompt/'
                save_folder = 'MISA'
                prompt_name = 'prompt'
                if not os.path.exists(os.path.join(prompt_save_root, save_folder)):
                    # Create the directory if it does not exist
                    os.makedirs(os.path.join(prompt_save_root, save_folder))
                print('saving to {}....'.format(os.path.join(prompt_save_root, save_folder)))
                if self.model_name != 'CodaPrompt':
                    if not prompt_name.endswith('.pt'):
                        prompt_name += '.pt'
                    e_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_e_'.format(task_id) + prompt_name)
                    eproj_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_eproj_'.format(task_id) + prompt_name)
                    g_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_g_'.format(task_id) + prompt_name)
                    gproj_prompt_save_path = os.path.join(prompt_save_root, save_folder, 'T{}_gproj_'.format(task_id) + prompt_name)
                    if self.distributed:
                        torch.save(self.model.module.e_prompt, e_prompt_save_path)
                        torch.save(self.model.module.g_prompt, g_prompt_save_path)
                    else:
                        torch.save(self.model.e_prompt, e_prompt_save_path)
                        torch.save(self.model.g_prompt, g_prompt_save_path)
                        if self.model.proj_g_pt is not None:
                            g_p_proj = self.model.proj_g_pt(self.model.g_prompt.prompts)+self.model.g_prompt.prompts
                            torch.save(g_p_proj, gproj_prompt_save_path)

                        if self.model.proj_e_pt is not None:
                            e_p_proj = self.model.proj_e_pt(self.model.e_prompt.prompts)+self.model.e_prompt.prompts
                            torch.save(e_p_proj, eproj_prompt_save_path)
                else:
                    prompt_name = 'prompt.pt'
                    for e in range(5):
                        K = getattr(self.model,f'e_k_{e}')
                        torch.save(K, os.path.join(prompt_save_root, save_folder, 'T{}_K_{}_'.format(task_id,e) + prompt_name))
                        A = getattr(self.model,f'e_a_{e}')
                        torch.save(A, os.path.join(prompt_save_root, save_folder, 'T{}_A_{}_'.format(task_id,e) + prompt_name))
                        p = getattr(self.model,f'e_p_{e}')
                        torch.save(p, os.path.join(prompt_save_root, save_folder, 'T{}_p_{}_'.format(task_id,e) + prompt_name))
                    if self.model.proj_e_pt is not None:
                        for e in range(5):
                            p = getattr(self.model,f'e_p_{e}')
                            p_proj = self.model.proj_e_pt(p)+p
                            torch.save(p_proj, os.path.join(prompt_save_root, save_folder, 'T{}_p_proj_{}_'.format(task_id,e) + prompt_name))

        if self.is_main_process():        

            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][self.n_tasks - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            # raise ValueError(cls_acc.shape) ValueError: (5, 200)
            acc_diff = []
            if self.n_tasks >1:
                for j in range(self.n_classes):
                    if np.max(cls_acc[:-1, j]) > 0:
                        acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
                F_last = np.mean(acc_diff)
            else:
                F_last = -999
            print(f"======== Summary =======")
            print(self.note)
            print(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            # for i in range(len(cls_acc)):
            #     print(f"Task {i}")
            #     print(cls_acc[i])
            print(f"="*24)
            print(eval_results['test_acc'])
        
            np.save(f"{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

            if self.eval_period is not None:
                np.save(f'{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_eval.npy', eval_results['test_acc'])
                np.save(f'{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_eval_time.npy', eval_results['data_cnt'])
    

    def profile_worker(self, gpu) -> None:
        self.memory = DummyMemory(datasize=self.memory_size, shape=(3,224,224))

        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                print(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                print(f"| Init Process group 0 : {self.local_rank}")
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

        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period
        
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
            for clas in exposed_classes:
                if clas not in self.exposed_classes:
                    self.exposed_classes.append(clas)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        # if len(new) > 0:
        #     for cc in new:
        #         self.mask[cc] = 0

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
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

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

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        # dist.all_gather(all_sizes, local_size, async_op=False)
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

        # dist.all_gather(all_qs_padded, item)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    
    def train_data_config(self, n_task, train_dataset,train_sampler):
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
            print(f"[Train] Task{t_i} Data Info")
            print(data_info);print()
            convert_data_info = self.convert_class_label(data_info)
            np.save(f"{self.log_path}/logs/{self.dataset}/{self.note}/seed_{self.rnd_seed}_task{t_i}_train_data.npy", convert_data_info)
            print(convert_data_info)
            
            print()
            
    def test_data_config(self, test_dataloader,task_id):
        data_info={}
        for i,data in enumerate(test_dataloader):
            _,label = data
            label = label.to(self.device)
            
            for b in range(len(label)):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())]+=1
                else:
                    data_info['Class_'+str(label[b].item())]=1
        
        print("<<Exposed Class>>")
        print(self.exposed_classes)
        
        print(f"[Test] Task {task_id} Data Info")
        print(data_info)
        print("<<Convert>>")
        convert_data_info = self.convert_class_label(data_info)
        print(convert_data_info)
        print()
        
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
        
        print("Current Task Data Info")
        print(data_info)
        print("<<Convert to str>>")
        convert_data_info = self.convert_class_label(data_info)
        print(convert_data_info)
        print()