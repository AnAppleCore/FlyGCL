import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import *
from methods._trainer import _Trainer
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.augment import Cutout
from utils.data_loader import get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

from fvcore.nn import FlopCountAnalysis


# https://github.com/facebookresearch/moco/blob/main/moco/loader.py#L13-L22
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, weak_transform, online_iter) -> None:
        self.weak_transform = weak_transform
        self.online_iter = online_iter

    def __call__(self, x):
        return_imgs = []
        for _ in range(int(self.online_iter)):
            weak1 = self.weak_transform(x)
            return_imgs.append(torch.stack([weak1]))
        return torch.stack(return_imgs)
    
class Trainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
    
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
        if self.model_name in ['vit', 'vit_base', 'DualPrompt', 'singlePrompt']:
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
                
        # weak aug
        self.train_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        print(f"Using train-transforms {train_transform}")
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        self.inp_size = inp_size
 
        self.train_dataset   = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=TwoCropsTransform(self.train_transform, self.online_iter))
        self.test_dataset    = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1) # datasets/OnlineIterDataset.py
        
        self.train_sampler   = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM)
        self.test_sampler    = OnlineTestSampler(self.test_dataset, [])
        
        if self.dataset == 'tinyimagenet': 
            self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=self.n_worker, drop_last=True) # num_workers=0
        else:
            self.train_dataloader    = DataLoader(self.online_iter_dataset, batch_size=self.temp_batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=self.n_worker) # num_workers=0
        
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def setup_distributed_model(self):
        print("Building model...")

        self.online_model = select_model('singlePrompt', self.dataset, self.n_classes, self.selection_size, self.pos_prompt, self.prompt_length, self.logit_type, self.kwargs).to(self.device)
                
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.online_model.to(self.device)
        
        self.model_without_ddp = self.online_model

        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.online_model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n,p in self.model_without_ddp.named_parameters() if p.requires_grad]
        print(learnables)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        #print(images.size()) # (B, online_iter, aug, 3, 224, 224)

        #self.curr_step += 1
        # inputs = torch.randn(1, 3, 224, 224).to(self.device)
        # flops = FlopCountAnalysis(self.online_model, inputs)
        # print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")
        for i in range(int(self.online_iter)):
            loss, acc = self.online_train([images[:, i].clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        del(images, labels)
        gc.collect()

        return _loss / _iter, _acc / _iter
    
    def online_train(self, data):
        self.online_model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data # x.size(): (B, 2, 3, 224, 224)
        if len(self.memory) > 0 and self.memory_batchsize > 0:
        # if self.memory_size > 0 and len(self.memory) > 0:
            memory_images, memory_labels = next(self.memory_provider)
            # for i in range(len(memory_labels)):
            #     memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images[:, 0]], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        x1  = x[:, 0].to(self.device)

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        y = y.to(self.device)
    
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.online_model(x1) + logit_mask
            loss = self.criterion(logit, y)
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        _, preds = logit.topk(self.topk, 1, True, True)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.online_model.eval()

        # for counting FLOPs
        # inputs = torch.randn(1, 3, 224, 224).to(self.device)
        # flops = FlopCountAnalysis(self.online_model, inputs)
        # print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)
                   
                logit = self.online_model(x)       

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
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()