import argparse
import logging
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import models.vit  # noqa: F401
from models.experts import LoRAExpert
from models.flyprompt_variants import FlyAdapterExpert


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ClassSubsetImageFolder(Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, class_ids: Sequence[int], transform=None):
        self.base_dataset = base_dataset
        self.class_ids = list(class_ids)
        self.class_id_set = set(self.class_ids)
        self.transform = transform
        self.indices = [idx for idx, (_, target) in enumerate(base_dataset.samples) if target in self.class_id_set]
        if len(self.indices) == 0:
            raise ValueError(f"No ImageFolder samples found for class ids: {self.class_ids[:10]}...")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, target = self.base_dataset.samples[self.indices[index]]
        image = self.base_dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ResampledClassSubsetImageFolder(ClassSubsetImageFolder):
    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        candidate_class_ids: Sequence[int],
        active_num_classes: int,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.candidate_class_ids = list(candidate_class_ids)
        self.active_num_classes = active_num_classes
        self.transform = transform
        self.class_ids = []
        self.class_id_set = set()
        self.indices = []
        self.resample_classes()

    def set_active_classes(self, active_classes: Sequence[int]):
        self.class_ids = sorted(active_classes)
        self.class_id_set = set(self.class_ids)
        self.indices = [
            idx for idx, (_, target) in enumerate(self.base_dataset.samples)
            if target in self.class_id_set
        ]
        if len(self.indices) == 0:
            raise ValueError(f"No ImageFolder samples found for active OOD class ids: {self.class_ids}")

    def resample_classes(self):
        if self.active_num_classes >= len(self.candidate_class_ids):
            active_classes = list(self.candidate_class_ids)
        else:
            active_classes = random.sample(self.candidate_class_ids, self.active_num_classes)
        self.set_active_classes(active_classes)


class PaperFAM(torch.optim.Optimizer):
    """Two-step FAM-style optimizer with configurable perturbation sign.

    Step 1 (first_step):
        - Perturb parameters along the OOD gradient direction scaled by rho.
        - ``perturb_sign=-1.0`` (default) descends along the OOD gradient
          (simulating one OOD gradient-descent step before the ID update).
        - ``perturb_sign=+1.0`` ascends along the OOD gradient (SAM-style
          worst-case perturbation).
    Step 2 (second_step):
        - Restore the original parameter values and apply the base
          optimizer step using the gradients accumulated from the ID pass.
    """

    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, perturb_sign=-1.0, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, perturb_sign=perturb_sign, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                perturb = (torch.pow(param, 2) if group["adaptive"] else 1.0) * param.grad * scale.to(param)
                param.add_(perturb, alpha=group["perturb_sign"])
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if "old_p" in self.state[param]:
                    param.data = self.state[param]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grads = [
            ((torch.abs(param) if group["adaptive"] else 1.0) * param.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if len(grads) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(grads), p=2)


class PromptAugmenter(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        return prompt + self.net(prompt)


class FlyPromptMISAInitModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_init_classes: int,
        len_prompt: int,
        pos_prompt: Iterable[int],
        aug_hidden_dim: int = None,
        expert_type: str = "prompt",
        fly_lora_rank: int = 5,
        fly_lora_alpha: float = 1.0,
        fly_lora_layers: int = 5,
        fly_adapter_down_dim: int = 10,
        fly_adapter_layers: int = 5,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.expert_type = expert_type
        self.len_prompt = len_prompt
        self.register_buffer("pos_prompt", torch.tensor(list(pos_prompt), dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_init_classes)
        self.embed_dim = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = aug_hidden_dim or self.embed_dim
        if self.expert_type == "prompt":
            self.prompt = nn.Parameter(torch.empty(self.num_layers, 1, len_prompt, self.embed_dim))
            nn.init.uniform_(self.prompt)
            self.prompt_augmenter = PromptAugmenter(self.embed_dim, hidden_dim)
            self.expert = None
        elif self.expert_type == "adapter":
            self.prompt = None
            self.prompt_augmenter = None
            self.expert = FlyAdapterExpert(
                num_experts=1,
                embed_dim=self.embed_dim,
                num_adapter_layers=fly_adapter_layers,
                adapter_down_dim=fly_adapter_down_dim,
            )
        elif self.expert_type == "lora":
            self.prompt = None
            self.prompt_augmenter = None
            self.expert = LoRAExpert(
                num_experts=1,
                embed_dim=self.embed_dim,
                num_lora_layers=fly_lora_layers,
                lora_rank=fly_lora_rank,
                lora_alpha=fly_lora_alpha,
            )
        else:
            raise ValueError(f"Unsupported expert_type: {self.expert_type}")
        self.classifier = nn.Linear(self.embed_dim, num_init_classes)

    def get_augmented_prompt(self) -> torch.Tensor:
        if self.expert_type != "prompt":
            raise RuntimeError("get_augmented_prompt is only available for prompt expert pretraining.")
        return self.prompt_augmenter(self.prompt)

    def _build_batched_prompts(self, batch_size: int) -> torch.Tensor:
        prompt = self.get_augmented_prompt().expand(-1, batch_size, -1, -1)
        prompt = prompt.permute(1, 0, 2, 3).contiguous()
        pos_bias = self.backbone.pos_embed[:, :1, :].unsqueeze(1).expand(
            batch_size,
            self.num_layers,
            self.len_prompt,
            self.embed_dim,
        )
        return prompt + pos_bias

    def forward_features_with_prompt(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        batch_size = x.size(0)
        cls_token = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        orig_num_tokens = x.size(1)

        prompts = self._build_batched_prompts(batch_size)
        for block_idx, block in enumerate(self.backbone.blocks):
            prompt_pos = (self.pos_prompt.eq(block_idx)).nonzero(as_tuple=False).flatten()
            if prompt_pos.numel() != 0:
                prompt_tokens = prompts.index_select(dim=1, index=prompt_pos.to(prompts.device))
                prompt_tokens = prompt_tokens.flatten(start_dim=1, end_dim=2)
                x = torch.cat((x, prompt_tokens), dim=1)
            x = block(x)
            x = x[:, :orig_num_tokens, :]

        x = self.backbone.norm(x)
        return x[:, 0]

    def forward_features_with_expert(self, inputs: torch.Tensor) -> torch.Tensor:
        expert_ids = torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)
        return self.expert(self.backbone, inputs, expert_ids)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.expert_type == "prompt":
            features = self.forward_features_with_prompt(inputs)
        else:
            features = self.forward_features_with_expert(inputs)
        return self.classifier(features)


class InfiniteLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.loader = None
        self.sampler = None
        self.iterator = None
        self._reset_loader()

    def _reset_loader(self):
        self.sampler = None
        if self.distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.sampler.set_epoch(self.epoch)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.iterator = iter(self.loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if isinstance(self.dataset, ResampledClassSubsetImageFolder):
                self.dataset.resample_classes()
                if self.distributed:
                    device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    classes = torch.tensor(self.dataset.class_ids, dtype=torch.long, device=device)
                    dist.broadcast(classes, src=0)
                    if self.rank != 0:
                        self.dataset.set_active_classes(classes.cpu().tolist())
                if self.rank == 0:
                    logger.info("Resampled OOD classes: %s", self.dataset.class_ids)
            self._reset_loader()
            return next(self.iterator)


def resolve_train_root(imagenet_root: str) -> str:
    root = Path(imagenet_root)
    train_root = root / "train"
    if train_root.is_dir():
        return str(train_root)
    return str(root)


def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    id_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')),
        transforms.ToTensor(),
        normalize,
    ])
    ood_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')),
        transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')),
        transforms.ToTensor(),
        normalize,
    ])
    return id_transform, ood_transform


def build_loaders(args):
    train_root = resolve_train_root(args.imagenet_root)
    metadata_dataset = datasets.ImageFolder(train_root)
    num_classes = len(metadata_dataset.classes)
    if num_classes < args.num_init_classes:
        raise ValueError(
            f"Expected at least {args.num_init_classes} ImageNet classes, found {num_classes} at {train_root}."
        )

    id_transform, ood_transform = build_transforms(args.image_size)
    id_dataset = ClassSubsetImageFolder(
        metadata_dataset,
        class_ids=range(args.num_id_classes),
        transform=id_transform,
    )
    ood_dataset = ResampledClassSubsetImageFolder(
        metadata_dataset,
        candidate_class_ids=range(args.num_id_classes, args.num_init_classes),
        active_num_classes=args.ood_active_classes,
        transform=ood_transform,
    )
    if args.distributed:
        device = torch.device(args.device)
        initial_ood_classes = torch.tensor(ood_dataset.class_ids, dtype=torch.long, device=device)
        dist.broadcast(initial_ood_classes, src=0)
        if args.rank != 0:
            ood_dataset.set_active_classes(initial_ood_classes.cpu().tolist())

    id_sampler = None
    if args.distributed:
        id_sampler = DistributedSampler(
            id_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )

    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=(id_sampler is None),
        sampler=id_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.rank == 0:
        logger.info("ImageNet train root: %s", train_root)
        logger.info("ID samples: %d | OOD active samples: %d", len(id_dataset), len(ood_dataset))
        logger.info("Initial OOD classes: %s", ood_dataset.class_ids)
    return metadata_dataset, id_loader, InfiniteLoader(
        dataset=ood_dataset,
        batch_size=args.ood_batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        world_size=args.world_size,
        rank=args.rank,
    ), id_sampler


def save_prompt_checkpoint(
    args,
    model: FlyPromptMISAInitModel,
    metadata_dataset: datasets.ImageFolder,
    save_path: str = None,
    epoch: int = None,
):
    save_path = Path(save_path or args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "expert_type": args.expert_type,
        "num_experts": 1,
        "embed_dim": model_to_save.embed_dim,
        "backbone": args.backbone,
        "init_method": "flyprompt_misa_isa_fam_aug",
        "epoch": epoch,
        "epochs": args.epochs,
        "fam_perturb_sign": 1.0 if args.fam_perturb_add else -1.0,
        "id_classes": list(range(args.num_id_classes)),
        "ood_classes": list(range(args.num_id_classes, args.num_init_classes)),
        "class_to_idx": metadata_dataset.class_to_idx,
    }
    if args.expert_type == "prompt":
        base_prompt = model_to_save.get_augmented_prompt().detach().cpu()
        checkpoint.update({
            "prompts": base_prompt,
            "base_prompt": base_prompt,
            "len_prompt": args.len_prompt,
            "pos_prompt": list(args.pos_prompt),
        })
        torch.save(checkpoint, save_path)
        logger.info("Saved FlyPrompt MISA prompt checkpoint to %s", save_path)
        logger.info("Saved base prompt shape: %s", tuple(base_prompt.shape))
    else:
        checkpoint.update({
            "expert_state_dict": {k: v.detach().cpu() for k, v in model_to_save.expert.state_dict().items()},
            "fly_lora_rank": args.fly_lora_rank,
            "fly_lora_alpha": args.fly_lora_alpha,
            "fly_lora_layers": args.fly_lora_layers,
            "fly_adapter_down_dim": args.fly_adapter_down_dim,
            "fly_adapter_layers": args.fly_adapter_layers,
        })
        torch.save(checkpoint, save_path)
        logger.info("Saved FlyPrompt MISA %s expert checkpoint to %s", args.expert_type, save_path)


def reduce_scalar(value: float, device: torch.device, average: bool = True) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= dist.get_world_size()
    return tensor.item()


def train(args):
    torch.manual_seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    metadata_dataset, id_loader, ood_provider, id_sampler = build_loaders(args)
    model = FlyPromptMISAInitModel(
        backbone_name=args.backbone,
        num_init_classes=args.num_init_classes,
        len_prompt=args.len_prompt,
        pos_prompt=args.pos_prompt,
        aug_hidden_dim=args.aug_hidden_dim,
        expert_type=args.expert_type,
        fly_lora_rank=args.fly_lora_rank,
        fly_lora_alpha=args.fly_lora_alpha,
        fly_lora_layers=args.fly_lora_layers,
        fly_adapter_down_dim=args.fly_adapter_down_dim,
        fly_adapter_layers=args.fly_adapter_layers,
    ).to(device)
    model_embed_dim = model.embed_dim

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    perturb_sign = 1.0 if args.fam_perturb_add else -1.0
    optimizer = PaperFAM(
        trainable_params,
        optim.Adam,
        lr=args.lr,
        rho=args.rho,
        weight_decay=args.weight_decay,
        perturb_sign=perturb_sign,
    )
    criterion = nn.CrossEntropyLoss()

    if args.rank == 0:
        logger.info("Backbone: %s | embed_dim: %d | expert_type: %s", args.backbone, model_embed_dim, args.expert_type)
        logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))
        logger.info("FAM perturbation sign: %s", "positive/add" if args.fam_perturb_add else "negative/sub")
        if args.distributed:
            logger.info(
                "DDP enabled | world_size: %d | per-rank ID batch: %d | global ID batch: %d | per-rank OOD batch: %d | global OOD batch: %d",
                args.world_size,
                args.batch_size,
                args.batch_size * args.world_size,
                args.ood_batch_size,
                args.ood_batch_size * args.world_size,
            )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        if id_sampler is not None:
            id_sampler.set_epoch(epoch)
        running_loss_id = 0.0
        running_loss_ood = 0.0
        running_correct = 0.0
        running_total = 0.0

        for step, (id_images, id_labels) in enumerate(id_loader):
            id_images = id_images.to(device, non_blocking=True)
            id_labels = id_labels.to(device, non_blocking=True)
            ood_images, ood_labels = ood_provider.next()
            ood_images = ood_images.to(device, non_blocking=True)
            ood_labels = ood_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                ood_logits = model(ood_images)
                loss_ood = criterion(ood_logits, ood_labels)
            loss_ood.backward()
            optimizer.first_step(zero_grad=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                id_logits = model(id_images)
                loss_id = criterion(id_logits, id_labels)
            loss_id.backward()
            optimizer.second_step(zero_grad=True)

            preds = id_logits.argmax(dim=-1)
            batch_correct = (preds == id_labels).sum().item()
            batch_total = id_labels.numel()
            batch_loss_id = loss_id.item()
            batch_loss_ood = loss_ood.item()

            if args.distributed:
                batch_correct = reduce_scalar(batch_correct, device, average=False)
                batch_total = reduce_scalar(batch_total, device, average=False)
                batch_loss_id = reduce_scalar(batch_loss_id, device, average=True)
                batch_loss_ood = reduce_scalar(batch_loss_ood, device, average=True)

            running_correct += batch_correct
            running_total += batch_total
            running_loss_id += batch_loss_id
            running_loss_ood += batch_loss_ood
            global_step += 1

            if args.rank == 0 and args.log_interval > 0 and global_step % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Step %d | loss_id %.4f | loss_ood %.4f | id_acc %.4f",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    running_loss_id / max(step + 1, 1),
                    running_loss_ood / max(step + 1, 1),
                    running_correct / max(running_total, 1),
                )

        if args.rank == 0:
            logger.info(
                "Epoch %d complete | loss_id %.4f | loss_ood %.4f | id_acc %.4f",
                epoch + 1,
                running_loss_id / max(len(id_loader), 1),
                running_loss_ood / max(len(id_loader), 1),
                running_correct / max(running_total, 1),
            )
            if args.save_every_epoch:
                base_save_path = Path(args.save_path)
                epoch_save_path = base_save_path.parent / f"epoch_{epoch + 1:03d}" / base_save_path.name
                save_prompt_checkpoint(args, model, metadata_dataset, save_path=str(epoch_save_path), epoch=epoch + 1)

    if args.rank == 0:
        save_prompt_checkpoint(args, model, metadata_dataset, epoch=args.epochs)
    if args.distributed:
        dist.barrier()


def init_distributed_args(args):
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.device != "cpu":
            torch.cuda.set_device(args.local_rank)
            args.device = f"cuda:{args.local_rank}"
        dist.init_process_group(backend="nccl")
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    return args


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone MISA-style prompt pretraining for FlyPrompt")
    parser.add_argument("--imagenet_root", type=str, required=True, help="Path to ImageNet-1k root or train directory")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num_init_classes", type=int, default=1000)
    parser.add_argument("--num_id_classes", type=int, default=900)
    parser.add_argument("--ood_active_classes", type=int, default=10)
    parser.add_argument("--expert_type", type=str, default="prompt", choices=["prompt", "adapter", "lora"])
    parser.add_argument("--len_prompt", type=int, default=20)
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--aug_hidden_dim", type=int, default=None)
    parser.add_argument("--fly_lora_rank", type=int, default=5)
    parser.add_argument("--fly_lora_alpha", type=float, default=1.0)
    parser.add_argument("--fly_lora_layers", type=int, default=5)
    parser.add_argument("--fly_adapter_down_dim", type=int, default=10)
    parser.add_argument("--fly_adapter_layers", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ood_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--fam_perturb_add",
        action="store_true",
        default=False,
        help="Use positive/add OOD-gradient perturbation. Default uses negative/sub perturbation to simulate one OOD gradient-descent step.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./checkpoints/FlyPrompt_MISA_Pretrain_Prompt/flyprompt_misa_prompt.pt")
    parser.add_argument("--save_every_epoch", action="store_true", default=False)
    args = parser.parse_args()
    args = init_distributed_args(args)
    if args.ood_batch_size is None:
        args.ood_batch_size = args.batch_size
    if args.use_amp:
        raise ValueError("--use_amp is disabled for this standalone FAM pretraining script to avoid unscaled two-step FAM gradients.")
    if args.num_id_classes >= args.num_init_classes:
        raise ValueError("num_id_classes must be smaller than num_init_classes to form an OOD split.")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        train(parsed_args)
    finally:
        cleanup_distributed(parsed_args)
