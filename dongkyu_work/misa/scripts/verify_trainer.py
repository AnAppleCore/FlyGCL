"""Stage 7: CPU-only MISA Trainer integration smoke test with fake CIFAR-100.

Exercises REAL:
  config parser -> METHODS['misa'] -> shared _Trainer setup_distributed_dataset
  -> shared OnlineSampler/DataLoader -> MISA online_step + batch logit mask
  -> MISA Prompt/model loss -> Backward -> Adam optimizer step
  -> OnlineTestSampler -> MISA seen-class-masked evaluation -> task boundary + next update.

Uses an intentionally small 5-block substitute backbone (feature dim=768)
to avoid external ViT checkpoint downloads and keep CPU CI fast. Stage 6
already independently tested the full real ViT-B/16 with official prompts.
No GPU training or accuracy benchmark is performed.
"""

from pathlib import Path
from unittest.mock import patch
import math
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MiniImageDataset(torch.utils.data.Dataset):
    """In-memory synthetic images, 100 labels, no CIFAR download."""

    def __init__(self, root, train, download=True, transform=None):
        self.classes = list(range(100))
        self.targets = [i for i in range(100) for _ in range(2 if train else 1)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = int(self.targets[index])
        # A nonconstant pixel pattern helps check actual optimizer updates.
        pixel = np.empty((32, 32, 3), dtype=np.uint8)
        pixel[:, :, 0] = (label * 7 + index) % 256
        pixel[:, :, 1] = (label * 3 + np.arange(32)[None, :]) % 256
        pixel[:, :, 2] = (index * 5 + np.arange(32)[:, None]) % 256
        image = Image.fromarray(pixel, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ToyInteractionBlock(nn.Module):
    """Cheap differentiable attention proxy: prompts influence CLS feature."""

    def forward(self, x):
        return x + 0.1 * x.mean(dim=1, keepdim=True)


class LightweightBackbone(nn.Module):
    """Only used by Stage 7; preserves ViT prompt insertion interfaces."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.num_features = 768
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=32, stride=32)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, 768))
        self.pos_drop = nn.Identity()
        self.blocks = nn.Sequential(*(ToyInteractionBlock() for _ in range(5)))
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, num_classes)

    def patch_embed_forward(self, x):
        return x

    def forward_patch(self, x):
        z = self.patch_embed(x)
        return z.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=32, stride=32)

    def forward(self, x):
        y = self.proj(x)
        return y.flatten(2).transpose(1, 2)


def lightweight_backbone(*args, **kwargs):
    assert kwargs.get("pretrained") is True, "Production MISA must request pretrained backbone"
    net = LightweightBackbone(num_classes=kwargs["num_classes"])
    net.patch_embed = PatchEmbed()
    # 224x224 with patch32 => 49 patches + CLS token
    return net


def make_args():
    from configuration.config import base_parser

    args_argv = [
        "main.py", "--method", "misa", "--dataset", "cifar100",
        "--data_dir", "/unused/synthetic/data", "--n_tasks", "5",
        "--n", "50", "--m", "10", "--rnd_NM", "--batchsize", "2",
        "--online_iter", "1", "--num_epochs", "1", "--opt_name", "adam",
        "--sched_name", "default", "--lr", "0.005",
        "--eval_period", "1000", "--transforms", "autoaug",
        "--seeds", "1993", "--note", "misa_stage7_cpu",
        "--misa_prompt_dir", str(ROOT / "dongkyu_work/misa/checkpoints"),
    ]
    with patch.object(sys, "argv", args_argv):
        return base_parser()


def first_multiclass_batch(data_loader):
    """Pick an actual streamed batch with >=2 unique labels (fixed seed)."""
    for images, labels, indices in data_loader:
        if len(set(labels.tolist())) >= 2:
            return images, labels, indices
    raise AssertionError("No multi-class batch in this tiny Si-Blurry task stream")


def assert_batch_mask(mask, labels):
    assert isinstance(mask, torch.Tensor) and mask.ndim == 1
    actual = torch.nonzero(torch.isfinite(mask)).flatten().tolist()
    expected = sorted(set(int(c) for c in labels))
    assert actual == expected, f"Unexpected train batch mask: {actual} != {expected}"
    assert all(float(mask[i]) == 0 for i in expected)


def main():
    torch.manual_seed(1993)
    np.random.seed(1993)
    torch.set_num_threads(2)
    from methods import METHODS
    from utils.onlinesampler import OnlineTestSampler
    import methods._trainer as trainer_module
    import models.vit as vit

    args = make_args()
    assert args.method == "misa" and args.rnd_NM and args.batchsize == 2
    assert args.no_batchmask is False
    print("PASS CLI: registered MISA method + Si-Blurry config parsed", flush=True)

    # _Trainer initializes device/world-size from torch.cuda.device_count().
    # This CPU-only test explicitly supplies an equivalent single-worker CPU
    # device before shared setup. It does NOT claim main.py GPU entrypoint runs.
    with tempfile.TemporaryDirectory() as logs, \
            patch.object(trainer_module, "DATASETS", {"cifar100": MiniImageDataset}), \
            patch.object(vit, "vit_base_patch16_224", side_effect=lightweight_backbone):
        cfg = vars(args).copy()
        cfg.update({"rnd_seed": 1993, "log_path": logs})
        trainer = METHODS["misa"](**cfg)
        trainer.device = torch.device("cpu")
        trainer.world_size = 1
        trainer.ngpus_per_nodes = 1
        trainer.distributed = False

        trainer.setup_distributed_dataset()
        assert type(trainer.train_sampler).__name__ == "OnlineSampler"
        assert isinstance(trainer.train_dataloader, DataLoader)
        assert trainer.online_iter_dataset.classes == list(range(100))
        trainer.setup_distributed_model()
        assert type(trainer.model_without_ddp).__name__ == "MISA"
        assert tuple(trainer.model_without_ddp.g_prompt.prompts.shape) == (1, 20, 768)
        assert tuple(trainer.model_without_ddp.e_prompt.prompts.shape) == (10, 60, 768)
        print("PASS setup: real _Trainer, synthetic dataset, shared DataLoader/Sampler, official Prompt", flush=True)

        from dongkyu_work.misa.implementation.trainer import MISA as MISACls
        assert isinstance(trainer, MISACls)
        assert trainer.criterion == trainer.model_without_ddp.loss_fn

        observed = []
        original_forward = trainer.model_forward

        def inspect_model_forward(x, y, mask=None):
            assert mask is not None, "MISA requires batch logit mask by default"
            assert_batch_mask(mask, y.tolist())
            observed.append((mask.detach().clone(), y.detach().clone()))
            output, loss = original_forward(x, y, mask=mask)
            assert torch.isfinite(loss).item(), "MISA batch loss nonfinite"
            assert torch.isfinite(output[:, y.unique()]).all().item()
            assert torch.isneginf(output[:, mask.isneginf()]).all().item()
            return output, loss

        with patch.object(trainer, "model_forward", side_effect=inspect_model_forward):
            trainer.train_sampler.set_task(0)
            trainer.online_before_task(0)
            images, labels, indices = first_multiclass_batch(trainer.train_dataloader)
            assert len(set(labels.tolist())) >= 2, "Need two batch classes to test masked CE"
            fc_before = trainer.model_without_ddp.backbone.fc.weight.detach().clone()
            prompt_before = trainer.model_without_ddp.e_prompt.prompts.detach().clone()
            opt_before = trainer.optimizer.state.get(trainer.model_without_ddp.backbone.fc.weight)
            assert opt_before is None or not opt_before, "Adam state should start empty"

            loss0, acc0 = trainer.online_step(images, labels, indices)
            assert math.isfinite(loss0) and 0 <= acc0 <= 1
            assert len(observed) == 1
            assert len(trainer.exposed_classes) >= 2
            assert trainer.optimizer.state[trainer.model_without_ddp.backbone.fc.weight]["step"].item() == 1
            assert not torch.equal(fc_before, trainer.model_without_ddp.backbone.fc.weight.detach())
            assert not torch.equal(prompt_before, trainer.model_without_ddp.e_prompt.prompts.detach())
            assert trainer.model_without_ddp.task_count == 0
            print("PASS online_step: batch mask -> Loss -> Backward -> Adam.step, FC & E-Prompt updated", flush=True)

            test_sampler = OnlineTestSampler(trainer.test_dataset, trainer.exposed_classes)
            test_loader = DataLoader(trainer.test_dataset, sampler=test_sampler, batch_size=2)
            test_result = trainer.online_evaluate(test_loader, task_id=0, end=True)
            assert set(test_result) == {"avg_loss", "avg_acc", "cls_acc"}
            assert math.isfinite(test_result["avg_loss"])
            assert 0 <= test_result["avg_acc"] <= 1
            assert len(test_result["cls_acc"]) == 100
            print("PASS evaluation: shared exposed-class test sampler + MISA seen-class-masked metrics", flush=True)

            trainer.online_after_task(0)
            assert trainer.model_without_ddp.task_count == 1
            trainer.train_sampler.set_task(1)
            trainer.online_before_task(1)
            second_images, second_labels, second_indices = first_multiclass_batch(trainer.train_dataloader)
            assert len(set(second_labels.tolist())) >= 2
            loss1, acc1 = trainer.online_step(second_images, second_labels, second_indices)
            assert math.isfinite(loss1) and 0 <= acc1 <= 1
            assert trainer.model_without_ddp.task_count == 1
            assert trainer.model_without_ddp.e_prompt.last_topk is not None
            assert trainer.model_without_ddp.e_prompt.last_topk.max().item() < 2
            assert len(observed) == 2
            assert trainer.optimizer.state[trainer.model_without_ddp.backbone.fc.weight]["step"].item() == 2
            print("PASS task boundary: task_count=1, task-1 DataLoader, second Adam update", flush=True)

    print("PASS Stage 7: real MISA Trainer pipeline with two toy online updates (CPU only)", flush=True)
    print("LIMIT: synthetic images + small proxy backbone; no real dataset, full ViT/GPU, or accuracy benchmark")


if __name__ == "__main__":
    main()
