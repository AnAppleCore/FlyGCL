"""Stage 3: SinglePrompt real FlyGCL CLI + shared Si-Blurry CPU smoke test.

Exercises the real argument parser, method/model registry, shared OnlineSampler
and DataLoader, DualPrompt training logic (batch mask, backward, Adam), task
boundaries, both evaluation policies, and full main.py metric saving.
Only CPU device / synthetic images / tiny substitute ViT are test-only shims.
"""
from pathlib import Path
import math
import runpy
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SyntheticCIFAR(torch.utils.data.Dataset):
    """100 real label IDs, two fake train images per class."""
    def __init__(self, root, train, download=True, transform=None):
        self.classes = list(range(100))
        self.targets = [i for i in range(100) for _ in range(2 if train else 1)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        cls = int(self.targets[i])
        a = np.empty((32, 32, 3), dtype=np.uint8)
        a[:, :, 0] = (7 * cls + i) % 256
        a[:, :, 1] = (np.arange(32)[None, :] + 3 * cls) % 256
        a[:, :, 2] = (np.arange(32)[:, None] + 11 * i) % 256
        image = Image.fromarray(a, mode="RGB")
        return (self.transform(image) if self.transform else image), cls


def proxy_vit(*args, **kwargs):
    """Small, genuine FlyGCL ViT blocks, not a random logits stub."""
    assert kwargs.get("pretrained") is True
    from models.vit import VisionTransformer
    return VisionTransformer(
        img_size=224, patch_size=32, embed_dim=32, depth=5,
        num_heads=4, num_classes=kwargs["num_classes"],
        qkv_bias=True, drop_rate=0.0, drop_path_rate=0.0,
    )


class CpuDeviceShim:
    def device(self, value, *args, **kwargs):
        if isinstance(value, int):
            assert value == 0
            return torch.device("cpu")
        return torch.device(value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(torch, name)


class UnseenWins(nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.size(0), 100, device=x.device)
        logits[:, 0] = 2
        logits[:, 50] = 10  # unseen class must win only in raw mode
        return logits


def check_eval_policies():
    from methods import METHODS
    from configuration.config import base_parser

    with tempfile.TemporaryDirectory() as td:
        with patch.object(sys, "argv", [
            "main.py", "--method", "singleprompt", "--dataset", "cifar100",
            "--log_path", td, "--seeds", "1993",
        ]):
            args = base_parser()
        assert args.singleprompt_eval_mode == "seen"
        assert args.singleprompt_logit_type == "cos_sim"
        assert args.pos_prompt == [0, 1, 2, 3, 4] and args.len_prompt == 20
        method = METHODS["singleprompt"](**vars(args))
        method.device = torch.device("cpu")
        method.n_classes = 100
        method.exposed_classes = [3, 5]
        method.mask = torch.full((100,), -torch.inf)
        method.mask[:2] = 0
        method.model = UnseenWins()
        method.model_without_ddp = method.model
        method.criterion = nn.CrossEntropyLoss(reduction="mean")
        data = [(torch.zeros(1, 3, 8, 8), torch.tensor([3]))]

        seen = method.online_evaluate(data)
        assert seen["avg_acc"] == 1.0
        assert math.isfinite(seen["avg_loss"])
        reference_mask = method.mask
        method.singleprompt_eval_mode = "raw"
        # Shared evaluator maps label tensors in place; provide a fresh batch.
        raw = method.online_evaluate([(torch.zeros(1, 3, 8, 8), torch.tensor([3]))])
        assert raw["avg_acc"] == 0.0
        assert raw["avg_loss"] > seen["avg_loss"]
        assert method.mask is reference_mask, "Evaluation mode leaked into train mask"
        assert set(raw) == set(seen) == {"avg_loss", "avg_acc", "cls_acc"}
        print("PASS: aligned seen-class and official raw-logit modes distinct", flush=True)

        for overrides in ({"no_batchmask": True},
                          {"use_rp_gate": True},
                          {"use_ema_head": True},
                          {"step_num": 10}):
            opts = vars(args).copy()
            opts.update(overrides)
            try:
                METHODS["singleprompt"](**opts)
            except ValueError:
                continue
            raise AssertionError("Invalid SinglePrompt option accepted: " + repr(overrides))
        print("PASS: rejects silent algorithm deviations", flush=True)


def check_real_main_entrypoint():
    from methods import METHODS
    from models import MODELS
    from methods import _trainer as shared_trainer
    from models import vit as shared_vit

    assert "singleprompt" in MODELS and "singleprompt" in METHODS

    base = METHODS["singleprompt"]
    original_worker = base.main_worker
    globals_copy = original_worker.__globals__.copy()
    assert globals_copy["torch"] is torch
    globals_copy["torch"] = CpuDeviceShim()
    cpu_worker = types.FunctionType(
        original_worker.__code__, globals_copy,
        original_worker.__name__, original_worker.__defaults__,
        original_worker.__closure__,
    )
    assert cpu_worker.__code__ is original_worker.__code__

    calls = {"updates": 0, "forward_masks": 0, "after_tasks": [], "trainer": None}
    original_step = base.online_step
    original_after = base.online_after_task
    original_forward = base.model_forward
    original_setup = base.setup_distributed_model

    def observed_setup(self):
        original_setup(self)
        assert type(self.model_without_ddp).__name__ == "SinglePrompt"
        assert len(self.model_without_ddp.backbone.blocks) == 5
        assert len(self.model_without_ddp.pos_prompt) == 5
        assert set(name for name, p in self.model_without_ddp.named_parameters()
                   if p.requires_grad) == {"prompt", "fc.weight"}
        calls["trainer"] = self
        calls["first_prompt"] = self.model_without_ddp.prompt.detach().clone()
        calls["first_fc"] = self.model_without_ddp.fc.weight.detach().clone()
        calls["prompt_id"] = id(self.model_without_ddp.prompt)

    def observed_forward(self, x, y, mask=None):
        assert mask is not None, "train step must use mini-batch mask"
        observed = torch.nonzero(torch.isfinite(mask)).flatten().tolist()
        expected = sorted(set(y.cpu().tolist()))
        assert observed == expected, f"train mask mismatch: {observed} / {expected}"
        calls["forward_masks"] += 1
        return original_forward(self, x, y, mask=mask)

    def observed_step(self, images, labels, idx):
        calls["updates"] += 1
        return original_step(self, images, labels, idx)

    def observed_after(self, task_id):
        calls["after_tasks"].append(task_id)
        assert id(self.model_without_ddp.prompt) == calls["prompt_id"]
        return original_after(self, task_id)

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="singleprompt-stage3-") as td:
        opts = [
            "main.py", "--method", "singleprompt", "--dataset", "cifar100",
            "--backbone", "vit_base_patch16_224",
            "--data_dir", td, "--log_path", td, "--note", "singleprompt_cpu",
            "--n_tasks", "5", "--n", "50", "--m", "10", "--rnd_NM",
            "--batchsize", "4", "--online_iter", "1", "--num_epochs", "1",
            "--lr", "0.005", "--opt_name", "adam",
            "--sched_name", "default", "--eval_period", "40",
            "--transforms", "autoaug", "--seeds", "1993",
            "--singleprompt_eval_mode", "seen",
        ]
        with patch.object(sys, "argv", opts), \
             patch.object(shared_trainer, "DATASETS", {"cifar100": SyntheticCIFAR}), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(shared_vit, "vit_base_patch16_224", side_effect=proxy_vit), \
             patch.object(base, "main_worker", cpu_worker), \
             patch.object(base, "setup_distributed_model", observed_setup), \
             patch.object(base, "online_step", observed_step), \
             patch.object(base, "model_forward", observed_forward), \
             patch.object(base, "online_after_task", observed_after):
            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")

        output = Path(td) / "logs/cifar100/singleprompt_cpu"
        # _Trainer log directory is log_path/logs/dataset/note, where
        # task metrics are saved at the same location.
        files = [output / "seed_1993.npy",
                 output / "seed_1993_eval.npy",
                 output / "seed_1993_eval_time.npy"]
        for f in files:
            assert f.exists(), f"Missing real shared Trainer metric file: {f}"
        task_acc, eval_acc, eval_time = (np.load(f) for f in files)
        assert task_acc.shape == (5,), task_acc.shape
        assert all(np.isfinite(arr).all() for arr in (task_acc, eval_acc, eval_time))
        assert len(eval_acc) > 0 and eval_time.shape == eval_acc.shape
        assert calls["after_tasks"] == [0, 1, 2, 3, 4]
        assert calls["updates"] > 0 and calls["updates"] == calls["forward_masks"]
        assert calls["trainer"].task_id == 5
        model = calls["trainer"].model_without_ddp
        assert id(model.prompt) == calls["prompt_id"]
        assert not torch.equal(model.prompt.detach(), calls["first_prompt"])
        assert not torch.equal(model.fc.weight.detach(), calls["first_fc"])
        assert any(state.get("step", 0) for state in calls["trainer"].optimizer.state.values())
        print("PASS: real main.py --method singleprompt, real shared Si-Blurry 5 tasks", flush=True)
        print("PASS: {} optimizer updates, batch mask on each step, fixed prompt".format(
              calls["updates"]), flush=True)
        print("PASS: {} evaluations, saved FlyGCL metrics in three .npy files".format(
              len(eval_acc)), flush=True)
        print("LIMIT: synthetic images, CPU shim, small random backbone; no real GPU/pretrained checkpoint", flush=True)


if __name__ == "__main__":
    check_eval_policies()
    check_real_main_entrypoint()
    print("PASS: SinglePrompt Stage 3 integration CPU smoke test", flush=True)
