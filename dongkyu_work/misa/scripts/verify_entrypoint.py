"""Stage 8: invoke the REAL main.py entrypoint and shared multi-task runner on CPU.

This is NOT a benchmark. Synthetic CIFAR100-like PIL images, the real MISA
Prompt checkpoints, the real _Trainer.main_worker code, the shared Si-Blurry
OnlineSampler/DataLoader, full eval/saving logic, and a small fast substitute
backbone keep the test independent from CUDA/dataset/backbone downloads.

Only the TEST provides a CPU stand-in for torch.device(int), and reports the
normal GPU entrypoint as not independently verified. Production source intact.
"""

import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dongkyu_work.misa.scripts.verify_trainer import (
    MiniImageDataset,
    lightweight_backbone,
)


class CpuDeviceShim:
    """Redirect only a CUDA-rank integer device to CPU in the runner's globals."""

    def device(self, value, *args, **kwargs):
        if isinstance(value, int):
            assert value == 0
            return torch.device("cpu")
        return torch.device(value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(torch, name)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1993)
    from methods import METHODS
    import methods._trainer as trainer_module
    import models.vit as vit

    # Execute the *identical bytecode* of the original _Trainer.main_worker;
    # only its global torch.device(0) lookup is bridged for this CPU-only CI.
    original_worker = METHODS["misa"].main_worker
    patched_globals = original_worker.__globals__.copy()
    assert patched_globals["torch"] is torch
    patched_globals["torch"] = CpuDeviceShim()
    cpu_worker = types.FunctionType(
        original_worker.__code__, patched_globals,
        original_worker.__name__, original_worker.__defaults__,
        original_worker.__closure__,
    )
    assert cpu_worker.__code__ is original_worker.__code__

    with tempfile.TemporaryDirectory(prefix="misa-stage8-") as temp_dir:
        args = [
            "main.py", "--method", "misa", "--dataset", "cifar100",
            "--data_dir", str(Path(temp_dir) / "fake-cifar"),
            "--log_path", str(Path(temp_dir) / "results"),
            "--note", "misa_cpu_entrypoint",
            "--n_tasks", "5", "--n", "50", "--m", "10", "--rnd_NM",
            "--batchsize", "4", "--online_iter", "1",
            "--num_epochs", "1", "--lr", "0.005", "--opt_name", "adam",
            "--sched_name", "default", "--eval_period", "10",
            "--transforms", "autoaug", "--seeds", "1993",
            "--misa_prompt_dir", str(ROOT / "dongkyu_work/misa/checkpoints"),
        ]

        calls = {"epochs": 0, "updates": 0, "task_ids": []}
        actual_step = METHODS["misa"].online_step
        actual_after_task = METHODS["misa"].online_after_task

        def counted_step(self, images, labels, idx):
            calls["updates"] += 1
            return actual_step(self, images, labels, idx)

        def counted_after_task(self, task_id):
            calls["task_ids"].append(task_id)
            return actual_after_task(self, task_id)

        # Fake dataset and lightweight ViT-shaped feature extractor only.
        # The optimizer, task loop, sampler and metric writer are unmodified.
        with patch.object(sys, "argv", args), \
             patch.object(trainer_module, "DATASETS", {"cifar100": MiniImageDataset}), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(vit, "vit_base_patch16_224",
                          side_effect=lightweight_backbone), \
             patch.object(METHODS["misa"], "main_worker", cpu_worker), \
             patch.object(METHODS["misa"], "online_step", counted_step), \
             patch.object(METHODS["misa"], "online_after_task", counted_after_task):
            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")

        task_path = (Path(temp_dir) / "results/logs/cifar100/"
                     "misa_cpu_entrypoint/seed_1993.npy")
        eval_path = task_path.with_name("seed_1993_eval.npy")
        times_path = task_path.with_name("seed_1993_eval_time.npy")
        for filename in (task_path, eval_path, times_path):
            assert filename.exists(), f"Missing shared metric file: {filename}"
        task_acc = np.load(task_path)
        eval_acc = np.load(eval_path)
        eval_times = np.load(times_path)
        assert task_acc.shape == (5,), f"Expected 5 task scores: {task_acc.shape}"
        assert np.isfinite(task_acc).all()
        assert eval_acc.ndim == 1 and len(eval_acc) > 0
        assert np.isfinite(eval_acc).all()
        assert eval_times.shape == eval_acc.shape
        assert (np.diff(eval_times) >= 0).all()

        assert calls["updates"] > 0, "No online updates executed"
        assert calls["task_ids"] == [0, 1, 2, 3, 4], calls["task_ids"]
        print("PASS real main.py entrypoint: CLI, MISA registry and run() invoked", flush=True)
        print("PASS real _Trainer.main_worker bytecode: 5 tasks, "
              f"{calls['updates']} online updates and evaluations", flush=True)
        print("PASS shared metric files: task accuracy + eval accuracy + time "
              f"({len(eval_acc)} evaluations)", flush=True)
        print("LIMIT CPU device shim and synthetic data/proxy backbone were "
              "test-only; actual pretrained ViT/CUDA/main GPU path untested", flush=True)

    print("PASS Stage 8 full main.py workflow CPU smoke test", flush=True)


if __name__ == "__main__":
    main()
