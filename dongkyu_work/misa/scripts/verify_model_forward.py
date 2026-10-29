"""Real ViT-B/16 + official MISA prompts, CPU-only forward/loss smoke test.

Uses the *actual* repository ViT-B/16 transformer (12 layers, 768 hidden)
but overrides only `pretrained=True` at the backbone factory for this test.
The test does NOT require/download external ViT-B/16 backbone weights and
does NOT train on data or evaluate accuracy.

Run from repository root:
    python dongkyu_work/misa/scripts/verify_model_forward.py
"""

from pathlib import Path
from unittest.mock import patch
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    torch.manual_seed(1993)
    torch.set_num_threads(2)
    from models import MODELS
    from methods import METHODS
    import models.vit as vit
    from utils.train_utils import select_model
    from dongkyu_work.misa.implementation.model import MISAExpertPrompt
    from dongkyu_work.misa.implementation.trainer import MISA as MISATrainer

    assert MODELS["misa"].__name__ == "MISA", "MISA not registered in MODELS"
    assert METHODS["misa"] is MISATrainer, "MISA not registered in METHODS"
    print("PASS registry: method and model resolve to MISA")

    # Only replace the forced pretrained=True factory call; this constructs
    # the REAL project's full ViT-B/16 module without downloading .npz weights.
    # Normal --method misa uses pretrained=True and requires backbone weights.
    original_factory = vit.vit_base_patch16_224
    seen_calls = []

    def offline_vit_factory(*args, **kwargs):
        seen_calls.append(dict(kwargs))
        if kwargs.get("pretrained") is not True:
            raise AssertionError("Production model must request pretrained ViT")
        kwargs = dict(kwargs)
        kwargs["pretrained"] = False
        return original_factory(*args, **kwargs)

    prompt_dir = ROOT / "dongkyu_work/misa/checkpoints"
    with patch.object(vit, "vit_base_patch16_224", side_effect=offline_vit_factory):
        model = select_model(
            method="misa",
            backbone="vit_base_patch16_224",
            num_classes=100,
            n_tasks=5,
            kwargs={"step_num": -1, "misa_prompt_dir": str(prompt_dir)}
        )
    assert seen_calls, "Real ViT factory patch was not called"
    assert model.backbone.num_features == 768
    assert len(model.backbone.blocks) == 12
    assert isinstance(model.e_prompt, MISAExpertPrompt)
    assert model.num_pt_per_task == 2 and model.e_pool == 10

    # Both official checkpoint tensors must actually be installed in model.
    g = torch.load(prompt_dir / "g_prompt.pt", map_location="cpu", weights_only=True)
    e = torch.load(prompt_dir / "e_prompt.pt", map_location="cpu", weights_only=True)
    assert tuple(model.g_prompt.prompts.shape) == (1, 20, 768)
    assert tuple(model.e_prompt.prompts.shape) == (10, 60, 768)
    assert torch.equal(model.g_prompt.prompts.detach(), g)
    assert torch.equal(model.e_prompt.prompts.detach(), e)
    print("PASS instantiation: full ViT-B/16, 12 blocks, official Prompt tensors exact")

    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    y = torch.tensor([0], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (1, 100), f"Eval logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Eval logits contain NaN/Inf"
    print("PASS eval forward: synthetic [1,3,224,224] -> [1,100], finite logits")

    model.train()
    logits = model(x)
    assert logits.shape == (1, 100)
    loss = model.loss_fn(logits, y)
    assert torch.isfinite(loss), "Train loss is not finite"
    assert loss.ndim == 0 and loss.requires_grad
    # The training path must limit MISA key search to two expert prompts.
    assert model.e_prompt.last_topk is not None
    assert model.e_prompt.last_topk.max().item() < model.num_pt_per_task
    print(f"PASS train forward/loss: shape={tuple(logits.shape)}, finite loss={loss.item():.6f}")

    # Test autograd wiring, but NO optimizer step or parameter updates.
    loss.backward()
    assert model.e_prompt.prompts.grad is not None
    assert torch.isfinite(model.e_prompt.prompts.grad).all()
    assert model.backbone.fc.weight.grad is not None
    assert torch.isfinite(model.backbone.fc.weight.grad).all()
    print("PASS backward wiring: finite e_prompt and classifier gradients; no optimizer step")
    model.zero_grad(set_to_none=True)

    # MISA train-time task boundary increments expert-key slice; check no shape
    # regressions in later tasks (where original upstream local-index bug matters).
    model.process_task_count()
    assert model.task_count == 1
    model.eval()
    with torch.inference_mode():
        advanced = model(x)
    assert advanced.shape == (1, 100) and torch.isfinite(advanced).all()
    print("PASS task transition: task_count=1, eval Forward still [1,100]")
    print("PASS: Full random-initialized ViT-B/16 + official ISA Prompt CPU forward/backward smoke test")
    print("LIMIT: external pretrained ViT .npz not used; this is not a GPU training/performance reproduction")


if __name__ == "__main__":
    main()
