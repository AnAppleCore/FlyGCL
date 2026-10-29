"""Stage 2: official SinglePrompt vs isolated FlyGCL model parity (CPU only).

No pretrained network download, no actual data, no full training loop.
"""
import gc
import torch
from torch import nn

from dongkyu_work.singleprompt.implementation.model import SinglePrompt
from dongkyu_work.singleprompt.original.SinglePrompt.models.singlePrompt import (
    PreAttention,
    SinglePromptViT,
)


def check_attention():
    torch.manual_seed(7)
    attention = PreAttention(dim=32, num_heads=4, qkv_bias=True)
    x = torch.randn(2, 4, 32, requires_grad=True)
    prompt = torch.randn(2, 2, 3, 32, requires_grad=True)
    official = attention(x, prompt)
    aligned = SinglePrompt._prefix_attention(attention, x, prompt)
    torch.testing.assert_close(official, aligned, atol=0, rtol=0)
    official.square().mean().backward(retain_graph=True)
    original_gradient = prompt.grad.clone()
    prompt.grad.zero_()
    aligned.square().mean().backward()
    torch.testing.assert_close(original_gradient, prompt.grad, atol=0, rtol=0)
    print("PASS: prefix attention official forward and gradient parity")


def check_full_model():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    source = SinglePromptViT(
        img_size=224, patch_size=16, embed_dim=768, depth=12,
        num_heads=12, num_classes=100,
        pos_prompt=[0, 1, 2, 3, 4], prompt_length=20,
        logit_type="cos_sim", drop_rate=0.0, drop_path_rate=0.0
    )
    torch.manual_seed(19)
    target = SinglePrompt(pretrained=False, num_classes=100)

    original = source.state_dict()
    local = target.backbone.state_dict()
    common = {k: v for k, v in original.items()
              if k in local and v.shape == local[k].shape}
    missing = set(local).difference(common)
    # FlyGCL's frozen base head can have a bias that upstream's
    # cosine head omits; it is not used by SinglePrompt forward.
    assert missing <= {"fc.bias"}, f"Backbone state mismatch: {missing}"
    assert len(common) > 100, "Full ViT backbone weight comparison missing"
    target.backbone.load_state_dict(common, strict=False)
    with torch.no_grad():
        target.prompt.copy_(source.prompt)
        target.fc.weight.copy_(source.fc.weight)

    source.eval()
    target.eval()
    active = [name for name, p in target.named_parameters() if p.requires_grad]
    assert active == ["prompt", "fc.weight"], active
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    assert trainable == 230400, trainable
    assert len(source.blocks) == len(target.backbone.blocks) == 12
    assert tuple(target.pos_prompt.tolist()) == (0, 1, 2, 3, 4)
    print("PASS: frozen 12-block ViT, 230400 trainable parameters, no router")

    torch.manual_seed(5)
    sample = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([7])

    # Full ViT Forward and prompt/head gradients.
    official_output = source(sample)
    target_output = target(sample)
    assert official_output.shape == target_output.shape == (1, 100)
    torch.testing.assert_close(official_output, target_output, rtol=2e-5, atol=2e-5)
    print("PASS: official full ViT-B/16 logits match on synthetic image")

    loss_official = nn.functional.cross_entropy(official_output, labels)
    loss_target = nn.functional.cross_entropy(target_output, labels)
    loss_official.backward()
    loss_target.backward()
    torch.testing.assert_close(source.prompt.grad, target.prompt.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(source.fc.weight.grad, target.fc.weight.grad, rtol=3e-4, atol=3e-5)
    assert source.prompt.grad.abs().sum().item() > 0
    assert target.prompt.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in target.backbone.parameters())
    print("PASS: CE loss, prompt and cosine-head gradients match official model")
    del official_output, target_output, sample, source, target
    gc.collect()


def check_options():
    for args in (
        {"pos_prompt": [0, 2, 4]},
        {"pos_prompt": []},
        {"len_prompt": 0},
        {"backbone_name": "vit_small_patch16_224"},
    ):
        try:
            SinglePrompt(pretrained=False, **args)
        except ValueError:
            continue
        raise AssertionError("Expected invalid configuration to be rejected: " + str(args))
    print("PASS: reject silent deviations from official default")


if __name__ == "__main__":
    check_attention()
    check_full_model()
    check_options()
    print("PASS: Stage 2 CPU model-only parity complete (not GPU/benchmark validation)")
