"""Optional CPU check with the actual common pretrained ViT-B/16 checkpoint.

Place the official Google ViT-B_16.npz in checkpoints/ before running. This uses
full 768-dimensional ViT features and default head sizes, not the tiny CI model.
No real dataset or accuracy benchmark is involved.
"""
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def check_checkpoint(backbone, checkpoint):
    assert backbone.num_features == 768 and len(backbone.blocks) == 12
    with np.load(checkpoint) as data:
        expected = torch.from_numpy(data['embedding/kernel'].copy()).permute(3, 2, 0, 1)
        torch.testing.assert_close(backbone.patch_embed.proj.weight.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(backbone.pos_embed.cpu(), torch.from_numpy(data['Transformer/posembed_input/pos_embedding'].copy()), rtol=0, atol=0)
        torch.testing.assert_close(backbone.norm.weight.cpu(), torch.from_numpy(data['Transformer/encoder_norm/scale'].copy()), rtol=0, atol=0)


def main():
    from models import MODELS
    from configuration.config import base_parser
    from unittest.mock import patch
    import sys
    checkpoint = ROOT / 'checkpoints/ViT-B_16.npz'
    if not checkpoint.is_file():
        raise FileNotFoundError('Place official ViT-B_16.npz in checkpoints/; this optional check does not download it')
    torch.set_num_threads(2)
    torch.manual_seed(19)
    images = torch.randn(2, 3, 224, 224)
    reference_features = None
    for method in ('air', 'gacl', 'slda'):
        # CPU smoke test uses a small AIR head; official 8192 needs >=50 GiB.
        head_args = {'air_expansion_size': 1024} if method == 'air' else {}
        model = MODELS[method](backbone_name='vit_base_patch16_224', num_classes=100, **head_args)
        check_checkpoint(model.backbone, checkpoint)
        model.initialize_statistics()
        model.eval()
        with torch.no_grad():
            features = model.features(images)
            assert features.shape == (2, 768)
            if reference_features is None:
                reference_features = features.clone()
            else:
                torch.testing.assert_close(features, reference_features, rtol=0, atol=0)
            expanded = model.expand(features)
            model.fit_features(expanded, torch.tensor([0, 1]))
            scores = model.scores(expanded, 'cpu')
            assert scores.shape == (2, 100) and torch.isfinite(scores).all()
        assert all(not p.requires_grad for p in model.parameters()) and not model.backbone.training
        print(f'PASS actual pretrained {method}: feature_dim=768, expansion={expanded.shape[1]}, default head fit/forward', flush=True)
        del model, features, expanded, scores
        gc.collect()
    for method in ('misa', 'singleprompt'):
        with patch.object(sys, 'argv', ['main.py', '--method', method, '--dataset', 'cifar100']):
            options = vars(base_parser())
        options.pop('backbone', None)
        model = MODELS[method](backbone_name='vit_base_patch16_224', num_classes=100, task_num=5, **options)
        model.eval()
        check_checkpoint(model.backbone, checkpoint)
        with torch.no_grad():
            scores = model(images[:1])
            assert scores.shape == (1, 100) and torch.isfinite(scores).all()
        print(f'PASS existing {method}: actual pretrained full ViT forward after NPZ loader fix', flush=True)
        del model, scores
        gc.collect()
    with checkpoint.open('rb') as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(2**20), b''):
            digest.update(chunk)
    print('Checkpoint SHA256:', digest.hexdigest())
    print('LIMIT: AIR smoke test uses 1024, not the official 8192. GPU/data accuracy experiments not performed')


if __name__ == '__main__':
    main()
