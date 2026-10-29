"""Actual FlyGCL CLI/stream/evaluation check with CPU, fake images and tiny ViT.

Test-only substitutions: integer CUDA device -> CPU; SyntheticCIFAR; pretrained
factory -> small real ViT. The shared worker bytecode and sampler are unchanged.
"""
import copy
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from dongkyu_work.singleprompt.scripts.verify_integration import SyntheticCIFAR, CpuDeviceShim

ROOT = Path(__file__).resolve().parents[3]


def proxy_vit(*args, **kwargs):
    assert kwargs['pretrained'] is True
    from models.vit import VisionTransformer
    return VisionTransformer(img_size=224, patch_size=32, embed_dim=32,
                             depth=2, num_heads=4, num_classes=kwargs['num_classes'], qkv_bias=True)


class UnseenWins(nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x), 100)
        logits[:, 0], logits[:, 50] = 2., 10.
        return logits


def check_evaluator(method):
    from configuration.config import base_parser
    from methods import METHODS
    from methods.dualprompt import DualPrompt
    with tempfile.TemporaryDirectory() as td, patch.object(sys, 'argv', ['main.py', '--method', method, '--log_path', td]):
        args = base_parser()
        if method == 'air':
            assert args.air_expansion_size == 8192 and args.air_max_stats_gib == 64.0
            assert args.analytic_stats_device == 'auto'
        trainer = METHODS[method](**vars(args))
        trainer.device = torch.device('cpu')
        trainer.n_classes = 100
        trainer.exposed_classes = [37, 4]
        trainer.mask = torch.full((100,), -torch.inf)
        trainer.mask[:2] = 0
        trainer.model = trainer.model_without_ddp = UnseenWins()
        trainer.criterion = nn.CrossEntropyLoss()
        for topk in (1, 2):
            trainer.topk = topk
            data = [(torch.zeros(2, 3, 8, 8), torch.tensor([37, 4])),
                    (torch.zeros(1, 3, 8, 8), torch.tensor([37]))]
            original = [labels.clone() for _, labels in data]
            actual = trainer.online_evaluate(data)
            expected = DualPrompt.online_evaluate(trainer, copy.deepcopy(data))
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0)
            assert all(torch.equal(labels, saved) for (_, labels), saved in zip(data, original))
            assert actual['avg_acc'] == (2/3 if topk == 1 else 1.)
        for changes in ({'online_iter': 0}, {'online_iter': 1.5}, {'use_amp': True}, {'step_num': 10}):
            try:
                METHODS[method](**(vars(args) | changes))
            except ValueError:
                pass
            else:
                raise AssertionError(changes)
    print(f'PASS {method}: evaluator equals DualPrompt, unseen logits excluded, labels preserved')


def check_launcher(method):
    with tempfile.TemporaryDirectory(prefix='analytic launch ') as td:
        fake = Path(td) / 'python substitute'
        output = Path(td) / 'args.json'
        fake.write_text('#!' + sys.executable + '\nimport json,sys,os\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps([os.getcwd(),sys.argv[1:]]))\n')
        fake.chmod(0o755)
        env = os.environ | {'PYTHON_BIN': str(fake), 'CAPTURE': str(output), 'DATA_DIR': '/data/with spaces'}
        script = ROOT / 'dongkyu_work' / method / 'scripts' / f'run_{method}.sh'
        subprocess.run(['bash', str(script), '--seeds', '7'], cwd=td, env=env, check=True)
        cwd, argv = json.loads(output.read_text())
        assert cwd == str(ROOT)
        assert argv[argv.index('--method') + 1] == method
        assert argv[argv.index('--data_dir') + 1] == '/data/with spaces'
        assert argv[-2:] == ['--seeds', '7']
    print(f'PASS {method}: launcher cwd, spaces, extra arguments')


def check_main(method):
    from methods import METHODS
    from models import MODELS
    from methods import _trainer as shared
    from models import vit
    assert method in MODELS and method in METHODS
    base = METHODS[method]
    worker = base.main_worker
    globals_copy = worker.__globals__.copy()
    globals_copy['torch'] = CpuDeviceShim()
    cpu_worker = types.FunctionType(worker.__code__, globals_copy, worker.__name__, worker.__defaults__, worker.__closure__)
    assert cpu_worker.__code__ is worker.__code__
    setup, step, after, evaluate = base.setup_distributed_model, base.online_step, base.online_after_task, base.online_evaluate
    calls = {'batches': 0, 'samples': 0, 'tasks': [], 'evals': 0, 'labels': []}

    def observed_setup(self):
        setup(self)
        model = self.model
        assert all(not p.requires_grad for p in model.parameters())
        assert not model.backbone.training
        assert model.projection is None or model.projection.out_features == 12
        calls['trainer'] = self
        calls['backbone'] = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        calls['head_id'] = id(model.head)

    def observed_step(self, images, labels, idx):
        saved = labels.clone()
        calls['labels'].extend(labels.tolist())
        result = step(self, images, labels, idx)
        assert torch.equal(saved, labels)
        assert all(math.isfinite(x) for x in result)
        calls['batches'] += 1
        calls['samples'] += len(labels) * int(self.online_iter)
        assert int(self.model.head.num_updates) == calls['samples']
        return result

    def observed_after(self, task_id):
        assert id(self.model.head) == calls['head_id']
        calls['tasks'].append(task_id)
        return after(self, task_id)

    def observed_evaluate(self, *args, **kwargs):
        before = int(self.model.head.num_updates)
        result = evaluate(self, *args, **kwargs)
        assert int(self.model.head.num_updates) == before
        calls['evals'] += 1
        return result

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='analytic-main-') as td:
        argv = ['main.py', '--method', method, '--dataset', 'cifar100', '--backbone', 'vit_base_patch16_224',
                '--data_dir', td, '--log_path', td, '--note', method + '_cpu',
                '--n_tasks', '5', '--n', '50', '--m', '10', '--rnd_NM',
                '--batchsize', '8', '--online_iter', '2', '--num_epochs', '1',
                '--eval_period', '40', '--transforms', 'autoaug', '--seeds', '1993',
                '--air_expansion_size', '12', '--gacl_expansion_size', '12', '--analytic_stats_device', 'cpu']
        with patch.object(sys, 'argv', argv), patch.object(shared, 'DATASETS', {'cifar100': SyntheticCIFAR}), \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(vit, 'vit_base_patch16_224', side_effect=proxy_vit), \
             patch.object(base, 'main_worker', cpu_worker), \
             patch.object(base, 'setup_distributed_model', observed_setup), \
             patch.object(base, 'online_step', observed_step), \
             patch.object(base, 'online_after_task', observed_after), \
             patch.object(base, 'online_evaluate', observed_evaluate):
            runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
        folder = Path(td) / 'logs/cifar100' / (method + '_cpu')
        arrays = [np.load(folder / f'seed_1993{suffix}.npy') for suffix in ('', '_eval', '_eval_time')]
        assert arrays[0].shape == (5,)
        assert all(np.isfinite(a).all() for a in arrays)
        assert len(arrays[1]) > 0 and arrays[1].shape == arrays[2].shape
        assert calls['tasks'] == list(range(5)) and calls['samples'] == 400
        model = calls['trainer'].model
        for key, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, calls['backbone'][key], rtol=0, atol=0)
        with patch.object(vit, 'vit_base_patch16_224', side_effect=proxy_vit):
            restored = MODELS[method](backbone_name='vit_base_patch16_224', num_classes=100,
                                     air_expansion_size=12, gacl_expansion_size=12)
            restored.load_state_dict(model.state_dict())
        sample = torch.randn(2, 3, 224, 224)
        torch.testing.assert_close(model(sample), restored(sample), rtol=0, atol=0)
        print(f'PASS {method}: real main.py, {calls["batches"]} batches, {calls["samples"]} statistic observations, '
              f'{calls["evals"]} evaluations, 5 tasks, 3 metric files, frozen backbone and state round-trip', flush=True)
    return calls['labels']


def verify_method(method):
    check_evaluator(method)
    check_launcher(method)
    return check_main(method)


if __name__ == '__main__':
    streams = [verify_method(method) for method in ('air', 'gacl', 'slda')]
    assert streams[0] == streams[1] == streams[2]
    print('PASS identical shared Si-Blurry sample order across all three methods')
    print('LIMIT: CPU shim, synthetic images, tiny random backbone; not a GPU accuracy experiment')
