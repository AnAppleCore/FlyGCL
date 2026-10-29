"""Compare adapters with pinned official cores and test checkpoint continuation.

No backbone/data downloads. SLDA's hardcoded CUDA initializer is replaced by CPU
in a test-only AST; official files are never modified.
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def check_sources(method):
    folder = ROOT / method
    manifest = json.loads((folder / 'original/SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        data = (folder / entry['path']).read_bytes()
        sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert sha == entry['git_blob_sha'], entry['path']
    print(f'PASS {method}: {len(manifest["files"])} unchanged upstream files')


def load_class(path, name, cpu_init=False):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    replacements = 0
    if cpu_init:
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == 'cuda':
                assert ast.unparse(node.targets[0]) == 'self.device'
                node.value = ast.Constant(value='cpu')
                replacements += 1
        assert replacements == 1
    namespace = dict(torch=torch, nn=nn, F=F, os=os, _Trainer=object)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def make_head(method):
    from dongkyu_work.air.implementation.head import AIRHead
    from dongkyu_work.gacl.implementation.head import GACLHead
    from dongkyu_work.slda.implementation.head import SLDAHead
    return {'air': lambda: AIRHead(7, 4, gamma=.7),
            'gacl': lambda: GACLHead(7, 4, gamma=.7),
            'slda': lambda: SLDAHead(7, 4)}[method]()


def make_reference(method):
    folder = ROOT / method / 'original' / method
    if method == 'gacl':
        cls = load_class(folder / 'methods/GACL.py', 'GACL')
        head = cls.__new__(cls)
        head.model = SimpleNamespace(expansion=lambda z: z)
        head.device = torch.device('cpu')
        head.out_features = 0
        head.W = torch.zeros(7, 0, dtype=torch.double)
        head.R = torch.eye(7, dtype=torch.double) / .7
        return head
    if method == 'air':
        spec = importlib.util.spec_from_file_location('reference_analytic', folder / 'analytic/AnalyticLinear.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GeneralizedARM(7, gamma=.7, device='cpu')
    return load_class(folder / 'SLDA_Model.py', 'StreamingLDA', True)(7, 4)


def verify_method(method):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    check_sources(method)
    x = torch.randn(19, 7, dtype=torch.float32 if method == 'slda' else torch.float64)
    y = torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 0, 2, 3, 1, 0, 2])
    for chunks in ([19], [1] * 19, [3, 6, 2, 8]):
        head, reference = make_head(method), make_reference(method)
        end = 0
        for size in chunks:
            start, end = end, end + size
            head.fit(x[start:end], y[start:end])
            if method == 'slda':
                for features, label in zip(x[start:end], y[start:end]):
                    reference.fit(features, label)
                for name in ('muK', 'cK', 'Sigma'):
                    torch.testing.assert_close(getattr(head, name), getattr(reference, name), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(head(x), reference.predict(x), rtol=3e-4, atol=3e-4)
            else:
                reference.fit(x[start:end], y[start:end])
                if method == 'air':
                    head.update(); reference.update()
                    n = reference.out_features
                    for name in ('A', 'cnt'):
                        torch.testing.assert_close(getattr(head, name)[:n], getattr(reference, name), check_dtype=False)
                    torch.testing.assert_close(head.C[:, :n], reference.C, rtol=1e-9, atol=1e-10)
                    expected = reference.weight
                else:
                    n = reference.out_features
                    torch.testing.assert_close(head.R, reference.R, rtol=1e-9, atol=1e-10)
                    targets = F.one_hot(y[:end], 4).double()
                    joint = torch.linalg.solve(x[:end].T @ x[:end] + .7 * torch.eye(7, dtype=x.dtype), x[:end].T @ targets)
                    torch.testing.assert_close(head.weight, joint, rtol=1e-9, atol=1e-10)
                    expected = reference.W
                torch.testing.assert_close(head.weight[:, :n], expected, rtol=1e-9, atol=1e-10)
            assert int(head.num_updates) == end
    # Round-trip before and after score-cache population, then continue updates.
    for predict_first in (False, True):
        head = make_head(method)
        head.fit(x[:9], y[:9])
        if predict_first:
            head(x)
        stream = io.BytesIO()
        torch.save(head.state_dict(), stream); stream.seek(0)
        restored = make_head(method)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        head.fit(x[9:], y[9:]); restored.fit(x[9:], y[9:])
        torch.testing.assert_close(head(x), restored(x), rtol=0, atol=0)
        assert int(restored.num_updates) == len(y)
    if method == 'air':
        from dongkyu_work.air.implementation.head import AIRHead
        try:
            AIRHead(8192, 100, max_stats_gib=8.0)
        except ValueError as error:
            assert '50.00 GiB' in str(error)
        else:
            raise AssertionError('AIR must reject oversized statistics before allocation')
        # Official dimensions pass the configured cap but must not allocate if
        # the server has too little RAM (mock the preflight: no 50 GiB allocation).
        with patch.object(AIRHead, '_host_available_bytes', return_value=2 * 2**30):
            try:
                AIRHead(8192, 100)
            except MemoryError as error:
                assert '54.0 GiB' in str(error)
            else:
                raise AssertionError('AIR must reject insufficient available memory')
    if method == 'air':
        cached = make_head('air')
        cached.fit(x[:3], y[:3])
        assert bool(cached.dirty)
        torch.testing.assert_close(cached.forward_cached(x[:2]), torch.zeros(2, 4, dtype=x.dtype))
        assert bool(cached.dirty)
        cached(x[:2]); assert not bool(cached.dirty)
    print(f'PASS {method}: official-core parity at every prefix, 3 partitions, checkpoint continuation')


if __name__ == '__main__':
    for method in ('air', 'gacl', 'slda'):
        verify_method(method)
