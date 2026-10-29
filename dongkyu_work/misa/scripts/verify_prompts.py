"""Check official MISA prompt integrity and CPU tensor loading. No training.

Run from any current working directory:
    python dongkyu_work/misa/scripts/verify_prompts.py
Optional:
    python dongkyu_work/misa/scripts/verify_prompts.py --prompt-dir /path/to/dir
"""

import argparse
import hashlib
from pathlib import Path

EXPECTED = {
    "g_prompt.pt": {
        "git_sha1": "b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53",
        "shape": (1, 20, 768),
    },
    "e_prompt.pt": {
        "git_sha1": "3b88950a0cabbdd533357fb84f83445fd1f47463",
        "shape": (10, 60, 768),
    },
}


def git_blob_sha(path):
    data = path.read_bytes()
    header = ("blob %d\0" % len(data)).encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Verify upstream MISA ISA pretrained prompts")
    parser.add_argument(
        "--prompt-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
    )
    args = parser.parse_args()
    import torch

    for filename, spec in EXPECTED.items():
        path = args.prompt_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing MISA prompt checkpoint: {path}")
        sha = git_blob_sha(path)
        if sha != spec["git_sha1"]:
            raise ValueError(f"{filename} SHA mismatch: {sha} != {spec['git_sha1']}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{filename}: expected torch.Tensor, got {type(tensor)!r}")
        if tuple(tensor.shape) != spec["shape"]:
            raise ValueError(
                f"{filename}: shape {tuple(tensor.shape)} != expected {spec['shape']}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"{filename}: dtype {tensor.dtype} != torch.float32")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{filename}: tensor has NaN or Inf values")
        print(
            f"PASS {filename}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, upstream_sha={sha}"
        )
    print("PASS: Both official MISA prompt checkpoints are CPU-loadable and shape-compatible.")


if __name__ == "__main__":
    main()
