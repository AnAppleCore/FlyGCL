"""Stage 9: reproducible static source and scope audit (no GPU training)."""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "30b102c26e6c129d6f134a53552a409ad552f2ad"
CORE = [
    "main.py", "models/dualprompt.py", "methods/dualprompt.py",
    "methods/flyprompt.py", "methods/_trainer.py", "models/l2p.py",
    "utils/onlinesampler.py", "utils/train_utils.py",
    "requirements.txt", "datasets/OnlineIterDataset.py",
]
# Reviewed NPZ dispatch-only compatibility fix; all other core files stay frozen.
VIT_COMPAT_BLOB = "3cccc835249cdec5421493eeb9596614e37f8589"
ORIGINAL = "dongkyu_work/misa/original/MISA"
EXPECTED = {
    "g_prompt.pt": "b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53",
    "e_prompt.pt": "3b88950a0cabbdd533357fb84f83445fd1f47463",
}


def run_git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def blob_sha(path):
    b = path.read_bytes()
    return hashlib.sha1(("blob %d\0" % len(b)).encode() + b).hexdigest()


def main():
    assert run_git("cat-file", "-t", BASE) == "commit", "checkout needs fetch-depth: 0"
    modified = run_git("diff", "--name-only", BASE,
                       "--", *CORE, ORIGINAL)
    assert modified == "", "Shared baseline or original MISA modified: " + modified
    originals = [p for p in (ROOT / ORIGINAL).rglob("*") if p.is_file()]
    assert len(originals) == 60, f"Expected 60 official original files, found {len(originals)}"
    assert blob_sha(ROOT / "models/vit.py") == VIT_COMPAT_BLOB, "Unexpected ViT change beyond approved NPZ compatibility fix"
    print("PASS original sources: 60 MISA files and 10 core files unchanged; exact approved NPZ loader fix")

    for filename, expected in EXPECTED.items():
        path = ROOT / "dongkyu_work/misa/checkpoints" / filename
        assert path.is_file() and blob_sha(path) == expected, filename
    print("PASS checkpoints: both official ISA Prompt blobs match upstream SHAs")

    required = [
        "dongkyu_work/misa/implementation/model.py",
        "dongkyu_work/misa/implementation/trainer.py",
        "dongkyu_work/misa/scripts/run_misa.sh",
        "dongkyu_work/misa/scripts/verify_prompts.py",
        "dongkyu_work/misa/scripts/verify_parity.py",
        "dongkyu_work/misa/scripts/verify_model_forward.py",
        "dongkyu_work/misa/scripts/verify_trainer.py",
        "dongkyu_work/misa/scripts/verify_entrypoint.py",
        "dongkyu_work/misa/scripts/verify_launcher.py",
        "dongkyu_work/misa/scripts/verify_final_audit.py",
        "dongkyu_work/misa/notes/CPU_VERIFICATION.md",
        "dongkyu_work/misa/notes/PARITY_VERIFICATION.md",
        "dongkyu_work/misa/notes/FORWARD_VERIFICATION.md",
        "dongkyu_work/misa/notes/TRAINER_VERIFICATION.md",
        "dongkyu_work/misa/notes/ENTRYPOINT_VERIFICATION.md",
        "dongkyu_work/misa/notes/FINAL_REPORT.md",
        "models/misa.py", "methods/misa.py",
    ]
    for rel in required:
        path = ROOT / rel
        assert path.is_file(), rel
        if rel.endswith(".py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=rel)

    model = (ROOT / "dongkyu_work/misa/implementation/model.py").read_text()
    trainer = (ROOT / "dongkyu_work/misa/implementation/trainer.py").read_text()
    assert "self.e_prompt = MISAExpertPrompt(self.e_prompt)" in model
    assert "logits = logits + self.mask" in trainer
    assert '"misa": MISA' in (ROOT / "models/__init__.py").read_text()
    assert '"misa": MISATrainer' in (ROOT / "methods/__init__.py").read_text()
    for shim, import_line in [
        ("models/misa.py", "from dongkyu_work.misa.implementation.model import MISA"),
        ("methods/misa.py", "from dongkyu_work.misa.implementation.trainer import MISA"),
    ]:
        content = (ROOT / shim).read_text()
        assert import_line in content and len(content.splitlines()) <= 8
    print("PASS implementation: import-only shims, registries, prompt and evaluation rules")
    print("PASS docs and Python syntax: all required MISA files present")
    print("PASS Stage 9 source-integrity audit (CPU only, no performance reproduction)")


if __name__ == "__main__":
    main()
