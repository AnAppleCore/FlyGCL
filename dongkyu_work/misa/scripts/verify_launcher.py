"""Verify the actual Bash launcher forwards correct args from any CWD.

No training is launched: a temporary 'python' executable captures
the command line that run_misa.sh would pass to the real main.py.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "dongkyu_work/misa/scripts/run_misa.sh"


def main():
    with tempfile.TemporaryDirectory(prefix="misa-launcher-") as d:
        tmp = Path(d)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "python"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "open(os.environ['MISA_LAUNCH_CAPTURE'], 'w').write("
            "json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]}))\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        capture = tmp / "args.json"
        env = os.environ.copy()
        env.update({
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "MISA_LAUNCH_CAPTURE": str(capture),
            "DATA_DIR": str(tmp / "imaginary-dataset"),
            "MISA_PROMPT_DIR": str(ROOT / "dongkyu_work/misa/checkpoints"),
        })

        # Deliberately run from a directory other than the repository root.
        subprocess.run(["bash", str(LAUNCHER)], cwd=tmp, env=env,
                       check=True, capture_output=True, text=True)
        actual = json.loads(capture.read_text(encoding="utf-8"))
        args = actual["argv"]
        assert Path(actual["cwd"]) == ROOT, "Launcher must cd to repository root"
        assert args[0] == "main.py"
        expected = {
            "--method": "misa",
            "--backbone": "vit_base_patch16_224",
            "--dataset": "cifar100",
            "--data_dir": env["DATA_DIR"],
            "--n_tasks": "5", "--n": "50", "--m": "10",
            "--batchsize": "64", "--online_iter": "3",
            "--num_epochs": "1", "--lr": "0.005",
            "--opt_name": "adam", "--sched_name": "default",
            "--eval_period": "1000", "--misa_prompt_dir": env["MISA_PROMPT_DIR"],
        }
        for flag, value in expected.items():
            assert flag in args and args[args.index(flag) + 1] == value, (
                f"Missing or incorrect {flag}: {args}"
            )
        assert "--rnd_NM" in args
        assert "--no_batchmask" not in args, "Official MISA batch mask disabled"
        assert args[args.index("--transforms") + 1] == "autoaug"
        print("PASS launcher: arbitrary CWD and exact MISA CIFAR-100 arguments")

        # Missing prompts must fail before any code launches.
        capture.unlink()
        env["MISA_PROMPT_DIR"] = str(tmp / "missing-prompts")
        failure = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp, env=env,
                                 check=False, capture_output=True, text=True)
        assert failure.returncode == 2
        assert "required official prompt missing" in failure.stderr
        assert not capture.exists(), "Training attempted before checkpoint preflight"
        print("PASS launcher: missing checkpoint fails before calling Python")

    print("PASS Stage 8 MISA production launcher checks (without training)")


if __name__ == "__main__":
    main()
