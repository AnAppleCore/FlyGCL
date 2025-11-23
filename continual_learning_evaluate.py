import argparse
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from methods import METHODS
from utils.onlinesampler import OnlineTestSampler

# Filter out the noisy FutureWarning from torch.load(weights_only=False)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline continual learning evaluation from saved checkpoints.")

    # Path resolution: either direct log_dir or components mirroring run_baselines*.sh
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Full path to log directory (results/logs/<dataset>/<note>).")
    parser.add_argument("--log_path", type=str, default="results",
                        help="Base log path used during training (default: results).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., cub200). Used when log_dir is not given.")
    parser.add_argument("--method", type=str, default=None,
                        help="Method name (e.g., dualprompt). Used when log_dir is not given.")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224",
                        help="Backbone name (e.g., vit_base_patch16_224). Used when log_dir is not given.")
    parser.add_argument("--extra_note", type=str, default="baseline_standard",
                        help="Extra note used in training scripts. Used when log_dir is not given.")

    parser.add_argument("--seed", type=int, required=True,
                        help="Seed index to evaluate (matches rnd_seed during training).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use, e.g., cuda:0 or cpu. Default: cuda if available else cpu.")

    args = parser.parse_args()

    if args.log_dir is None:
        missing = [k for k in ["dataset", "method"] if getattr(args, k) is None]
        if missing:
            raise ValueError(
                f"When --log_dir is not provided, you must specify --dataset and --method (missing: {missing}).")
        note = f"{args.method}_{args.backbone}_{args.dataset}_{args.extra_note}"
        args.log_dir = os.path.join(args.log_path, "logs", args.dataset, note)

    return args


def build_trainer_from_config(config_dict, device):
    # Reconstruct trainer with the same hyper-parameters used during training
    cfg = dict(config_dict)
    # Ensure single-seed semantics
    if "rnd_seed" in cfg:
        rnd_seed = cfg["rnd_seed"]
    else:
        rnd_seed = None
    cfg.setdefault("seeds", [rnd_seed] if rnd_seed is not None else [1])

    trainer = METHODS[cfg["method"]](**cfg)

    # Force non-distributed single-device setup for offline eval
    trainer.world_size = 1
    trainer.ngpus_per_nodes = 1
    trainer.distributed = False
    trainer.device = device
    trainer.gpu = device.index if device.type == "cuda" else 0

    # Seed for reproducibility
    if rnd_seed is not None:
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(rnd_seed)
        cudnn.deterministic = True
    cudnn.benchmark = False

    # Build dataset and model
    trainer.setup_distributed_dataset()
    trainer.total_samples = len(trainer.train_dataset)
    trainer.setup_distributed_model()

    return trainer


def main():
    args = parse_args()

    # Resolve paths
    log_dir = args.log_dir
    seed = args.seed
    ckpt_path = os.path.join(log_dir, f"seed_{seed}_ckpt.pth")
    cfg_path = os.path.join(log_dir, f"seed_{seed}_config.json")
    task_acc_path = os.path.join(log_dir, f"seed_{seed}.npy")
    eval_path = os.path.join(log_dir, f"seed_{seed}_eval.npy")

    for p in [ckpt_path, cfg_path, task_acc_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # Choose device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and checkpoint
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Sanity check on seed
    ckpt_seed = ckpt.get("rnd_seed", None)
    if ckpt_seed is not None and ckpt_seed != seed:
        print(f"[Warning] ckpt rnd_seed={ckpt_seed} != requested seed={seed}")

    trainer = build_trainer_from_config(cfg_dict, device)

    # Load model weights (handle minor shape changes for buffers like 'similarity')
    state_dict = ckpt["model_state_dict"]
    sim = state_dict.get("similarity", None)
    if isinstance(sim, torch.Tensor) and sim.dim() == 0:
        state_dict["similarity"] = sim.view(1)
    trainer.model_without_ddp.load_state_dict(state_dict)
    trainer.model.to(device)

    # Restore exposed_classes and mask
    exposed = ckpt.get("exposed_classes", None)
    if exposed is not None:
        trainer.exposed_classes = list(exposed)
        trainer.mask = torch.zeros(trainer.n_classes, device=trainer.device) - torch.inf
        trainer.mask[:len(trainer.exposed_classes)] = 0

    # Build test loader exactly as in training summary
    test_sampler = OnlineTestSampler(trainer.test_dataset, trainer.exposed_classes)
    test_loader = DataLoader(
        trainer.test_dataset,
        batch_size=trainer.batchsize * 2,
        sampler=test_sampler,
        num_workers=trainer.n_worker,
    )

    trainer.model.eval()
    with torch.no_grad():
        eval_dict = trainer.online_evaluate(test_loader, task_id=trainer.n_tasks - 1, end=True)

    offline_acc = float(eval_dict["avg_acc"])
    offline_cls_acc = eval_dict.get("cls_acc", None)

    # Load npy results and recompute basic metrics
    task_acc = np.load(task_acc_path)
    n_tasks_npy = int(task_acc.shape[0])
    A_last_npy = float(task_acc[-1])
    A_avg_npy = float(np.mean(task_acc))
    A_auc_npy = None
    eval_series = None
    if os.path.exists(eval_path):
        eval_series = np.load(eval_path)
        if eval_series.size > 0:
            A_auc_npy = float(np.mean(eval_series))

    diff = abs(offline_acc - A_last_npy)

    print("============================================")
    print(f"Log dir: {log_dir}")
    print(f"Seed: {seed}")
    print(f"#Tasks (trainer / npy): {trainer.n_tasks} / {n_tasks_npy}")
    print(f"Per-task accuracies from training (seed_{seed}.npy):")
    for t_id, acc_t in enumerate(task_acc):
        print(f"  Task {t_id:02d}: {acc_t:.6f}")
    print(f"Offline avg_acc (recomputed on final model): {offline_acc:.6f}")
    print(f"A_last from npy (after last task): {A_last_npy:.6f}")
    print(f"Abs diff between offline avg_acc and A_last: {diff:.6e}")
    print(f"A_avg (mean over tasks) from npy: {A_avg_npy:.6f}")
    if A_auc_npy is not None:
        print(f"A_auc (mean over eval series) from npy: {A_auc_npy:.6f}")
        print(f"  #Eval points: {eval_series.shape[0]}")
        head = min(5, eval_series.shape[0])
        tail = min(5, eval_series.shape[0])
        print(f"  First {head} eval accs: {eval_series[:head]}")
        if eval_series.shape[0] > head:
            print(f"  Last {tail} eval accs:  {eval_series[-tail:]}")
    else:
        print("A_auc from npy: N/A (no seed_*_eval.npy found)")
    if offline_cls_acc is not None:
        cls_acc_np = np.array(offline_cls_acc, dtype=float)
        print(f"Offline cls_acc: {cls_acc_np.shape[0]} classes")
        k = min(10, cls_acc_np.shape[0])
        print(f"  First {k} class accs: {cls_acc_np[:k]}")
    if diff < 1e-4:
        print("[Result] Offline evaluation matches npy A_last within 1e-4 tolerance.")
    else:
        print("[Warning] Offline evaluation and npy A_last differ beyond 1e-4 tolerance.")
    print("============================================")


if __name__ == "__main__":
    main()

