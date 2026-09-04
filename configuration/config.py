import argparse

from datasets import DATASETS
from methods import METHODS


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # ========== Experiment configuration ==========
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--note", type=str, default="", help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")

    # ============ Model configuration =============
    parser.add_argument("--method", type=str, default="l2p", help="Method name", choices=METHODS.keys())
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone name")

    # =========== Dataset configuration ============
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name", choices=DATASETS.keys())
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root directory (see README for expected layouts per dataset).")
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--step_num", type=int, default=-1,
                        help="Number of internal steps for DualPrompt/MVP/FlyPrompt. Only values > 1 enable sample-based internal step scheduling; otherwise these methods follow task boundaries.")

    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch.")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'autoaug'], help="Additional train transforms [cutmix, cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=False, help="Disable batch mask, use seen mask")

    # ========== Evaluation configuration ==========
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    # ============= ViT configurations =============
    parser.add_argument('--profile', action='store_true', default=False, help='enable profiling for ViT_Prompt')

    # ============= EWC / LwF configurations ============
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC regularization strength.")
    parser.add_argument("--ewc_gamma", type=float, default=1.0, help="Decay for consolidated EWC Fisher before adding the current task Fisher.")
    parser.add_argument("--ewc_fisher_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="Store EWC Fisher and parameter snapshot on GPU.")
    parser.add_argument("--ewc_empirical_labels", action=argparse.BooleanOptionalAction, default=True, help="Use ground-truth labels for empirical Fisher; otherwise use model predictions.")
    parser.add_argument("--lwf_lambda", type=float, default=1.0, help="LwF distillation loss weight.")
    parser.add_argument("--lwf_temperature", type=float, default=2.0, help="Temperature for LwF knowledge distillation.")
    parser.add_argument("--lwf_teacher_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="Keep the frozen LwF teacher model on GPU.")

    # ============= MISA configurations ============
    parser.add_argument('--load_pt', action='store_true', default=False, help='load pretrained prompts (MISA)')
    parser.add_argument('--flyprompt_pt_path', type=str, default='./checkpoints/flyprompt_misa_prompt.pt',
                        help='Path to MISA-style pretrained FlyPrompt prompt checkpoint.')

    # ============= MePo configurations ============
    parser.add_argument('--mepo_backbone_path', type=str, default=None,
                        help='Path to pretrained backbone checkpoint for MEPO backbone override.')
    parser.add_argument('--cov_path', type=str, default=None,
                        help='Path to covariance matrix .npy for MEPO CLS calibration.')
    parser.add_argument('--cov_coef', type=float, default=0.7,
                        help='Interpolation coeff between original and MEPO-calibrated CLS (0-1).')

    # ======== HiDe / NoRGa configurations =========
    parser.add_argument("--lam_orth", type=float, default=1, help="Orthogonal loss weight for HiDe/NoRGa.")
    parser.add_argument("--ca_num_per_class", type=int, default=200, help="Number of CA samples per class for HiDe/NoRGa.")
    parser.add_argument("--ca_steps", type=int, default=200, help="Number of CA optimization steps for HiDe/NoRGa.")

    # ========== SD-LoRA configurations ==========
    parser.add_argument("--sdlora_rank", type=int, default=10, help="LoRA rank for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_alpha", type=float, default=0.8, help="Scaling factor alpha for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_layers", type=str, default="all", help="Which ViT blocks to apply LoRA to (e.g., 'all', 'last4').")
    parser.add_argument("--sdlora_ortho_weight", type=float, default=0.0, help="Orthogonal loss weight for SD-LoRA (0 means disabled).")

    # ========== FlyPrompt configurations ==========
    parser.add_argument("--len_prompt", type=int, default=20, help="The length of the prompt for each expert")
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="The position of the prompt")
    parser.add_argument("--rp_dim", type=int, default=10000, help="The dimension of the random projection head")
    parser.add_argument("--rp_ridge", type=float, default=1e4, help="The ridge parameter for the random projection head")
    parser.add_argument("--ema_ratio", type=float, nargs="+", default=[0.9, 0.99], help="The EMA ratio for the expert FCs")
    parser.add_argument("--ensemble_method", type=str, default="softmax_max_prob", choices=["mean", "max_prob", "min_entropy", "softmax_mean", "softmax_max_prob", "softmax_min_entropy"],
                        help="Ensemble method for combining expert outputs: mean (average), max (maximum), min_entropy (minimum entropy), and softmax variants of these.")
    parser.add_argument("--router_type", type=str, default="rpfc", choices=["rpfc", "ws"],
                        help="Routing head type: rpfc (random projection FC) or ws (whitened subspace).")
    parser.add_argument("--ws_k", type=int, default=32,
                        help="Subspace rank for whitened subspace router.")
    parser.add_argument("--no_ema_ensemble", action="store_true", default=False,
                        help="Disable EMA head bank for FlyPrompt (only): skip EMA updates during training and use only the online fc head at evaluation.")
    parser.add_argument("--use_analytic_head", action="store_true", default=False,
                        help="Enable FlyPrompt frozen-feature analytic class head and extra per-eval diagnostics.")
    parser.add_argument("--use_analytic_gain", action="store_true", default=False,
                        help="Use stage-dependent analytic DAN gain as the FlyPrompt evaluation output.")
    parser.add_argument("--analytic_gain_max_lambda", type=float, default=0.5,
                        help="Maximum strength for the stage-dependent analytic DAN gain.")
    parser.add_argument("--analytic_gain_schedule", type=str, default="quadratic", choices=["quadratic", "linear", "constant"],
                        help="Stage schedule for analytic DAN gain strength.")

    # ========== FlyAdapter / FlyLoRA configurations ==========
    parser.add_argument("--fly_lora_rank", type=int, default=5,
                        help="LoRA rank for FlyLoRA expert modules.")
    parser.add_argument("--fly_lora_alpha", type=float, default=1.0,
                        help="Scaling factor alpha for FlyLoRA expert modules.")
    parser.add_argument("--fly_lora_layers", type=int, default=5,
                        help="Number of early ViT blocks using FlyLoRA K/V experts.")
    parser.add_argument("--fly_adapter_down_dim", type=int, default=10,
                        help="Adapter bottleneck dimension for FlyAdapter expert modules.")
    parser.add_argument("--fly_adapter_layers", type=int, default=5,
                        help="Number of early ViT blocks using FlyAdapter experts.")

    # ========== RPFC gating configurations ==========
    parser.add_argument("--use_rp_gate", action="store_true", default=False,
                        help="Use FlyPrompt-style RPFC head for task gating in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ========== EMA head bank configurations ==========
    parser.add_argument("--use_ema_head", action="store_true", default=False,
                        help="Use EMA-based classifier head bank and ensemble in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ======== Expert similarity analysis ==========
    parser.add_argument("--analysis_expert_similarity", action="store_true", default=False,
                        help="If set, run expert feature similarity / CKA (including residual vs common) analysis after training.")

    args = parser.parse_args()
    return args
