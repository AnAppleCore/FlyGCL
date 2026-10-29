import argparse

def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")
    parser.add_argument(
        "--mode",
        type=str,
        default="DualPrompt",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")

    parser.add_argument(
        "--log_path",
        type=str,
        default="result",
        help="The path logs are saved.",
    )
    parser.add_argument(
        "--model_name", type=str, default="singlePrompt", help="Model name"
    )

    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch.")
    #parser.add_argument("--load_pt", action="store_true", default=False)
    parser.add_argument("--nobatchmask", action="store_true", default=False) # If false, use seen mask
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])

    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )

    parser.add_argument(
        "--transforms",
        nargs="*",
        default=['cutmix', 'autoaug'],
        help="Additional train transforms [cutmix, cutout, randaug]",
    )

    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")
    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    parser.add_argument("--note", type=str, help="Short description of the exp")

    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")
    parser.add_argument('--selection_size', type=int, default=1, help='# candidates to use for ViT_Prompt')

    # for memory buffer
    parser.add_argument(
        "--memory_size", type=int, default=0, help="Episodic memory size"
    )
    parser.add_argument("--pos_prompt", type=int, default=5, help="prompt positons")
    parser.add_argument("--prompt_length", type=int, default=10, help="prompt length")
    
    # for logit type experiment
    parser.add_argument("--logit_type", type=str, default='linear', choices=['linear', 'cos_sim'])

    args = parser.parse_args()
    return args
