from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .ranpac import RanPAC
from .slca import SLCA
from .hide_norga_trainer import HiDeGCLTrainer, NoRGaGCLTrainer
from .sdlora import SDLoRAGCL
from .sprompt import SPrompt as SPromptTrainer
from .ewc import EWC
from .lwf import LwF
from .seq_baselines import LinearProbe, SeqFinetune, SeqFinetuneSmallLR

METHODS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "flyadapter": FlyPrompt,
    "flylora": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "ranpac": RanPAC,
    "slca": SLCA,
    "hide": HiDeGCLTrainer,
    "hide_lora": HiDeGCLTrainer,
    "hide_adapter": HiDeGCLTrainer,
    "norga": NoRGaGCLTrainer,
    "sdlora": SDLoRAGCL,
    "sprompt": SPromptTrainer,
    "ewc": EWC,
    "lwf": LwF,
    "seq_finetune": SeqFinetune,
    "linear_probe": LinearProbe,
    "seq_finetune_small_lr": SeqFinetuneSmallLR,
}