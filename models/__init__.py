from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .ranpac import RanPAC
from .hide_norga_prefix_vit import HiDePrefixModel, NoRGaPrefixModel
from .hide_lora_vit import HiDeLoRAModel
from .hide_adapter_vit import HiDeAdapterModel
from .sdlora import SDLoRAModel
from .sprompt import SPrompt
from .finetune_vit import FinetuneViT

MODELS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "ranpac": RanPAC,
    "hide": HiDePrefixModel,
    "hide_lora": HiDeLoRAModel,
    "hide_adapter": HiDeAdapterModel,
    "norga": NoRGaPrefixModel,
    "sdlora": SDLoRAModel,
    "sprompt": SPrompt,
    "ewc": FinetuneViT,
    "lwf": FinetuneViT,
}