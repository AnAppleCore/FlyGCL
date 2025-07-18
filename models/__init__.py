from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .moeranpac import MoERanPAC
from .ranpac import RanPAC

MODELS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "moeranpac": MoERanPAC,
    "ranpac": RanPAC,
}