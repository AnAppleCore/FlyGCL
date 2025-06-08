from .clib import CLIB
from .codaprompt import CodaPrompt
from .derpp import DERPP
from .dualprompt import DualPrompt
from .dualprompt_fam import DualPrompt as fam
from .dualprompt_sam import DualPrompt as sam
from .er_ace import ERACE
from .er_acep import ERACEP
from .er_baseline import ER
from .Finetuning import FT
from .FlyPrompt import FlyPrompt
from .L2P import L2P
from .lwf import LwF
from .mvp import MVP
from .rainbow_memory import RM
from .slca import SLCA

METHODS = {"er": ER, "clib":CLIB, "L2P":L2P, "rm":RM, "Finetuning":FT, "mvp":MVP, "DualPrompt":DualPrompt, 
           "sam": sam, "fam":fam, "lwf":LwF, "derpp": DERPP,  "erace": ERACE, "eracep": ERACEP, "slca":SLCA, 
           "CodaPrompt": CodaPrompt, "FlyPrompt":FlyPrompt}