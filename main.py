# import torch
import os
import time
import torch
from configuration import config
from datasets import *
from methods.er_baseline import ER
from methods.clib import CLIB
from methods.L2P import L2P
from methods.rainbow_memory import RM
from methods.Finetuning import FT
from methods.slca import SLCA
from methods.lwf import LwF
from methods.derpp import DERPP 
from methods.er_ace import ERACE
from methods.er_acep import ERACEP
from methods.mvp import MVP
from methods.dualprompt import DualPrompt
from methods.dualprompt_sam import DualPrompt as sam
from methods.dualprompt_fam import DualPrompt as fam
from methods.codaprompt import CodaPrompt
import random
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# torch.backends.cudnn.enabled = False
methods = { "er": ER, "clib":CLIB, 'L2P':L2P, 'rm':RM, 'Finetuning':FT, 'mvp':MVP, 'DualPrompt':DualPrompt, 
           'sam': sam, 'fam':fam, 'lwf':LwF, 'derpp': DERPP,  'erace': ERACE, 'eracep': ERACEP, 'slca':SLCA, 
           'CodaPrompt': CodaPrompt}

torch.backends.cudnn.enabled = False
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # Get Configurations
    args = config.base_parser()
    seed_lst = args.seeds
    if args.isa:
        seed_lst = [1]
    print('>>>>>>>running for seed: {}'.format(seed_lst))
    for seed in seed_lst: # ,2,3,4,5
        setattr(args, 'rnd_seed', seed)
        print(args)

        trainer = methods[args.mode](**vars(args))

        trainer.run()
        print(args.note)
        print(args.dataset)

if __name__ == "__main__":
    main()
    
    time.sleep(3)
