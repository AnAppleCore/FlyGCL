import logging.config

logging.config.fileConfig('configuration/logging.conf')

import os
import time

from configuration import config
from datasets import *
from methods import METHODS

os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import torch
import torch.multiprocessing

torch.backends.cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # Get Configurations
    args = config.base_parser()
    seed_lst = args.seeds
    if args.isa:
        seed_lst = [1]
    print('>>>>>>>running for seed: {}'.format(seed_lst))
    for seed in seed_lst:
        setattr(args, 'rnd_seed', seed)
        print(args)

        trainer = METHODS[args.mode](**vars(args))

        trainer.run()
        print(args.note)
        print(args.dataset)

if __name__ == "__main__":
    main()
    
    time.sleep(3)
