import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple, Callable, Optional

import glob
from shutil import move, rmtree
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image
from torch.utils.data import Subset

class NCH(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        self.fpath = os.path.join(root, 'NCH')

        if self.train:
            fpath = self.fpath + '/NCT-CRC-HE-100K'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)

        else:
            fpath = self.fpath + '/CRC-VAL-HE-7K'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)



    def __len__(self):
        return len(self.targets)