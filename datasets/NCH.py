import glob
import os
import os.path
import pathlib
from pathlib import Path
from shutil import move, rmtree
from typing import Any, Callable, Optional, Tuple

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive,
                                        download_url, verify_str_arg)


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