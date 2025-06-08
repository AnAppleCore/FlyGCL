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
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive,
                                        download_url, verify_str_arg)


class ImageNet(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        # self.fpath = os.path.join(root, 'imgnt')
        self.fpath = root

        if not os.path.exists(self.fpath):
            if not download:
                print(self.fpath)
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            fpath = self.fpath + '/train'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            # print(self.__dir__())
            # raise ValueError('stop')
            # self.classes = [i for i in range(1000)]
            # self.class_to_idx = [i for i in range(1000)]


        else:
            fpath = self.fpath + '/val'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            # self.classes = [i for i in range(1000)]
            # self.class_to_idx = [i for i in range(1000)]

        # self.data = datasets.ImageFolder(fpath, transform=transform)