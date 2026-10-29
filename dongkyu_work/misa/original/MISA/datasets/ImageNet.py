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


class ImageNet(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        self.fpath = os.path.join(root, 'imgnt')

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