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

class ImageNet900(ImageFolder):
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
        
        keep = [i for i in range(100)]
        
        temp_cls = []
        temp_dist = {}
        count = 0
        for idx, clss in zip(range(len(self.classes)), self.classes):
            if idx in keep:
                temp_cls.append(clss)
                temp_dist[clss] = count
                count += 1
        temp_img = []
        temp_targets = []
        for img in self.imgs:
            if img[1] in keep:
                temp_img.append((img[0], keep.index(img[1])))
                temp_targets.append(keep.index(img[1]))

        self.classes = temp_cls
        self.class_to_idx = temp_dist
        self.imgs = temp_img
        self.targets = temp_targets

        # raise ValueError(self.classes, self.class_to_idx, self.imgs[:5], len(self.targets) )
        # self.classes = [i for i in range(len(keep))]
        # self.class_to_idx = []
        # if self.dataset == 'imagenetsub':   
        #     idx = [i for i in range(len(self.train_dataset)) if self.train_dataset.imgs[i][1] in keep ]
        #     self.online_iter_dataset = Subset(self.online_iter_dataset, idx)

    def __len__(self):
        return len(self.targets)