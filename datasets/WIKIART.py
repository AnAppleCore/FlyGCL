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


class WIKIART(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.path = self.root + '/wikiart/wikiart/'
        super().__init__(self.path, transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)]) if transform is None else transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),transform]), target_transform=target_transform)
        generator = torch.Generator().manual_seed(0)
        len_train = int(len(self.samples) * 0.8)
        len_test = len(self.samples) - len_train
        self.train_sample = torch.randperm(len(self.samples), generator=generator)
        self.test_sample = self.train_sample[len_train:].sort().values.tolist()
        self.train_sample = self.train_sample[:len_train].sort().values.tolist()

        if train:
            self.classes = [i for i in range(43)]
            self.class_to_idx = [i for i in range(43)]
            samples = []
            for idx in self.train_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        else:
            self.classes = [i for i in range(43)]
            self.class_to_idx = [i for i in range(43)]
            samples = []
            for idx in self.test_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)