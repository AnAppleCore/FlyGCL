from typing import Callable, Optional
import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class Imagenet_R(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        
        self.fpath = root

        if not os.path.exists(self.fpath):
            if not download:
                print(self.fpath)
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        if train:
            fpath = self.fpath + '/train'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            # print(self.__dir__())
            # raise ValueError('stop')
            # self.classes = [i for i in range(1000)]
            # self.class_to_idx = [i for i in range(1000)]

        else:
            fpath = self.fpath + '/test'
            super().__init__(fpath, transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
        
    #     self.root = os.path.expanduser(root)
    #     self.url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    #     self.filename = 'imagenet-r.tar'

    #     fpath = os.path.join(self.root, self.filename)
    #     if not os.path.isfile(fpath):
    #         if not download:
    #            raise RuntimeError('Dataset not found. You can use download=True to download it')
    #         else:
    #             print('Downloading from '+ self.url)
    #             download_url(self.url, self.root, filename=self.filename)
    #     if not os.path.exists(os.path.join(self.root, 'imagenet-r')):
    #         import tarfile
    #         tar = tarfile.open(fpath, 'r')
    #         tar.extractall(os.path.join(self.root))
    #         tar.close()

    #     self.path = self.root + '/imagenet-r/'
    #     super().__init__(self.path, transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)]) if transform is None else transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),transform]), target_transform=target_transform)
    #     generator = torch.Generator().manual_seed(0)
    #     len_train = int(len(self.samples) * 0.8)
    #     len_test = len(self.samples) - len_train
    #     self.train_sample = torch.randperm(len(self.samples), generator=generator)
    #     self.test_sample = self.train_sample[len_train:].sort().values.tolist()
    #     self.train_sample = self.train_sample[:len_train].sort().values.tolist()

    #     if train:
    #         self.classes = [i for i in range(200)]
    #         self.class_to_idx = [i for i in range(200)]
    #         samples = []
    #         for idx in self.train_sample:
    #             samples.append(self.samples[idx])
    #         self.targets = [s[1] for s in samples]
    #         self.samples = samples

    #     else:
    #         self.classes = [i for i in range(200)]
    #         self.class_to_idx = [i for i in range(200)]
    #         samples = []
    #         for idx in self.test_sample:
    #             samples.append(self.samples[idx])
    #         self.targets = [s[1] for s in samples]
    #         self.samples = samples

    # def __len__(self) -> int:
    #     return len(self.samples)