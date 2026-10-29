from typing import Callable, Optional

import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class CUBRandom(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False,
                 cls_lst=None
                 ) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(self.root)
            zip_ref.close()
            import tarfile
            tar_ref = tarfile.open(os.path.join(self.root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(self.root)
            tar_ref.close()
    
        self.dataset = ImageFolder(self.root + '/CUB_200_2011/images', transforms.ToTensor() if transform is None else transform, target_transform)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train_data, test_data  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train_data if train else test_data


        if cls_lst is None:
            keep = [i for i in range(175)]
        else:
            keep = cls_lst   

        temp_cls = []
        temp_dist = {}
        count = 0
        for idx, clss in zip(range(len(self.dataset.dataset.classes)), self.dataset.dataset.classes):
            if idx in keep:
                temp_cls.append(clss)
                temp_dist[clss] = count
                count += 1
        temp_img = []
        temp_targets = []
        keep_idx = []

        for idx in self.dataset.indices:
            if self.dataset.dataset.targets[idx] in keep:
                temp_targets.append(keep.index(self.dataset.dataset.targets[idx]))
                keep_idx.append(idx)

        self.classes = temp_cls
        self.class_to_idx = temp_dist

        self.targets = temp_targets
        self.dataset = torch.utils.data.Subset(self.dataset.dataset, keep_idx)
        self.keep = keep

        
    
    def __getitem__(self, index):
        # print(index)
        img, label = self.dataset.__getitem__(index)
        label = self.keep.index(label)
        return img, label

    def __len__(self):
        return len(self.dataset)
