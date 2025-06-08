import os
from typing import Callable, Optional

import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import transforms


class CARS196(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = True
                 ) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.url = 'http://ftp.cs.stanford.edu/cs/cvgl/CARS196.zip'
        self.filename = 'CARS196.zip'

        fpath = os.path.join(self.root, self.filename)
        # if not os.path.isfile(fpath):
        #     if not download:
        #         raise RuntimeError('Dataset not found. You can use download=True to download it')
        #     else:
        #         print('Downloading from '+self.url)
        #         download_url(self.url, self.root, filename=self.filename)
        # if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
        #     import zipfile
        #     zip_ref = zipfile.ZipFile(fpath, 'r')
        #     zip_ref.extractall(self.root)
        #     zip_ref.close()
        #     import tarfile
        #     tar_ref = tarfile.open(os.path.join(self.root, 'CUB_200_2011.tgz'), 'r')
        #     tar_ref.extractall(self.root)
        #     tar_ref.close()

        ## split data according to classess
        # import scipy.io
        # import os
        # import re
        # import shutil

        # source = '/data/datasets/CARS196'
        # target = '/data/datasets/CARS196/split_imgs'
        # data = scipy.io.loadmat('/data/datasets/CARS196/cars_annos.mat')
        # class_names = data['class_names']
        # annotations = data['annotations']
        # #print(annotations)

        # for i in range(annotations.shape[1]):
        #     name = str(annotations[0, i][0])[2:-2]
        #     image_path = os.path.join(source, name)
        #     print(image_path)
        #     clas = int(annotations[0, i][5])
        #     class_name = str(class_names[0, clas-1][0]).replace(' ', '_')
        #     class_name = class_name.replace('/', '')
        #     target_path = os.path.join(target, class_name)
        #     if not os.path.isdir(target_path):
        #         os.mkdir(target_path)
        #     print(target_path)
        #     shutil.copy(image_path, target_path)
    
        self.dataset = ImageFolder(self.root, transforms.ToTensor() if transform is None else transform, target_transform)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train_data, test_data  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train_data if train else test_data
        self.classes = self.dataset.dataset.classes
        self.targets = []
        for i in self.dataset.indices:
            self.targets.append(self.dataset.dataset.targets[i])
        pass
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)
    
    # def prepare(self):
    #     annotations = loadmat(self.dataset_folder / 'cars_annos.mat')['annotations'][0]

    #     file_names = []
    #     class_ids = []
    #     is_tests = []
    #     bboxes = []

    #     for annotation in annotations:
    #         file_name = annotation[0][0]
    #         x1 = annotation[1][0][0]
    #         y1 = annotation[2][0][0]
    #         x2 = annotation[3][0][0]
    #         y2 = annotation[4][0][0]
    #         class_id = annotation[5][0][0]
    #         is_test = annotation[6][0][0]

    #         file_names.append(file_name)
    #         bboxes.append(f'{x1} {y1} {x2} {y2}')
    #         class_ids.append(class_id)
    #         is_tests.append(is_test)

    #     df_info = pd.DataFrame(
    #         list(
    #             zip(file_names, 
    #                 class_ids, 
    #                 bboxes,
    #                 is_tests)
    #             ), 
    #         columns=[
    #             'file_name', 
    #             'class_id', 
    #             'bbox',
    #             'is_test'
    #         ]
    #     )
        
    #     df_info['label'] = df_info['class_id'].apply(lambda x: f'cars_{x}')    
    #     df_info = self.add_image_sizes(df_info, self.dataset_folder)

    #     df_info = df_info[[
    #         'file_name', 
    #         'label',
    #         'bbox', 
    #         'width', 
    #         'height',
    #         'is_test'
    #     ]]
    
    #     print(df_info)
    #     print(df_info.dtypes)

    #     # img_idx = 28
    #     # fname = str(self.dataset_folder / df_info.iloc[img_idx].file_name)
    #     # bbox  = [int(x) for x in df_info.iloc[img_idx].bbox.split(' ')]
    #     # image = cv2.imread(fname)
    #     # image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0))
    #     # cv2.imwrite('img.jpg', image)
        
    #     df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False)
