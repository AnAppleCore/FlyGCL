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

class ImageNetSub(ImageFolder):
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
        
        keep = [0, 5, 6, 8, 10, 11, 12, 13, 15, 21, 22, 23, 27, 29, 33, 34, 36, 37, 38, 39, 43, 47, 51, 52, 53, 56, 57, 58, 59, 60, 64, 66, 72, 73, 74, 77, 80, 82, 84, 91, 95, 100, 101, 102, 103, 110, 119, 120, 121, 134, 136, 137, 141, 152, 156, 158, 161, 162, 164, 165, 166, 168, 169, 174, 177, 181, 182, 184, 185, 189, 192, 193, 194, 198, 202, 204, 205, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 232, 233, 247, 250, 251, 252, 253, 254, 255, 257, 258, 259, 261, 262, 276, 290, 292, 293, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 310, 312, 316, 317, 318, 320, 321, 322, 324, 326, 327, 328, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 346, 348, 350, 351, 352, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 366, 368, 369, 370, 371, 373, 374, 375, 376, 377, 378, 379, 380, 381, 395, 396, 441, 442, 443, 444, 446, 449, 450, 451, 452, 453, 454, 455, 456, 459, 460, 461, 464, 465, 468, 469, 472, 473, 475, 476, 477, 478, 479, 481, 482, 483, 484, 486, 487, 489, 490, 491, 493, 494, 495, 497, 501, 502, 503, 504, 505, 506, 507, 510, 512, 513, 514, 515, 516, 518, 519, 520, 521, 522, 523, 524, 527, 528, 529, 530, 531, 533, 534, 535, 536, 537, 538, 539, 540, 541, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 558, 559, 560, 561, 563, 564, 566, 569, 571, 572, 574, 575, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 606, 607, 608, 609, 610, 611, 613, 615, 616, 617, 618, 620, 622, 623, 624, 626, 628, 629, 630, 631, 632, 633, 634, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 653, 654, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 676, 679, 680, 681, 684, 685, 686, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 705, 706, 708, 709, 710, 711, 712, 713, 714, 715, 717, 718, 719, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 732, 736, 738, 740, 741, 742, 743, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 762, 763, 764, 766, 767, 769, 770, 771, 772, 773, 775, 776, 777, 778, 780, 782, 783, 784, 785, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 802, 803, 804, 805, 807, 809, 810, 812, 813, 814, 816, 818, 819, 820, 822, 823, 824, 825, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 838, 840, 841, 843, 844, 846, 847, 848, 851, 852, 854, 855, 856, 857, 858, 859, 860, 861, 863, 864, 865, 867, 868, 869, 870, 871, 872, 875, 876, 878, 880, 881, 882, 883, 884, 885, 886, 889, 891, 892, 893, 894, 895, 896, 897, 898, 901, 902, 903, 904, 905, 906, 907, 908, 911, 912, 913, 914, 915, 916, 918, 919, 920, 921, 922, 925, 926, 927, 930, 931, 933, 934, 936, 937, 939, 940, 941, 942, 943, 944, 946, 948, 949, 952, 953, 955, 956, 958, 959, 960, 961, 965, 966, 968, 969, 971, 974, 976, 977, 979, 980, 981, 982, 983, 984, 985, 986, 987, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
        
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