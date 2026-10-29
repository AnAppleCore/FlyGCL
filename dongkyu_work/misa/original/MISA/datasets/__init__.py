from .CUB200 import CUB200
from .TinyImageNet import TinyImageNet
from .OnlineIterDataset import OnlineIterDataset
from .Imagenet_R import Imagenet_R
from .ImageNet import ImageNet
from .ImageNetSub import ImageNetSub
from .ImageNet100 import ImageNet100
from .ImageNet900 import ImageNet900
from .ImageNetRandom import ImageNetRandom
from .NCH import NCH
from .CUB175 import CUB175
from .CUBrandom import CUBRandom
from .GTSRB import GTSRB
from .WIKIART import WIKIART

from torchvision.datasets import CIFAR10, CIFAR100, Places365 #, ImageNet

__all__ = [
    "CUB200",
    "TinyImageNet",
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    "Imagenet_R",
    "ImageNetSub",
    "ImageNet100",
    "ImageNet900",
    "ImageNetRandom",
    "OnlineIterDataset",
    "NCH",
    "CUB175",
    "CUBRandom",
    "Places365",
    "GTSRB",
    "WIKIART"
]