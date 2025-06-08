from torchvision.datasets import CIFAR10, CIFAR100, Places365

from .CARS196 import CARS196
from .CUB175 import CUB175
from .CUB200 import CUB200
from .CUBrandom import CUBRandom
from .GTSRB import GTSRB
from .ImageNet import ImageNet
from .ImageNet100 import ImageNet100
from .ImageNet900 import ImageNet900
from .Imagenet_R import Imagenet_R
from .ImageNetRandom import ImageNetRandom
from .ImageNetSub import ImageNetSub
from .NCH import NCH
from .OnlineIterDataset import OnlineIterDataset
from .TinyImageNet import TinyImageNet
from .WIKIART import WIKIART

__all__ = [
    "CUB200",
    "CARS196",
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