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

DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "tinyimagenet": TinyImageNet,
    "cub200": CUB200,
    "cars196": CARS196,
    "cub175": CUB175,
    "cubrandom": CUBRandom,
    "imagenet": ImageNet,
    "imagenet100": ImageNet100,
    "imagenet900": ImageNet900,
    "imagenetsub": ImageNetSub,
    "imagenet-r": Imagenet_R,
    "imagenetrandom": ImageNetRandom,
    'nch': NCH,
    'places365': Places365,
    "gtsrb": GTSRB,
    "wikiart": WIKIART
}

__all__ = [cls.__name__ for cls in DATASETS.values()] + ["DATASETS", "OnlineIterDataset"]