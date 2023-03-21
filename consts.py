from config import *

import torch
from torchvision import transforms

import modules.xfrm as xfrm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}, Device: {device}")

test_transform = transforms.Compose(
    [
        xfrm.PilToNumpy(),
        xfrm.ResizeKeypointAndImage((IMAGE_SIZE, IMAGE_SIZE)),
        xfrm.KeypointImageToTensor(),
    ]
)

pretrain_transform = transforms.Compose(
    [
        xfrm.PilToNumpy(),
        xfrm.ResizeKeypointAndImage((IMAGE_SIZE, IMAGE_SIZE)),
        xfrm.KeypointImageToTensor(),
    ]
)

retrain_transform = transforms.Compose(
    [
        xfrm.PilToNumpy(),
        xfrm.ResizeKeypointAndImage((IMAGE_SIZE, IMAGE_SIZE)),
        xfrm.RandomHorizontalFlip(),
        xfrm.RandomRotation(45),
        xfrm.RandomTranslation((0.1, 0.1)),
        xfrm.RandomBrightnessAdjust(0.5),
        xfrm.RandomJitter(0.5),
        xfrm.KeypointImageToTensor(),
    ]
)
