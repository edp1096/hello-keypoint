import torch

from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


class FacialKeypointsDataset(Dataset):
    def __init__(self, image_path, annotation_path, transforms=None):
        self.transforms = transforms

        self.images = []
        self.keypoints = []

        for i, fname in enumerate(os.listdir(image_path)):
            fname_base = os.path.splitext(fname)[0]

            # Load image
            ipath = os.path.join(image_path, fname)
            image = Image.open(ipath).convert("RGB")
            self.images.append(image)

            # Load keypoints
            kpath = os.path.join(annotation_path, f"{fname_base}.csv")
            keypoints = np.genfromtxt(kpath, delimiter=",", dtype=np.float32)
            self.keypoints.append(keypoints)

    def __getitem__(self, idx):
        return_dict = {"image": self.images[idx], "keypoints": self.keypoints[idx]}

        if self.transforms:
            return_dict = self.transforms(return_dict)

        return return_dict

    def __len__(self):
        return len(self.images)
