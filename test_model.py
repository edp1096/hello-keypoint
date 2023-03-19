from config import *
from common import device, test_transform

import torch
from torchvision import datasets, models, transforms

from modules.net import NetHead
from modules.file import loadWeights
from modules.plot import plotTrainResults, plotTrainResultsLoss, plotFaceWithKeypoints

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


test_set_path = os.listdir(f"{DATA_ROOT}/test")
rand_num = random.randint(0, len(test_set_path) - 1)
image = Image.open(f"{DATA_ROOT}/test/{test_set_path[rand_num]}").convert("RGB")
data = {"image": image, "keypoints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
data = test_transform(data)

num_classes = 6


model = NetHead(num_classes)
model.to(device)
model.load_state_dict(loadWeights()["model"])
model.eval()

with torch.no_grad():
    embed, pred = model(data["image"].unsqueeze(dim=0).to(device))

sample = {"image": data["image"], "keypoints": pred.cpu().squeeze(dim=0)}
plotFaceWithKeypoints(sample)
