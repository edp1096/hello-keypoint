from config import *
from consts import device, test_transform

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


sample_path = "data/sample"
fname = "will_farrell_crop.jpg"

image = Image.open(f"{sample_path}/{fname}").convert("RGB")
data = {"image": image, "keypoints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
data = test_transform(data)

num_classes = 6


model = NetHead(num_classes, pretrained=False)
model.to(device)
model.load_state_dict(loadWeights()["model"])
model.eval()

with torch.no_grad():
    embed, pred = model(data["image"].unsqueeze(dim=0).to(device))

sample = {"image": data["image"], "keypoints": pred.cpu().squeeze(dim=0)}
plotFaceWithKeypoints(sample, is_save=True)
