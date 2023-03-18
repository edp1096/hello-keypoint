from config import *
from common import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.cuda.amp as amp

from modules.dataset import FacialKeypointsDataset
from modules.net import NetHead
import modules.fit as fit
import modules.valid as valid
import modules.loss as myloss
from modules.file import saveWeights, saveEpochInfo, loadWeights
from modules.plot import plotTrainResults

import os
import time


def trainMain():
    image_path = f"{DATA_ROOT}/train/images"
    annotation_path = f"{DATA_ROOT}/train/annotations"

    data_set = FacialKeypointsDataset(image_path, annotation_path, transforms=image_transform)

    train_size = int(0.8 * len(data_set))
    valid_size = len(data_set) - train_size
    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = train_set[0]["keypoints"].shape[0]  # 6

    model = NetHead(num_classes, pretrained=True)
    model.to(device)

    print(model)

    total_batch = len(train_loader)
    print("Batch count : {}".format(total_batch))

    criterion = nn.MSELoss().to(device)
    # criterion = myloss.NaNMSELoss
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    # 학습 시작
    total_start_time = time.time()
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")
        epoch_start_time = time.time()

        if USE_AMP:
            train_loss = fit.runAMP(device, train_loader, model, criterion, optimizer, scaler)
        else:
            train_loss = fit.run(device, train_loader, model, criterion, optimizer)
        valid_loss = valid.run(device, valid_loader, model, criterion)

        print(f"Train - Loss: {train_loss:>3.5f}")
        print(f"Valid - Loss: {valid_loss:>3.5f}")

        print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

    result = {}

    return result


torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

scaler = None
if USE_AMP:
    BATCH_SIZE *= 2
    scaler = amp.GradScaler()


image_transform = data_transform
epoch_num = EPOCHS_PRETRAIN
result_pretrain = trainMain()  # pretrain

print("Training done!")
