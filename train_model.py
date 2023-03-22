from config import *
from consts import *

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
from modules.plot import plotTrainResults, plotTrainResultsLoss, plotFaceWithKeypoints
from modules.xfrm import *

import os
import time


def trainMain():
    image_path = f"{DATA_ROOT}/images"
    annotation_path = f"{DATA_ROOT}/annotations"

    data_set = FacialKeypointsDataset(image_path, annotation_path, transforms=None)
    num_classes = POINT_NUM

    data_set.transforms = image_transform

    train_size = int(0.8 * len(data_set))
    valid_size = len(data_set) - train_size
    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

    # # sample
    # sample = train_loader.dataset[1]
    # plotFaceWithKeypoints(sample)
    # exit()

    model = NetHead(num_classes, pretrained=True)
    model.to(device)

    print(model)

    total_batch = len(train_loader)
    print("Batch count : {}".format(total_batch))

    criterion = myloss.NaNMSELoss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    # 학습 시작
    total_start_time = time.time()
    train_losses, valid_losses = [], []
    train_loss, valid_loss, best_loss = 9999, 9999, 9999
    best_epoch = 0
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

        train_losses.append(train_loss), valid_losses.append(valid_loss)

        print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

        # 모델 저장
        if valid_loss < best_loss:
            saveWeights(model.state_dict(), optimizer.state_dict(), best_loss, valid_loss)  # Save weights
            saveEpochInfo(epoch, train_loss, train_loss, valid_loss, valid_loss)  # Write epoch info

            best_epoch = epoch
            best_loss = valid_loss

    print(f"Total time: {time.time() - total_start_time:.2f} seconds")
    print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss:.5f}")

    result = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }

    return result


torch.manual_seed(7)
if device == "cuda":
    torch.cuda.manual_seed_all(7)

scaler = None
if USE_AMP:
    BATCH_SIZE *= 2
    scaler = amp.GradScaler()


image_transform = pretrain_transform

if os.path.isfile(WEIGHT_FILE):
    print("Mode: retrain")
    epoch_num = EPOCHS
    image_transform = retrain_transform
    result = trainMain()
else:
    print("Mode: train")
    epoch_num = EPOCHS_PRETRAIN
    result_pretrain = trainMain()

    print("Mode: retrain")
    epoch_num = EPOCHS
    image_transform = retrain_transform
    result_retrain = trainMain()

    result = {
        "train_losses": result_pretrain["train_losses"] + result_retrain["train_losses"],
        "valid_losses": result_pretrain["valid_losses"] + result_retrain["valid_losses"],
        "best_epoch": result_retrain["best_epoch"] + EPOCHS_PRETRAIN,
        "best_loss": result_retrain["best_loss"],
    }


# 학습 결과 그래프
plotTrainResultsLoss(
    result["train_losses"],
    result["valid_losses"],
    result["best_epoch"],
    result["best_loss"],
)

print("Training done!")
