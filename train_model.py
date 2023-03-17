from config import *
from common import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.cuda.amp as amp

from modules.net import NetHead
import modules.fit as fit
import modules.valid as valid
import modules.loss as myloss
from modules.file import saveWeights, saveEpochInfo, loadWeights
from modules.plot import plotTrainResults

import os
import time


def trainMain():
    train_set = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=image_transform)
    valid_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=image_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(train_set.classes)

    model = NetHead(num_classes, pretrained=True)
    model.to(device)

    print(model)

    if os.path.isfile(WEIGHT_FILE):
        model.load_state_dict(torch.load(WEIGHT_FILE)["model"])

    total_batch = len(train_loader)
    print("Batch count : {}".format(total_batch))

    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = myloss.FocalLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # optim_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    optim_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=LEARNING_RATE,
        step_size_up=total_batch,
        cycle_momentum=False,
        mode="triangular2",
    )

    # 학습 시작
    total_start_time = time.time()
    train_accs, valid_accs, train_losses, valid_losses = [], [], [], []
    train_acc, valid_acc, best_acc = 0, 0, 0
    best_epoch = 0
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")
        epoch_start_time = time.time()

        if USE_AMP:
            train_acc, train_loss = fit.runAMP(device, train_loader, model, criterion, optimizer, scaler)
        else:
            train_acc, train_loss = fit.run(device, train_loader, model, criterion, optimizer)
        valid_acc, valid_loss = valid.run(device, valid_loader, model, criterion)

        optim_scheduler.step()

        print(f"Train - Acc: {(100*train_acc):>3.2f}%, Loss: {train_loss:>3.5f}")
        print(f"Valid - Acc: {(100*valid_acc):>3.2f}%, Loss: {valid_loss:>3.5f}")

        train_accs.append(train_acc * 100), valid_accs.append(valid_acc * 100)
        train_losses.append(train_loss), valid_losses.append(valid_loss)

        print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

        # 모델 저장
        if valid_acc > best_acc:
            saveWeights(model.state_dict(), optimizer.state_dict(), best_acc, valid_acc)  # Save weights
            saveEpochInfo(epoch, train_acc, train_loss, valid_acc, valid_loss)  # Write epoch info

            best_epoch = epoch
            best_acc = valid_acc

    print(f"Total time: {time.time() - total_start_time:.2f} seconds")
    print(f"Best epoch: {best_epoch+1}, Best acc: {best_acc * 100:>2.5f}%")

    result = {
        "train_accs": train_accs,
        "valid_accs": valid_accs,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "best_epoch": best_epoch,
        "best_acc": best_acc,
    }

    return result


torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

scaler = None
if USE_AMP:
    BATCH_SIZE *= 2
    scaler = amp.GradScaler()


image_transform = data_transform
if os.path.isfile(WEIGHT_FILE):
    print("Mode: retrain")
    image_transform = retrain_transform
    result = trainMain()
else:
    print("Mode: train")
    epoch_num = EPOCHS_PRETRAIN
    image_transform = data_transform
    result_pretrain = trainMain()  # pretrain

    print("Mode: retrain")
    epoch_num = EPOCHS
    image_transform = retrain_transform
    result_retrain = trainMain()  # finetune

    result = {
        "train_accs": result_pretrain["train_accs"] + result_retrain["train_accs"],
        "valid_accs": result_pretrain["valid_accs"] + result_retrain["valid_accs"],
        "train_losses": result_pretrain["train_losses"] + result_retrain["train_losses"],
        "valid_losses": result_pretrain["valid_losses"] + result_retrain["valid_losses"],
        "best_epoch": result_retrain["best_epoch"] + EPOCHS_PRETRAIN,
        "best_acc": result_retrain["best_acc"],
    }

# 학습 결과 그래프
plotTrainResults(
    result["train_accs"],
    result["valid_accs"],
    result["train_losses"],
    result["valid_losses"],
    result["best_epoch"],
    result["best_acc"],
)

print("Training done!")
