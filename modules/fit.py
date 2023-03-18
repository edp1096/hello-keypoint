from config import *

import torch
import torch.cuda.amp as amp


def run(device, loader, model, loss_fn, optimizer):
    model.train()

    dataset_size = len(loader.dataset)
    loss_total = 0.0

    for batch, data in enumerate(loader):
        image, keypoints = data["image"].to(device), data["keypoints"].to(device)

        with torch.set_grad_enabled(True):
            embeds, logits = model(image)
            loss = loss_fn(logits, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    train_loss = loss_total / dataset_size

    return train_loss


def runAMP(device, loader, model, loss_fn, optimizer, scaler):
    model.train()

    dataset_size = len(loader.dataset)
    loss_total = 0.0

    for batch, data in enumerate(loader):
        image, keypoints = data["image"].to(device), data["keypoints"].to(device)

        with amp.autocast():
            with torch.set_grad_enabled(True):
                embed, logits = model(image.float())
                loss = loss_fn(logits, keypoints)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_total += loss.item()

    train_loss = loss_total / dataset_size

    return train_loss
