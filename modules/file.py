from config import *

import torch

import os


def saveWeights(model_state_dict, optimizer_state_dict, best_acc, current_acc):
    os.makedirs(os.path.dirname(f"{OUTPUT_SAVE_ROOT}/"), exist_ok=True)

    state_dicts = {"model": model_state_dict, "optimizer": optimizer_state_dict}
    torch.save(state_dicts, f"{WEIGHT_FILE}")
    print(f"Saved best model and optimizer state to {WEIGHT_FILE}")
    print(f"Valid acc: {best_acc:>2.5f} -> {current_acc:>2.5f}\n")


def loadWeights():
    state_dicts = torch.load(f"{WEIGHT_FILE}")
    print(f"Loaded model and optimizer state from {WEIGHT_FILE}")

    return state_dicts


def saveEpochInfo(epoch, train_acc, train_loss, valid_acc, valid_loss):
    with open(WEIGHT_INFO_FILE, "w") as f:
        f.write(f"Epoch: {epoch+1}\n")
        f.write(f"Train acc: {train_acc * 100:>2.5f}%\n")
        f.write(f"Valid acc: {valid_acc * 100:>2.5f}%\n")
        f.write(f"Train loss: {train_loss:>2.5f}\n")
        f.write(f"Valid loss: {valid_loss:>2.5f}\n")
