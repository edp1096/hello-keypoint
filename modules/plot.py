from config import *

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os


def plotTrainResults(train_accs, valid_accs, train_losses, valid_losses, best_epoch, best_acc):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{DATASET_NAME} - {MODEL_NAME}")

    ax[0].set_title("Accuracy")
    ax[0].plot(train_accs, label="train")
    ax[0].plot(valid_accs, label="valid")
    ax[0].plot([best_epoch + 1], [best_acc * 100], marker="o", markersize=5, color="red", label="best")
    ax[0].set_ylim(0, 100)
    ax[0].legend()

    ax[1].set_title("Loss")
    ax[1].plot(train_losses, label="train")
    ax[1].plot(valid_losses, label="valid")
    ax[1].legend()

    plt.savefig(LOSS_RESULT_FILE)
    plt.show()


def scatter(x, labels, root=".", subtitle=None, dataset="MNIST"):
    num_classes = len(set(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")

    if dataset == "MNIST":
        idx2name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif dataset == "CIFAR10" or dataset == "STL10":
        idx2name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        dst_path = "data/dst"
        idx2name = []
        for file in os.listdir(os.path.join(dst_path, "train")):
            if os.path.isdir(os.path.join(dst_path, "train", file)):
                idx2name.append(file)

        if len(idx2name) == 0:
            raise Exception("Please specify the dataset")

    txts = []
    for i in range(num_classes):
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, idx2name[i], fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(os.path.join(root, str(subtitle)))
