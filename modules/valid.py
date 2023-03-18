import torch


def run(device, dataloader, model, loss_fn):
    model.eval()

    dataset_size = len(dataloader.dataset)
    loss_total = 0.0

    for data in dataloader:
        image, keypoints = data["image"].to(device), data["keypoints"].to(device)

        with torch.no_grad():
            embed, logits = model(image)
            loss = loss_fn(logits, keypoints)

        loss_total += loss.item()

    valid_loss = loss_total / dataset_size

    return valid_loss
