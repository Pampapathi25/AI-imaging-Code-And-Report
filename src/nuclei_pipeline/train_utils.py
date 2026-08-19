import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .metrics import dice_score_from_logits, iou_score_from_logits


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    total_loss = total_dice = total_iou = 0.0
    n_batches = 0
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_dice += float(dice_score_from_logits(logits.detach(), masks).cpu())
        total_iou += float(iou_score_from_logits(logits.detach(), masks).cpu())
        n_batches += 1
    return total_loss/n_batches, total_dice/n_batches, total_iou/n_batches


def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
