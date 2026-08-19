import torch


def dice_score_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2 * intersection + eps) / (denom + eps)).mean()


def iou_score_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def binary_dice(pred, target, eps=1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def binary_iou(pred, target, eps=1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return float((inter + eps) / (union + eps))
