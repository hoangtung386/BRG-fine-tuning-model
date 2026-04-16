import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def combined_loss(pred, target, dice_weight=0.5, bce_weight=0.5):
    dice = dice_loss(pred, target)
    bce = bce_loss(pred, target)
    return dice_weight * dice + bce_weight * bce


def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
