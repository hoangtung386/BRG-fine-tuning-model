import torch
import torch.nn.functional as F


def _get_pred_tensor(pred):
    if pred is None:
        raise ValueError("Model prediction is None")
    if isinstance(pred, (list, tuple)):
        # Get the last non-None tensor
        for p in reversed(pred):
            if p is not None and isinstance(p, torch.Tensor):
                pred = p
                break
        else:
            raise ValueError("No valid tensor found in model output list")
    if not isinstance(pred, torch.Tensor):
        raise ValueError(f"Model prediction is not a Tensor, got {type(pred)}")
    return pred


def dice_loss(pred, target, smooth=1.0):
    pred = _get_pred_tensor(pred)
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_loss(pred, target):
    pred = _get_pred_tensor(pred)
    return F.binary_cross_entropy_with_logits(pred, target)


def ssim_loss(pred, target, window_size=11):
    pred = _get_pred_tensor(pred)
    from pytorch_msssim import ssim

    pred = torch.sigmoid(pred)
    return 1 - ssim(pred, target, window_size=window_size, size_average=True)


def combined_loss(pred, target, ssim_weight=10, bce_weight=90, iou_weight=0.25):
    pred = _get_pred_tensor(pred)
    ssim = ssim_loss(pred, target)
    bce = bce_loss(pred, target)
    iou = 1 - iou_score(pred, target)
    return ssim_weight * ssim + bce_weight * bce + iou_weight * iou


def iou_score(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def boundary_iou(pred, target, boundary_width=5, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    kernel = torch.ones(1, 1, boundary_width, boundary_width, device=pred.device)
    pred_dilated = F.conv2d(pred, kernel, padding=boundary_width // 2) > 0
    target_dilated = F.conv2d(target, kernel, padding=boundary_width // 2) > 0
    pred_boundary = pred_dilated & ~pred
    target_boundary = target_dilated & ~target
    pred_boundary = pred_boundary.float()
    target_boundary = target_boundary.float()
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def dice_coefficient(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def precision_score(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + 1e-6) / (tp + fp + 1e-6)


def recall_score(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + 1e-6) / (tp + fn + 1e-6)


def f1_score(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def compute_all_metrics(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    return {
        "iou": iou_score(pred, target, threshold),
        "boundary_iou": boundary_iou(pred, target, threshold=threshold),
        "dice": dice_coefficient(pred, target, threshold),
        "precision": precision_score(pred, target, threshold),
        "recall": recall_score(pred, target, threshold),
        "f1": f1_score(pred, target, threshold),
    }
