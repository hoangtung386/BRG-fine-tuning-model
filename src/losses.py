import torch
import torch.nn.functional as F


def _get_pred_tensor(pred):
    if pred is None:
        raise ValueError("Model prediction is None")
    if isinstance(pred, (list, tuple)):
        for p in reversed(pred):
            if p is not None and isinstance(p, torch.Tensor):
                return p
        raise ValueError("No valid tensor found in model output")
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


def focal_bce_loss(pred, target, alpha=0.75, gamma=2.0):
    pred = _get_pred_tensor(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = torch.sigmoid(pred)
    pt = prob * target + (1 - prob) * (1 - target)
    focal_weight = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = focal_weight * (1 - pt).pow(gamma)
    return (focal_weight * bce).mean()


def ssim_loss(pred, target):
    pred = _get_pred_tensor(pred)
    pred = torch.sigmoid(pred)
    from pytorch_msssim import ssim

    return 1 - ssim(pred, target, size_average=True)


def filled_region_loss(pred, target, kernel_size=15, threshold=0.1):
    pred = _get_pred_tensor(pred)
    pred_sigmoid = torch.sigmoid(pred)

    kernel = torch.ones(
        1, 1, kernel_size, kernel_size, device=pred_sigmoid.device, dtype=pred_sigmoid.dtype
    )
    kernel_area = float(kernel_size * kernel_size)

    pred_dilated = F.conv2d(pred_sigmoid, kernel, padding=kernel_size // 2) / kernel_area
    pred_dilated = (pred_dilated > threshold).float()
    pred_closed = F.conv2d(pred_dilated, kernel, padding=kernel_size // 2) / kernel_area
    pred_closed = (pred_closed > threshold).float()

    holes = (pred_closed - pred_sigmoid.detach()).clamp(0, 1)
    if holes.sum() <= 0:
        return pred_sigmoid.new_tensor(0.0)
    return F.mse_loss(pred_sigmoid * holes, target * holes)


def combined_loss(
    pred,
    target,
    ssim_weight=5,
    bce_weight=1,
    dice_weight=1,
    iou_weight=1,
    hole_weight=2,
):
    pred = _get_pred_tensor(pred)
    ssim = ssim_loss(pred, target)
    bce = focal_bce_loss(pred, target, alpha=0.75, gamma=2.0)
    iou = 1 - iou_score(pred, target)
    dice = dice_loss(pred, target)
    hole = filled_region_loss(pred, target)
    return (
        ssim_weight * ssim
        + bce_weight * bce
        + iou_weight * iou
        + dice_weight * dice
        + hole_weight * hole
    )


def iou_score(pred, target, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def boundary_iou(pred, target, boundary_width=5, threshold=0.5):
    pred = _get_pred_tensor(pred)
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    kernel = torch.ones(1, 1, boundary_width, boundary_width, device=pred.device)
    pred_dilated = F.conv2d(pred.float(), kernel, padding=boundary_width // 2) > 0
    target_dilated = F.conv2d(target.float(), kernel, padding=boundary_width // 2) > 0
    pred_boundary = pred_dilated.float() - pred.float()
    target_boundary = target_dilated.float() - target.float()
    pred_boundary = (pred_boundary > 0).float()
    target_boundary = (target_boundary > 0).float()
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
