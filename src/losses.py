import torch
import torch.nn.functional as F


def _get_pred_tensor(pred):
    print(f"DEBUG _get_pred_tensor: type={type(pred)}, is None={pred is None}")
    if pred is None:
        raise ValueError("Model prediction is None")

    # If it's a list/tuple, find the first valid tensor
    if isinstance(pred, (list, tuple)):
        print(f"DEBUG: pred is list/tuple with len={len(pred)}")

        # Flatten nested lists and find valid tensor
        def flatten_and_find(l, depth=0):
            print(f"DEBUG flatten depth={depth}, type={type(l)}")
            for i, item in enumerate(l):
                if isinstance(item, (list, tuple)):
                    result = flatten_and_find(item, depth + 1)
                    if result is not None:
                        return result
                elif item is not None and isinstance(item, torch.Tensor):
                    print(f"DEBUG: found valid tensor at index {i}, shape={item.shape}")
                    return item
                else:
                    print(
                        f"DEBUG: item[{i}] is None or not tensor: {type(item) if item is not None else None}"
                    )
            return None

        valid_pred = flatten_and_find(pred)
        if valid_pred is None:
            print(f"DEBUG: final pred is None!")
            raise ValueError("No valid tensor found in model output")
        pred = valid_pred

    if not isinstance(pred, torch.Tensor):
        print(f"DEBUG: pred is not tensor: {type(pred)}")
        raise ValueError(f"Model prediction is not a Tensor, got {type(pred)}")
    print(f"DEBUG: returning pred with shape {pred.shape}")
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
