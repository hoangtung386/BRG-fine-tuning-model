import os
import torch
import numpy as np
from PIL import Image


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mount_drive():
    from google.colab import drive

    drive.mount("/content/drive")


def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available")


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def find_largest_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        result = None
        best_size = 0
        for item in output:
            if isinstance(item, torch.Tensor):
                size = item.shape[-2] * item.shape[-1]
                if size > best_size:
                    best_size = size
                    result = item
        return result
    return output


def find_largest_tensor_recursive(obj, best_tensor=None, best_size=0):
    if obj is None:
        return best_tensor, best_size
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 4:
            size = obj.shape[-2] * obj.shape[-1]
            if size > best_size:
                return obj, size
        return best_tensor, best_size
    if isinstance(obj, (list, tuple)):
        for item in obj:
            best_tensor, best_size = find_largest_tensor_recursive(
                item, best_tensor, best_size
            )
    return best_tensor, best_size


def compute_metrics_batch(model, dataloader, device="cuda"):
    from src.losses import iou_score

    model.eval()
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)[-1]
            iou = iou_score(preds, masks)

            total_iou += iou.item() * imgs.size(0)
            total_samples += imgs.size(0)

    return total_iou / total_samples
