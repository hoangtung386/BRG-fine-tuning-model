import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def fill_holes(binary_mask):
    """Fill enclosed holes in a binary mask."""
    try:
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        return binary_mask
    return binary_fill_holes(binary_mask)


def predict_with_fill(model, img_tensor, device="cuda", threshold=0.5):
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))[-1]
        pred = torch.sigmoid(pred[0, 0]).cpu().numpy()
    binary = pred > threshold
    return fill_holes(binary)


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def visualize_prediction(model, dataset, idx=0, device="cuda"):
    model.eval()

    img, mask = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_batch)[-1]
        pred = torch.sigmoid(pred[0, 0]).cpu().numpy()

    img = denormalize(img).permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()
    pred_binary = fill_holes(pred > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Predicted Mask (prob)")
    axes[2].axis("off")

    axes[3].imshow(img)
    axes[3].imshow(pred_binary, alpha=0.5, cmap="jet")
    axes[3].set_title("Overlay Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_batch(model, dataloader, device="cuda", n_images=4):
    model.eval()

    batch = next(iter(dataloader))
    imgs, masks = batch[:n_images]
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = model(imgs)
        preds = torch.sigmoid(preds[-1])

    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))

    for i in range(n_images):
        img = denormalize(imgs[i]).permute(1, 2, 0).numpy()
        mask = masks[i].squeeze().numpy()
        pred = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("GT Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Pred Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


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


def visualize_training_progress(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    axes[0].plot(history.get("val_iou", []), label="Val IoU")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.get("val_iou", []), label="Val IoU", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation IoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
