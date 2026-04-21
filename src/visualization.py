import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.utils import denormalize, compute_metrics_batch


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
