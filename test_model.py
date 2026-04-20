import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def load_trained_model(checkpoint_path, model_name="briaai/RMBG-2.0", device="cuda"):
    print(f"Loading model: {model_name}")
    from transformers import AutoModelForImageSegmentation
    
    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded weights from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found, using pretrained model")
    
    model = model.to(device)
    model.eval()
    return model


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


def preprocess_image(image_path, img_size=1024):
    from torchvision import transforms
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img), img


def resize_mask_to_size(mask_array, target_size, resample=Image.NEAREST):
    """
    Resize a mask/probability map to target PIL size (width, height).
    mask_array is expected in [0, 1].
    """
    mask_uint8 = np.clip(mask_array * 255.0, 0, 255).astype(np.uint8)
    resized = Image.fromarray(mask_uint8).resize(target_size, resample=resample)
    return np.array(resized).astype(np.float32) / 255.0


def predict(model, image_tensor, device="cuda"):
    img_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = find_largest_tensor(output)
        pred = torch.sigmoid(pred)[0, 0]
    return pred.cpu().numpy()


def denormalize(tensor):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)
    
    tp = intersection
    fp = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {"IoU": iou, "Dice": dice, "Precision": precision, "Recall": recall, "F1": f1}


def test_model_on_data(model, data_dir, mask_dir=None, img_size=1024, threshold=0.5, device="cuda", output_dir=None):
    data_dir = Path(data_dir)
    output_root = Path(output_dir) if output_dir else None
    
    # Find all images
    extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    image_paths = []
    for ext in extensions:
        for p in data_dir.rglob(f"*.{ext}"):
            image_paths.append(p)
    
    print(f"Found {len(image_paths)} images in {data_dir}")
    
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    category_metrics = {}
    
    for img_path in tqdm(image_paths, desc="Testing"):
        try:
            rel_path = img_path.relative_to(data_dir)
            category = rel_path.parts[0]
            
            img_tensor, original_img = preprocess_image(str(img_path), img_size)
            pred_mask = predict(model, img_tensor, device)
            pred_mask_original_size = resize_mask_to_size(
                pred_mask, original_img.size, resample=Image.BILINEAR
            )
            mask_binary = (pred_mask_original_size > threshold).astype(np.uint8) * 255
            
            if output_root:
                rel_path = img_path.relative_to(data_dir)
                out_path = output_root / rel_path.parent / f"{img_path.stem}_mask.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask_binary).save(out_path)
            
            if mask_dir:
                mask_path = Path(mask_dir) / category / (img_path.stem + ".png")
                if not mask_path.exists():
                    mask_path = Path(mask_dir) / (img_path.stem + ".png")
                
                if mask_path.exists():
                    gt_mask = np.array(Image.open(mask_path).convert("L"))
                    pred_for_eval = pred_mask
                    if gt_mask.shape != pred_mask.shape:
                        pred_for_eval = resize_mask_to_size(
                            pred_mask,
                            (gt_mask.shape[1], gt_mask.shape[0]),
                            resample=Image.BILINEAR,
                        )
                    metrics = compute_metrics(pred_for_eval, gt_mask, threshold)
                    metrics["category"] = category
                    all_metrics.append(metrics)
                    
                    if category not in category_metrics:
                        category_metrics[category] = []
                    category_metrics[category].append(metrics)
        
        except Exception as e:
            print(f"Error: {img_path.name} - {e}")
    
    if all_metrics:
        print("\n========== OVERALL METRICS ==========")
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in ["IoU", "Dice", "Precision", "Recall", "F1"]}
        for k, v in avg.items():
            print(f"{k}: {v:.4f}")
        
        if category_metrics:
            print("\n========== METRICS BY CATEGORY ==========")
            for cat, metrics in category_metrics.items():
                cat_avg = {k: np.mean([m[k] for m in metrics]) for k in ["IoU", "Dice", "Precision", "Recall", "F1"]}
                print(f"\n{cat}:")
                for k, v in cat_avg.items():
                    print(f"  {k}: {v:.4f}")
    
    return all_metrics


def visualize_sample(model, data_dir, mask_dir=None, img_size=1024, threshold=0.5, device="cuda", num_samples=4):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
    data_dir = Path(data_dir)
    
    extensions = ["jpg", "jpeg", "png"]
    image_paths = []
    for ext in extensions:
        for p in data_dir.rglob(f"*.{ext}"):
            image_paths.append(p)
            if len(image_paths) >= num_samples:
                break
    
    if not image_paths:
        print("No images found")
        return
    
    num_samples = len(image_paths)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_paths):
        img_tensor, _ = preprocess_image(str(img_path), img_size)
        pred_mask = predict(model, img_tensor, device)
        mask_binary = (pred_mask > threshold).astype(np.uint8) * 255
        
        img_denorm = denormalize(img_tensor.cpu()).permute(1, 2, 0).numpy()
        img_denorm = np.clip(img_denorm, 0, 1)
        
        axes[i, 0].imshow(img_denorm)
        axes[i, 0].set_title(img_path.name)
        axes[i, 0].axis("off")
        
        if mask_dir:
            mask_path = Path(mask_dir) / img_path.relative_to(data_dir).parts[0] / (img_path.stem + ".png")
            if mask_path.exists():
                gt_mask = np.array(Image.open(mask_path).convert("L"))
                axes[i, 1].imshow(gt_mask, cmap="gray")
                axes[i, 1].set_title("Ground Truth")
            else:
                axes[i, 1].text(0.5, 0.5, "No GT", ha="center", va="center", transform=axes[i, 1].transAxes)
                axes[i, 1].set_title("GT (not found)")
            axes[i, 1].axis("off")
        
        axes[i, 2].imshow(mask_binary, cmap="gray")
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test trained RMBG-2.0 model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = load_trained_model(args.checkpoint, device=device)
    
    if args.visualize:
        visualize_sample(model, args.data_dir, args.mask_dir, args.img_size, args.threshold, device)
    else:
        test_model_on_data(model, args.data_dir, args.mask_dir, args.img_size, args.threshold, device, args.output_dir)


if __name__ == "__main__":
    main()