import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from src.utils import find_largest_tensor


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


def remove_background(model, image_path, output_path=None, img_size=1024, threshold=0.5, device="cuda"):
    img_tensor, original_img = preprocess_image(image_path, img_size)
    pred_mask = predict(model, img_tensor, device)
    pred_mask_original_size = resize_mask_to_size(
        pred_mask, original_img.size, resample=Image.BILINEAR
    )
    
    mask_binary = (pred_mask_original_size > threshold).astype(np.uint8)
    
    original_array = np.array(original_img)
    
    result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
    result_array[:, :, :3] = original_array
    result_array[:, :, 3] = mask_binary * 255
    
    result_img = Image.fromarray(result_array, mode="RGBA")
    
    if output_path:
        result_img.save(output_path)
        print(f"Saved: {output_path}")
    
    return result_img


def process_folder(model, input_dir, output_dir=None, img_size=1024, threshold=0.5, device="cuda"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir.parent / (input_dir.name + "_nobg")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    image_paths = []
    for ext in extensions:
        for p in input_dir.rglob(f"*.{ext}"):
            image_paths.append(p)
    
    print(f"Found {len(image_paths)} images")
    
    for img_path in image_paths:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{img_path.stem}_nobg.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            remove_background(model, str(img_path), str(out_path), img_size, threshold, device)
        except Exception as e:
            print(f"Error: {img_path.name} - {e}")
    
    print(f"Done! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Remove background using trained RMBG-2.0 model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--input", type=str, required=True, help="Input image path or folder")
    parser.add_argument("--output", type=str, default=None, help="Output path (for single image) or folder")
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = load_trained_model(args.checkpoint, device=device)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        output_path = args.output if args.output else input_path.parent / f"{input_path.stem}_nobg.png"
        remove_background(model, str(input_path), str(output_path), args.img_size, args.threshold, device)
    elif input_path.is_dir():
        process_folder(model, str(input_path), args.output, args.img_size, args.threshold, device)
    else:
        print(f"Error: {args.input} not found")


if __name__ == "__main__":
    main()
