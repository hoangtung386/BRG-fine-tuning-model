import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import random

try:
    from scipy.ndimage import binary_dilation, binary_erosion

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LineArtDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        img_size=1024,
        augment=True,
        img_exts=(".jpg", ".jpeg", ".png"),
        mask_ext=".png",
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.img_exts = img_exts
        self.mask_ext = mask_ext
        self.samples = []

        self._scan_subfolders()

        self.tf_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
            ]
        )

    def _scan_subfolders(self):
        """Scan subfolders and create image-mask pairs"""
        for subfolder in sorted(os.listdir(self.img_dir)):
            img_sub_path = os.path.join(self.img_dir, subfolder)
            mask_sub_path = os.path.join(self.mask_dir, subfolder)

            if os.path.isdir(img_sub_path) and os.path.isdir(mask_sub_path):
                for img_name in sorted(os.listdir(img_sub_path)):
                    if not img_name.lower().endswith(self.img_exts):
                        continue

                    base_name = os.path.splitext(img_name)[0]
                    mask_name = base_name + self.mask_ext
                    mask_full_path = os.path.join(mask_sub_path, mask_name)

                    if os.path.exists(mask_full_path):
                        self.samples.append(
                            {
                                "image": os.path.join(img_sub_path, img_name),
                                "mask": mask_full_path,
                                "subfolder": subfolder,  # Track source subfolder for debugging
                                "base_name": base_name,
                            }
                        )

        print(f"Loaded {len(self.samples)} image-mask pairs from {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def _augment_mask(self, mask):
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            if random.random() < 0.5:
                mask = mask.filter(ImageFilter.MinFilter(kernel_size))
            else:
                mask = mask.filter(ImageFilter.MaxFilter(kernel_size))

        if random.random() < 0.15:
            mask = self._line_thickness_jitter(mask)

        if random.random() < 0.05:
            mask = self._drop_random_stroke(mask)

        return mask

    def _line_thickness_jitter(self, mask):
        arr = np.array(mask)
        jitter = random.uniform(-2, 2)
        kernel = int(abs(jitter)) + 1
        if jitter > 0 and HAS_SCIPY:
            struct = np.ones((kernel, kernel))
            arr = binary_dilation(arr > 127, structure=struct).astype(np.uint8) * 255
        elif jitter < 0 and HAS_SCIPY:
            struct = np.ones((kernel, kernel))
            arr = binary_erosion(arr > 127, structure=struct).astype(np.uint8) * 255
        elif jitter > 0:
            arr = np.array(mask.filter(ImageFilter.MaxFilter(kernel)))
        elif jitter < 0:
            arr = np.array(mask.filter(ImageFilter.MinFilter(kernel)))
        return Image.fromarray(arr)

    def _drop_random_stroke(self, mask):
        arr = np.array(mask)
        h, w = arr.shape
        for _ in range(random.randint(1, 3)):
            y = random.randint(2, h - 3)
            x = random.randint(2, w - 3)
            size = random.randint(1, 3)
            arr[y - size : y + size, x - size : x + size] = 0
        return Image.fromarray(arr)

    def _augment_image(self, img):
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.9, 1.1)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        if random.random() < 0.2:
            contrast_factor = random.uniform(0.9, 1.1)
            img = transforms.functional.adjust_contrast(img, contrast_factor)

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")

        # Verify mapping for debugging
        if self.augment and random.random() < 0.001:  # Rare debug
            print(f"DEBUG: {sample['subfolder']}/{sample['base_name']} -> OK")

        if self.augment:
            img = self._augment_image(img)
            mask = self._augment_mask(mask)

        img = self.tf_img(img)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()

        return img, mask


def create_train_val_split(dataset, train_ratio=0.9, shuffle=True, seed=42):
    """Create train/val split without using random_split"""
    n = len(dataset)
    n_train = int(train_ratio * n)

    if shuffle:
        indices = torch.randperm(
            n, generator=torch.Generator().manual_seed(seed)
        ).tolist()
    else:
        indices = list(range(n))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    return train_dataset, val_dataset
