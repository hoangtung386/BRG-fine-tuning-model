import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import random

try:
    from scipy.ndimage import binary_dilation, binary_erosion, binary_propagation

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _disk_kernel(radius):
    diameter = radius * 2 + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


class LineArtDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        img_size=1024,
        augment=True,
        img_exts=(".jpg", ".jpeg", ".png"),
        mask_ext=".png",
        trapped_ball_radius=3,
        use_structural_guidance=True,
        guidance_strength=0.35,
        use_progressive_patch_shuffle=True,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.img_exts = img_exts
        self.mask_ext = mask_ext
        self.trapped_ball_radius = trapped_ball_radius
        self.use_structural_guidance = use_structural_guidance
        self.guidance_strength = guidance_strength
        self.use_progressive_patch_shuffle = use_progressive_patch_shuffle
        self.shuffle_grids = [2, 4, 8, 16, 32]
        self._shuffle_calls = 0
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

    def _trapped_ball_close(self, img):
        """Close small line gaps using trapped-ball style morphology (R=3 by default)."""
        if not HAS_SCIPY:
            return img

        arr = np.array(img.convert("RGB"))
        gray = arr.mean(axis=2) / 255.0
        line_mask = gray < 0.85
        background = ~line_mask

        # Flood-fill from image borders to estimate reachable background.
        border_seed = np.zeros_like(background, dtype=bool)
        border_seed[0, :] = background[0, :]
        border_seed[-1, :] = background[-1, :]
        border_seed[:, 0] = background[:, 0]
        border_seed[:, -1] = background[:, -1]
        flooded_background = binary_propagation(border_seed, mask=background)

        kernel = _disk_kernel(self.trapped_ball_radius)
        trapped_background = background & (~flooded_background)
        trapped_background = binary_erosion(trapped_background, structure=kernel)
        trapped_background = binary_dilation(trapped_background, structure=kernel)

        # Reinforce closed boundaries around trapped regions to prevent leakage.
        closed_lines = line_mask | binary_dilation(trapped_background, structure=kernel)
        out = arr.copy()
        out[closed_lines] = np.minimum(out[closed_lines], 25)
        return Image.fromarray(out.astype(np.uint8))

    def _compute_skeleton_map(self, mask):
        """Extract a thin structural skeleton map from GT mask."""
        mask_arr = np.array(mask, dtype=np.uint8) > 127
        if not HAS_SCIPY:
            return mask_arr.astype(np.float32)

        skel = np.zeros_like(mask_arr, dtype=bool)
        element = _disk_kernel(1)
        work = mask_arr.copy()

        # Morphological skeletonization.
        while work.any():
            eroded = binary_erosion(work, structure=element)
            opened = binary_dilation(eroded, structure=element)
            skel |= work & (~opened)
            work = eroded
        return skel.astype(np.float32)

    def _inject_structural_guidance(self, img, skeleton_map):
        if not self.use_structural_guidance:
            return img
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        sk = np.clip(skeleton_map, 0.0, 1.0)
        arr = arr * (1.0 - self.guidance_strength * sk[..., None])
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _progressive_patch_shuffle(self, img, mask):
        """Shuffle local patches (2x2 -> 32x32 progression) to enforce local matching."""
        if not (self.augment and self.use_progressive_patch_shuffle):
            return img, mask

        if random.random() > 0.35:
            return img, mask

        self._shuffle_calls += 1
        stage_span = max(1, len(self.samples) // 2)
        stage_idx = min(len(self.shuffle_grids) - 1, self._shuffle_calls // stage_span)
        grid = self.shuffle_grids[stage_idx]

        img_arr = np.array(img.convert("RGB"))
        mask_arr = np.array(mask.convert("L"))
        h, w = mask_arr.shape

        ph = max(1, h // grid)
        pw = max(1, w // grid)

        img_patches = []
        mask_patches = []
        coords = []
        for gy in range(grid):
            for gx in range(grid):
                y0 = gy * ph
                x0 = gx * pw
                y1 = h if gy == grid - 1 else (gy + 1) * ph
                x1 = w if gx == grid - 1 else (gx + 1) * pw
                coords.append((y0, y1, x0, x1))
                img_patches.append(img_arr[y0:y1, x0:x1].copy())
                mask_patches.append(mask_arr[y0:y1, x0:x1].copy())

        perm = list(range(len(coords)))
        random.shuffle(perm)

        out_img = img_arr.copy()
        out_mask = mask_arr.copy()
        for dst_idx, src_idx in enumerate(perm):
            y0, y1, x0, x1 = coords[dst_idx]
            patch_img = img_patches[src_idx]
            patch_mask = mask_patches[src_idx]
            if patch_img.shape[:2] != (y1 - y0, x1 - x0):
                continue
            out_img[y0:y1, x0:x1] = patch_img
            out_mask[y0:y1, x0:x1] = patch_mask

        return Image.fromarray(out_img), Image.fromarray(out_mask)

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

        img = self._trapped_ball_close(img)
        img, mask = self._progressive_patch_shuffle(img, mask)
        skeleton_map = self._compute_skeleton_map(mask)
        img = self._inject_structural_guidance(img, skeleton_map)

        img = self.tf_img(img)
        # For very bright line-art inputs, invert to better match pretrained feature statistics.
        if img.mean() > 0.8:
            img = 1.0 - img
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
