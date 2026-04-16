import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


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
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def _augment_mask(self, mask):
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            if random.random() < 0.5:
                mask = mask.filter(ImageFilter.MinFilter(kernel_size))
            else:
                mask = mask.filter(ImageFilter.MaxFilter(kernel_size))
        return mask

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

        if self.augment:
            img = self._augment_image(img)
            mask = self._augment_mask(mask)

        img = self.tf_img(img)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()

        return img, mask
