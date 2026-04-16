import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class LineArtDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=1024, augment=True):
        self.files = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

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

    def __len__(self):
        return len(self.files)

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
        name = self.files[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            img = self._augment_image(img)
            mask = self._augment_mask(mask)

        img = self.tf_img(img)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()

        return img, mask
