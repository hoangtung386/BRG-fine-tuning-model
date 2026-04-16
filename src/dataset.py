import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LineArtDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=2048):
        self.files = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.tf_img = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.tf_img(img)

        mask = transforms.Resize((self.img_size, self.img_size))(mask)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()

        return img, mask
