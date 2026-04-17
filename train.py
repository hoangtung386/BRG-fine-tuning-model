#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms.functional as TF

load_dotenv()
token = os.getenv("HF_TOKEN")

from transformers import AutoModelForImageSegmentation
from src.freeze_strategy import BiRefNetFreezeStrategy
from src.dataset import LineArtDataset, create_train_val_split


class BoundaryLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_edge = self._compute_edge(pred)
        target_edge = self._compute_edge(target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        boundary = F.mse_loss(pred_edge, target_edge)
        return bce + self.alpha * boundary

    def _compute_edge(self, x):
        return x - F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.3, boundary_weight=0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(alpha=boundary_weight)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        total = 0
        for p in pred:
            total += (
                self.bce_weight * self.bce(p, target)
                + self.dice_weight * self.dice(p, target)
                + self.boundary_weight * self.boundary(p, target)
            )
        return total


class LineArtAugmentation:
    def __init__(self, p_dilation=0.2, p_erode=0.2, p_netal_drop=0.05):
        self.p_dilation = p_dilation
        self.p_erode = p_erode
        self.p_netal_drop = p_netal_drop

    def __call__(self, img, mask):
        if random.random() < self.p_dilation:
            kernel_size = random.choice([3, 5])
            mask = mask.filter(ImageFilter.MaxFilter(kernel_size))

        if random.random() < self.p_erode:
            kernel_size = random.choice([3, 5])
            mask = mask.filter(ImageFilter.MinFilter(kernel_size))

        if random.random() < self.p_netal_drop:
            mask = self._drop_random_stroke(mask)

        return img, mask

    def _drop_random_stroke(self, mask):
        arr = np.array(mask)
        h, w = arr.shape
        n_drops = random.randint(1, 3)
        for _ in range(n_drops):
            y = random.randint(2, h - 3)
            x = random.randint(2, w - 3)
            size = random.randint(1, 3)
            arr[y - size : y + size, x - size : x + size] = 0
        return Image.fromarray(arr)


class BiRefNetTrainer:
    PHASE_CONFIGS = {
        1: {
            "name": "decoder_squeeze",
            "epochs": 3,
            "lr": 5e-4,
            "freeze_fn": "apply_phase_1",
        },
        2: {"name": "stage_0_1", "epochs": 8, "lr": 2e-4, "freeze_fn": "apply_phase_2"},
        3: {
            "name": "full_finetune",
            "epochs": 10,
            "lr": 1e-4,
            "freeze_fn": "apply_phase_3",
        },
    }

    def __init__(
        self,
        model_name="briaai/RMBG-2.0",
        img_dir="data/images",
        mask_dir="data/masks",
        img_size=1024,
        batch_size=2,
        accumulation_steps=4,
        checkpoint_dir="checkpoints",
        device="cuda",
    ):
        self.device = device
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        print("Loading model...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, token=token, trust_remote_code=True
        ).to(device)

        self.freeze_strategy = BiRefNetFreezeStrategy(self.model)
        self.criterion = CombinedLoss()
        self.scaler = GradScaler()

    def setup_dataloaders(self):
        print("Setting up dataloaders...")
        self.train_dataset = LineArtDataset(
            self.img_dir, self.mask_dir, img_size=self.img_size, augment=True
        )
        self.val_dataset = LineArtDataset(
            self.img_dir, self.mask_dir, img_size=self.img_size, augment=False
        )

        train_indices = list(range(int(0.9 * len(self.train_dataset))))
        val_indices = list(
            range(int(0.9 * len(self.val_dataset)), len(self.val_dataset))
        )

        self.train_loader = DataLoader(
            torch.utils.data.Subset(self.train_dataset, train_indices),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            torch.utils.data.Subset(self.val_dataset, val_indices),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

        print(f"Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")

    def train_phase(self, phase: int, config: dict):
        freeze_fn = getattr(self.freeze_strategy, config["freeze_fn"])
        freeze_fn()
        self.freeze_strategy.print_summary(config["name"])

        lr = config["lr"]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if p.requires_grad and "bb." not in n
                    ],
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if p.requires_grad and "bb." in n
                    ],
                    "lr": lr * 0.1,
                },
            ],
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"] * len(self.train_loader)
        )

        for epoch in range(config["epochs"]):
            self.model.train()
            total_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Phase {phase} Ep {epoch + 1}")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                scheduler.step()
                total_loss += loss.item() * self.accumulation_steps
                pbar.set_postfix(
                    {"loss": f"{loss.item() * self.accumulation_steps:.4f}"}
                )

            avg_loss = total_loss / len(self.train_loader)
            print(f"Phase {phase} Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            self._save_checkpoint(phase, epoch, avg_loss)

    def _save_checkpoint(self, phase, epoch, loss):
        path = os.path.join(
            self.checkpoint_dir, f"phase{phase}_ep{epoch + 1}_loss{loss:.4f}.pt"
        )
        torch.save(
            {
                "phase": phase,
                "epoch": epoch,
                "loss": loss,
                "model_state": self.model.state_dict(),
            },
            path,
        )
        print(f"Saved: {path}")

    def train(self):
        self.setup_dataloaders()

        for phase, config in self.PHASE_CONFIGS.items():
            print(f"\n{'=' * 60}")
            print(f"Starting Phase {phase}: {config['name']}")
            print(f"{'=' * 60}")
            self.train_phase(phase, config)

        print("\nTraining complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--mask_dir", default="data/masks")
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    trainer = BiRefNetTrainer(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()
