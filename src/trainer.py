import os
import torch
from tqdm import tqdm

from src.losses import combined_loss, iou_score, boundary_iou


class Trainer:
    def __init__(
        self, model, optimizer, device, ckpt_dir, num_epochs=30, use_boundary_iou=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.best_iou = 0.0
        self.best_boundary_iou = 0.0
        self.use_boundary_iou = use_boundary_iou

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            preds = self.model(imgs)[-1]
            loss = combined_loss(preds, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_iou = 0
        total_boundary_iou = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                preds = self.model(imgs)[-1]
                total_iou += iou_score(preds, masks).item()
                total_boundary_iou += boundary_iou(preds, masks).item()

        avg_iou = total_iou / len(val_loader)
        avg_boundary_iou = total_boundary_iou / len(val_loader)

        return avg_iou, avg_boundary_iou

    def save_checkpoint(self, epoch, val_iou, val_boundary_iou, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_iou": val_iou,
            "val_boundary_iou": val_boundary_iou,
        }

        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.ckpt_dir, "last_model.pth"))

        if is_best:
            torch.save(
                self.model.state_dict(), os.path.join(self.ckpt_dir, "best_model.pth")
            )

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_iou, val_boundary_iou = self.validate(val_loader)

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val IoU={val_iou:.4f}, Val Boundary IoU={val_boundary_iou:.4f}"
            )

            if self.use_boundary_iou:
                is_best = val_boundary_iou > self.best_boundary_iou
                if is_best:
                    self.best_boundary_iou = val_boundary_iou
                    self.best_iou = val_iou
                    print(f"  -> New best! Boundary IoU={self.best_boundary_iou:.4f}")
            else:
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                    print(f"  -> New best! IoU={self.best_iou:.4f}")

            self.save_checkpoint(epoch, val_iou, val_boundary_iou, is_best)

        print(
            f"Training complete! Best IoU: {self.best_iou:.4f}, Best Boundary IoU: {self.best_boundary_iou:.4f}"
        )
