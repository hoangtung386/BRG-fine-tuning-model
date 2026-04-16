import os
import torch
from tqdm import tqdm

from src.losses import combined_loss, iou_score, boundary_iou


def _get_valid_pred(output):
    """Find the tensor with largest spatial dimensions (last scale) in model output"""
    if output is None:
        raise ValueError("Model output is None")

    def find_largest_tensor(obj, best_tensor=None, best_size=0):
        """Recursively find tensor with largest spatial size"""
        if obj is None:
            return best_tensor, best_size
        if isinstance(obj, torch.Tensor):
            # Check spatial dimensions (H, W) - last 2 dims
            if obj.dim() >= 4:
                size = obj.shape[-2] * obj.shape[-1]
                if size > best_size:
                    return obj, size
            return best_tensor, best_size
        if isinstance(obj, (list, tuple)):
            for item in obj:
                best_tensor, best_size = find_largest_tensor(
                    item, best_tensor, best_size
                )
        return best_tensor, best_size

    pred, size = find_largest_tensor(output)
    if pred is None:
        raise ValueError("No valid tensor in model output")
    return pred


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        ckpt_dir,
        num_epochs=30,
        use_boundary_iou=True,
        use_wandb=False,
        wandb_project="rmbg-lineart",
        wandb_run_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.best_iou = 0.0
        self.best_boundary_iou = 0.0
        self.use_boundary_iou = use_boundary_iou
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        if use_wandb:
            import wandb

            self.wandb = wandb
            run_name = wandb_run_name or f"run_{num_epochs}ep"
            wandb.init(project=wandb_project, name=run_name)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            output = self.model(imgs)
            pred = _get_valid_pred(output)
            loss = combined_loss(pred, masks)

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

                output = self.model(imgs)
                pred = _get_valid_pred(output)

                total_iou += iou_score(pred, masks).item()
                total_boundary_iou += boundary_iou(pred, masks).item()

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

            if self.use_wandb:
                self.wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_iou": val_iou,
                        "val_boundary_iou": val_boundary_iou,
                        "epoch": epoch,
                    }
                )

        print(
            f"Training complete! Best IoU: {self.best_iou:.4f}, Best Boundary IoU: {self.best_boundary_iou:.4f}"
        )

        if self.use_wandb:
            self.wandb.finish()
