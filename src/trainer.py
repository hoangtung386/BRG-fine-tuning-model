import os
import torch
from tqdm import tqdm

from src.losses import combined_loss, iou_score, boundary_iou
from src.utils import find_largest_tensor_recursive


def _get_valid_pred(output):
    """Find the tensor with largest spatial dimensions (last scale) in model output"""
    if output is None:
        raise ValueError("Model output is None")

    pred, size = find_largest_tensor_recursive(output)
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
        phase=1,
        resume_from=None,
        use_boundary_iou=True,
        patience=4,
        use_wandb=False,
        wandb_project="rmbg-lineart",
        wandb_run_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.phase = phase
        self.start_epoch = 1

        # Resume from checkpoint
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = ckpt.get("epoch", 1) + 1
            self.phase = ckpt.get("phase", phase)
            self.best_iou = ckpt.get("val_iou", 0.0)
            self.best_boundary_iou = ckpt.get("val_boundary_iou", 0.0)
            print(f"  Resumed from epoch {ckpt.get('epoch', 0)}, phase {self.phase}")
            print(
                f"  Best IoU: {self.best_iou:.4f}, Boundary IoU: {self.best_boundary_iou:.4f}"
            )
        else:
            self.best_iou = 0.0
            self.best_boundary_iou = 0.0

        self.use_boundary_iou = use_boundary_iou
        self.patience = patience
        self.epochs_without_improvement = 0
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

    def save_checkpoint(
        self, epoch, val_iou, val_boundary_iou, is_best=False, phase=1, train_loss=0.0
    ):
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Always save last model
        last_ckpt = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_iou": val_iou,
            "val_boundary_iou": val_boundary_iou,
            "train_loss": train_loss,
        }
        torch.save(last_ckpt, os.path.join(self.ckpt_dir, "last_model.pth"))

        # Save per-epoch checkpoint
        epoch_ckpt = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "val_iou": val_iou,
            "val_boundary_iou": val_boundary_iou,
            "train_loss": train_loss,
        }
        torch.save(
            epoch_ckpt, os.path.join(self.ckpt_dir, f"phase{phase}_ep{epoch}.pt")
        )

        # Save best model if improved
        if is_best:
            torch.save(
                self.model.state_dict(), os.path.join(self.ckpt_dir, "best_model.pth")
            )
            print(f"  -> Saved best_model.pth (Boundary IoU: {val_boundary_iou:.4f})")

    def train(self, train_loader, val_loader):
        print(f"Starting training from epoch {self.start_epoch} to {self.num_epochs}")
        for epoch in range(self.start_epoch, self.num_epochs + 1):
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
                    self.epochs_without_improvement = 0
                    print(f"  -> New best! Boundary IoU={self.best_boundary_iou:.4f}")
                else:
                    self.epochs_without_improvement += 1
            else:
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                    self.epochs_without_improvement = 0
                    print(f"  -> New best! IoU={self.best_iou:.4f}")
                else:
                    self.epochs_without_improvement += 1

            self.save_checkpoint(
                epoch, val_iou, val_boundary_iou, is_best, self.phase, train_loss
            )

            if self.use_wandb:
                self.wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_iou": val_iou,
                        "val_boundary_iou": val_boundary_iou,
                        "epoch": epoch,
                    }
                )

            if self.epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)"
                )
                break

        print(
            f"Training complete! Best IoU: {self.best_iou:.4f}, Best Boundary IoU: {self.best_boundary_iou:.4f}"
        )

        if self.use_wandb:
            self.wandb.finish()
