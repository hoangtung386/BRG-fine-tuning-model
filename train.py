import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_NAME,
    DATASET_PATH,
    MASK_PATH,
    CKPT_DIR,
    TRAIN_RATIO,
)
from src.dataset import LineArtDataset
from src.model import load_model, unfreeze_all_params, create_optimizer
from src.trainer import Trainer
from src.utils import get_device, mount_drive, print_gpu_info


def main():
    device = get_device()
    print(f"Using device: {device}")
    print_gpu_info()

    mount_drive()

    dataset = LineArtDataset(DATASET_PATH, MASK_PATH, IMG_SIZE)
    train_len = int(TRAIN_RATIO * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_len}, Val samples: {val_len}")

    model = load_model(MODEL_NAME, device)
    unfreeze_all_params(model)

    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ckpt_dir=CKPT_DIR,
        num_epochs=NUM_EPOCHS,
    )

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
