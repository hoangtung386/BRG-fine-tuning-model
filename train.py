import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.dataset import LineArtDataset
from src.model import load_model, unfreeze_all_params, create_optimizer
from src.trainer import Trainer
from src.utils import get_device, mount_drive, print_gpu_info


def main():
    device = get_device()
    print(f"Using device: {device}")
    print_gpu_info()

    mount_drive()

    dataset = LineArtDataset(
        config.DATASET_PATH, config.MASK_PATH, config.IMG_SIZE, augment=True
    )
    train_len = int(config.TRAIN_RATIO * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_len}, Val samples: {val_len}")

    model = load_model(
        config.MODEL_NAME,
        device,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    )
    unfreeze_all_params(model)

    optimizer = create_optimizer(
        model,
        lr_decoder=config.LEARNING_RATE,
        lr_encoder=config.LEARNING_RATE_ENCODER,
        weight_decay=config.WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ckpt_dir=config.CKPT_DIR,
        num_epochs=config.NUM_EPOCHS,
        use_boundary_iou=True,
    )

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
