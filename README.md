# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task. Optimized for Colab Pro GPU (100GB VRAM).

## Project Structure

```
BRG-fine-tuning-model/
├── config/              # Configuration
│   └── __init__.py
├── src/                 # Source code
│   ├── __init__.py
│   ├── dataset.py       # Dataset class with subfolder support & augmentation
│   ├── model.py        # Model loading, freezing options
│   ├── losses.py       # Loss functions (SSIM + BCE + IoU + Boundary IoU)
│   ├── trainer.py     # Training loop with wandb logging
│   ├── utils.py       # Utility functions
│   └── visualization.py  # Visualization functions
├── data/                # Dataset files (images/, masks/)
├── notebooks/           # Colab notebooks
│   └── train.ipynb    # Colab training notebook (clone & run)
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- **IMG_SIZE**: 1024 (optimal for RMBG-2.0)
- **Loss**: 10×SSIM + 90×BCE + 0.25×IoU (BiRefNet formula)
- **Optimizer**: AdamW with trainable params only
- **Freezing**: Freeze encoder to speed up training (~5-10x faster)
- **Augmentation**: Random erode/dilate, brightness/contrast
- **Metric**: Boundary IoU (5px edge) for best model selection
- **WandB**: Integrated logging for experiment tracking

## Configuration

Edit `config/__init__.py`:
- `IMG_SIZE`: Input image size (default: 1024)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Training epochs (default: 30)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `FREEZE_ENCODER`: Freeze encoder (default: True) - set False for full fine-tune
- `FREEZE_DECODER_EXCEPT_LAST`: Freeze decoder except last layer
- `CKPT_DIR`: Checkpoint save path (Google Drive)

## Training Modes

| Mode | Speed | Best For |
|------|-------|---------|
| `FREEZE_ENCODER=True` | ~5-10x faster | Quick fine-tuning, limited GPU |
| `FREEZE_ENCODER=False` | Slower | Full fine-tune, best quality |

## Colab Usage

1. Clone project from GitHub to Colab
2. Add HuggingFace & W&B tokens in Colab secrets
3. Select GPU runtime (G4 recommended)
4. Run cells sequentially
5. Checkpoints saved to `/content/drive/MyDrive/rmbg_checkpoints/`

## Checkpoints

- `last_model.pth`: Latest checkpoint (includes optimizer state, epoch, IoU)
- `best_model.pth`: Best model by validation Boundary IoU