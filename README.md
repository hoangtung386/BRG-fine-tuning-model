# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task. Optimized for Colab Pro GPU (100GB VRAM).

## Project Structure

```
BRG-fine-tuning-model/
├── config/           # Configuration
│   └── __init__.py
├── src/              # Source code
│   ├── __init__.py
│   ├── dataset.py    # Dataset class (with augmentation)
│   ├── model.py      # Model loading & optimizer
│   ├── losses.py     # Loss functions (SSIM + BCE + IoU + Boundary IoU)
│   ├── trainer.py    # Training loop
│   ├── utils.py      # Utility functions
│   └── visualization.py  # Visualization functions
├── data/             # Dataset files
├── notebooks/        # Colab notebooks
│   └── train.ipynb   # Colab training notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- **IMG_SIZE**: 1024 (optimal for RMBG-2.0)
- **Loss**: 10×SSIM + 90×BCE + 0.25×IoU (BiRefNet formula)
- **Optimizer**: AdamW with differential LR (encoder: 5e-6, decoder: 2e-5)
- **Gradient Checkpointing**: Enabled to save VRAM
- **Augmentation**: Random erode/dilate, brightness/contrast
- **Metric**: Boundary IoU (5px edge) for best model selection

## Configuration

Edit `config/__init__.py`:
- `IMG_SIZE`: Input image size (default: 1024)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Training epochs (default: 30)
- `LEARNING_RATE`: Decoder LR (default: 2e-5)
- `LEARNING_RATE_ENCODER`: Encoder LR (default: 5e-6)
- `USE_GRADIENT_CHECKPOINTING`: Enable gradient checkpointing (default: True)
- `CKPT_DIR`: Checkpoint save path (Google Drive)

## Colab Usage

1. Open `notebooks/train.ipynb` in Colab
2. Select GPU runtime (G4 recommended)
3. Run cells sequentially
4. Checkpoints are saved to `/content/drive/MyDrive/rmbg_checkpoints/`

## Checkpoints

- `last_model.pth`: Latest checkpoint (includes optimizer state, epoch, IoU)
- `best_model.pth`: Best model by validation Boundary IoU