# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task.

## Project Structure

```
BRG-fine-tuning-model/
├── config/           # Configuration
│   └── __init__.py
├── src/              # Source code
│   ├── __init__.py
│   ├── dataset.py    # Dataset class
│   ├── model.py      # Model loading & optimizer
│   ├── losses.py     # Loss functions (Dice + BCE, IoU)
│   └── trainer.py    # Training loop
├── data/             # Dataset files
│   ├── images/       # Input images
│   └── masks/        # Mask images
├── notebooks/        # Colab notebooks
│   └── train.ipynb   # Colab training notebook
├── utils.py          # Utility functions
├── train.py          # Main entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare dataset:**
   - Place images in `data/images/`
   - Place masks in `data/masks/`
   - Images and masks must have matching filenames

3. **Run training locally:**
   ```bash
   python train.py
   ```

## Configuration

Edit `config/__init__.py` to customize:
- `IMG_SIZE`: Input image size (default: 2048)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Training epochs (default: 30)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `CKPT_DIR`: Checkpoint save path (default: Google Drive)

## Colab Usage (G4 100GB VRAM)

1. Push project to GitHub
2. Open `notebooks/train.ipynb` in Colab
3. Update `GITHUB_URL` in cell 1
4. Select GPU runtime (G4 recommended)
5. Run cells sequentially
6. Checkpoints are saved to `/content/drive/MyDrive/rmbg_checkpoints/`

## Model

- Base model: `briaai/RMBG-2.0`
- Full fine-tuning (all parameters unfrozen)
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Loss: Dice + BCE
- Metric: IoU

## Checkpoints

- `last_model.pth`: Latest checkpoint (includes optimizer state)
- `best_model.pth`: Best model by validation IoU