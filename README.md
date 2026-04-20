# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task. Optimized for Colab Pro GPU (100GB VRAM).

## Project Structure

```
BRG-fine-tuning-model/
├── config/                  # Configuration
│   └── __init__.py
├── src/                     # Source code
│   ├── __init__.py
│   ├── dataset.py           # Dataset + line art augmentation
│   ├── model.py             # Model loading
│   ├── freeze_strategy.py  # 3-phase freeze strategy
│   ├── losses.py           # Loss functions (SSIM + BCE + IoU + Boundary)
│   ├── trainer.py          # Training loop with wandb
│   ├── utils.py            # Utilities
│   └── visualization.py   # Visualization
├── data/                    # Dataset (images/, masks/)
├── notebooks/             # Colab notebooks
│   └── train.ipynb       # Colab training notebook
├── train.py               # Local training script
├── birefnet_architecture.mermaid  # Architecture diagram
├── requirements.txt
├── .env                   # Environment variables
└── README.md
```

## 3-Phase Freeze Strategy

Domain adaptation from natural images to line art (black & white anime):

| Phase | Description | Trainable Params | Epochs |
|-------|------------|------------------|-------|
| 1 | Decoder + Squeeze + PatchEmbed | ~25M (11.5%) | 2-3 |
| 2 | + Stage 0-1 + Alternating Stage 2 | ~74M (33.5%) | 5-8 |
| 3 | Full fine-tune | ~173M (78.6%) | 5-10 |

### Why Alternating Freeze?

- **Line art** has different feature distribution than natural images
- **Low-level** (Stage 0-1): Need to relearn edge detection for thin strokes
- **Mid-level** (Stage 2): Alternating freeze preserves semantic while adapting
- **High-level** (Stage 3): Keep object-level features frozen

## Features

- **IMG_SIZE**: 1024 (optimal for RMBG-2.0)
- **Loss**: BCE + Dice + Boundary (combined)
- **LR Scheduler**: CosineAnnealing
- **Freeze Strategy**: 3-phase alternating for domain adaptation
- **Augmentation**: Line thickness jitter, stroke drop, erode/dilate
- **Metric**: Boundary IoU (5px edge)
- **Mixed Precision**: AMP training

## Line Art Augmentation

- **Line thickness jitter** (±2px using scipy dilation/erosion)
- **Random stroke drop** (5%) - prevents overfitting to line thickness
- **Dilation/Erosion** for mask augmentation

## Configuration

Edit `config/__init__.py`:
- `IMG_SIZE`: Input size (default: 1024)
- `BATCH_SIZE`: Batch size (default: 2-4)
- `NUM_EPOCHS`: Training epochs per phase
- `LEARNING_RATE`: Decoder LR (default: 5e-4)
- `LEARNING_RATE_ENCODER`: Encoder LR (default: 1e-5)

## Local Training

```bash
conda activate fine_tune
python train.py --img_dir data/images --mask_dir data/masks --batch_size 2
```

## Colab Usage

1. Clone project from GitHub to Colab
2. Add HuggingFace & W&B tokens in Colab secrets
3. Select GPU runtime (G4 recommended, 100GB VRAM)
4. Run cells sequentially
5. Checkpoints saved to Google Drive

### Training Phases in Notebook

- **Cell 7**: Load model + apply_phase_1() + create optimizer
- **Cell 10**: Train Phase 1 (2-3 epochs)
- **Cell 11**: apply_phase_2() + train Phase 2
- **Cell 12**: apply_phase_3() + train Phase 3

## Checkpoints

- `phase{1,2,3}_ep{epoch}_loss{loss}.pt`: Phase checkpoints
- Best model saved by validation Boundary IoU

## Key Differences from Original

| Aspect | Original | This Version |
|--------|----------|--------------|
| Freezing | Full encoder | 3-phase alternating |
| Params Phase 1 | ~220M (all) | ~25M (11.5%) |
| VRAM usage | ~100GB | ~30GB |
| Time (30 epochs) | ~30h | ~8-12h |
| Augmentation | Basic | Line art specific |