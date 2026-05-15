# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task. Runs on local machine with GPU.

## Project Structure

```
BRG-fine-tuning-model/
├── config/              # Configuration
│   └── __init__.py
├── src/                 # Source code
│   ├── __init__.py
│   ├── dataset.py       # Dataset with subfolder support
│   ├── model.py        # Model loading & freezing
│   ├── losses.py       # Masked redrawing loss + boundary boost + hole penalty
│   ├── trainer.py     # Training loop with wandb + early stopping
│   ├── utils.py      # Utilities
│   └── visualization.py  # Visualization + optional hole filling
├── dataset_lie_art/     # Dataset (images + masks)
├── notebooks/           # Jupyter notebooks
│   └── train.ipynb  # Training notebook
├── checkpoints/        # Model checkpoints (created during training)
├── test_model.py       # Test script
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Place training data in `dataset_lie_art/`:

```
dataset_lie_art/
├── images/
│   ├── foreground_00/   # <category folders>
│   │   ├── image001.png
│   │   └── ...
│   └── ...
└── masks/
    ├── foreground_00/   # Same structure as images
    │   ├── image001.png  # Binary mask (0 or 255)
    │   └── ...
    └── ...
```

## Checkpoints

Checkpoints are saved to `checkpoints/` directory:
- `last_model.pth`: Latest checkpoint (includes optimizer state, epoch, IoU)
- `best_model.pth`: Best model by validation Boundary IoU
- `phase{1,2,3}_ep{epoch}.pt`: Per-phase checkpoints

## Features

- **IMG_SIZE**: 1024 (optimal for RMBG-2.0)
- **Loss**: Masked redrawing loss (`Mask1 + Mask2 + 100xBoundaryMask`) + hole penalty
- **Optimizer**: AdamW with trainable params only
- **Freezing**: Progressive unfreezing (`patch_embed + norms` -> `stage 0/1` -> deeper blocks)
- **Preprocessing**: Trapped-ball style gap closing (`R=3`) to seal broken line boundaries
- **Structural guidance**: Skeleton-map guided input enhancement for edge-focused learning
- **Augmentation**: Random erode/dilate, brightness/contrast + progressive patch shuffle (2x2 -> 32x32)
- **Metric**: Boundary IoU (5px edge) for best model selection
- **Early stopping**: Patience-based stop to reduce overfitting (default patience=4)
- **Post-processing**: Optional hole filling for cleaner binary masks in visualization/inference helpers
- **WandB**: Integrated logging

## Configuration

Edit `config/__init__.py`:
- `IMG_SIZE`: Input image size (default: 256)
- `BATCH_SIZE`: Batch size (default: 8)
- `NUM_EPOCHS`: Training epochs (default: 30)
- `FREEZE_ENCODER`: Keep encoder trainable by default (default: False)
- `CKPT_DIR`: Checkpoint save path (default: `checkpoints/`)

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (tested on 8GB+ VRAM)
- Conda (recommended)

```bash
conda create -n fine_tune python=3.11
conda activate fine_tune
pip install -r requirements.txt
```

## HuggingFace & WandB API Keys

Model `briaai/RMBG-2.0` requires HuggingFace login. Export before running notebook:

```bash
export HF_TOKEN="hf_your_token_here"
export WANDB_API_KEY="your_wandb_key_here"  # optional
```

Get token at https://huggingface.co/settings/tokens

## Training

Open the notebook with the correct kernel:

```bash
conda activate fine_tune
jupyter notebook notebooks/train.ipynb
```

Or register the kernel (one-time):
```bash
conda activate fine_tune
python -m ipykernel install --user --name fine_tune --display-name "fine_tune"
```

Then run all cells sequentially. Training has 3 phases:
- **Phase 1**: Decoder + Squeeze (2-3 epochs)
- **Phase 2**: + Stage 0-1 + Alternating Stage 2 (5-8 epochs)
- **Phase 3**: Full Fine-tune (5-10 epochs)

## Test Trained Model

```bash
python test_model.py --checkpoint checkpoints/best_model.pth --data_dir dataset_lie_art/images --mask_dir dataset_lie_art/masks --output_dir test_output

# Visualize samples
python test_model.py --checkpoint checkpoints/best_model.pth --data_dir dataset_lie_art/images --visualize
```

## Test Output Metrics

The test script will compute per-category and overall:
- **IoU**: Intersection over Union
- **Dice**: Dice coefficient
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **F1**: Harmonic mean of precision and recall

## Training Notes

- Phase 3 full fine-tuning uses lower learning rates to stabilize training:
  - Decoder LR: `5e-5`
  - Encoder LR: `1e-6`