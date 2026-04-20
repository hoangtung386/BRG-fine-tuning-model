# Fine-tune RMBG-2.0 for Line Art Removal

Fine-tuning script for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model on line art background removal task. Optimized for Colab Pro GPU (100GB VRAM).

## Project Structure

```
BRG-fine-tuning-model/
├── config/              # Configuration
│   └── __init__.py
├── src/                 # Source code
│   ├── __init__.py
│   ├── dataset.py       # Dataset with subfolder support
│   ├── model.py        # Model loading & freezing
│   ├── losses.py       # Loss functions (SSIM + BCE + IoU + Boundary IoU)
│   ├── trainer.py     # Training loop with wandb
│   ├── utils.py      # Utilities
│   └── visualization.py  # Visualization
├── data/                # Dataset (training)
├── test_data/           # Test data for evaluation
├── test_output/        # Predicted masks output
├── notebooks/          # Colab notebooks
│   └── train.ipynb  # Colab training notebook
├── best_model.pth     # Best trained model (download from Drive)
├── test_model.py       # Test script
├── requirements.txt
├── .gitignore
└── README.md
```

## Prepare Test Data

Place test images in folder structure:

```
test_data/
├── <category1>/
│   ├── image001.jpg
│   ├── image002.jpg
│   ...
├── <category2>/
│   └── ...
```

### Optional Ground Truth Masks

If you have ground truth masks for metrics computation:

```
test_output/
├── <category1>/         # Same structure as test_data
│   ├── image001.png    # Binary mask (0 or 255)
│   ├── image002.png
│   └── ...
```

## Download Best Model

After training, download `best_model.pth` from Google Drive to project root:

```
/content/drive/MyDrive/rmbg_checkpoints/best_model.pth → ./best_model.pth
```

## Features

- **IMG_SIZE**: 1024 (optimal for RMBG-2.0)
- **Loss**: 10×SSIM + 90×BCE + 0.25×IoU (BiRefNet formula)
- **Optimizer**: AdamW with trainable params only
- **Freezing**: Freeze encoder for faster training
- **Augmentation**: Random erode/dilate, brightness/contrast
- **Metric**: Boundary IoU (5px edge) for best model selection
- **WandB**: Integrated logging

## Configuration

Edit `config/__init__.py`:
- `IMG_SIZE`: Input image size (default: 1024)
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Training epochs (default: 30)
- `FREEZE_ENCODER`: Freeze encoder (default: True)
- `CKPT_DIR`: Checkpoint save path

## Test Trained Model

After training, download checkpoint and test:

```bash
# Test all images in test_data/ folder
python test_model.py --checkpoint best_model.pth --data_dir test_data --output_dir test_output

# Test with ground truth masks for metrics
python test_model.py --checkpoint best_model.pth --data_dir test_data --mask_dir test_output --output_dir test_output

# Visualize samples
python test_model.py --checkpoint best_model.pth --data_dir test_data --visualize
```

### Test Output Metrics

The test script will compute:
- **IoU**: Intersection over Union
- **Dice**: Dice coefficient
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **F1**: Harmonic mean of precision and recall

And breakdown by category/subfolder.

## Colab Usage

1. Clone project from GitHub to Colab
2. Add HuggingFace & W&B tokens in Colab secrets
3. Select GPU runtime (G4 recommended)
4. Run cells sequentially
5. Download checkpoints from Drive

## Checkpoints

- `last_model.pth`: Latest checkpoint (includes optimizer state, epoch, IoU)
- `best_model.pth`: Best model by validation Boundary IoU