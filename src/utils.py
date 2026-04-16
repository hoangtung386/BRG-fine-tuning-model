import os
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mount_drive():
    from google.colab import drive

    drive.mount("/content/drive")


def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available")
