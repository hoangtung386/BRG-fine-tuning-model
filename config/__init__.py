class Config:
    IMG_SIZE = 1024
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    LEARNING_RATE_ENCODER = 5e-6
    WEIGHT_DECAY = 0.01

    MODEL_NAME = "briaai/RMBG-2.0"

    DATASET_PATH = "/content/drive/MyDrive/Projects/dataset_line-art"
    MASK_PATH = "/content/drive/MyDrive/Projects/ground_truth_anime"

    CKPT_DIR = "/content/drive/MyDrive/rmbg_checkpoints"
    TRAIN_RATIO = 0.9

    USE_GRADIENT_CHECKPOINTING = False


config = Config()
