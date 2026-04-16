class Config:
    IMG_SIZE = 2048
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    MODEL_NAME = "briaai/RMBG-2.0"

    DATASET_PATH = "/content/dataset/images"
    MASK_PATH = "/content/dataset/masks"

    CKPT_DIR = "/content/drive/MyDrive/rmbg_checkpoints"
    TRAIN_RATIO = 0.9


config = Config()
