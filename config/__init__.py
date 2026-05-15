class Config:
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    LEARNING_RATE_ENCODER = 5e-6
    WEIGHT_DECAY = 0.01
    FREEZE_ENCODER = False

    MODEL_NAME = "briaai/RMBG-2.0"

    DATASET_PATH = "dataset_lie_art/images"
    MASK_PATH = "dataset_lie_art/masks"

    CKPT_DIR = "checkpoints"
    TRAIN_RATIO = 0.9

    USE_GRADIENT_CHECKPOINTING = False


config = Config()
