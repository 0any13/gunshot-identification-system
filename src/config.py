SAMPLE_RATE = 22050
CLIP_DURATION = 4
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MAX = 8000

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

RAW_DATA_DIR = "data/raw"
PROCESSED_STAGE1_DIR = "data/processed/stage1"
PROCESSED_STAGE2_DIR = "data/processed/stage2"
CHECKPOINT_DIR = "checkpoints"

# stage 1: coarse classification
STAGE1_CLASSES = [
    "gunshot",
    "siren",
    "background",
]
NUM_STAGE1_CLASSES = len(STAGE1_CLASSES)

# stage 2: fine siren classification
# only runs when stage 1 predicts "siren"
STAGE2_CLASSES = [
    "police",
    "ambulance",
    "firetruck",
    "unknown",
]
NUM_STAGE2_CLASSES = len(STAGE2_CLASSES)

# if stage 2 max softmax probability is below this, return "unknown"
STAGE2_CONFIDENCE_THRESHOLD = 0.60