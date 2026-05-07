SAMPLE_RATE = 22050
CLIP_DURATION = 4         # seconds - all clips padded/truncated to this
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MAX = 8000

# training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# test split is the remainder: 0.15

# paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"

# class names - adjust if you merge/split siren types
CLASSES = [
    "gunshot",
    "police_siren",
    "ambulance_siren",
    "firetruck_siren",
    "background",
]
NUM_CLASSES = len(CLASSES)
