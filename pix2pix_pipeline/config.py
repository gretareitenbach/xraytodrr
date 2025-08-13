import torch

# --- Weights & Biases (wandb) Configuration ---
WANDB_PROJECT = "XRay_to_DRR_Pix2Pix"
WANDB_ENTITY = None

# --- Cross-Validation Configuration ---
NUM_FOLDS = 4
DATA_DIR = None

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = None
LAMBDA_L1 = 100

# --- Model and Data Parameters ---
IMAGE_SIZE = 256
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# --- Dataset and Checkpoint Paths ---
CHECKPOINT_DIR = "checkpoints"
SAVED_IMAGES_DIR = "saved_images"

# --- Optimizer Parameters ---
BETA1 = 0.5
BETA2 = 0.999

# --- Training Configuration Flags ---
LOAD_MODEL = False
SAVE_MODEL = True