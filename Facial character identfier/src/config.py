import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "archive")
IMG_DIR = os.path.join(DATA_DIR, "img_align_celeba", "img_align_celeba") + os.sep
MERGED_CSV = os.path.join(DATA_DIR, "celeba_merged.csv")
LANDMARKS_CSV = os.path.join(DATA_DIR, "list_landmarks_align_celeba.csv")

IMG_SIZE = 128
BATCH_SIZE = 64

ORIG_IMG_WIDTH = 178.0
ORIG_IMG_HEIGHT = 218.0
BBOX_COLS = ["x_1", "y_1", "width", "height"]

# --- Stage 1: normal (frozen-backbone) training ---
TOTAL_EPOCHS = 15
EPOCHS_PER_SESSION = 2
CHECKPOINT_DIR = os.path.join(DATA_DIR, "training_checkpoints")   # in-progress checkpoints
FINAL_MODEL_DIR = os.path.join(DATA_DIR, "final_model")           # dumped here at epoch 15
FINAL_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "multitask_face_model.keras")