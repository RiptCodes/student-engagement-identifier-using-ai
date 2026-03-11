# ============================================================
# CONFIG - all paths and hyperparameters in one place
# ============================================================

# paths - change these to match your setup
DATASET_PATH = r'/mnt/c/Users/raysh/OneDrive - University of Reading (1)/DAiSEE-PROJ/DAiSEE'
PROJECT_PATH = r'/mnt/c/Users/raysh/OneDrive - University of Reading (1)/DAiSEE-PROJ/projects'

TRAIN_PATH  = f'{DATASET_PATH}/DataSet/Train'
VAL_PATH    = f'{DATASET_PATH}/DataSet/Validation'
TEST_PATH   = f'{DATASET_PATH}/DataSet/Test'
LABELS_PATH = f'{DATASET_PATH}/Labels'

SAVE_DIR    = f'{PROJECT_PATH}/processed_data'
MODEL_PATH  = f'{PROJECT_PATH}/best_model.keras'

# model params
IMG_SIZE   = (224, 224)
BATCH      = 64
N_CLASSES = 2
LABELS    = ['Not Engaged', 'Engaged']
EPOCHS     = 20
LR         = 1e-4
PATIENCE   = 4

# data params
FRAME_STEP = 8
MIN_FRAMES = 5
GEN_STEP   = 3
