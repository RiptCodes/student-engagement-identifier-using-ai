import os
import matplotlib.pyplot as plt

from config import *
from preprocessing import (FacePreprocessor, load_labels,
                            plot_distribution, load_or_process,
                            find_video)

# ============================================================
# OUTPUT DIRS
# ============================================================
os.makedirs(SAVE_DIR,          exist_ok=True)
os.makedirs('outputs/previews', exist_ok=True)
os.makedirs('outputs/plots',    exist_ok=True)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print(f"TRAIN_PATH exists: {os.path.exists(TRAIN_PATH)}")
    print(f"VAL_PATH exists:   {os.path.exists(VAL_PATH)}")
    print(f"TEST_PATH exists:  {os.path.exists(TEST_PATH)}")

    train_df, val_df, test_df = load_labels()

    plot_distribution(train_df)

    preprocessor = FacePreprocessor(target_size=IMG_SIZE)

    FORCE_REDO = False

    load_or_process(TRAIN_PATH, train_df, SAVE_DIR, "train",
                    preprocessor, max_per_class=None, preview=True, force=FORCE_REDO)

    load_or_process(VAL_PATH, val_df, SAVE_DIR, "val",
                    preprocessor, max_per_class=None, preview=False, force=FORCE_REDO)

    load_or_process(TEST_PATH, test_df, SAVE_DIR, "test",
                    preprocessor, max_per_class=None, preview=False, force=FORCE_REDO)

    print("\nData preparation complete.")
    print(f"Saved to: {SAVE_DIR}")