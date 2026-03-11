# prepare_data.py

import os
import matplotlib.pyplot as plt

from config import *
from preprocessing import (FacePreprocessor, load_labels, load_or_process,
                            find_video, plot_distribution, show_frames)


if __name__ == '__main__':
    print(f"TRAIN_PATH exists: {os.path.exists(TRAIN_PATH)}")
    print(f"VAL_PATH exists: {os.path.exists(VAL_PATH)}")
    print(f"TEST_PATH exists: {os.path.exists(TEST_PATH)}")

    train_df, val_df, test_df = load_labels()

    plot_distribution(train_df)

    preprocessor = FacePreprocessor(target_size=IMG_SIZE)

    # face detection demo
    clip_id = train_df.iloc[1]['ClipID']
    vpath   = find_video(TRAIN_PATH, clip_id)
    frames  = show_frames(vpath, max_frames=20)

    plt.figure(figsize=(12, 5))
    for i, frame in enumerate(frames[:10]):
        boxed, n_faces = preprocessor.draw_boxes(frame)
        plt.subplot(2, 5, i+1)
        plt.imshow(boxed)
        plt.title(f"{n_faces} face{'s' if n_faces != 1 else ''}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("face_detection_demo.png")

    FORCE_REDO = False

    load_or_process(TRAIN_PATH, train_df, SAVE_DIR, "train",
                    preprocessor, max_per_class=None, preview=True, force=FORCE_REDO)

    load_or_process(VAL_PATH, val_df, SAVE_DIR, "val",
                    preprocessor, max_per_class=None, preview=False, force=FORCE_REDO)

    load_or_process(TEST_PATH, test_df, SAVE_DIR, "test",
                    preprocessor, max_per_class=None, preview=False, force=FORCE_REDO)

    print(f"\nDone. Saved to: {SAVE_DIR}")