import os
from config import *
from preprocessing import FacePreprocessor, load_labels, plot_distribution, load_or_process

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('outputs/previews', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)

# another .py that runs preprocessing.py basically
if __name__ == '__main__':
    train_df, val_df, test_df = load_labels()
    plot_distribution(train_df)

    preprocessor = FacePreprocessor()

    # train + val: per-class frame sampling (Not Engaged step 2, Engaged step 4)
    # to boost the minority class. force=True regenerates the tfrecords so the
    # new sampling takes effect (delete *_progress.json too if resuming).
    load_or_process(TRAIN_PATH, train_df, SAVE_DIR, 'train', preprocessor,
                    preview=True, force=True, per_class_step=True)
    load_or_process(VAL_PATH, val_df, SAVE_DIR, 'val', preprocessor,
                    preview=False, force=True, per_class_step=True)
    # test: reuse the existing test.tfrecord if present (force=False) so the new
    # model is evaluated on exactly the same frames as the old one. If it has to
    # be (re)built, it uses a uniform frame step (per_class_step=False).
    load_or_process(TEST_PATH, test_df, SAVE_DIR, 'test', preprocessor,
                    preview=False, force=False, per_class_step=False)

    print(f"Done. Saved to {SAVE_DIR}")