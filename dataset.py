import numpy as np
import tensorflow as tf

from config import *



class DataGenerator:
    def __init__(self, frames_file, labels_file, batch_size=BATCH, step=GEN_STEP):
        self.frames     = np.load(frames_file, mmap_mode='r')
        self.labels     = np.load(labels_file)
        self.batch_size = batch_size

        self.samples = []
        for v in range(len(self.labels)):
            vid   = self.frames[v]
            mask  = vid.reshape(vid.shape[0], -1).any(axis=1)
            valid = np.where(mask)[0][::step]
            for fi in valid:
                self.samples.append((v, fi))

        self.n_samples = len(self.samples)
        self.n_batches = max(1, self.n_samples // batch_size)
        print(f"  {self.n_samples} samples, {self.n_batches} batches")

    def get_batch(self, b):
        start = b * self.batch_size
        end   = min(start + self.batch_size, self.n_samples)
        batch = self.samples[start:end]

        imgs, lbls = [], []
        for v, fi in batch:
            imgs.append(self.frames[v][fi].astype(np.float32))
            lbls.append(int(self.labels[v]))

        X = tf.keras.applications.resnet_v2.preprocess_input(
                np.array(imgs, dtype=np.float32))
        y = np.eye(N_CLASSES, dtype=np.float32)[lbls]
        return X, y

    def shuffle(self):
        np.random.shuffle(self.samples)
