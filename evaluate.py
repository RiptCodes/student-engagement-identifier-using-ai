import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import *
from dataset import DataGenerator


# ============================================================
# EVALUATION
# ============================================================
def evaluate(test_gen):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    all_preds, all_true = [], []

    for b in range(test_gen.n_batches):
        X_t, y_t = test_gen.get_batch(b)
        preds     = model(tf.constant(X_t), training=False)
        all_preds.extend(np.argmax(preds.numpy(), axis=1))
        all_true.extend(np.argmax(y_t,            axis=1))

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    print(f"Test Accuracy: {np.mean(all_preds == all_true):.4f}")
    print(classification_report(all_true, all_preds, target_names=LABELS, zero_division=0))

    return model, all_preds, all_true


def plot_confusion(all_true, all_preds):
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix (Figure 3)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_confidence(model, test_gen, all_true):
    all_confs, all_correct = [], []

    for b in range(test_gen.n_batches):
        X_t, y_t = test_gen.get_batch(b)
        preds     = model(tf.constant(X_t), training=False).numpy()
        for j in range(len(preds)):
            all_confs.append(np.max(preds[j]))
            all_correct.append(np.argmax(preds[j]) == np.argmax(y_t[j]))

    all_confs   = np.array(all_confs)
    all_correct = np.array(all_correct)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(all_confs[all_correct],  bins=20, alpha=0.7, label='Correct',   color='green')
    ax1.hist(all_confs[~all_correct], bins=20, alpha=0.7, label='Incorrect', color='red')
    ax1.set_title('Confidence Distribution (Figure 4a)')
    ax1.set_xlabel('Confidence'); ax1.set_ylabel('Count')
    ax1.legend()

    for i, l in enumerate(LABELS):
        mask = all_true == i
        if mask.sum() > 0:
            ax2.bar(l, np.mean(all_confs[mask]), alpha=0.7)
    ax2.set_title('Mean Confidence per Class (Figure 4b)')
    ax2.set_xlabel('Engagement Level'); ax2.set_ylabel('Mean Confidence')

    plt.tight_layout()
    plt.show()

    print(f"Overall: {np.mean(all_confs):.4f} | Correct: {np.mean(all_confs[all_correct]):.4f} | Incorrect: {np.mean(all_confs[~all_correct]):.4f}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    test_gen = DataGenerator(f'{SAVE_DIR}/test_frames.npy', f'{SAVE_DIR}/test_labels.npy')

    model, all_preds, all_true = evaluate(test_gen)
    plot_confusion(all_true, all_preds)
    plot_confidence(model, test_gen, all_true)
