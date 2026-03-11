import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from config import *
from dataset import DataGenerator
from model import build_model


train_loss_avg = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
val_accuracy   = tf.keras.metrics.CategoricalAccuracy()

model     = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def train_step(x, y, sample_weights):
    with tf.GradientTape() as tape:
        logits     = model(x, training=True)
        loss_value = tf.keras.losses.categorical_crossentropy(y, logits)
        loss_value = loss_value * sample_weights
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss_avg.update_state(loss_value)
    train_accuracy.update_state(y, logits)

@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    val_accuracy.update_state(y, logits)


def train(train_gen, val_gen):
    os.makedirs(PROJECT_PATH, exist_ok=True)

    # class weights
    raw_labels  = np.load(f'{SAVE_DIR}/train_labels.npy')
    weights     = compute_class_weight('balanced', classes=np.unique(raw_labels), y=raw_labels)
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"Class weights: {weight_dict}")

    history    = {'loss': [], 'acc': [], 'val_acc': []}
    best_acc   = 0.0
    no_improve = 0

    print("\nTraining started")
    model.summary()

    for epoch in range(EPOCHS):
        train_gen.shuffle()
        train_loss_avg.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

        for b in tqdm(range(train_gen.n_batches), desc=f"Epoch {epoch+1}/{EPOCHS}"):
            X, y      = train_gen.get_batch(b)
            # build sample weights from class weights
            y_indices = np.argmax(y, axis=1)
            s_weights = np.array([weight_dict[i] for i in y_indices], dtype=np.float32)
            train_step(tf.constant(X), tf.constant(y), tf.constant(s_weights))

        for b in range(val_gen.n_batches):
            X_v, y_v = val_gen.get_batch(b)
            val_step(tf.constant(X_v), tf.constant(y_v))

        loss    = float(train_loss_avg.result())
        acc     = float(train_accuracy.result())
        val_acc = float(val_accuracy.result())

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            model.save(MODEL_PATH)
            print(f"  Saved best model (Val Acc: {best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Val Acc: {best_acc:.4f}")
    return history


def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['loss'], label='Train Loss', color='#3498db', linewidth=2)
    ax1.set_title('Loss (Figure 2a)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history['acc'],     label='Train Acc', color='#2ecc71', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc',   color='#e74c3c', linewidth=2, linestyle='--')
    ax2.set_title('Accuracy (Figure 2b)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1]); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle('Training Curves')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_gen = DataGenerator(f'{SAVE_DIR}/train_frames.npy', f'{SAVE_DIR}/train_labels.npy')
    val_gen   = DataGenerator(f'{SAVE_DIR}/val_frames.npy',   f'{SAVE_DIR}/val_labels.npy')

    history = train(train_gen, val_gen)
    plot_training(history)