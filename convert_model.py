"""
Convert the trained Keras engagement model to TensorFlow.js so it can run
in the browser on the GitHub Pages demo (docs/index.html).

Usage:
    pip install tensorflow "tensorflowjs>=4.0.0"
    python convert_model.py                          # uses MODEL_PATH from config.py
    python convert_model.py path/to/model.keras      # explicit path

Output: docs/model/model.json + group*.bin  (loaded by docs/js/app.js)
"""
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs

# default to the path recorded in config.py, but allow an override argument
try:
    from config import MODEL_PATH as CFG_MODEL_PATH
except Exception:
    CFG_MODEL_PATH = None

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "model")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else CFG_MODEL_PATH
    if not model_path:
        sys.exit("No model path given and config.MODEL_PATH is unset.")
    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}")

    print(f"Loading: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Converting to TensorFlow.js -> {OUT_DIR}")
    tfjs.converters.save_keras_model(model, OUT_DIR)

    print("\nDone. Commit the docs/model/ folder and push to GitHub.")
    print("The demo page will pick it up automatically.")


if __name__ == "__main__":
    main()
