"""
Convert the trained Keras engagement model to TensorFlow.js so it can run
in the browser on the GitHub Pages demo (docs/index.html).

Usage:
    python convert_model.py                          # uses MODEL_PATH from config.py
    python convert_model.py path/to/model.keras      # explicit path

Output: docs/model/model.json + group*.bin  (loaded by docs/js/app.js)
"""
import os
import sys
import tempfile

import tensorflow as tf

# default to the path recorded in config.py, but allow an override argument
try:
    from config import MODEL_PATH as CFG_MODEL_PATH
except Exception:
    CFG_MODEL_PATH = None

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "model")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else CFG_MODEL_PATH
    if not model_path:
        sys.exit("No model path given and config.MODEL_PATH is unset.")
    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}")

    print(f"Loading: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = os.path.join(tmpdir, "saved_model")
        print(f"Exporting SavedModel to temp dir...")
        # Keras 3 export: produces an inference-only SavedModel
        model.export(saved_model_path)

        print(f"Converting SavedModel -> TensorFlow.js -> {OUT_DIR}")
        from tensorflowjs.converters import tf_saved_model_conversion_v2
        tf_saved_model_conversion_v2.convert_tf_saved_model(
            saved_model_path,
            OUT_DIR,
            signature_def="serving_default",
            saved_model_tags="serve",
        )

    # confirm output
    files = os.listdir(OUT_DIR)
    print(f"\nDone! {len(files)} files written to docs/model/:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}  ({size/1024/1024:.1f} MB)" if size > 1024*1024 else f"  {f}  ({size/1024:.1f} KB)")
    print("\nCommit docs/model/ and push to GitHub — the demo page picks it up automatically.")


if __name__ == "__main__":
    main()
