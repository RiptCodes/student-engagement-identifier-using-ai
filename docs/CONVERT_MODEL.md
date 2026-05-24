# Adding the trained model to the web demo

The web demo (`docs/index.html`) runs the engagement model **in the browser**
using TensorFlow.js. To make scoring work you need to convert your trained
Keras model (`.keras`) into the TF.js format and drop it into `docs/model/`.

This is a one-time step. Face detection works without it; only the engagement
score needs the model.

## 1. Install the converter

```bash
pip install tensorflow "tensorflowjs>=4.0.0"
```

(Use the same Python environment you trained in, or any env with a matching
TensorFlow version, so the `.keras` file loads cleanly.)

## 2. Run the conversion

From the repository root:

```bash
# Uses MODEL_PATH from config.py
python convert_model.py

# ...or pass the file explicitly:
python convert_model.py "C:\path\to\model_20260314_1941.keras"
```

This writes `docs/model/model.json` plus one or more `group*.bin` weight files.

> ResNet50V2 is ~25M parameters, so the converted weights are roughly 90–100 MB.
> That is within GitHub's limits but will take a moment to download on first
> visit. The browser caches it afterwards.

## 3. Commit and push

```bash
git add docs/model
git commit -m "Add converted TF.js engagement model"
git push
```

GitHub Pages redeploys automatically. Reload the demo page — the banner about a
missing model disappears and live scoring starts working.

## Troubleshooting

- **`model.json` 404 in the console** — the `docs/model/` folder wasn't pushed,
  or Pages hasn't finished redeploying (wait ~1 min).
- **Weights fail to load** — make sure every `group*.bin` next to `model.json`
  was committed (none ignored by `.gitignore`).
- **Loads but scores look wrong** — confirm the model outputs a 2-class softmax
  in the order `["Not Engaged", "Engaged"]` (matches `config.py`).
