# model/

The converted TensorFlow.js engagement model goes here:

```
model/
  model.json
  group1-shard1of*.bin
  ...
```

It is **not** committed yet. Generate it with `python convert_model.py` from the
repo root (see `../CONVERT_MODEL.md`), then commit this folder and push.

Until the files exist, the live demo runs face detection only and shows a
"model not loaded" notice.
