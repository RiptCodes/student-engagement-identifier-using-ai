import os
import json
import cv2
import signal
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

from config import *


# ============================================================
# UTILS
# ============================================================
def timeout_handler(signum, frame):
    raise TimeoutError()


def find_video(video_dir, clip_id):
    clean_id = str(clip_id).replace('.avi', '').replace('.mp4', '')
    per_id   = clean_id[:6]
    paths    = [
        f"{video_dir}/{per_id}/{clean_id}/{clean_id}.avi",
        f"{video_dir}/{per_id}/{clean_id}/{clean_id}.mp4",
        f"{video_dir}/{clean_id}.avi",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def show_preview(frames, frame_indices, label_text, clip_id):
    n         = min(6, len(frames))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes      = axes.flatten()
    for i in range(n):
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {frame_indices[i]}\n{label_text}", fontsize=9)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.suptitle(f"Preview: {clip_id}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"preview_{label_text}.png")
    plt.close()


# ============================================================
# FACE PREPROCESSOR
# ============================================================
class FacePreprocessor:
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size
        self.detector    = YOLO('yolov8n-face-lindevs.pt')
        self.detector.to('cuda')

    def extract_face(self, result, frame):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best  = int(areas.argmax())

        x1, y1, x2, y2 = map(int, boxes.xyxy[best].tolist())
        h, w = frame.shape[:2]

        pad = int(0.3 * (x2 - x1))
        x1  = max(0, x1 - pad);  y1 = max(0, y1 - pad)
        x2  = min(w, x2 + pad);  y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(crop, self.target_size)

    def process_frame(self, frame):
        results = self.detector(frame, verbose=False, imgsz=640)[0]
        return self.extract_face(results, frame)

    def draw_boxes(self, frame):
        results = self.detector(frame, verbose=False, imgsz=640)[0]
        boxes   = results.boxes
        out     = frame.copy()
        n       = 0

        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                    (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best  = int(areas.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best].tolist())
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n = 1

        return out, n


# ============================================================
# PROCESS ONE VIDEO - returns frames array or None
# ============================================================
BATCH_DETECT = 8

def process_video(vpath, preprocessor):
    signal.signal(signal.SIGALRM, timeout_handler)

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None

    raw_frames = []
    try:
        signal.alarm(30)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % FRAME_STEP == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                raw_frames.append((idx, frame))
            idx += 1
        signal.alarm(0)
    except TimeoutError:
        cap.release()
        return None
    cap.release()

    if len(raw_frames) == 0:
        return None

    frames, indices = [], []
    try:
        signal.alarm(60)
        for b in range(0, len(raw_frames), BATCH_DETECT):
            batch   = [f for _, f in raw_frames[b:b+BATCH_DETECT]]
            results = preprocessor.detector(batch, verbose=False, imgsz=640)
            for j, res in enumerate(results):
                face = preprocessor.extract_face(res, raw_frames[b+j][1])
                if face is not None:
                    frames.append(face)
                    indices.append(raw_frames[b+j][0])
        signal.alarm(0)
    except TimeoutError:
        return None

    if len(frames) < MIN_FRAMES:
        return None

    return np.array(frames, dtype=np.uint8), indices


# ============================================================
# DATASET PROCESSING - saves per-video files, no RAM buildup
# ============================================================
def process_split(video_dir, labels_df, out_dir, split_name, preprocessor,
                  max_per_class=None, preview=True):

    # per-video output dir
    vid_dir       = os.path.join(out_dir, f"{split_name}_videos")
    frames_path   = os.path.join(out_dir, f"{split_name}_frames.npy")
    labels_path   = os.path.join(out_dir, f"{split_name}_labels.npy")
    meta_path     = os.path.join(out_dir, f"{split_name}_meta.json")
    progress_path = os.path.join(out_dir, f"{split_name}_progress.json")

    os.makedirs(vid_dir, exist_ok=True)

    if max_per_class is not None:
        parts = []
        for cls in labels_df['Label'].unique():
            parts.append(labels_df[labels_df['Label'] == cls].head(max_per_class))
        labels_df = pd.concat(parts).sample(frac=1).reset_index(drop=True)

    # load progress
    processed_ids = set()
    failed        = []
    shown         = set()
    start_idx     = 0

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        processed_ids = set(progress['processed_ids'])
        failed        = progress['failed']
        shown         = set(progress['shown'])
        start_idx     = progress['start_idx']
        print(f"\nResuming {split_name} — {len(processed_ids)} videos already done...")
    else:
        print(f"\nProcessing {split_name}... ({len(labels_df)} videos)")

    for i, (_, row) in enumerate(tqdm(labels_df.iterrows(), total=len(labels_df),
                                       initial=start_idx)):
        if i < start_idx:
            continue

        clip_id    = str(row['ClipID'])
        label      = int(row['Label'])
        label_text = LABELS[label]
        vid_path   = os.path.join(vid_dir, f"{clip_id}.npy")

        # already saved to disk - skip
        if clip_id in processed_ids and os.path.exists(vid_path):
            continue

        vpath = find_video(video_dir, clip_id)
        if vpath is None:
            failed.append(clip_id)
        else:
            result = process_video(vpath, preprocessor)
            if result is None:
                failed.append(clip_id)
            else:
                frames, indices = result

                if preview and label not in shown:
                    print(f"\nNew class: {label_text}")
                    show_preview(frames, indices, label_text, clip_id)
                    shown.add(label)

                # save immediately to disk - no RAM buildup
                np.save(vid_path, frames)
                processed_ids.add(clip_id)

        # checkpoint every 50 videos
        if (i + 1) % 50 == 0:
            with open(progress_path, 'w') as f:
                json.dump({
                    'start_idx':     i + 1,
                    'processed_ids': list(processed_ids),
                    'failed':        failed,
                    'shown':         list(shown)
                }, f)
            print(f"  Checkpoint at {i+1}/{len(labels_df)}")

    print(f"\nAll videos processed. Assembling final arrays...")

    # assemble final npy from per-video files
    all_frames, all_labels, all_ids = [], [], []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        clip_id  = str(row['ClipID'])
        label    = int(row['Label'])
        vid_path = os.path.join(vid_dir, f"{clip_id}.npy")

        if os.path.exists(vid_path):
            frames = np.load(vid_path)
            all_frames.append(frames)
            all_labels.append(label)
            all_ids.append(clip_id)

    if len(all_frames) == 0:
        print("No videos processed.")
        return None, None

    max_len = max(len(f) for f in all_frames)
    h, w    = all_frames[0][0].shape[:2]

    padded = []
    for frames in all_frames:
        if len(frames) < max_len:
            p      = np.zeros((max_len-len(frames), h, w, 3), dtype=np.uint8)
            frames = np.vstack([frames, p])
        padded.append(frames)

    frames_np = np.array(padded, dtype=np.uint8)
    labels_np = np.array(all_labels, dtype=np.int64)

    np.save(frames_path, frames_np)
    np.save(labels_path, labels_np)

    meta = {"split": split_name, "n_videos": len(all_ids), "n_failed": len(failed)}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"{split_name.upper()} COMPLETE - processed: {len(all_ids)}, failed: {len(failed)}")
    return frames_np, labels_np


def load_or_process(video_dir, labels_df, out_dir, split_name, preprocessor,
                    max_per_class=None, preview=True, force=False):
    frames_path = os.path.join(out_dir, f"{split_name}_frames.npy")
    labels_path = os.path.join(out_dir, f"{split_name}_labels.npy")
    meta_path   = os.path.join(out_dir, f"{split_name}_meta.json")

    if not force and os.path.exists(frames_path) and os.path.exists(labels_path):
        print(f"\nLoading {split_name} from cache...")
        frames_np = np.load(frames_path)
        labels_np = np.load(labels_path)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Videos: {meta['n_videos']}, Failed: {meta['n_failed']}")
        return frames_np, labels_np

    return process_split(video_dir, labels_df, out_dir, split_name,
                         preprocessor, max_per_class=max_per_class, preview=preview)


def load_labels():
    train_df = pd.read_csv(f'{LABELS_PATH}/TrainLabels.csv')
    val_df   = pd.read_csv(f'{LABELS_PATH}/ValidationLabels.csv')
    test_df  = pd.read_csv(f'{LABELS_PATH}/TestLabels.csv')

    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.strip()

    train_df['Label'] = train_df['Engagement'].apply(lambda x: 0 if int(x) <= 1 else 1)
    val_df['Label']   = val_df['Engagement'].apply(lambda x: 0 if int(x) <= 1 else 1)
    test_df['Label']  = test_df['Engagement'].apply(lambda x: 0 if int(x) <= 1 else 1)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def show_preview(frames, frame_indices, label_text, clip_id):
    n         = min(6, len(frames))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes      = axes.flatten()
    for i in range(n):
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {frame_indices[i]}\n{label_text}", fontsize=9)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.suptitle(f"Preview: {clip_id}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/previews/preview_{label_text}.png")
    plt.close()


def plot_distribution(train_df):
    plt.figure(figsize=(6, 4))
    counts     = train_df['Label'].value_counts().sort_index()
    bar_labels = [LABELS[i] for i in counts.index]
    bars       = plt.bar(bar_labels, counts.values)
    plt.title("Training Set Engagement Distribution (Figure 1)")
    plt.xlabel("Engagement Level")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=30)
    max_h = max(counts.values)
    for bar in bars:
        h = bar.get_height()
        if h < max_h:
            plt.text(bar.get_x() + bar.get_width()/2, h + max_h*0.01,
                     f'{int(h)}', ha='center', va='bottom', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width()/2, h/2,
                     f'{int(h)}', ha='center', va='center', color='white', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/plots/engagement_distribution.png")
    plt.close()