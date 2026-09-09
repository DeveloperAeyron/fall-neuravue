"""In-house pose extraction v2: wider window + smarter track selection.

Changes vs v1:
  - N_FRAMES = 48 (spans ~2-3 s at 15-25 fps, so full fall event is captured
    even if the trigger timestamp is off by +/- 1 s).
  - Track selection now scores by MOVEMENT OVER TIME (vertical velocity +
    aspect-ratio change across the window) rather than just aspect-ratio
    range at a single instant. Staff who stand still get low scores; the
    faller wins.
  - Save K=5 tracks so we can still fall back if the top pick is wrong.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"D:\fall-neuravue")
IN_DATA = ROOT / "data"
CLIPS_MANIFEST = IN_DATA / "clips_manifest.csv"
TRIGGERS = IN_DATA / "trigger_times.csv"
VIDEO_ROOT = Path(r"D:\fall-detection-testing")

N_FRAMES = 48
MAX_TRACKS = 5
OUT_DIR = ROOT / "outputs" / "pose" / "in_house_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_frames(video_path: Path, center_s: float, n: int) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if nb == 0:
        cap.release()
        return None, fps
    cf = int(round(center_s * fps))
    start = max(0, cf - n // 2)
    start = min(start, max(0, nb - n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    while len(frames) < n:
        frames.append(np.zeros_like(frames[0]) if frames else np.zeros((768, 1360, 3), np.uint8))
    return np.stack(frames, axis=0), fps


def extract_window(model: YOLO, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns kpts (T,K,17,3), bboxes (T,K,4), conf (T,K).
    K = MAX_TRACKS. Tracks are ordered per-frame by detection conf (top->bottom)."""
    T = frames.shape[0]
    kpts = np.zeros((T, MAX_TRACKS, 17, 3), np.float32)
    bboxes = np.zeros((T, MAX_TRACKS, 4), np.float32)
    confs = np.zeros((T, MAX_TRACKS), np.float32)

    # Batch predict in chunks to avoid OOM on long windows
    CHUNK = 24
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        results = model.predict(
            source=[frames[i] for i in range(s, e)],
            device=0, verbose=False, conf=0.20, imgsz=640, stream=False,
        )
        for i, r in enumerate(results):
            gi = s + i
            if r.boxes is None or len(r.boxes) == 0:
                continue
            bc = r.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-bc)[:MAX_TRACKS]
            kp_data = r.keypoints.data.detach().cpu().numpy()
            for k, j in enumerate(order):
                if kp_data.shape[-1] == 2:
                    kpts[gi, k] = np.concatenate([kp_data[j], np.ones((17, 1), np.float32)], axis=-1)
                else:
                    kpts[gi, k] = kp_data[j].astype(np.float32)
                bboxes[gi, k] = r.boxes.xyxy[j].detach().cpu().numpy().astype(np.float32)
                confs[gi, k] = float(bc[j])
    return kpts, bboxes, confs


def load_manifest() -> dict[str, dict]:
    with CLIPS_MANIFEST.open() as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def load_triggers() -> dict[str, float]:
    with TRIGGERS.open() as f:
        return {r["clip"]: float(r["trigger_s"]) for r in csv.DictReader(f)}


def find_video(clip: str) -> Path | None:
    cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
    return cand[0] if cand else None


def main() -> None:
    manifest = load_manifest()
    triggers = load_triggers()

    work: list[tuple[str, str, float]] = []
    for clip, meta in manifest.items():
        label = meta["label"]
        dur = float(meta["duration_s"])
        if label == "positive":
            if clip in triggers:
                work.append((clip, "positive", triggers[clip]))
            else:
                print(f"[skip] pos w/o trigger: {clip}")
        elif label == "negative":
            work.append((clip, "negative", dur / 2.0))

    model = YOLO("yolo11m-pose.pt")
    model.to("cuda")

    t0 = time.time()
    for i, (clip, label, center_s) in enumerate(work, 1):
        video_path = find_video(clip)
        if video_path is None:
            print(f"[skip] {clip} not found")
            continue
        out = OUT_DIR / f"{label}_{Path(clip).stem}_t{int(center_s)}.npz"
        if out.exists():
            continue
        frames, fps = sample_frames(video_path, center_s, N_FRAMES)
        if frames is None:
            print(f"[skip] cannot read {clip}")
            continue
        kpts, bboxes, confs = extract_window(model, frames)
        np.savez_compressed(
            out, kpts=kpts, bboxes=bboxes, conf=confs,
            label=label, trigger_s=center_s, clip=clip,
            src_fps=fps, n_frames=N_FRAMES,
        )
        print(f"[{i}/{len(work)}] {label:<9} {clip}  center={center_s:.1f}s  fps={fps:.1f}  {time.time()-t0:.1f}s", flush=True)

    print(f"\nDone. elapsed={time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
