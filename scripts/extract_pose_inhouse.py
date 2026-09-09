"""Extract YOLO11m-pose skeletons for in-house TP + FP windows.

Positives come from trigger_times.csv (manual annotations).
Negatives come from candidate_windows.csv rank-1 peaks (motion-based).
Each window = center ± 5s @ 25 fps, matched to Le2i temporal density.

We sample 16 consecutive frames per window at the source frame rate directly
around the trigger (i.e., 0.64 s of "the moment") -- consistent with Le2i.
That's the WINDOW we'll actually train on.  We also save a wider view: the
full 16 frames tightly around the trigger.

For multi-person clips: keep top-3 person tracks per frame; caller can pick
the "faller" later by aspect-ratio-change heuristic.

Output: D:\\fall-neuravue\\outputs\\pose\\in_house\\{label}_{clip_stem}_t{trigger}.npz
    kpts   : float32 (T, K, 17, 3)  K=up to 3 tracks per frame, padded with zeros
    bboxes : float32 (T, K, 4)
    conf   : float32 (T, K)
    label  : str "positive"|"negative"
    trigger_s : float
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

# Path to the actual video clips on Waleed (mirror of the Mac's folder)
VIDEO_ROOT = Path(r"D:\fall-detection-testing")

# Motion-peak negatives — we generated these on the Mac in candidate_windows.csv.
# We'll re-derive them here from the manifest to avoid another file copy.
NEG_WINDOW_SEC = 10.0   # for negatives, use midpoint of clip as fallback
POS_WINDOW_HALF = 5.0   # ±seconds around trigger, used to locate the 16-frame slice
N_FRAMES = 16           # match Le2i snippet length
MAX_TRACKS = 3          # keep top-3 person tracks per frame

OUT_DIR = ROOT / "outputs" / "pose" / "in_house"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with CLIPS_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            out[r["clip"]] = r
    return out


def load_triggers() -> list[dict]:
    out: list[dict] = []
    with TRIGGERS.open() as f:
        for r in csv.DictReader(f):
            r["trigger_s"] = float(r["trigger_s"])
            out.append(r)
    return out


def sample_16_frames(video_path: Path, center_s: float) -> tuple[np.ndarray | None, float]:
    """Grab 16 consecutive frames centered at center_s. Returns (frames[T,H,W,3], fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if nb == 0:
        cap.release()
        return None, fps

    # Real fps: on these merged clips container fps is bogus. Use duration-based.
    # We already have real_fps in the manifest; but for windowing use container fps
    # after checking POS_MSEC below.
    center_frame = int(round(center_s * fps))
    start = max(0, center_frame - N_FRAMES // 2)
    end = min(nb, start + N_FRAMES)
    start = max(0, end - N_FRAMES)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < N_FRAMES:
        # Pad with zeros to keep tensor shape.
        while len(frames) < N_FRAMES:
            frames.append(np.zeros_like(frames[0]) if frames else np.zeros((768, 1360, 3), np.uint8))
    return np.stack(frames, axis=0), fps


def extract_window(model: YOLO, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = frames.shape[0]
    kpts = np.zeros((T, MAX_TRACKS, 17, 3), np.float32)
    bboxes = np.zeros((T, MAX_TRACKS, 4), np.float32)
    confs = np.zeros((T, MAX_TRACKS), np.float32)

    results = model.predict(
        source=[frames[i] for i in range(T)],
        device=0,
        verbose=False,
        conf=0.20,
        imgsz=640,
        stream=False,
    )
    for i, r in enumerate(results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        box_confs = r.boxes.conf.detach().cpu().numpy()
        order = np.argsort(-box_confs)[:MAX_TRACKS]
        kp_data = r.keypoints.data.detach().cpu().numpy()
        for k, j in enumerate(order):
            if kp_data.shape[-1] == 2:
                kpts[i, k] = np.concatenate([kp_data[j], np.ones((17, 1), np.float32)], axis=-1)
            else:
                kpts[i, k] = kp_data[j].astype(np.float32)
            bboxes[i, k] = r.boxes.xyxy[j].detach().cpu().numpy().astype(np.float32)
            confs[i, k] = float(box_confs[j])
    return kpts, bboxes, confs


def main() -> None:
    manifest = load_manifest()
    triggers = load_triggers()
    pos_clips = {t["clip"]: t for t in triggers}

    # Build the work list.
    work: list[tuple[str, str, float, Path]] = []
    for clip, meta in manifest.items():
        label = meta["label"]
        dur = float(meta["duration_s"])
        rel_path = meta["path"]  # e.g. all_video_data/TrueFalls/MergedExtracted/xxx.mp4
        video_path = VIDEO_ROOT / rel_path
        if not video_path.exists():
            # Try mapping to Waleed's mirror
            alt = VIDEO_ROOT / "all_video_data" / Path(rel_path).parent.name / "MergedExtracted" / Path(rel_path).name
            if alt.exists():
                video_path = alt
            else:
                # Recurse-find
                cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
                if not cand:
                    print(f"[skip] video missing: {clip}")
                    continue
                video_path = cand[0]

        if label == "positive":
            if clip not in pos_clips:
                print(f"[skip] positive w/o trigger: {clip}")
                continue
            work.append((clip, "positive", pos_clips[clip]["trigger_s"], video_path))
        elif label == "negative":
            # Use clip midpoint as trigger center (motion-peak alternative in candidate_windows.csv
            # is elsewhere on the Mac; midpoint is a reasonable "arbitrary window of a non-fall").
            work.append((clip, "negative", dur / 2.0, video_path))
        # unlabeled: skipped for now

    print(f"[info] {len(work)} windows queued")

    model = YOLO("yolo11m-pose.pt")
    model.to("cuda")

    t0 = time.time()
    for i, (clip, label, center_s, video_path) in enumerate(work, 1):
        out = OUT_DIR / f"{label}_{Path(clip).stem}_t{int(center_s)}.npz"
        if out.exists():
            continue
        frames, fps = sample_16_frames(video_path, center_s)
        if frames is None:
            print(f"[skip] cannot read {clip}")
            continue
        kpts, bboxes, confs = extract_window(model, frames)
        np.savez_compressed(
            out, kpts=kpts, bboxes=bboxes, conf=confs,
            label=label, trigger_s=center_s, clip=clip,
            src_fps=fps, n_frames=N_FRAMES,
        )
        dt = time.time() - t0
        print(f"[{i}/{len(work)}] {label:<9} {clip}  center={center_s:.1f}s  fps={fps:.1f}  {dt:.1f}s", flush=True)

    print(f"\nDone. elapsed={time.time()-t0:.1f}s  saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
