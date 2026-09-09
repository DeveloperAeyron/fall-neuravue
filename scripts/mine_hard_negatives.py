"""Mine hard-negative windows from FP + HB clips.

For each clip:
  1. Run YOLO11m-pose at 5 fps across the ENTIRE clip.
  2. Track the top-conf person per frame (single-track for speed).
  3. Compute per-frame "fall-likeness":
        - aspect ratio (h/w) instantaneous
        - vertical velocity (centroid dy)
  4. Slide a 3-second window (~15 samples at 5 fps). Score each window by:
        w_score = max(dy_norm)  +  0.5 * (ar_range)  +  0.3 * (final_ar)
  5. Pick top-K non-overlapping peak windows per clip.
  6. For each peak, RE-EXTRACT a 48-frame window at native fps around that
     center using the same v2 in-house extractor logic (multi-track K=5).

Output: D:\\fall-neuravue\\outputs\\pose\\hard_negs\\{bucket}_{clip}_p{rank}_t{center}.npz
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"D:\fall-neuravue")
CLIPS_MANIFEST = ROOT / "data" / "clips_manifest.csv"
VIDEO_ROOT = Path(r"D:\fall-detection-testing")

OUT_DIR = ROOT / "outputs" / "pose" / "hard_negs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_FPS = 5.0
WINDOW_SEC = 3.0
TOP_K = 3
MAX_TRACKS = 5
N_FRAMES_OUT = 48

BUCKETS = {"FlasePositives": "FP", "NVR_manual_records": "HB"}


def scan_clip_for_peaks(model: YOLO, video_path: Path) -> list[float]:
    """Return list of TOP_K peak-center timestamps (seconds) in this clip."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / SCAN_FPS)))

    times, cys, ars, confs = [], [], [], []
    idx = 0
    frames_buf = []
    BATCH = 64
    idxs_buf = []

    def flush():
        if not frames_buf:
            return
        results = model.predict(source=frames_buf, device=0, verbose=False,
                                conf=0.20, imgsz=480, stream=False)
        for r, ii in zip(results, idxs_buf):
            t_s = ii / fps
            if r.boxes is None or len(r.boxes) == 0:
                times.append(t_s); cys.append(np.nan); ars.append(np.nan); confs.append(0.0)
                continue
            bc = r.boxes.conf.detach().cpu().numpy()
            j = int(np.argmax(bc))
            x1, y1, x2, y2 = r.boxes.xyxy[j].detach().cpu().numpy()
            w = float(x2 - x1); h = float(y2 - y1)
            times.append(t_s)
            cys.append(float((y1 + y2) / 2.0))
            ars.append(h / max(w, 1.0))
            confs.append(float(bc[j]))
        frames_buf.clear(); idxs_buf.clear()

    while True:
        # advance step-1 frames via grab
        skipped = True
        for _ in range(step - 1):
            skipped = cap.grab()
            if not skipped: break
        if not skipped: break
        ok, f = cap.read()
        if not ok: break
        frames_buf.append(f); idxs_buf.append(idx)
        idx += step
        if len(frames_buf) >= BATCH:
            flush()
    flush()
    cap.release()

    if len(times) < 5:
        return []

    t_arr = np.asarray(times); cy = np.asarray(cys); ar = np.asarray(ars); cf = np.asarray(confs)
    frame_h_guess = float(np.nanmax(cy) * 2) if np.any(~np.isnan(cy)) else 768.0
    # Per-sample velocity (dy per sample step)
    dy = np.abs(np.diff(cy)) / max(frame_h_guess, 1.0)
    dy = np.concatenate([[0.0], dy])
    # Sliding window
    W = max(3, int(round(WINDOW_SEC * SCAN_FPS)))
    scores = np.full(len(t_arr), -np.inf)
    for i in range(len(t_arr) - W + 1):
        seg_dy = dy[i:i+W]
        seg_ar = ar[i:i+W]
        seg_cf = cf[i:i+W]
        if np.nanmean(seg_cf) < 0.15:
            continue
        peak_dy = float(np.nanmax(seg_dy)) if np.any(~np.isnan(seg_dy)) else 0.0
        ar_valid = seg_ar[~np.isnan(seg_ar)]
        if len(ar_valid) < 3:
            continue
        ar_range = float(ar_valid.max() - ar_valid.min())
        final_ar = float(ar_valid[-1])
        scores[i + W // 2] = peak_dy + 0.5 * ar_range + 0.3 * final_ar

    # Top-K non-overlapping peaks (suppression window = 2*WINDOW_SEC in samples)
    picks: list[float] = []
    supp_samples = int(round(2 * WINDOW_SEC * SCAN_FPS))
    scores_work = scores.copy()
    for _ in range(TOP_K):
        i = int(np.argmax(scores_work))
        if not np.isfinite(scores_work[i]):
            break
        picks.append(float(t_arr[i]))
        lo = max(0, i - supp_samples); hi = min(len(scores_work), i + supp_samples)
        scores_work[lo:hi] = -np.inf
    return picks


def sample_frames(video_path: Path, center_s: float, n: int) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if nb == 0: cap.release(); return None, fps
    cf = int(round(center_s * fps))
    start = max(0, cf - n // 2); start = min(start, max(0, nb - n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    while len(frames) < n:
        frames.append(np.zeros_like(frames[0]) if frames else np.zeros((768, 1360, 3), np.uint8))
    return np.stack(frames, axis=0), fps


def extract_multitrack(model: YOLO, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = frames.shape[0]
    kpts = np.zeros((T, MAX_TRACKS, 17, 3), np.float32)
    bboxes = np.zeros((T, MAX_TRACKS, 4), np.float32)
    confs = np.zeros((T, MAX_TRACKS), np.float32)
    CHUNK = 24
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        results = model.predict(source=[frames[i] for i in range(s, e)],
                                device=0, verbose=False, conf=0.20, imgsz=640, stream=False)
        for i, r in enumerate(results):
            gi = s + i
            if r.boxes is None or len(r.boxes) == 0: continue
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


def main() -> None:
    with CLIPS_MANIFEST.open() as f:
        clips = [r for r in csv.DictReader(f)
                 if r["bucket"] in BUCKETS]
    print(f"[mine] {len(clips)} clips ({sum(1 for c in clips if c['bucket']=='FlasePositives')} FP, "
          f"{sum(1 for c in clips if c['bucket']=='NVR_manual_records')} HB)")

    model = YOLO("yolo11m-pose.pt"); model.to("cuda")

    t0 = time.time()
    for ci, c in enumerate(clips, 1):
        clip = c["clip"]
        bucket_tag = BUCKETS[c["bucket"]]
        cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
        if not cand:
            print(f"[skip] {clip} not found"); continue
        video_path = cand[0]

        tscan = time.time()
        peaks = scan_clip_for_peaks(model, video_path)
        print(f"[{ci:2}/{len(clips)}] {bucket_tag} {clip}  scan={time.time()-tscan:.1f}s  peaks={len(peaks)}", flush=True)

        for rank, center_s in enumerate(peaks, 1):
            out = OUT_DIR / f"{bucket_tag}_{Path(clip).stem}_p{rank}_t{int(center_s)}.npz"
            if out.exists(): continue
            frames, fps = sample_frames(video_path, center_s, N_FRAMES_OUT)
            if frames is None: continue
            kpts, bboxes, confs = extract_multitrack(model, frames)
            np.savez_compressed(out, kpts=kpts, bboxes=bboxes, conf=confs,
                                label="negative", trigger_s=center_s, clip=clip,
                                bucket=bucket_tag, peak_rank=rank, src_fps=fps, n_frames=N_FRAMES_OUT)

    print(f"\n[mine] done in {time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
