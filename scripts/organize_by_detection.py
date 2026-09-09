"""Score every clip with Model B v2, extract detected windows to fall-detected/,
and write CSV of clips with no fall detected.

Sources of candidate windows per clip:
  - in_house_v2/positive_*.npz            (4 TP clips, human trigger)
  - hard_negs/FP_*.npz                    (13 FP clips, 3 peaks each)
  - hard_negs/HB_*.npz                    (24 HB clips, 3 peaks each)
  - TP_Ch44_1 has no windows yet -> mine on the fly

For each clip: pick the window with the highest Model B v2 score.
  - If max_score >= THRESHOLD: extract a WINDOW_SEC clip around that timestamp
    via ffmpeg -> D:\\fall-neuravue\\fall-detected\\{clip_stem}_t{sec}.mp4
  - Else: add row to no_fall_detected.csv
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor

sys.path.insert(0, r"D:\fall-neuravue\scripts")
from train_model_b_videomae import (
    build_model, read_and_prep, read_video_window,
    MODEL_NAME, N_FRAMES, IMG_SIZE, DEVICE,
)
from mine_hard_negatives import scan_clip_for_peaks
from ultralytics import YOLO

ROOT = Path(r"D:\fall-neuravue")
CLIPS_MANIFEST = ROOT / "data" / "clips_manifest.csv"
INHOUSE_V2 = ROOT / "outputs" / "pose" / "in_house_v2"
HARD_NEGS = ROOT / "outputs" / "pose" / "hard_negs"
MODEL_HEAD_PT = ROOT / "outputs" / "models" / "model_b_v2_head.pt"
VIDEO_ROOT = Path(r"D:\fall-detection-testing")

OUT_DIR = ROOT / "fall-detected"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = ROOT / "no_fall_detected.csv"

THRESHOLD = 0.20
WINDOW_SEC = 10.0  # ffmpeg-extracted clip length (5s each side of trigger)


def load_model():
    proc = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)
    backbone, head = build_model(mean, std)
    head.load_state_dict(torch.load(MODEL_HEAD_PT, map_location=DEVICE))
    head.eval(); backbone.eval()
    return backbone, head, mean, std


def score_window(backbone, head, video_path: Path, center_s: float, mean, std) -> float:
    frames = read_video_window(video_path, center_s)
    x = read_and_prep(frames, mean, std).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = backbone(pixel_values=x).last_hidden_state.mean(dim=1)
        p = F.softmax(head(emb), dim=-1)[0, 1].item()
    return float(p)


def collect_candidate_centers() -> dict[str, list[float]]:
    """clip_name -> list of candidate trigger timestamps to score."""
    out: dict[str, list[float]] = {}
    for p in list(INHOUSE_V2.glob("*.npz")) + list(HARD_NEGS.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        clip = str(d["clip"])
        t = float(d["trigger_s"])
        out.setdefault(clip, []).append(t)
    return out


def find_video(clip: str) -> Path | None:
    cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
    return cand[0] if cand else None


def extract_clip(video_path: Path, center_s: float, out_path: Path) -> bool:
    start = max(0, center_s - WINDOW_SEC / 2)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{start:.2f}",
        "-i", str(video_path),
        "-t", f"{WINDOW_SEC:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-an",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr[:200]}")
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def main() -> None:
    # Load clip list
    with CLIPS_MANIFEST.open() as f:
        manifest = list(csv.DictReader(f))

    # Load Model B v2
    print("[model] loading Model B v2...")
    backbone, head, mean, std = load_model()

    candidates = collect_candidate_centers()

    # Ch44 has no candidate windows yet -> mine on the fly
    ch44 = next((c for c in manifest if c["clip"].startswith("TP_Ch44_1")), None)
    if ch44 and ch44["clip"] not in candidates:
        print(f"[mine] Ch44 has no windows -> quick pose scan...")
        model_pose = YOLO("yolo11m-pose.pt"); model_pose.to("cuda")
        vp = find_video(ch44["clip"])
        if vp is not None:
            peaks = scan_clip_for_peaks(model_pose, vp)
            candidates[ch44["clip"]] = peaks
            print(f"  found {len(peaks)} peaks")

    detected_rows = []
    not_detected_rows = []

    t0 = time.time()
    for i, c in enumerate(manifest, 1):
        clip = c["clip"]
        video_path = find_video(clip)
        if video_path is None:
            print(f"[{i:2}/{len(manifest)}] MISSING {clip}")
            not_detected_rows.append({"clip": clip, "bucket": c["bucket"],
                                       "label_original": c["label"],
                                       "n_windows_scored": 0, "max_score": None,
                                       "reason": "video file not found"})
            continue

        centers = candidates.get(clip, [])
        # If still no centers (unlabeled + not mined), pick midpoint + ~30% + ~70%
        if not centers:
            dur = float(c["duration_s"])
            centers = [dur * 0.3, dur * 0.5, dur * 0.7]

        # Score each candidate
        best = {"score": -1.0, "center": None}
        for cs in centers:
            s = score_window(backbone, head, video_path, cs, mean, std)
            if s > best["score"]:
                best = {"score": s, "center": cs}

        dt = time.time() - t0
        if best["score"] >= THRESHOLD:
            out_name = f"{Path(clip).stem}_t{int(best['center'])}_score{int(best['score']*100)}.mp4"
            out_path = OUT_DIR / out_name
            ok = extract_clip(video_path, best["center"], out_path)
            status = "OK" if ok else "FAIL"
            detected_rows.append({"clip": clip, "bucket": c["bucket"],
                                   "label_original": c["label"],
                                   "trigger_s": round(best["center"], 2),
                                   "score": round(best["score"], 4),
                                   "extracted_to": str(out_path.relative_to(ROOT)) if ok else "",
                                   "extract_status": status})
            print(f"[{i:2}/{len(manifest)}] DETECT {clip[:50]:<50} score={best['score']:.3f} t={best['center']:.0f}s -> {out_name}  ({dt:.1f}s)")
        else:
            not_detected_rows.append({"clip": clip, "bucket": c["bucket"],
                                       "label_original": c["label"],
                                       "n_windows_scored": len(centers),
                                       "max_score": round(best["score"], 4),
                                       "reason": f"max_score {best['score']:.3f} < threshold {THRESHOLD}"})
            print(f"[{i:2}/{len(manifest)}] no    {clip[:50]:<50} max={best['score']:.3f}  ({dt:.1f}s)")

    # Write CSVs
    if detected_rows:
        det_csv = ROOT / "fall-detected" / "_detections.csv"
        with det_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detected_rows[0].keys()))
            w.writeheader(); w.writerows(detected_rows)
        print(f"\n[detections] {len(detected_rows)} clips -> {det_csv.relative_to(ROOT)}")

    with CSV_OUT.open("w", newline="") as f:
        fn = ["clip", "bucket", "label_original", "n_windows_scored", "max_score", "reason"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader(); w.writerows(not_detected_rows)
    print(f"[no_fall] {len(not_detected_rows)} clips -> {CSV_OUT.relative_to(ROOT)}")

    # Summary
    n_by_bucket = {}
    for r in detected_rows:
        n_by_bucket.setdefault(r["bucket"], {"detect": 0, "nodetect": 0})["detect"] += 1
    for r in not_detected_rows:
        n_by_bucket.setdefault(r["bucket"], {"detect": 0, "nodetect": 0})["nodetect"] += 1
    print("\n[summary]")
    print(f"  {'bucket':<20}{'detected':>10}{'not_detected':>14}")
    for b, x in n_by_bucket.items():
        print(f"  {b:<20}{x['detect']:>10}{x['nodetect']:>14}")


if __name__ == "__main__":
    main()
