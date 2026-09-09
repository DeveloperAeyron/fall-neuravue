"""Full sliding-window scan of every TP + FP clip with Model B v2.

For each clip:
  - stride the whole duration with a WINDOW_STRIDE_SEC step
  - at each center, sample 16 consecutive frames at native fps
  - score with Model B v2
  - keep max score and its timestamp
Also record how many windows in the clip scored above THRESHOLD.

Output: D:\\fall-neuravue\\full_scan_report.csv
  clip, bucket, duration_s, n_windows_scored, max_score, argmax_t,
  n_above_threshold, detected
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor

sys.path.insert(0, r"D:\fall-neuravue\scripts")
from train_model_b_videomae import (
    build_model, read_and_prep, read_video_window,
    MODEL_NAME, N_FRAMES, DEVICE,
)
from lighting_guard import build_luma_timeline

ROOT = Path(r"D:\fall-neuravue")
CLIPS_MANIFEST = ROOT / "data" / "clips_manifest.csv"
MODEL_HEAD_PT = ROOT / "outputs" / "models" / "model_b_v2_head.pt"
VIDEO_ROOT = Path(r"D:\fall-detection-testing")
OUT_CSV = ROOT / "full_scan_report.csv"

WINDOW_STRIDE_SEC = 2.0      # scan every 2s (dense)
THRESHOLD = 0.20
BATCH = 8                    # windows per VideoMAE forward pass


def load_model():
    proc = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)
    backbone, head = build_model(mean, std)
    head.load_state_dict(torch.load(MODEL_HEAD_PT, map_location=DEVICE))
    head.eval(); backbone.eval()
    return backbone, head, mean, std


def find_video(clip: str) -> Path | None:
    cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
    return cand[0] if cand else None


@torch.no_grad()
def scan_clip(backbone, head, video_path: Path, duration_s: float, mean, std):
    """Score every window. Lighting-transition windows are zeroed and skipped."""
    centers = np.arange(1.0, max(1.5, duration_s - 1.0), WINDOW_STRIDE_SEC)
    scores = [0.0] * len(centers)
    guarded = [False] * len(centers)
    timeline = build_luma_timeline(video_path)

    pending_idx: list[int] = []
    buf_x: list[torch.Tensor] = []

    def flush() -> None:
        if not buf_x:
            return
        x = torch.stack(buf_x, dim=0).to(DEVICE)
        emb = backbone(pixel_values=x).last_hidden_state.mean(dim=1)
        p = F.softmax(head(emb), dim=-1)[:, 1].cpu().numpy().tolist()
        for j, s in zip(pending_idx, p):
            scores[j] = float(s)
        buf_x.clear()
        pending_idx.clear()

    for i, cs in enumerate(centers):
        decision = timeline.decide(float(cs))
        if decision.rejected:
            guarded[i] = True
            continue
        frames = read_video_window(video_path, float(cs))
        buf_x.append(read_and_prep(frames, mean, std))
        pending_idx.append(i)
        if len(buf_x) == BATCH:
            flush()
    flush()
    return centers.tolist(), scores, guarded


def main() -> None:
    filters = [a for a in sys.argv[1:] if not a.endswith(".csv")]
    if any(a.endswith(".csv") for a in sys.argv[1:]):
        global OUT_CSV
        OUT_CSV = ROOT / Path(next(a for a in sys.argv[1:] if a.endswith(".csv"))).name
    with CLIPS_MANIFEST.open() as f:
        clips = [r for r in csv.DictReader(f)
                 if r["bucket"] in ("TrueFalls", "NewTPRecords", "FlasePositives")]
    if filters:
        clips = [r for r in clips if any(f in r["clip"] for f in filters)]
    print(f"[scan] {len(clips)} clips ({sum(1 for c in clips if c['bucket'] in ('TrueFalls','NewTPRecords'))} TP, "
          f"{sum(1 for c in clips if c['bucket']=='FlasePositives')} FP)"
          + (f"  filter={filters}" if filters else "")
          + f"  -> {OUT_CSV.name}")

    backbone, head, mean, std = load_model()

    rows = []
    t_all = time.time()
    for i, c in enumerate(clips, 1):
        clip = c["clip"]
        bucket = c["bucket"]
        dur = float(c["duration_s"])
        vp = find_video(clip)
        if vp is None:
            print(f"[{i:2}/{len(clips)}] MISSING {clip}"); continue
        t0 = time.time()
        centers, scores, guarded = scan_clip(backbone, head, vp, dur, mean, std)
        dt = time.time() - t0
        max_i = int(np.argmax(scores))
        max_score = float(scores[max_i])
        n_above = int(sum(1 for s in scores if s >= THRESHOLD))
        n_guarded = int(sum(guarded))
        detected = max_score >= THRESHOLD
        expected = "TP" if bucket in ("TrueFalls", "NewTPRecords") else "FP"
        if expected == "TP" and detected: outcome = "TP_DETECT"
        elif expected == "TP" and not detected: outcome = "TP_MISS"
        elif expected == "FP" and detected: outcome = "FP_SURVIVED"
        else: outcome = "FP_SUPPRESSED"
        rows.append({
            "clip": clip, "bucket": bucket, "duration_s": round(dur, 1),
            "n_windows_scored": len(scores),
            "n_windows_lighting_guarded": n_guarded,
            "max_score": round(max_score, 4),
            "argmax_t": round(centers[max_i], 1),
            "n_windows_above_threshold": n_above,
            "detected": detected,
            "outcome": outcome,
        })
        print(f"[{i:2}/{len(clips)}] {outcome:<14} {bucket[:15]:<15} {clip[:44]:<44} "
              f"max={max_score:.3f}@{centers[max_i]:.0f}s  n>={THRESHOLD}: {n_above}/{len(scores)}  "
              f"guarded={n_guarded}  ({dt:.1f}s)")

    # write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summary
    tp_hit = sum(1 for r in rows if r["outcome"] == "TP_DETECT")
    tp_miss = sum(1 for r in rows if r["outcome"] == "TP_MISS")
    fp_survived = sum(1 for r in rows if r["outcome"] == "FP_SURVIVED")
    fp_suppressed = sum(1 for r in rows if r["outcome"] == "FP_SUPPRESSED")
    n_tp = tp_hit + tp_miss
    n_fp = fp_survived + fp_suppressed
    print(f"\n[summary]  scan in {time.time()-t_all:.0f}s  ->  {OUT_CSV.relative_to(ROOT)}")
    print(f"  TP clips: {tp_hit}/{n_tp} detected  ({100*tp_hit/max(n_tp,1):.0f}%)")
    print(f"  FP clips: {fp_suppressed}/{n_fp} suppressed  ({100*fp_suppressed/max(n_fp,1):.0f}%)")


if __name__ == "__main__":
    main()
