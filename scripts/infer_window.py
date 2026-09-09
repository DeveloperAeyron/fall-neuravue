"""Score one candidate window: lighting guard, then Model B v2.

This is the production path for a second-stage verifier:

    candidate (video + timestamp)
        → lighting_guard.decide_window()     # IR↔colour / lights on-off
        → if rejected: score = 0.0
        → else VideoMAE-base (frozen) + MLP head
        → alarm if score >= THRESHOLD (0.20)

Usage (on the GPU box):

    python infer_window.py path/to/clip.mp4 51
    python infer_window.py path/to/clip.mp4 251 --threshold 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighting_guard import decide_window
from train_model_b_videomae import (
    DEVICE,
    MODEL_NAME,
    build_model,
    read_and_prep,
    read_video_window,
)

ROOT = Path(r"D:\fall-neuravue")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
MODEL_HEAD_PT = ROOT / "outputs" / "models" / "model_b_v2_head.pt"
if not MODEL_HEAD_PT.exists():
    MODEL_HEAD_PT = ROOT / "models" / "model_b_v2_head.pt"

THRESHOLD = 0.20


def load_model():
    proc = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)
    backbone, head = build_model(mean, std)
    head.load_state_dict(torch.load(MODEL_HEAD_PT, map_location=DEVICE))
    head.eval()
    backbone.eval()
    return backbone, head, mean, std


def score_window(video_path: Path, center_s: float, backbone, head, mean, std) -> dict:
    guard = decide_window(video_path, center_s)
    if guard.rejected:
        return {
            "score": 0.0,
            "alarm": False,
            "guarded": True,
            "reason": guard.reason,
            "luma_range": guard.lum_range,
            "luma_jump": guard.lum_jump,
        }

    frames = read_video_window(video_path, center_s)
    x = read_and_prep(frames, mean, std).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = backbone(pixel_values=x).last_hidden_state.mean(dim=1)
        p = F.softmax(head(emb), dim=-1)[0, 1].item()
    return {
        "score": float(p),
        "alarm": float(p) >= THRESHOLD,
        "guarded": False,
        "reason": "ok",
        "luma_range": guard.lum_range,
        "luma_jump": guard.lum_jump,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score one fall-candidate window")
    ap.add_argument("video", type=Path)
    ap.add_argument("center_s", type=float, help="window center in seconds")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    global THRESHOLD
    THRESHOLD = args.threshold

    backbone, head, mean, std = load_model()
    out = score_window(args.video, args.center_s, backbone, head, mean, std)
    print(
        f"{args.video.name}  t={args.center_s:.1f}s  "
        f"score={out['score']:.3f}  alarm={out['alarm']}  "
        f"guarded={out['guarded']}  luma={out['luma_range']:.1f}/{out['luma_jump']:.1f}  "
        f"{out['reason']}"
    )


if __name__ == "__main__":
    main()
