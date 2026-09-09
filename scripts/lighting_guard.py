"""Reject VideoMAE windows that sit on an IR↔colour / lights-on-off swap.

VideoMAE treats a whole-frame brightness jump as fall-like motion. The 16-frame
scoring window is only ~0.6 s, so the high-score peak is often *after* the
swap, when luma inside the window is already stable. This guard looks at a
wider context around the window center.

Calibrated 2026-09-10 on the labeled peaks (0–255 Rec.601 luma):

    kind                         range±5s   jump±5s
    4 TP peaks + annots          ≤ 4.9      ≤ 1.4
    walker (Ch56_1_181000)       24.2       8.7
    lights ON  Ch58_2_191724     58–68      58
    lights ON  Ch58_2_014952     65.6       25.6   ← 16-frame range was only 3.5
    lights OFF Ch58_1_190523     216        125

Thresholds sit in the gap: range 30, jump 18.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CONTEXT_RADIUS_SEC = 5.0
SAMPLE_FPS = 8.0
LUM_RANGE_THRESHOLD = 30.0
LUM_JUMP_THRESHOLD = 18.0
PREVIEW_WIDTH = 160


@dataclass(frozen=True)
class LightingDecision:
    rejected: bool
    lum_range: float
    lum_jump: float
    reason: str

    def as_score(self, model_score: float) -> float:
        return 0.0 if self.rejected else model_score


def _luma_bgr(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (PREVIEW_WIDTH, 90), interpolation=cv2.INTER_AREA)
    bgr = small.reshape(-1, 3).mean(axis=0)
    return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])


def _stats(lum: np.ndarray) -> tuple[float, float]:
    if lum.size == 0:
        return 0.0, 0.0
    rng = float(lum.max() - lum.min())
    jump = float(np.abs(np.diff(lum)).max()) if lum.size > 1 else 0.0
    return rng, jump


def _decide(rng: float, jump: float) -> LightingDecision:
    reasons = []
    if rng > LUM_RANGE_THRESHOLD:
        reasons.append(f"luma_range {rng:.1f}>{LUM_RANGE_THRESHOLD}")
    if jump > LUM_JUMP_THRESHOLD:
        reasons.append(f"luma_jump {jump:.1f}>{LUM_JUMP_THRESHOLD}")
    return LightingDecision(
        rejected=bool(reasons),
        lum_range=rng,
        lum_jump=jump,
        reason="; ".join(reasons) if reasons else "ok",
    )


@dataclass
class LumaTimeline:
    """Precomputed 8 fps luma for one clip — use this on dense scans."""

    times: np.ndarray  # seconds
    lum: np.ndarray

    def decide(self, center_s: float, radius_s: float = CONTEXT_RADIUS_SEC) -> LightingDecision:
        if self.lum.size == 0:
            return LightingDecision(False, 0.0, 0.0, "empty-timeline")
        lo, hi = center_s - radius_s, center_s + radius_s
        mask = (self.times >= lo) & (self.times <= hi)
        return _decide(*_stats(self.lum[mask]))


def build_luma_timeline(video_path: Path | str) -> LumaTimeline:
    """Walk the file once with grab() skips. Cheap vs VideoMAE."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    times: list[float] = []
    lum: list[float] = []
    i = 0
    while True:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(i / fps)
            lum.append(_luma_bgr(frame))
        else:
            if not cap.grab():
                break
        i += 1
        if n and i >= n:
            break
    cap.release()
    return LumaTimeline(np.asarray(times, dtype=np.float32), np.asarray(lum, dtype=np.float32))


def decide_window(video_path: Path | str, center_s: float) -> LightingDecision:
    """Local ±5 s read — for scoring a handful of candidate windows."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = int(round(max(0.0, center_s - CONTEXT_RADIUS_SEC) * fps))
    end_f = int(round((center_s + CONTEXT_RADIUS_SEC) * fps))
    if n:
        end_f = min(end_f, n - 1)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    lum: list[float] = []
    fidx = start_f
    while fidx <= end_f:
        if (fidx - start_f) % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            lum.append(_luma_bgr(frame))
        else:
            if not cap.grab():
                break
        fidx += 1
    cap.release()
    return _decide(*_stats(np.asarray(lum, dtype=np.float32)))
