"""Render pose overlays on the 4 in-house TP windows for visual QA.

For each positive window: re-read the same 16 frames, overlay all K=3 person tracks
we saved, colored by track index (0=red, 1=green, 2=blue). Highlight the track
Model C actually picked (biggest AR-range) with a thick outline.

Output: D:\\fall-neuravue\\outputs\\pose\\qa\\{clip}_frame{n}.jpg
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\fall-neuravue")
INHOUSE_POSE_DIR = ROOT / "outputs" / "pose" / "in_house"
QA_DIR = ROOT / "outputs" / "pose" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_ROOT = Path(r"D:\fall-detection-testing")

# COCO 17 skeleton connections
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),  # head
]

COLORS = [(0, 0, 255), (0, 200, 0), (255, 100, 0)]  # BGR: red, green, blue-ish
LSHO, RSHO, LHIP, RHIP = 5, 6, 11, 12


def frame_bbox(kpts_t: np.ndarray) -> tuple[int, int, int, int] | None:
    visible = kpts_t[kpts_t[:, 2] > 0.1]
    if visible.shape[0] < 3:
        return None
    x1, y1 = visible[:, 0].min(), visible[:, 1].min()
    x2, y2 = visible[:, 0].max(), visible[:, 1].max()
    return int(x1), int(y1), int(x2), int(y2)


def pick_best_track(kpts_mk: np.ndarray, conf_mk: np.ndarray) -> int:
    T, K = kpts_mk.shape[0], kpts_mk.shape[1]
    best_k, best_score = 0, -1.0
    for k in range(K):
        if conf_mk[:, k].mean() < 0.05:
            continue
        ars = []
        for t in range(T):
            b = frame_bbox(kpts_mk[t, k])
            if b is None:
                continue
            x1, y1, x2, y2 = b
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                ars.append((y2 - y1) / (x2 - x1))
        if len(ars) >= 2:
            score = float(np.nanmax(ars) - np.nanmin(ars))
            if score > best_score:
                best_score, best_k = score, k
    return best_k


def load_clip_path(clip_stem: str) -> Path | None:
    # clip_stem like "positive_TP_Ch53_1_2026-08-26_123330_t217"
    parts = clip_stem.split("_", 1)[1]  # drop label prefix
    parts = parts.rsplit("_", 1)[0]     # drop _t{n}
    cand = list((VIDEO_ROOT / "all_video_data").rglob(parts + ".mp4"))
    return cand[0] if cand else None


def re_sample_frames(video_path: Path, center_s: float, n: int = 16) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
        frames.append(np.zeros((768, 1360, 3), np.uint8))
    return np.stack(frames, axis=0)


def draw_skeleton(img: np.ndarray, kp: np.ndarray, color: tuple[int, int, int], thick: int = 2) -> None:
    for a, b in SKELETON:
        if kp[a, 2] > 0.1 and kp[b, 2] > 0.1:
            pa = (int(kp[a, 0]), int(kp[a, 1]))
            pb = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(img, pa, pb, color, thick)
    for i in range(17):
        if kp[i, 2] > 0.1:
            cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 3, color, -1)


def main() -> None:
    pos_files = sorted(INHOUSE_POSE_DIR.glob("positive_*.npz"))
    print(f"[qa] {len(pos_files)} positive windows to render")

    strip_rows = []

    for p in pos_files:
        d = np.load(p, allow_pickle=True)
        clip = str(d["clip"])
        center_s = float(d["trigger_s"])
        kpts_mk = d["kpts"].astype(np.float32)   # (16, 3, 17, 3)
        conf_mk = d["conf"].astype(np.float32)   # (16, 3)

        best_k = pick_best_track(kpts_mk, conf_mk)
        print(f"\n  {p.stem}  best_track={best_k}")

        video_path = load_clip_path(p.stem)
        if video_path is None:
            print("    [!] video not found, skipping render")
            continue
        frames = re_sample_frames(video_path, center_s, 16)

        strip = []
        for t in range(16):
            img = frames[t].copy()
            for k in range(kpts_mk.shape[1]):
                if conf_mk[t, k] > 0.1:
                    color = COLORS[k % len(COLORS)]
                    draw_skeleton(img, kpts_mk[t, k], color, thick=2 if k != best_k else 4)
                    # Draw bbox for best track
                    if k == best_k:
                        b = frame_bbox(kpts_mk[t, k])
                        if b is not None:
                            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 3)
            # Label
            cv2.putText(img, f"t={t}  best_k={best_k}  clip={clip[:30]}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # Downsize for the strip
            small = cv2.resize(img, (480, 270))
            strip.append(small)

        # Save individual frames of the middle 8 for detail
        for t in [0, 2, 4, 6, 8, 10, 12, 14]:
            out = QA_DIR / f"{p.stem}_frame{t:02d}.jpg"
            cv2.imwrite(str(out), frames[t])  # save raw so user can compare

        # 4x4 grid strip
        rows = []
        for r in range(4):
            row = np.concatenate(strip[r*4:(r+1)*4], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        out_strip = QA_DIR / f"{p.stem}_grid.jpg"
        cv2.imwrite(str(out_strip), grid)
        print(f"    grid -> {out_strip.name}")
        strip_rows.append(out_strip)

    print(f"\n[qa] wrote {len(strip_rows)} grids to {QA_DIR}")


if __name__ == "__main__":
    main()
