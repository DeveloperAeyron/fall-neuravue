"""Score already-found ST-GCN cluster times with Model B + lighting guard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\fall-neuravue\scripts")
from infer_window import load_model, score_window

VIDEO = Path(
    r"D:\fall-detection-testing\all_video_data\NewTPRecords"
    r"\MergedExtracted\TP_Ch53_1_2026-08-26_124000.mp4"
)
CENTERS = [
    27.2, 58.3, 170.2, 180.3, 184.4, 190.5, 198.2,
    214.3, 220.6, 225.3, 235.6, 245.8, 259.6, 274.1, 338.2,
]
ANNOTATED = 251.0


def main() -> None:
    print("loading Model B...", flush=True)
    backbone, head, mean, std = load_model()
    print("t_s,score,alarm,guarded,reason,near_annot", flush=True)
    kept = 0
    for c in CENTERS:
        o = score_window(VIDEO, c, backbone, head, mean, std)
        if o["alarm"]:
            kept += 1
        near = "YES" if abs(c - ANNOTATED) <= 15 else ""
        print(
            f"{c:.1f},{o['score']:.4f},{o['alarm']},{o['guarded']},{o['reason']},{near}",
            flush=True,
        )
    print(
        f"ST-GCN clusters={len(CENTERS)}  ModelB_kept={kept}  "
        f"suppressed={len(CENTERS) - kept}  annotated_fall={ANNOTATED}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
