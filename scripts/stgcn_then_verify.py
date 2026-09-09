"""ST-GCN fires, then Model B v2 + lighting guard scores those windows.

    python stgcn_then_verify.py --all --workers 2
    python stgcn_then_verify.py --one TP_Ch53_1_2026-08-26_124000.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STGCN_ROOT = Path(r"D:\fall-detection-testing")
NEURAVUE_ROOT = Path(r"D:\fall-neuravue")
VIDEO_ROOT = STGCN_ROOT / "all_video_data"
OUT_DIR = NEURAVUE_ROOT / "outputs"
CLUSTER_CSV = OUT_DIR / "stgcn_then_verify.csv"
STAMPS_CSV = OUT_DIR / "stgcn_fire_stamps.csv"
SUMMARY_JSON = OUT_DIR / "stgcn_then_verify.json"
CLUSTER_GAP_S = 3.0
BUCKETS = ("TrueFalls", "NewTPRecords", "FlasePositives")

sys.path.insert(0, str(STGCN_ROOT))
sys.path.insert(0, str(NEURAVUE_ROOT / "scripts"))


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    m, s = divmod(int(round(t)), 60)
    return f"{m:02d}:{s:02d}"


def find_video(name: str) -> Path:
    hits = list(VIDEO_ROOT.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def list_clips() -> list[tuple[str, str, str]]:
    """(clip, bucket, label) for TP+FP only."""
    rows = []
    man = NEURAVUE_ROOT / "data" / "clips_manifest.csv"
    if not man.exists():
        man = STGCN_ROOT / "verifier" / "outputs" / "clips_manifest.csv"
    if man.exists():
        with man.open() as f:
            for r in csv.DictReader(f):
                if r["bucket"] in BUCKETS:
                    rows.append((r["clip"], r["bucket"], r["label"]))
        return rows
    for bucket, label in (
        ("NewTPRecords", "positive"),
        ("TrueFalls", "positive"),
        ("FlasePositives", "negative"),
    ):
        d = VIDEO_ROOT / bucket
        if d.exists():
            for p in sorted(d.rglob("*.mp4")):
                if not p.name.startswith("._"):
                    rows.append((p.name, bucket, label))
    return rows


def cluster_times(times: list[float], gap: float = CLUSTER_GAP_S):
    if not times:
        return [], []
    groups = [[times[0]]]
    for t in times[1:]:
        if t - groups[-1][-1] > gap:
            groups.append([t])
        else:
            groups[-1].append(t)
    centers = [float(np.median(g)) for g in groups]
    return centers, groups


def done_clips(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {r["clip"] for r in csv.DictReader(f) if r.get("clip")}


def append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    for _ in range(200):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise TimeoutError(f"lock timeout {lock}")
    try:
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerows(rows)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def process_one(clip: str, bucket: str, label: str, models, score_fn, ns) -> dict:
    video = find_video(clip)
    print(f"[stgcn] {clip}  {bucket}", flush=True)
    t0 = time.time()
    import torch
    with torch.no_grad():
        r = models["run_video"](str(video), models["stgcn"], ns, writer_dir=None)
    fires = [float(x) for x in (r.get("fall_times_s") or [])]
    centers, groups = cluster_times(fires)
    wall = time.time() - t0
    print(
        f"[stgcn] {clip} done {wall:.0f}s  fires={len(fires)}  "
        f"clusters={len(centers)}  hist={r.get('actions_histogram')}",
        flush=True,
    )

    stamp_rows = [
        {
            "clip": clip,
            "bucket": bucket,
            "label": label,
            "fire_t_s": round(t, 2),
            "fire_mmss": mmss(t),
        }
        for t in fires
    ]

    cluster_rows = []
    if not centers:
        cluster_rows.append({
            "clip": clip, "bucket": bucket, "label": label,
            "stgcn_fired": False,
            "cluster_t_s": "", "cluster_mmss": "",
            "n_fire_frames": 0, "b_score": "",
            "b_alarm": False, "guarded": False, "reason": "stgcn_no_fall",
        })
    else:
        for c, g in zip(centers, groups):
            out = score_fn(video, c)
            cluster_rows.append({
                "clip": clip, "bucket": bucket, "label": label,
                "stgcn_fired": True,
                "cluster_t_s": round(c, 2),
                "cluster_mmss": mmss(c),
                "n_fire_frames": len(g),
                "b_score": round(out["score"], 4),
                "b_alarm": out["alarm"],
                "guarded": out["guarded"],
                "reason": out["reason"],
            })
            print(
                f"  {clip}  ST-GCN {mmss(c)} ({c:.1f}s, n={len(g)}) -> "
                f"B={out['score']:.3f} keep={out['alarm']} {out['reason']}",
                flush=True,
            )

    part = OUT_DIR / "stgcn_parts" / f"{Path(clip).stem}.json"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps({
        "clip": clip, "bucket": bucket, "label": label,
        "wall_s": round(wall, 1),
        "n_fires": len(fires),
        "n_clusters": len(centers),
        "actions": r.get("actions_histogram"),
        "clusters": cluster_rows,
    }, indent=2))
    return {"stamps": stamp_rows, "clusters": cluster_rows}


def worker_main(clip_rows: list[tuple[str, str, str]], fps: float, n_steps: int, device: str) -> None:
    os.chdir(STGCN_ROOT)
    sys.path.insert(0, str(STGCN_ROOT))
    sys.path.insert(0, str(NEURAVUE_ROOT / "scripts"))

    from run_pipeline_test_10fps import (
        ResizePadding, patch_torch_load, run_video,
    )
    from DetectorLoader import TinyYOLOv3_onecls
    from PoseEstimateLoader import SPPE_FastPose
    from ActionsEstLoader import TSSTG
    from infer_window import load_model, score_window
    import torch

    patch_torch_load(device)
    print(f"[worker {os.getpid()}] loading ST-GCN + Model B ({len(clip_rows)} clips)", flush=True)
    with torch.no_grad():
        detect = TinyYOLOv3_onecls(384, device=device)
        pose = SPPE_FastPose("resnet50", 224, 160, device=device)
        action = TSSTG(device=device)
    stgcn = (detect, pose, action, ResizePadding(384, 384))
    backbone, head, mean, std = load_model()

    class A:
        pass
    ns = A()
    ns.device = device
    ns.fps = fps
    ns.stgcn_frames = n_steps
    ns.detection_input_size = 384
    ns.pose_input_size = "224x160"
    ns.pose_backbone = "resnet50"
    ns.show = False

    models = {"stgcn": stgcn, "run_video": run_video}

    def score_fn(video, c):
        return score_window(video, c, backbone, head, mean, std)

    stamp_fields = ["clip", "bucket", "label", "fire_t_s", "fire_mmss"]
    cluster_fields = [
        "clip", "bucket", "label", "stgcn_fired", "cluster_t_s", "cluster_mmss",
        "n_fire_frames", "b_score", "b_alarm", "guarded", "reason",
    ]
    for clip, bucket, label in clip_rows:
        try:
            out = process_one(clip, bucket, label, models, score_fn, ns)
            append_csv(STAMPS_CSV, out["stamps"], stamp_fields)
            append_csv(CLUSTER_CSV, out["clusters"], cluster_fields)
        except Exception as e:
            print(f"[FAIL] {clip}: {e}", flush=True)
            append_csv(CLUSTER_CSV, [{
                "clip": clip, "bucket": bucket, "label": label,
                "stgcn_fired": False, "cluster_t_s": "", "cluster_mmss": "",
                "n_fire_frames": 0, "b_score": "", "b_alarm": False,
                "guarded": False, "reason": f"error: {e}",
            }], cluster_fields)


def seed_first_clip() -> None:
    """Write the already-finished sanity clip so workers skip it."""
    if CLUSTER_CSV.exists():
        return
    clip = "TP_Ch53_1_2026-08-26_124000.mp4"
    scored = [
        (27.2, 0.0233, False), (58.3, 0.0649, False), (170.2, 0.0963, False),
        (180.3, 0.3884, True), (184.4, 0.2047, True), (190.5, 0.0539, False),
        (198.2, 0.1923, False), (214.3, 0.3416, True), (220.6, 0.0878, False),
        (225.3, 0.1192, False), (235.6, 0.0227, False), (245.8, 0.1648, False),
        (259.6, 0.4353, True), (274.1, 0.2855, True), (338.2, 0.2467, True),
    ]
    rows = [{
        "clip": clip, "bucket": "NewTPRecords", "label": "positive",
        "stgcn_fired": True, "cluster_t_s": t, "cluster_mmss": mmss(t),
        "n_fire_frames": "", "b_score": s, "b_alarm": a, "guarded": False, "reason": "ok",
    } for t, s, a in scored]
    append_csv(CLUSTER_CSV, rows, list(rows[0].keys()))
    print(f"[seed] skipped already-scored {clip}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stgcn_frames", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.one:
        clips = [(args.one, "", "")]
        for name, bucket, label in list_clips():
            if name == args.one:
                clips = [(name, bucket, label)]
                break
    else:
        clips = list_clips()
        seed_first_clip()
        skip = done_clips(CLUSTER_CSV)
        clips = [c for c in clips if c[0] not in skip]
        # TPs first
        clips.sort(key=lambda x: (0 if x[2] == "positive" else 1, x[0]))

    print(f"[plan] {len(clips)} clips  workers={args.workers}", flush=True)
    for c, b, lab in clips:
        print(f"  - {lab or '?':9} {c}", flush=True)

    if args.workers <= 1 or len(clips) <= 1:
        worker_main(clips, args.fps, args.stgcn_frames, args.device)
        return

    shards = [[] for _ in range(args.workers)]
    for i, row in enumerate(clips):
        shards[i % args.workers].append(row)

    from multiprocessing import Process
    procs = []
    for shard in shards:
        if not shard:
            continue
        p = Process(target=worker_main, args=(shard, args.fps, args.stgcn_frames, args.device))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[done] all workers exited", flush=True)


if __name__ == "__main__":
    main()
