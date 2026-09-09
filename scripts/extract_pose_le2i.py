"""Extract 17-keypoint COCO skeletons for every Le2i snippet folder using YOLO11m-pose.

Input:  D:\\fall-neuravue\\data\\external\\le2i\\raw\\{train,val}\\{class}\\{clip}\\{n}.jpg
Output: D:\\fall-neuravue\\outputs\\pose\\le2i\\{split}_{class}_{clip}.npz
    kpts   : float32 (T, 17, 3)  -- (x, y, conf); (0,0,0) if no person that frame
    bboxes : float32 (T, 4)      -- (x1, y1, x2, y2); zeros if no person
    conf   : float32 (T,)        -- detection confidence of chosen track
    class  : str
    split  : str

Multi-person policy: this dataset is single-person, so we take the top-confidence
person detection per frame. Frames with no person get a zero row (masked later).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(r"D:\fall-neuravue")
LE2I_ROOT = ROOT / "data" / "external" / "le2i" / "raw"
OUT_DIR = ROOT / "outputs" / "pose" / "le2i"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo11m-pose.pt"


def sorted_frames(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".jpg"]
    # names look like "212.jpg" -> sort numerically when possible
    def key(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(files, key=key)


def extract_one(model: YOLO, folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = sorted_frames(folder)
    if not frames:
        return np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    # Batch all frames of this snippet in one predict() call (only ~16 frames).
    results = model.predict(
        source=[str(p) for p in frames],
        device=0,
        verbose=False,
        conf=0.25,
        imgsz=640,
        stream=False,
    )

    T = len(frames)
    kpts = np.zeros((T, 17, 3), dtype=np.float32)
    bboxes = np.zeros((T, 4), dtype=np.float32)
    confs = np.zeros((T,), dtype=np.float32)

    for i, r in enumerate(results):
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        # Pick highest-confidence person track for this frame.
        box_confs = r.boxes.conf.detach().cpu().numpy()
        j = int(np.argmax(box_confs))
        # r.keypoints.data: (num_persons, 17, 3) if kp confs available; else (n,17,2)
        kp_data = r.keypoints.data.detach().cpu().numpy()
        if kp_data.shape[-1] == 2:
            k = np.concatenate([kp_data[j], np.ones((17, 1), np.float32)], axis=-1)
        else:
            k = kp_data[j]
        kpts[i] = k.astype(np.float32)
        bboxes[i] = r.boxes.xyxy[j].detach().cpu().numpy().astype(np.float32)
        confs[i] = float(box_confs[j])

    return kpts, bboxes, confs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only process this many snippets (0=all)")
    args = parser.parse_args()

    model = YOLO(MODEL_NAME)
    model.to("cuda")

    total = 0
    ok = 0
    empty = 0
    t0 = time.time()

    for split_dir in sorted(LE2I_ROOT.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name  # train|val
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for snippet in sorted(cls_dir.iterdir()):
                if not snippet.is_dir():
                    continue
                total += 1
                out_path = OUT_DIR / f"{split}_{cls}_{snippet.name}.npz"
                if out_path.exists():
                    continue
                kpts, bboxes, confs = extract_one(model, snippet)
                if kpts.size == 0:
                    empty += 1
                    continue
                np.savez_compressed(
                    out_path,
                    kpts=kpts, bboxes=bboxes, conf=confs,
                    cls=cls, split=split, snippet=snippet.name,
                )
                ok += 1
                if ok % 50 == 0:
                    dt = time.time() - t0
                    print(f"[{ok}/{total}] {dt:.1f}s  last={out_path.name}", flush=True)
                if args.limit and ok >= args.limit:
                    break
            if args.limit and ok >= args.limit:
                break
        if args.limit and ok >= args.limit:
            break

    dt = time.time() - t0
    print(f"\nDone. saved={ok} empty={empty} total_scanned={total} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
