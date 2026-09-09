"""Model B v2: VideoMAE head trained on Le2i + HB hard negatives.

Change vs v1: mix 72 HB peak windows (surveillance-camera background clips that
never triggered the deployed detector) into the training set as extra negatives.
Each HB peak is a 48-frame extraction; we sample 16 consecutive frames from its
middle (matches Le2i's 16-frame training density).

FP peaks stay pure holdout for evaluation - never touched during training.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
sys.path.insert(0, r"D:\fall-neuravue\scripts")
from train_model_b_videomae import (
    ROOT, LE2I_ROOT, INHOUSE_POSE_V2, HARD_NEGS, OUT_DIR, VIDEO_ROOT,
    MODEL_NAME, N_FRAMES, IMG_SIZE, BATCH, LR, DEVICE, SEED,
    FALL_CLASSES, build_model, embed, read_and_prep, sample_le2i_snippet,
    read_video_window, Le2iDataset, score_inhouse,
)

EPOCHS = 12
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)


class HBNegDataset(Dataset):
    """Load HB peak .npz files -> read 16 frames from the corresponding video window."""
    def __init__(self, mean, std):
        self.items = []
        for p in sorted(HARD_NEGS.glob("HB_*.npz")):
            d = np.load(p, allow_pickle=True)
            clip = str(d["clip"])
            center_s = float(d["trigger_s"])
            cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
            if cand:
                self.items.append((cand[0], center_s))
        self.mean = mean; self.std = std

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        video_path, center_s = self.items[i]
        frames = read_video_window(video_path, center_s)
        return read_and_prep(frames, self.mean, self.std), 0  # label 0 = not_fall


class ConcatDataset(Dataset):
    def __init__(self, ds_list):
        self.ds_list = ds_list
        self.offsets = np.cumsum([0] + [len(d) for d in ds_list])
    def __len__(self): return int(self.offsets[-1])
    def __getitem__(self, i):
        for k, ds in enumerate(self.ds_list):
            if i < self.offsets[k+1]:
                return ds[i - self.offsets[k]]
        raise IndexError(i)


def train_head(backbone, head, loader_tr, loader_va, class_weights, epochs):
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    best = {"epoch": 0, "auc": 0.0, "state": None}
    for ep in range(1, epochs + 1):
        head.train()
        tot = 0.0; n = 0
        for x, y in loader_tr:
            x = x.to(DEVICE); y = y.to(DEVICE)
            emb = embed(backbone, x)
            logits = head(emb)
            loss = F.cross_entropy(logits, y, weight=class_weights.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * y.size(0); n += y.size(0)
        head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader_va:
                x = x.to(DEVICE); y = y.to(DEVICE)
                emb = embed(backbone, x)
                p = F.softmax(head(emb), dim=-1)[:, 1]
                ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
        y_arr = np.concatenate(ys); p_arr = np.concatenate(ps)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) > 1 else float("nan")
        ap = float(average_precision_score(y_arr, p_arr))
        print(f"  ep{ep:>2}  train_loss={tot/max(n,1):.3f}  val_AP={ap:.3f}  val_AUC={auc:.3f}")
        if auc > best["auc"]:
            best = {"epoch": ep, "auc": auc, "ap": ap,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    print(f"[best] ep={best['epoch']}  val_AUC={best['auc']:.3f}  val_AP={best['ap']:.3f}")


def main() -> None:
    proc = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)

    print("[data] Le2i …")
    ds_tr_le = Le2iDataset("train", mean, std)
    ds_va = Le2iDataset("val", mean, std)
    print(f"  le2i train={len(ds_tr_le)}  val={len(ds_va)}")

    print("[data] HB negatives …")
    ds_hb = HBNegDataset(mean, std)
    print(f"  hb={len(ds_hb)}")

    ds_tr = ConcatDataset([ds_tr_le, ds_hb])
    print(f"[data] total train={len(ds_tr)}")

    loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    loader_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    ys_tr = np.array([lbl for _, lbl in ds_tr_le.items] + [0] * len(ds_hb))
    w = np.array([1.0 / max((ys_tr == c).sum(), 1) for c in range(2)], dtype=np.float32)
    w = w / w.sum() * 2.0
    class_weights = torch.tensor(w, dtype=torch.float32)
    print(f"  class weights: {w.tolist()}   (pos={int((ys_tr==1).sum())} neg={int((ys_tr==0).sum())})")

    print("[model] VideoMAE …")
    backbone, head = build_model(mean, std)

    print("\n[train] fine-tuning head …")
    t0 = time.time()
    train_head(backbone, head, loader_tr, loader_va, class_weights, EPOCHS)
    print(f"[train] done in {time.time()-t0:.1f}s")

    print("\n[eval] in-house …")
    rows = score_inhouse(backbone, head, mean, std)
    y = np.array([1 if r["label"] == "positive" else 0 for r in rows])
    p = np.array([r["score"] for r in rows])
    from sklearn.metrics import roc_auc_score, average_precision_score
    ap = average_precision_score(y, p); auc = roc_auc_score(y, p)
    print(f"\n[in-house v2] AP={ap:.3f}  AUC={auc:.3f}  n_pos={int(y.sum())} n_neg={int((1-y).sum())}")

    print("\n[ranked]")
    for r in sorted(rows, key=lambda x: -x["score"]):
        m = "POS" if r["label"] == "positive" else "neg"
        print(f"{m:<4}{r['score']:>8.3f}   {r['clip']}")

    # Count at threshold 0.95
    above = [(r["label"], r["score"]) for r in rows if r["score"] >= 0.95]
    tp_hit = sum(1 for l, _ in above if l == "positive")
    fp_hit = sum(1 for l, _ in above if l == "negative")
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    print(f"\n[operating point @ 0.95]  tp_caught={tp_hit}/{n_pos}   fp_kept={fp_hit}/{n_neg}   "
          f"fp_reduction={(1 - fp_hit/max(n_neg,1))*100:.0f}%")

    (OUT_DIR / "model_b_v2_report.json").write_text(json.dumps({
        "rows": rows, "ap": float(ap), "auc": float(auc),
        "tp_at_95": tp_hit, "fp_at_95": fp_hit,
    }, indent=2))
    torch.save(head.state_dict(), OUT_DIR / "model_b_v2_head.pt")
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
