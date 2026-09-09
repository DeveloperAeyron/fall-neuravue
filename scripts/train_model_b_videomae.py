"""Model B: fine-tune VideoMAE-base head on Le2i (Fall vs rest), then score
in-house TP + FP-peak windows. Ensemble with Model C (LightGBM) at the end.

Model: MCG-NJU/videomae-base (K400-pretrained) — swap the classifier head,
freeze the backbone by default, fine-tune head with class-balanced loss.

Input: 16-frame RGB clips at 224x224.
  - Le2i: sample 16 frames from each snippet's JPGs, spaced evenly.
  - In-house: read 16 frames from a ±1.5 s window around the trigger.

We fine-tune the classifier head only for speed (backbone frozen) — 800 samples
isn't enough to fine-tune the full 87M-param model without heavy overfit.
"""

from __future__ import annotations

import csv
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

ROOT = Path(r"D:\fall-neuravue")
LE2I_ROOT = ROOT / "data" / "external" / "le2i" / "raw"
INHOUSE_POSE_V2 = ROOT / "outputs" / "pose" / "in_house_v2"
HARD_NEGS = ROOT / "outputs" / "pose" / "hard_negs"
OUT_DIR = ROOT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = Path(r"D:\fall-detection-testing")
TRIGGERS = ROOT / "data" / "trigger_times.csv"

MODEL_NAME = "MCG-NJU/videomae-base"
N_FRAMES = 16
IMG_SIZE = 224
BATCH = 8
EPOCHS = 12
LR = 3e-4
DEVICE = "cuda"
SEED = 42

FALL_CLASSES = {"Fall"}

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)


# ------------ processors / model ------------

def build_model(processor_mean, processor_std) -> tuple[nn.Module, nn.Module]:
    backbone = VideoMAEModel.from_pretrained(MODEL_NAME)
    for p in backbone.parameters():
        p.requires_grad = False
    head = nn.Sequential(
        nn.LayerNorm(backbone.config.hidden_size),
        nn.Linear(backbone.config.hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return backbone.to(DEVICE).eval(), head.to(DEVICE).train()


# ------------ data ------------

def read_and_prep(frames_bgr: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Take a list of BGR frames -> (T, 3, H, W) normalized tensor."""
    out = np.zeros((N_FRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, f in enumerate(frames_bgr[:N_FRAMES]):
        img = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - mean) / std
        out[i] = img.transpose(2, 0, 1)
    return torch.from_numpy(out)


def sample_le2i_snippet(folder: Path) -> list[np.ndarray]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".jpg"],
                   key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    if not files:
        return []
    if len(files) >= N_FRAMES:
        idx = np.linspace(0, len(files) - 1, N_FRAMES).astype(int)
        files = [files[i] for i in idx]
    else:
        # pad by repeating last
        files = files + [files[-1]] * (N_FRAMES - len(files))
    return [cv2.imread(str(p)) for p in files]


def read_video_window(video_path: Path, center_s: float) -> list[np.ndarray]:
    """Read N_FRAMES centered on center_s at NATIVE fps (matches Le2i temporal density)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cf = int(round(center_s * fps))
    start = max(0, cf - N_FRAMES // 2); start = min(start, max(0, nb - N_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    while len(frames) < N_FRAMES:
        frames.append(np.zeros_like(frames[0]) if frames else np.zeros((768, 1360, 3), np.uint8))
    return frames


class Le2iDataset(Dataset):
    def __init__(self, split: str, mean: np.ndarray, std: np.ndarray):
        self.items = []
        for cls_dir in sorted((LE2I_ROOT / split).iterdir()):
            if not cls_dir.is_dir(): continue
            label = 1 if cls_dir.name in FALL_CLASSES else 0
            for snip in sorted(cls_dir.iterdir()):
                if snip.is_dir():
                    self.items.append((snip, label))
        self.mean = mean; self.std = std

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        folder, label = self.items[i]
        frames = sample_le2i_snippet(folder)
        if not frames:
            return torch.zeros(N_FRAMES, 3, IMG_SIZE, IMG_SIZE), label
        return read_and_prep(frames, self.mean, self.std), label


# ------------ train / eval ------------

@torch.no_grad()
def embed(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, 3, H, W)
    out = backbone(pixel_values=x)
    # VideoMAE base returns last_hidden_state (B, N_patches, D). Pool over patches.
    emb = out.last_hidden_state.mean(dim=1)  # (B, D)
    return emb


def train_head(backbone, head, loader_tr, loader_va, class_weights, epochs=EPOCHS):
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
        # val
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


def score_inhouse(backbone, head, mean, std) -> list[dict]:
    """Score TP + FP-peak windows (48-frame). Slide 16-frame sub-windows w/ stride 8 -> take max."""
    rows = []
    files = sorted(INHOUSE_POSE_V2.glob("positive_*.npz")) + sorted(HARD_NEGS.glob("FP_*.npz"))
    for p in files:
        d = np.load(p, allow_pickle=True)
        clip = str(d["clip"])
        center_s = float(d["trigger_s"])
        label = str(d["label"])
        cand = list((VIDEO_ROOT / "all_video_data").rglob(clip))
        if not cand: continue
        video_path = cand[0]
        # For evaluation, just sample N_FRAMES centered at trigger (matches training density).
        frames = read_video_window(video_path, center_s)
        x = read_and_prep(frames, mean, std).unsqueeze(0).to(DEVICE)
        head.eval()
        with torch.no_grad():
            emb = embed(backbone, x)
            prob = F.softmax(head(emb), dim=-1)[0, 1].item()
        rows.append({"clip": p.stem, "label": label, "score": float(prob)})
    return rows


def main() -> None:
    proc = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)

    print("[data] loading Le2i…")
    ds_tr = Le2iDataset("train", mean, std)
    ds_va = Le2iDataset("val", mean, std)
    print(f"  train={len(ds_tr)}  val={len(ds_va)}")
    loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    loader_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    # class weights for balancing
    ys_tr = np.array([lbl for _, lbl in ds_tr.items])
    w = np.array([1.0 / max((ys_tr == c).sum(), 1) for c in range(2)], dtype=np.float32)
    w = w / w.sum() * 2.0
    class_weights = torch.tensor(w, dtype=torch.float32)
    print(f"  class weights: {w.tolist()}")

    print("[model] building VideoMAE …")
    backbone, head = build_model(mean, std)
    print(f"  backbone params: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M (frozen)")
    print(f"  head params: {sum(p.numel() for p in head.parameters())/1e6:.2f}M (trainable)")

    print("\n[train] fine-tuning head …")
    t0 = time.time()
    train_head(backbone, head, loader_tr, loader_va, class_weights)
    print(f"[train] done in {time.time()-t0:.1f}s")

    print("\n[eval] scoring in-house windows …")
    rows = score_inhouse(backbone, head, mean, std)

    y = np.array([1 if r["label"] == "positive" else 0 for r in rows])
    p = np.array([r["score"] for r in rows])
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    ap = average_precision_score(y, p); auc = roc_auc_score(y, p)
    print(f"\n[in-house] AP={ap:.3f}  ROC-AUC={auc:.3f}  n_pos={int(y.sum())} n_neg={int((1-y).sum())}")

    print("\n[ranked]")
    for r in sorted(rows, key=lambda x: -x["score"]):
        m = "POS" if r["label"] == "positive" else "neg"
        print(f"{m:<4}{r['score']:>8.3f}   {r['clip']}")

    # Ensemble with Model C v3 report if available
    c_report = OUT_DIR / "model_c_v3_report.json"
    if c_report.exists():
        creport = json.loads(c_report.read_text())
        cmap = {r["clip"]: r["score"] for r in creport.get("eval_rows", [])}
        combo = []
        for r in rows:
            cs = cmap.get(r["clip"], None)
            if cs is None: continue
            combo.append({"clip": r["clip"], "label": r["label"],
                          "b": r["score"], "c": cs,
                          "avg": 0.5 * r["score"] + 0.5 * cs,
                          "max": max(r["score"], cs)})
        if combo:
            for how in ("avg", "max"):
                yy = np.array([1 if r["label"] == "positive" else 0 for r in combo])
                pp = np.array([r[how] for r in combo])
                print(f"\n[ensemble {how}] AP={average_precision_score(yy, pp):.3f}  AUC={roc_auc_score(yy, pp):.3f}")
            print("\n[ensemble avg ranked]")
            for r in sorted(combo, key=lambda x: -x["avg"]):
                m = "POS" if r["label"] == "positive" else "neg"
                print(f"{m:<4}avg={r['avg']:.3f}  b={r['b']:.3f}  c={r['c']:.3f}   {r['clip']}")

    (OUT_DIR / "model_b_report.json").write_text(json.dumps({"rows": rows,
                                                             "ap": float(ap), "auc": float(auc)}, indent=2))
    torch.save(head.state_dict(), OUT_DIR / "model_b_head.pt")
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
