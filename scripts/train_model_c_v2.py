"""Model C v2: Le2i-trained LightGBM, applied to in-house with sliding-window scoring.

At inference on in-house 48-frame windows:
  - slide a 16-frame window with stride 4 -> 9 sub-windows
  - for each sub-window, pick the person track with the largest
    "movement score" = vertical-velocity range + aspect-ratio range over that
    sub-window (staff standing still get low scores; the faller wins)
  - compute features on that best track, score with the Le2i-trained booster
  - return MAX over sub-windows as the final window score

Le2i training is unchanged from v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

# Reuse the v1 feature extractor
import sys
sys.path.insert(0, r"D:\fall-neuravue\scripts")
from train_model_c_lightgbm import (
    snippet_features, load_le2i, FEATURE_NAMES, FALL_CLASSES, _frame_bbox,
)

ROOT = Path(r"D:\fall-neuravue")
INHOUSE_POSE_DIR_V2 = ROOT / "outputs" / "pose" / "in_house_v2"
OUT_DIR = ROOT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LSHO, RSHO, LHIP, RHIP = 5, 6, 11, 12
FRAME_H_IN, FRAME_W_IN = 768.0, 1360.0


def track_movement_score(kpts_t: np.ndarray, conf_t: np.ndarray) -> float:
    """Score how much this track moves over the given sub-window.
    kpts_t: (T, 17, 3); conf_t: (T,)
    Combines: vertical-velocity peak + aspect-ratio range.
    """
    T = kpts_t.shape[0]
    ars, cys = [], []
    for t in range(T):
        b = _frame_bbox(kpts_t[t])
        if b is None:
            continue
        x1, y1, x2, y2 = b
        w = x2 - x1; h = y2 - y1
        if w > 5 and h > 5:
            ars.append(h / w)
            cys.append((y1 + y2) / 2.0)
    if len(ars) < 3:
        return -1.0
    ars = np.asarray(ars); cys = np.asarray(cys)
    ar_range = float(ars.max() - ars.min())
    dy = np.diff(cys)
    max_dy = float(np.abs(dy).max()) / FRAME_H_IN
    conf_mean = float(conf_t.mean())
    # Weighted score: aspect change matters, velocity matters more, confidence gates it
    return conf_mean * (ar_range + 8.0 * max_dy)


def score_inhouse_window(booster: lgb.Booster, npz_path: Path) -> tuple[float, dict]:
    d = np.load(npz_path, allow_pickle=True)
    kpts_mk = d["kpts"].astype(np.float32)   # (T=48, K=5, 17, 3)
    conf_mk = d["conf"].astype(np.float32)   # (48, K)
    T, K = kpts_mk.shape[0], kpts_mk.shape[1]

    SLIDE = 16
    STRIDE = 4
    starts = list(range(0, T - SLIDE + 1, STRIDE))
    if not starts:
        starts = [0]

    best_score = -1.0
    best_info = {"start": 0, "track": 0, "score": 0.0}
    per_slide = []

    for s in starts:
        e = s + SLIDE
        # Pick best track for this sub-window
        best_k, best_mov = 0, -1e9
        for k in range(K):
            mov = track_movement_score(kpts_mk[s:e, k], conf_mk[s:e, k])
            if mov > best_mov:
                best_mov, best_k = mov, k
        if best_mov < 0:
            continue
        feats = snippet_features(kpts_mk[s:e, best_k], conf_mk[s:e, best_k],
                                 frame_h=FRAME_H_IN, frame_w=FRAME_W_IN)
        p = float(booster.predict(feats.reshape(1, -1))[0])
        per_slide.append({"start": s, "track": int(best_k), "mov": float(best_mov), "prob": p})
        if p > best_score:
            best_score = p
            best_info = per_slide[-1]

    return max(best_score, 0.0), {"best": best_info, "per_slide": per_slide,
                                    "label": str(d["label"]), "clip": str(d["clip"])}


def train_le2i(spl_l, Xl, yl) -> lgb.Booster:
    Xtr, ytr = Xl[spl_l == "train"], yl[spl_l == "train"]
    Xva, yva = Xl[spl_l == "val"], yl[spl_l == "val"]
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    spw = n_neg / max(n_pos, 1)
    tset = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    vset = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES, reference=tset)
    params = dict(objective="binary", metric=["binary_logloss", "average_precision"],
                  learning_rate=0.05, num_leaves=31, min_data_in_leaf=8,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  scale_pos_weight=spw, verbose=-1)
    booster = lgb.train(params, tset, num_boost_round=400,
                        valid_sets=[tset, vset], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    p_val = booster.predict(Xva)
    print(f"[le2i val] AP={average_precision_score(yva, p_val):.3f}  AUC={roc_auc_score(yva, p_val):.3f}")
    return booster


def main() -> None:
    Xl, yl, spl_l, _ = load_le2i()
    print(f"[le2i] {Xl.shape[0]} snippets")
    booster = train_le2i(spl_l, Xl, yl)

    # In-house v2
    files = sorted(INHOUSE_POSE_DIR_V2.glob("*.npz"))
    print(f"\n[in-house v2] {len(files)} windows")

    rows = []
    for p in files:
        s, info = score_inhouse_window(booster, p)
        rows.append({"clip": p.stem, "label": info["label"], "score": s,
                     "best_start": info["best"]["start"], "best_track": info["best"]["track"],
                     "best_mov": info["best"].get("mov", 0.0)})

    y_true = np.array([1 if r["label"] == "positive" else 0 for r in rows])
    y_score = np.array([r["score"] for r in rows])
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        ap = average_precision_score(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        print(f"\n[in-house v2] AP={ap:.3f}  ROC-AUC={auc:.3f}")

    print("\n[in-house v2 per-window] (sorted by score desc)")
    print(f"{'lbl':<4}{'score':>8}{'startF':>8}{'trk':>5}{'mov':>7}   clip")
    for r in sorted(rows, key=lambda x: -x["score"]):
        mark = "POS" if r["label"] == "positive" else "neg"
        print(f"{mark:<4}{r['score']:>8.3f}{r['best_start']:>8}{r['best_track']:>5}{r['best_mov']:>7.2f}   {r['clip']}")

    out = OUT_DIR / "model_c_v2_report.json"
    out.write_text(json.dumps({"rows": rows,
                               "ap": float(ap), "auc": float(auc)}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
