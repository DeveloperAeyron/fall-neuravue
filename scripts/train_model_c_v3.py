"""Model C v3: Le2i + HB (surveillance-domain hard negatives) -> eval on TP vs FP peaks.

Training data:
  - Le2i: 777 snippets, 16-frame windows, Fall vs {Blank, Stand, Likefall, Lie}
  - HB peaks: from D:\\fall-neuravue\\outputs\\pose\\hard_negs\\HB_*.npz
    These are surveillance-camera clips that never triggered the deployed
    detector -> safe to fold in as label=0. For each 48-frame peak, we slide
    16-frame sub-windows with stride 8 and take features from the sub-window
    whose 'track movement' is largest (the same picker used at inference).

Evaluation:
  - In-house TP set (positive): 4 windows from in_house_v2 (48-frame @ human trigger)
  - In-house FP peaks (negative): FP_*.npz from hard_negs/ (48-frame @ auto-mined peak)
  - Both scored with the same sliding-window + track-picking inference used in v2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

import sys
sys.path.insert(0, r"D:\fall-neuravue\scripts")
from train_model_c_lightgbm import (
    snippet_features, load_le2i, FEATURE_NAMES, _frame_bbox,
)
from train_model_c_v2 import score_inhouse_window, track_movement_score

ROOT = Path(r"D:\fall-neuravue")
POSE_INHOUSE_V2 = ROOT / "outputs" / "pose" / "in_house_v2"
POSE_HARD_NEGS = ROOT / "outputs" / "pose" / "hard_negs"
OUT_DIR = ROOT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_H_IN, FRAME_W_IN = 768.0, 1360.0
SLIDE = 16
STRIDE = 8


def load_hard_neg_features(prefix: str) -> tuple[np.ndarray, list[str]]:
    """For each 48-frame npz starting with prefix, slide 16-frame sub-windows and
    return features from the sub-window with the largest 'movement score' on the
    best track. This mirrors inference-time behavior, so the training and eval
    both use the same worst-case slice logic.
    """
    X, names = [], []
    for p in sorted(POSE_HARD_NEGS.glob(f"{prefix}_*.npz")):
        d = np.load(p, allow_pickle=True)
        kpts_mk = d["kpts"].astype(np.float32)   # (48, K, 17, 3)
        conf_mk = d["conf"].astype(np.float32)   # (48, K)
        T, K = kpts_mk.shape[0], kpts_mk.shape[1]
        starts = list(range(0, T - SLIDE + 1, STRIDE)) or [0]
        best_feats, best_mov = None, -1e9
        for s in starts:
            e = s + SLIDE
            for k in range(K):
                mov = track_movement_score(kpts_mk[s:e, k], conf_mk[s:e, k])
                if mov > best_mov:
                    feats = snippet_features(kpts_mk[s:e, k], conf_mk[s:e, k],
                                             frame_h=FRAME_H_IN, frame_w=FRAME_W_IN)
                    best_mov = mov
                    best_feats = feats
        if best_feats is None:
            continue
        X.append(best_feats)
        names.append(p.stem)
    return (np.asarray(X, dtype=np.float32) if X else np.zeros((0, len(FEATURE_NAMES)), np.float32)), names


def eval_at_recall(y_true, y_score, target=0.75):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    good = rec[:-1] >= target
    if not good.any():
        return float("nan"), float("nan"), float("nan")
    i = np.where(good)[0][-1]
    return float(thr[i]), float(prec[i]), float(rec[i])


def main() -> None:
    # ---- Le2i ----
    Xl, yl, spl_l, _ = load_le2i()
    Xtr_le = Xl[spl_l == "train"]; ytr_le = yl[spl_l == "train"]
    Xva_le = Xl[spl_l == "val"]; yva_le = yl[spl_l == "val"]

    # ---- HB (unlabeled background) as extra hard-neg training ----
    Xhb, hb_names = load_hard_neg_features("HB")
    yhb = np.zeros(len(Xhb), dtype=np.int32)
    print(f"[hb hard-negs] {len(Xhb)} feature vectors from {len(hb_names)} HB peaks")

    # ---- Combine ----
    X_train = np.concatenate([Xtr_le, Xhb], axis=0)
    y_train = np.concatenate([ytr_le, yhb], axis=0)
    print(f"[train] total={len(X_train)}  pos={int((y_train==1).sum())}  neg={int((y_train==0).sum())}")

    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)

    tset = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    vset = lgb.Dataset(Xva_le, label=yva_le, feature_name=FEATURE_NAMES, reference=tset)
    params = dict(objective="binary", metric=["binary_logloss", "average_precision"],
                  learning_rate=0.05, num_leaves=31, min_data_in_leaf=8,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  scale_pos_weight=spw, verbose=-1)
    booster = lgb.train(params, tset, num_boost_round=600,
                        valid_sets=[tset, vset], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    p_va = booster.predict(Xva_le)
    print(f"\n[le2i val] AP={average_precision_score(yva_le, p_va):.3f}  "
          f"AUC={roc_auc_score(yva_le, p_va):.3f}")

    # ---- In-house evaluation ----
    # Positives: in_house_v2/positive_*.npz  (4 windows)
    # Negatives: hard_negs/FP_*.npz          (multiple peaks per FP clip)
    rows = []
    for p in sorted(POSE_INHOUSE_V2.glob("positive_*.npz")):
        s, info = score_inhouse_window(booster, p)
        rows.append({"clip": p.stem, "label": "positive", "score": s})
    for p in sorted(POSE_HARD_NEGS.glob("FP_*.npz")):
        s, info = score_inhouse_window(booster, p)
        rows.append({"clip": p.stem, "label": "negative", "score": s})

    y_true = np.array([1 if r["label"] == "positive" else 0 for r in rows])
    y_score = np.array([r["score"] for r in rows])
    n_pos = int(y_true.sum()); n_neg = int((1 - y_true).sum())
    ap = average_precision_score(y_true, y_score) if n_pos and n_neg else float("nan")
    auc = roc_auc_score(y_true, y_score) if n_pos and n_neg else float("nan")
    print(f"\n[eval TP vs FP-peaks]  n_pos={n_pos}  n_neg={n_neg}")
    print(f"  AP={ap:.3f}  ROC-AUC={auc:.3f}")

    for tr in (0.95, 0.75, 0.50):
        thr, prec, rec = eval_at_recall(y_true, y_score, tr)
        if np.isnan(thr):
            print(f"  @recall>={tr:.2f}: unattainable")
        else:
            fp_kept = int(round((1 - prec) * (rec * n_pos) / max(prec, 1e-9)))
            print(f"  @recall>={tr:.2f}: thr={thr:.3f}  precision={prec:.3f}  actual_recall={rec:.3f}")

    print(f"\n[per-window]  (sorted by score desc)")
    print(f"{'lbl':<4}{'score':>8}   {'clip'}")
    for r in sorted(rows, key=lambda x: -x["score"]):
        m = "POS" if r["label"] == "positive" else "neg"
        print(f"{m:<4}{r['score']:>8.3f}   {r['clip']}")

    # Save
    booster.save_model(str(OUT_DIR / "model_c_v3_lightgbm.txt"))
    (OUT_DIR / "model_c_v3_report.json").write_text(json.dumps({
        "ap": float(ap), "auc": float(auc),
        "n_train_pos": int((y_train == 1).sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "eval_rows": rows,
    }, indent=2))
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
