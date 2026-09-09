"""Model C: LightGBM on hand-crafted skeleton features.

Features per 16-frame snippet (COCO 17 keypoints):
  - bbox aspect ratio: first, last, max, delta_last_minus_first
  - bbox width, height (median across window, normalized)
  - centroid vertical velocity: max_dy, mean_abs_dy, dy_last3_minus_first3
  - torso angle vs vertical: first, last, |last-first|, max
  - hip-height fraction of frame (proxy for "on ground"): first, last, min
  - keypoint stillness in last 6 frames (mean per-kp variance, normalized)
  - detection confidence: mean, min, missing_frames_ratio

Labels: Le2i {Fall} => positive; {Blank, Stand, Likefall, Lie} => negative.
(We can revisit including Lie in positives later.)

Train / val: keep Le2i's own train/val split (filename prefix).
Cross-domain eval: apply to in-house 17 windows (4 pos, 13 neg).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, confusion_matrix,
)

ROOT = Path(r"D:\fall-neuravue")
LE2I_POSE_DIR = ROOT / "outputs" / "pose" / "le2i"
INHOUSE_POSE_DIR = ROOT / "outputs" / "pose" / "in_house"
OUT_DIR = ROOT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO indices: 5=Lshoulder 6=Rshoulder 11=Lhip 12=Rhip 0=nose
LSHO, RSHO, LHIP, RHIP, NOSE = 5, 6, 11, 12, 0

FALL_CLASSES = {"Fall"}


def _frame_bbox(kpts_t: np.ndarray) -> tuple[float, float, float, float]:
    """Bounding box from visible keypoints of ONE person at ONE frame."""
    visible = kpts_t[kpts_t[:, 2] > 0.1]
    if visible.shape[0] < 3:
        return 0.0, 0.0, 0.0, 0.0
    x1, y1 = visible[:, 0].min(), visible[:, 1].min()
    x2, y2 = visible[:, 0].max(), visible[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def _torso_angle(k: np.ndarray) -> float:
    """Angle between torso (mid-hip -> mid-shoulder) and vertical up. Radians, 0=upright."""
    sho = (k[LSHO, :2] + k[RSHO, :2]) / 2.0
    hip = (k[LHIP, :2] + k[RHIP, :2]) / 2.0
    if k[LSHO, 2] < 0.1 or k[RSHO, 2] < 0.1 or k[LHIP, 2] < 0.1 or k[RHIP, 2] < 0.1:
        return np.nan
    v = sho - hip
    # image coords: y grows downward; "up" is -y direction
    up = np.array([0.0, -1.0])
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.nan
    cos = np.clip(np.dot(v / n, up), -1.0, 1.0)
    return float(np.arccos(cos))


def snippet_features(kpts: np.ndarray, conf: np.ndarray, frame_h: float = 240.0, frame_w: float = 320.0) -> np.ndarray:
    """kpts: (T, 17, 3) for ONE person track. conf: (T,)."""
    T = kpts.shape[0]

    # Per-frame bbox
    bboxes = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        bboxes[t] = _frame_bbox(kpts[t])
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    valid = (w > 5) & (h > 5)

    # Aspect ratio h/w
    ar = np.where(valid, h / np.maximum(w, 1.0), np.nan)
    ar_first = np.nanmean(ar[:3]) if np.any(valid[:3]) else np.nan
    ar_last = np.nanmean(ar[-3:]) if np.any(valid[-3:]) else np.nan
    ar_min = float(np.nanmin(ar)) if np.any(valid) else np.nan
    ar_max = float(np.nanmax(ar)) if np.any(valid) else np.nan
    ar_range = ar_max - ar_min if not (np.isnan(ar_max) or np.isnan(ar_min)) else np.nan
    ar_delta = ar_last - ar_first if not (np.isnan(ar_first) or np.isnan(ar_last)) else np.nan

    # Normalized w/h
    w_norm = float(np.nanmedian(np.where(valid, w / frame_w, np.nan)))
    h_norm = float(np.nanmedian(np.where(valid, h / frame_h, np.nan)))

    # Centroid vertical velocity (normalized by frame_h)
    cy = np.where(valid, (bboxes[:, 1] + bboxes[:, 3]) / 2.0, np.nan)
    dy = np.diff(cy) / frame_h
    max_dy = float(np.nanmax(dy)) if np.any(~np.isnan(dy)) else np.nan          # down = positive
    mean_abs_dy = float(np.nanmean(np.abs(dy))) if np.any(~np.isnan(dy)) else np.nan
    cy_first3 = float(np.nanmean(cy[:3])) if np.any(~np.isnan(cy[:3])) else np.nan
    cy_last3 = float(np.nanmean(cy[-3:])) if np.any(~np.isnan(cy[-3:])) else np.nan
    cy_delta_norm = (cy_last3 - cy_first3) / frame_h if not (np.isnan(cy_first3) or np.isnan(cy_last3)) else np.nan

    # Torso angle
    torso = np.array([_torso_angle(kpts[t]) for t in range(T)], dtype=np.float32)
    torso_first = float(np.nanmean(torso[:3])) if np.any(~np.isnan(torso[:3])) else np.nan
    torso_last = float(np.nanmean(torso[-3:])) if np.any(~np.isnan(torso[-3:])) else np.nan
    torso_max = float(np.nanmax(np.abs(torso))) if np.any(~np.isnan(torso)) else np.nan
    torso_delta = torso_last - torso_first if not (np.isnan(torso_first) or np.isnan(torso_last)) else np.nan

    # Hip height fraction (y of mid-hip / frame_h). Higher y = lower in frame.
    hip_y = np.array([(kpts[t, LHIP, 1] + kpts[t, RHIP, 1]) / 2.0
                       if kpts[t, LHIP, 2] > 0.1 and kpts[t, RHIP, 2] > 0.1 else np.nan
                       for t in range(T)], dtype=np.float32)
    hip_first = float(np.nanmean(hip_y[:3] / frame_h)) if np.any(~np.isnan(hip_y[:3])) else np.nan
    hip_last = float(np.nanmean(hip_y[-3:] / frame_h)) if np.any(~np.isnan(hip_y[-3:])) else np.nan
    hip_max_norm = float(np.nanmax(hip_y / frame_h)) if np.any(~np.isnan(hip_y)) else np.nan

    # Stillness in last 6 frames (mean per-kp x/y variance, normalized)
    tail = kpts[-6:, :, :2]
    mask = kpts[-6:, :, 2] > 0.1
    var_x = np.nan_to_num(np.where(mask, tail[..., 0], np.nan)).var(axis=0)
    var_y = np.nan_to_num(np.where(mask, tail[..., 1], np.nan)).var(axis=0)
    stillness = float(np.sqrt((var_x + var_y).mean()) / max(frame_h, 1.0))

    # Detection quality
    conf_mean = float(conf.mean())
    conf_min = float(conf.min())
    missing_ratio = float((conf < 0.05).mean())

    feats = np.array([
        ar_first, ar_last, ar_min, ar_max, ar_range, ar_delta,
        w_norm, h_norm,
        max_dy, mean_abs_dy, cy_delta_norm,
        torso_first, torso_last, torso_max, torso_delta,
        hip_first, hip_last, hip_max_norm,
        stillness,
        conf_mean, conf_min, missing_ratio,
    ], dtype=np.float32)
    return feats


FEATURE_NAMES = [
    "ar_first", "ar_last", "ar_min", "ar_max", "ar_range", "ar_delta",
    "w_norm", "h_norm",
    "max_dy", "mean_abs_dy", "cy_delta_norm",
    "torso_first", "torso_last", "torso_max", "torso_delta",
    "hip_first", "hip_last", "hip_max_norm",
    "stillness",
    "conf_mean", "conf_min", "missing_ratio",
]


def load_le2i() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, y, split, names = [], [], [], []
    for p in sorted(LE2I_POSE_DIR.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        # Filename: {split}_{class}_{clip}.npz
        parts = p.stem.split("_", 2)
        if len(parts) < 3:
            continue
        spl, cls, _ = parts
        label = 1 if cls in FALL_CLASSES else 0
        kpts = d["kpts"].astype(np.float32)      # (T, 17, 3)
        conf = d["conf"].astype(np.float32)       # (T,)
        # Le2i frames were 320x240 (per README), but pose coords are from YOLO on that image
        feats = snippet_features(kpts, conf, frame_h=240.0, frame_w=320.0)
        X.append(feats); y.append(label); split.append(spl); names.append(p.stem)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(split), names


def load_inhouse() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Multi-track (T, K=3, 17, 3). Pick track with largest aspect-ratio range."""
    X, y, names = [], [], []
    for p in sorted(INHOUSE_POSE_DIR.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        label = 1 if str(d["label"]) == "positive" else 0
        kpts_mk = d["kpts"].astype(np.float32)   # (T, K, 17, 3)
        conf_mk = d["conf"].astype(np.float32)   # (T, K)
        T, K = kpts_mk.shape[0], kpts_mk.shape[1]

        # Pick the track with the largest aspect-ratio dynamic range across time
        # (proxy for "the one who moved from vertical to horizontal or vice versa").
        best_k, best_score = 0, -1.0
        for k in range(K):
            if conf_mk[:, k].mean() < 0.05:
                continue
            ars = []
            for t in range(T):
                x1, y1, x2, y2 = _frame_bbox(kpts_mk[t, k])
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    ars.append((y2 - y1) / (x2 - x1))
            if len(ars) < 2:
                continue
            score = float(np.nanmax(ars) - np.nanmin(ars))
            if score > best_score:
                best_score, best_k = score, k

        kpts = kpts_mk[:, best_k]   # (T, 17, 3)
        conf = conf_mk[:, best_k]   # (T,)
        # in-house frame size = 1360x768
        feats = snippet_features(kpts, conf, frame_h=768.0, frame_w=1360.0)
        X.append(feats); y.append(label); names.append(p.stem)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), names


def eval_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.95) -> tuple[float, float, float]:
    """Return (threshold, precision, false_positive_reduction) at target recall."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # thr has len = len(prec)-1
    good = rec[:-1] >= target_recall
    if not good.any():
        return float("nan"), float("nan"), float("nan")
    # Pick highest threshold that still meets recall target
    idx = np.where(good)[0]
    i = idx[-1]
    return float(thr[i]), float(prec[i]), float(rec[i])


def main() -> None:
    # ---- Le2i features ----
    Xl, yl, spl_l, names_l = load_le2i()
    print(f"[le2i] loaded {Xl.shape[0]} snippets, features={Xl.shape[1]}")
    print(f"  pos={int((yl==1).sum())}  neg={int((yl==0).sum())}")

    Xtr, ytr = Xl[spl_l == "train"], yl[spl_l == "train"]
    Xva, yva = Xl[spl_l == "val"], yl[spl_l == "val"]
    print(f"  train: pos={int((ytr==1).sum())} neg={int((ytr==0).sum())}")
    print(f"  val:   pos={int((yva==1).sum())} neg={int((yva==0).sum())}")

    # Class imbalance weighting
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    scale_pos = n_neg / max(n_pos, 1)

    train_set = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES, reference=train_set)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "average_precision"],
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=8,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        scale_pos_weight=scale_pos,
        verbose=-1,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # ---- Le2i val metrics ----
    p_val = booster.predict(Xva)
    ap_val = average_precision_score(yva, p_val)
    auc_val = roc_auc_score(yva, p_val)
    thr95, prec95, rec95 = eval_at_recall(yva, p_val, 0.95)
    print(f"\n[le2i val] AP={ap_val:.3f}  ROC-AUC={auc_val:.3f}")
    print(f"[le2i val @recall>=0.95] thr={thr95:.3f}  precision={prec95:.3f}  actual_recall={rec95:.3f}")

    # ---- In-house cross-domain eval ----
    Xh, yh, names_h = load_inhouse()
    p_h = booster.predict(Xh)
    print(f"\n[in-house] {Xh.shape[0]} windows  pos={int((yh==1).sum())}  neg={int((yh==0).sum())}")
    if (yh == 1).sum() > 0 and (yh == 0).sum() > 0:
        try:
            ap_h = average_precision_score(yh, p_h)
            auc_h = roc_auc_score(yh, p_h)
            print(f"[in-house] AP={ap_h:.3f}  ROC-AUC={auc_h:.3f}")
        except Exception as e:
            print(f"[in-house] scoring error: {e}")

    # Print per-clip scores
    print("\n[in-house per-window scores]")
    for n, y, s in sorted(zip(names_h, yh, p_h), key=lambda x: -x[2]):
        mark = "P" if y == 1 else "N"
        print(f"  {mark}  score={s:.3f}  {n}")

    # Save model + a small artifact
    booster.save_model(str(OUT_DIR / "model_c_lightgbm.txt"))
    out = {
        "le2i_val_ap": float(ap_val),
        "le2i_val_auc": float(auc_val),
        "le2i_val_prec_at_recall95": float(prec95),
        "in_house_scores": {n: float(s) for n, s in zip(names_h, p_h)},
        "in_house_labels": {n: int(y) for n, y in zip(names_h, yh)},
        "feature_importance": dict(zip(FEATURE_NAMES, [int(v) for v in booster.feature_importance()])),
    }
    (OUT_DIR / "model_c_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved model + report to {OUT_DIR}")


if __name__ == "__main__":
    main()
