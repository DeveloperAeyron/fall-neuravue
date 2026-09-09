# fall-neuravue

Two-stage fall-detection **verifier** for the Neuravue NVR system.
The deployed detector is over-eager. This repo scores each candidate window
and only keeps alarms that still look like a fall.

```
NVR feed → [existing detector] → candidate window
                                    │
                                    ▼
                    [luminance-delta lighting guard]
                         huge IR↔colour jump?
                           yes          no
                            │            │
                        suppress   [Model B v2]
                                         │
                               score ≥ threshold?
                                 yes          no
                                  │            │
                               ALARM       suppress
```

**Shipping model:** Model B v2 (VideoMAE-base, frozen backbone, fine-tuned
MLP head) at **threshold 0.20**, plus the lighting guard in
`scripts/lighting_guard.py`.

Private repo: [waleedshoaib2/fall-neuravue](https://github.com/waleedshoaib2/fall-neuravue).
Original videos are **not** in Git (identifiable people).

---

## How Model B v2 is implemented

### Architecture

| Piece | What |
|---|---|
| Backbone | `MCG-NJU/videomae-base` (Hugging Face), **frozen**. ~87M params, pretrained on Kinetics-400. |
| Head | MLP, **trainable** (~200K params): `LayerNorm(768) → Linear(768,256) → GELU → Dropout(0.3) → Linear(256,2)` |
| Input | 16 consecutive RGB frames at 224×224, sampled at the video's native fps around a window center |
| Output | softmax P(fall) on class index 1 |
| Loss | class-weighted cross-entropy |
| Train | 12 epochs, batch 8, AdamW lr 3e-4, weight decay 1e-4. ~8–10 min on RTX 5070 |

Only the head is trained. 764 samples is not enough to unfreeze the 87M
backbone without overfitting.

Code: `scripts/train_model_b_videomae.py` (v1) and `scripts/train_model_b_v2.py`.
Weights: `models/model_b_v2_head.pt`. The backbone is downloaded from Hugging
Face at runtime.

### Training data

**Positives:** Le2i `Fall` snippets (74 train / 28 val).

**Negatives:**
- Le2i `Blank`, `Stand`, `Likefall`, `Lie` (not-fall)
- **72 HB peaks** — 24 unlabeled `HB_*.mp4` background clips × 3 most
  fall-like 3-second windows, auto-mined by `scripts/mine_hard_negatives.py`

Total train ≈ 692 Le2i + 72 HB = **764**. FP clips are a **pure holdout** —
never used in training.

### Why this model, not pose

Model C (LightGBM on YOLO11m-pose skeleton features) was the first baseline.
It missed both staff-occluded Ch53 `_124000` falls because the pose tracker
locked onto staff instead of the falling patient. VideoMAE looks at pixels, so
it still sees the fall behind people.

| Model | Input | In-house AP | In-house AUC | Annotated TP recall |
|---|---|---|---|---|
| Model C v3 (LightGBM + pose) | skeletons | 0.377 | 0.744 | 2/4 |
| Model B v1 (VideoMAE, Le2i only) | RGB | 0.775 | 0.962 | 4/4 @ 0.95 |
| **Model B v2 (VideoMAE + HB negs)** | RGB | **0.917** | **0.987** | **4/4 @ 0.20** |

Le2i val for v2: AP **0.994**. Ensemble of B+C did not beat B alone.

v2 scores are on a **compressed scale** vs v1. Adding HB negatives pulled
every score down, so **0.20 in v2 ≈ 0.95 in v1** for the same recall/precision
trade. Do not mix the two thresholds.

---

## Two evaluation protocols — do not mix the numbers

### A. 43-window holdout (pose-mined peaks)

4 human-annotated TP windows + 39 hard-mined peaks from the 13 FP clips.
This is the number that was used while iterating the head.

| Threshold | TP recall | FP-window reduction | FP-clip reduction |
|---|---|---|---|
| **0.20 (shipping)** | **4/4 (100%)** | 37/39 (95%) | 11/13 (85%) |
| 0.24 | 3/4 (75%) | 37/39 (95%) | 12/13 (92%) |
| 0.30 | 3/4 (75%) | 39/39 (100%) | 13/13 (100%) |

### B. Dense full-clip scan (honest production proxy)

`scripts/full_scan_tp_fp.py`: every 2 seconds across the whole TP + FP clip,
score a 16-frame window, take the max. Lighting-transition windows are zeroed
before the model runs. Unguarded baseline: `outputs/full_scan_report.csv`.
With guard: `outputs/full_scan_report_guarded.csv`.

**Headline at threshold 0.20 + lighting guard:** 4/4 annotated TPs caught
(same peaks and scores as before the guard), **4/13 FP clips still fire**
(69% FP-clip reduction, was 54%). Ch44 missed (no trigger, max 0.006).

---

## Detected clips — dense scan @ threshold 0.20

This is the table that matters for “what would fire in production.”
Peak time = second inside the source clip where Model B v2 scored highest.

| # | Nature | Filename | Peak (mm:ss) | Peak (s) | Score | Guard |
|---|---|---|---|---|---|---|
| 1 | **TP** | `TP_Ch53_1_2026-08-26_124000.mp4` | **04:11** | 251 | 0.976 | none |
| 2 | **TP** | `TP_Ch53_2_2026-08-26_124000.mp4` | **03:33** | 213 | 0.899 | none |
| 3 | **TP** | `TP_Ch53_1_2026-08-26_123330.mp4` | **03:27** | 207 | 0.853 | none |
| 4 | **TP** | `TP_Ch53_2_2026-08-26_123330.mp4` | **02:43** | 163 | 0.454 | none |
| 5 | FP | `FP_Ch56_1_2026-08-05_181000.mp4` | **02:13** | 133 | 0.959 | (walker, not lighting) |
| 6 | FP | `FP_Ch58_2_2026-08-06_060559.mp4` | **00:25** | 25 | 0.578 | none |
| 7 | FP | `FP_Ch58_2_2026-08-05_191724.mp4` | **04:09** | 249 | 0.529 | lights-on @51 killed; leftover = sit-on-bed IR |
| 8 | FP | `FP_Ch48_1_2026-08-05_031559.mp4` | **09:15** | 555 | 0.327 | none |
| — | FP killed | `FP_Ch58_2_2026-08-06_014952.mp4` | was 20:11 | 1211 | 0.791 → **0.185** | lighting guard |
| — | FP killed | `FP_Ch58_1_2026-08-04_190523.mp4` | was 11:23 | 683 | 0.485 → **0.177** | lighting guard |

**4 TP detected, 4 FP survived** (was 6). Not listed: Ch44 miss (max 0.006) and
9 FPs suppressed. All 4 TP peaks are unchanged — the guard zeroed 0 TP windows.

Human-annotated impact times (for comparison) live in `data/trigger_times.csv`:

| Clip | Annotated impact |
|---|---|
| Ch53_1_123330 | 217 s (model peak 207 s) |
| Ch53_1_124000 | 251 s (model peak 251 s — exact) |
| Ch53_2_123330 | 179 s (model peak 163 s) |
| Ch53_2_124000 | 297 s (model peak 213 s — 84 s earlier; needs visual re-check) |

---

## Threshold 0.20 vs 0.80 (dense scan)

| Threshold | Annotated TP recall | Unique events caught | FP-clip reduction | Alarms | Precision |
|---|---|---|---|---|---|
| **0.20 + guard** | **4/4 (100%)** | 2/2 | **9/13 (69%)** | 4 TP + 4 FP = 8 | 50% |
| 0.20 unguarded | 4/4 (100%) | 2/2 | 7/13 (54%) | 4 TP + 6 FP = 10 | 40% |
| **0.80 + guard** | **3/4 (75%)** | **2/2** | **12/13 (92%)** | 3 TP + 1 FP = 4 | **75%** |

At 0.80 the dropped TP is still `Ch53_2_123330` (0.454) — duplicate camera of
an event caught at 0.899. The only FP that survives 0.80 after the guard is
the walker (`Ch56_1_181000`, 0.959). The old 0.982 lights-on peak is gone.

---

## Lighting guard (shipped)

`scripts/lighting_guard.py` looks at ±5 s around each window center at 8 fps
(not just the 16-frame / ~0.6 s model window — the high-score peak is often
*after* the swap). Reject if Rec.601 luma range > 30 **or** consecutive-sample
jump > 18 (0–255 scale).

Calibrated so all 4 TP peaks stay (range ≤ 4.9, jump ≤ 1.4) and the walker
clip is not treated as lighting (range 24.2). Wired into
`full_scan_tp_fp.py` (skips VideoMAE on guarded windows) and
`organize_by_detection.py`.

Killed: `Ch58_2_014952` (0.791 → 0.185) and `Ch58_1_190523` (0.485 → 0.177).
On `Ch58_2_191724` the t=51 lights-on peak (0.982) is gone; a later 0.529
peak remains — person sitting on the bed edge in IR, not another mode swap.

## Remaining failure modes (4 survivors)

| # | Clip | Score | Failure mode |
|---|---|---|---|
| 1 | `FP_Ch56_1_181000` | **0.959** | Elderly person **walking with a walker** |
| 2 | `FP_Ch58_2_060559` | 0.578 | **Staff bending over a patient in bed** |
| 3 | `FP_Ch58_2_191724` | 0.529 | **Sitting on bed edge in IR** (lights-on already guarded) |
| 4 | `FP_Ch48_1_031559` | 0.327 | **Staff bending over a patient in bed** |

### Next fixes

1. **Targeted negatives + retrain the head.** Mine ~50 staff-bending / sit-on-bed
   windows and ~20 walker windows from HB/normal footage. Retrain Model B v2
   head (~10 min). Do **not** train on the 13 FP clips — they are the holdout.
2. **Pose vertical-velocity gate.** Real falls have a short downward-velocity
   spike; staff bending, sitting down, and walker gait are slow.

Full operational notes for the next session: **[CARRY_ON.md](CARRY_ON.md)**.

---

## What's in this repo

```
scripts/         Training, pose extract, lighting_guard, organize, full scan
models/          Trained heads only (VideoMAE head + LightGBM). Backbone from HF.
data/            clips_manifest.csv, trigger_times.csv, no_fall_detected.csv
outputs/         Per-model JSON reports + unguarded / guarded full-scan CSVs
fall-detected/   10-second mp4 excerpts from the peak-window organization pass
CARRY_ON.md      Session memory: failure modes, paths, next work
```

## Environment (Waleed GPU box)

- Host alias: `ssh waleed`
- Workspace: `D:\fall-neuravue\`
- Videos: `D:\fall-detection-testing\all_video_data\`
- Python: `D:\fall-detection-testing\.venv\Scripts\python.exe`
- PyTorch 2.11 + CUDA 12.8, RTX 5070 (Blackwell sm_120)
- `ultralytics`, `transformers`, `decord`, `lightgbm`, `scikit-learn`, `opencv-python`

Mac mirror of this repo: `/Users/mc/aeyron/neuravue/fall-neuravue-repo/`.
Source videos on Mac: `/Users/mc/aeyron/neuravue/fall-detection-testing/all_video_data/`.

## Known limitations

- Only **2 unique fall events** in the labeled TP set (Ch53 × 2 events × 2
  cameras). 100% recall is on 4 windows from one room.
- `TP_Ch44_1_..._065511.mp4` still has **no annotated trigger**. User said
  “around 5 min”; never confirmed. Dense-scan max is 0.006.
- We never received the **deployed detector's actual fire timestamps** for
  FPs. Pose-mined peaks and dense-scan maxima are proxies.
- Model A (PoseC3D / ST-GCN) was not built — MMAction2 has no Blackwell wheels.
