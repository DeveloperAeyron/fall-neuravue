# fall-neuravue

Two-stage fall-detection **verifier** for the Neuravue NVR system.
The deployed detector is over-eager. This repo scores each candidate window
and only keeps alarms that still look like a fall.

```
NVR feed → [existing detector] → candidate window
                                    │
                                    ▼
                         [Model B v2 verifier]
                                    │
                          score ≥ threshold?
                           yes          no
                            │            │
                         ALARM       suppress
```

**Shipping model:** Model B v2 (VideoMAE-base, frozen backbone, fine-tuned
MLP head) at **threshold 0.20**.

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
score a 16-frame window, take the max. Every FP clip gets hundreds of chances
to fool the model. Raw report: `outputs/full_scan_report.csv`.

**Headline at threshold 0.20:** 4/4 annotated TPs caught, **6/13 FP clips
still fire** (54% FP-clip reduction), Ch44 missed (no trigger, max score 0.006).

---

## Detected clips — dense scan @ threshold 0.20

This is the table that matters for “what would fire in production.”
Peak time = second inside the source clip where Model B v2 scored highest.

| # | Nature | Filename | Peak (mm:ss) | Peak (s) | Score |
|---|---|---|---|---|---|
| 1 | **TP** | `TP_Ch53_1_2026-08-26_124000.mp4` | **04:11** | 251 | 0.976 |
| 2 | **TP** | `TP_Ch53_2_2026-08-26_124000.mp4` | **03:33** | 213 | 0.899 |
| 3 | **TP** | `TP_Ch53_1_2026-08-26_123330.mp4` | **03:27** | 207 | 0.853 |
| 4 | **TP** | `TP_Ch53_2_2026-08-26_123330.mp4` | **02:43** | 163 | 0.454 |
| 5 | FP | `FP_Ch58_2_2026-08-05_191724.mp4` | **00:51** | 51 | 0.982 |
| 6 | FP | `FP_Ch56_1_2026-08-05_181000.mp4` | **02:13** | 133 | 0.959 |
| 7 | FP | `FP_Ch58_2_2026-08-06_014952.mp4` | **20:11** | 1211 | 0.791 |
| 8 | FP | `FP_Ch58_2_2026-08-06_060559.mp4` | **00:25** | 25 | 0.578 |
| 9 | FP | `FP_Ch58_1_2026-08-04_190523.mp4` | **11:23** | 683 | 0.485 |
| 10 | FP | `FP_Ch48_1_2026-08-05_031559.mp4` | **09:15** | 555 | 0.327 |

**4 TP detected, 6 FP survived.** Not listed: 1 TP miss
(`TP_Ch44_1_2026-08-06_065511.mp4`, max 0.006, no annotated trigger) and 7 FPs
correctly suppressed (all max < 0.20). All 24 HB background clips stayed below
threshold on the peak-window organization pass.

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
| **0.20** | **4/4 (100%)** | 2/2 | 7/13 (54%) | 4 TP + 6 FP = 10 | 40% |
| **0.80** | **3/4 (75%)** | **2/2** | **11/13 (85%)** | 3 TP + 2 FP = 5 | **60%** |

At 0.80 the dropped TP is `Ch53_2_123330` (0.454). That is the **same fall** as
`Ch53_2_124000` from the other camera (0.899), so unique-event recall stays
100% if both cameras are live. The two FPs that still survive 0.80 are the
lights-on artefact (`Ch58_2_191724`, 0.982) and the walker (`Ch56_1_181000`, 0.959).

---

## Known failure modes (visual QA of the 6 surviving FPs)

Only **three** distinct modes. Half of them are lighting, not people.

| # | Clip | Score | Failure mode |
|---|---|---|---|
| 1 | `FP_Ch58_2_191724` | **0.982** | **Lights turn ON** at night (IR → colour). Global brightness jump looks like sudden motion. |
| 2 | `FP_Ch56_1_181000` | **0.959** | Elderly person **walking with a walker** — stooped posture reads as post-fall. |
| 3 | `FP_Ch58_2_014952` | 0.791 | **Lights-on transition** + staff enters to tend a sleeping patient. |
| 4 | `FP_Ch58_2_060559` | 0.578 | **Staff bending over a patient in bed.** |
| 5 | `FP_Ch58_1_190523` | 0.485 | **Lights turn OFF** + hunched sitting on the bed edge. |
| 6 | `FP_Ch48_1_031559` | 0.327 | **Staff bending over a patient in bed.** |

| Mode | Cases | Share of survivors |
|---|---|---|
| Lighting transitions (IR ↔ colour) | #1, #3, #5 | 50% |
| Staff bending over patient | #4, #6 | 33% |
| Elderly walking with walker | #2 | 17% |

**Canonical lighting example:** `FP_Ch58_2_191724` around t=50 s. t=42–49 dim
IR, someone lying still; **t=50 lights slam on** and the camera switches to
colour; t=51–61 someone moves near the head of the bed. VideoMAE score 0.982.
20 of 413 windows in that clip scored ≥ 0.20 because of lighting flicker, not
a fall.

### Next fixes (in this order)

1. **Lighting-transition guard (no retrain).** Reject a window if global
   luminance range across the 16 frames is huge, e.g.
   `(mean_lum.max() - mean_lum.min()) > 25` on 0–255. Should kill 3/6
   surviving FPs (dense-scan FP-clip suppression 54% → ~77%). Zero risk to
   the four annotated TPs (none of them are IR↔colour swaps).
2. **Targeted negatives + retrain the head.** Mine ~50 staff-bending-over-bed
   windows and ~20 walker windows from HB/normal footage. Retrain Model B v2
   head (~10 min). Do **not** train on the 13 FP clips — they are the holdout.
3. **Pose vertical-velocity gate.** Real falls have a short downward-velocity
   spike; staff bending and walker gait are slow. After VideoMAE fires, require
   a pose-track peak vertical velocity above a threshold.

Full operational notes for the next session: **[CARRY_ON.md](CARRY_ON.md)**.

---

## What's in this repo

```
scripts/         Training, pose extract, hard-neg mining, organize, full scan
models/          Trained heads only (VideoMAE head + LightGBM). Backbone from HF.
data/            clips_manifest.csv, trigger_times.csv, no_fall_detected.csv
outputs/         Per-model JSON reports + full_scan_report.csv
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
