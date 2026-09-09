# Carry-on memory — Neuravue fall-detection verifier

Read this at the start of the next session. Last updated 2026-09-10 (lighting guard shipped).

Repo: [waleedshoaib2/fall-neuravue](https://github.com/waleedshoaib2/fall-neuravue) (private).
Mac checkout: `/Users/mc/aeyron/neuravue/fall-neuravue-repo/`
Conversation workspace: `/Users/mc/aeyron/neuravue/fall-detection-testing/`
GPU box: `ssh waleed` (key auth works, no password). RTX 5070, Blackwell sm_120.

---

## What this project is

A **second-stage verifier**, not a from-scratch detector.

```
NVR → existing detector → candidate → lighting guard → Model B v2 → alarm if score ≥ 0.20
```

Goal: cut false positives from the deployed detector without dropping real falls.

Shipping model: **Model B v2** (`models/model_b_v2_head.pt`).
VideoMAE-base frozen backbone + ~200K MLP head. Input = 16 RGB frames @ 224×224.
Trained on Le2i Fall-vs-rest **plus 72 HB hard-negative peaks**. FP clips are holdout.

v2 scores are compressed vs v1. **0.20 in v2 ≈ 0.95 in v1.** Do not mix.

---

## Paths on Waleed

| What | Path |
|---|---|
| Scripts / models / outputs | `D:\fall-neuravue\` |
| Python | `D:\fall-detection-testing\.venv\Scripts\python.exe` |
| Videos | `D:\fall-detection-testing\all_video_data\` |
| Model B v2 head | `D:\fall-neuravue\outputs\models\model_b_v2_head.pt` |
| Pose (in-house v2, K=5 tracks) | `D:\fall-neuravue\outputs\pose\in_house_v2\` |
| Hard-neg pose peaks | `D:\fall-neuravue\outputs\pose\hard_negs\` |
| Le2i | `D:\fall-neuravue\data\external\le2i\raw\` |
| Dense-scan report | `D:\fall-neuravue\full_scan_report.csv` (copied to repo `outputs/`) |

Run inference:

```bash
ssh waleed 'D:\fall-detection-testing\.venv\Scripts\python.exe D:\fall-neuravue\scripts\full_scan_tp_fp.py'
```

---

## Data reality

| Bucket | Clips | Meaning |
|---|---|---|
| TP (`TrueFalls` / `NewTPRecords`) | 5 files, **2 unique events** | Ch53 event A (123330) × 2 cams, Ch53 event B (124000) × 2 cams, plus Ch44 unlabeled |
| FP (`FlasePositives` — folder name misspelled) | 13 | Deployed detector fired, labeled not-fall |
| HB (`NVR_manual_records`) | 24 | Background; detector never fired. Used as hard negatives |

All 1360×768. Total 42 clips.

### Annotated TP triggers (`data/trigger_times.csv`)

These were **corrected live** after the first timestamps pointed at aftermath / non-falls. Do not revert.

| Clip | trigger_s | Notes |
|---|---|---|
| `TP_Ch53_1_2026-08-26_123330.mp4` | 217 | Fall onto floor next to console; ~1.5 s collapse. Earlier voluntary lay-on-bed ~202 s is a hard negative in the *same* clip |
| `TP_Ch53_1_2026-08-26_124000.mp4` | 251 | Older man falls between staff; impact ~251–252 |
| `TP_Ch53_2_2026-08-26_123330.mp4` | 179 | Same event as Ch53_1_123330, opposite camera. Confirmed 2:55–3:03 |
| `TP_Ch53_2_2026-08-26_124000.mp4` | 297 | Same event as Ch53_1_124000, opposite cam. 3 people in frame |

**Still missing:** `TP_Ch44_1_2026-08-06_065511.mp4`. User said “around 5 min”. Dense-scan max = 0.006 at t=411 s. Need a real impact timestamp before using it as a positive.

---

## Two number sets (easy to confuse)

### 43-window holdout (used while training)

4 annotated TP windows + 39 pose-mined FP peaks.
Model B v2: in-house AP **0.917**, AUC **0.987**.
@ 0.20: 4/4 TP, 2/39 FP windows, 2/13 FP clips → 95% window / 85% clip FP reduction.

### Dense full-clip scan (honest)

Stride 2 s over every TP+FP clip.
- Unguarded baseline: `outputs/full_scan_report.csv` — 4/4 TP, 6/13 FP (54%).
- **With lighting guard: `outputs/full_scan_report_guarded.csv` — 4/4 TP (identical peaks/scores), 4/13 FP (69%).**
Ch44 miss. 24 HB clips were not in this dense scan (they were in the peak-window organize pass, all suppressed).

**When someone asks “does it work?”, quote the dense-scan numbers, not the 43-window ones.**

---

## Detected table @ 0.20 (dense scan)

| Nature | File | Peak | Score | Notes |
|---|---|---|---|---|
| TP | Ch53_1_124000 | 04:11 / 251 s | 0.976 | unchanged |
| TP | Ch53_2_124000 | 03:33 / 213 s | 0.899 | unchanged |
| TP | Ch53_1_123330 | 03:27 / 207 s | 0.853 | unchanged |
| TP | Ch53_2_123330 | 02:43 / 163 s | 0.454 | unchanged |
| FP | Ch56_1_181000 | 02:13 / 133 s | **0.959** | walker |
| FP | Ch58_2_060559 | 00:25 / 25 s | 0.578 | staff bend |
| FP | Ch58_2_191724 | 04:09 / 249 s | 0.529 | leftover sit-on-bed IR; lights-on @51 killed |
| FP | Ch48_1_031559 | 09:15 / 555 s | 0.327 | staff bend |
| FP dead | Ch58_2_014952 | — | 0.791 → 0.185 | lighting guard |
| FP dead | Ch58_1_190523 | — | 0.485 → 0.177 | lighting guard |

At **threshold 0.80 + guard**: 3/4 TP + **only the walker**. Lights-on 0.982 is gone.

`Ch53_2_124000`: model peak is **84 s before** the annotated impact (213 vs 297). Visual-check whether that earlier spike is pre-fall motion or a different event.

---

## Failure modes to carry forward

### 1. Sudden lighting / IR ↔ colour — DONE (2026-09-10)

`scripts/lighting_guard.py`: ±5 s context at 8 fps. Reject if luma range > 30 or jump > 18.
The 16-frame model window is only ~0.6 s, so the peak is often *after* the swap
(e.g. 014952 peak at 1211, blackout at 1206). Do not shrink the lookaround below ~4 s.

Verified: 4/4 TP windows untouched. Killed 014952 and 190523. On 191724 the
t=51 lights-on (0.982) is gone; leftover 0.529@249 is someone **sitting on the
bed edge in IR** (grid: `fp_check/ch58_2_p249_grid.jpg`), not another mode swap.

Do not retune the guard to kill t=249 — that would start eating sit-to-bed
and maybe real falls. That leftover is Fix 2/3 material.

### 2. Staff bending over a patient in bed

| Clip | Peak | Score |
|---|---|---|
| `FP_Ch58_2_2026-08-06_060559` | 25 s | 0.578 |
| `FP_Ch48_1_2026-08-05_031559` | 555 s | 0.327 |

Slow motion, low vertical velocity — pose miner missed these so they never entered HB training negatives.

### 3. Elderly walking with walker (stooped)

| Clip | Peak | Score |
|---|---|---|
| `FP_Ch56_1_2026-08-05_181000` | 133 s | **0.959** |

Looks like a post-fall crouch to a pixel model. Survives even threshold 0.80.

### 4. Sitting on bed edge in IR (new leftover after the lighting guard)

| Clip | Peak | Score |
|---|---|---|
| `FP_Ch58_2_2026-08-05_191724` | 249 s | 0.529 |

Same clip that used to be the 0.982 lights-on FP. Treat as a sit-down / hunched-posture negative, same bucket as staff-bending.

**Fix 2:** mine targeted negatives from *other* HB/normal footage (not the 13 FP holdout clips): ~50 staff-bending / sit-on-bed, ~20 walker. Retrain head only, ~10 min.

**Fix 3:** hybrid gate — after VideoMAE fires, require YOLO-pose peak vertical velocity above a threshold. Kills slow “bending”, “sitting down”, and “walker” without retraining.

---

## What HB peaks are

`HB_*.mp4` = 24 clips the deployed detector never fired on.
`mine_hard_negatives.py` ran YOLO11m-pose at 5 fps, scored vertical velocity + aspect-ratio range, took top-3 non-overlapping 3 s windows per clip → **72 hard negatives**.

They improved v1→v2 (FP-window filter 90%→95% on the 43-window test).
They **cannot** catch lighting transitions (no pose signal) or slow staff-bending / walker gait (low velocity). That is why those three modes survived.

---

## Do not do these

- Do **not** train on the 13 FP clips. They are the only in-house negative holdout. Even leave-one-clip-out leaks scene statistics.
- Do **not** unfreeze the VideoMAE backbone on 764 samples.
- Do **not** quote 43-window 85% FP-clip reduction as the production number. Dense scan is **69% at 0.20 with the lighting guard** (54% without).
- Do **not** mix v1 thresholds (0.95) with v2 (0.20).
- Do **not** commit `all_video_data/*.mp4` (patients/staff).
- Do **not** revert the four corrected TP timestamps.
- Model A (PoseC3D/ST-GCN via MMAction2) is blocked: no Blackwell wheels for current PyTorch. Not needed unless VideoMAE stalls.

---

## Next session checklist

1. ~~Lighting luminance guard~~ **done**. Next: mine staff-bending / sit-on-bed + walker negatives from HB (not from FP holdout). Retrain Model B v2 head → `model_b_v3_head.pt`. Keep v2 as backup (`verifier/outputs/models_backup/` on Mac already has v1 and v2).
2. Get a confirmed Ch44 trigger (user: ~5 min). Add to `trigger_times.csv`. Re-extract pose window. Re-run dense scan on that clip.
3. Optionally add the pose-velocity gate for remaining slow FPs (walker 0.959, staff bend, sit-on-bed 0.529).
4. More TP annotations from other rooms (Ch44, Ch58, …) before trusting recall.
5. If labeling at scale: a review UI / active-learning loop. Not started.

---

## Model C (kept as interpretable baseline, not shipping)

LightGBM on 22 hand-crafted skeleton features from YOLO11m-pose (aspect ratio, torso angle, vertical velocity, hip height, stillness, det confidence).
In-house AP 0.377 / AUC 0.744. Misses both staff-occluded `_124000` falls because track selection picked staff.
Weights: `models/model_c_v3_lightgbm.txt`. Ensemble with B did not help.

Pose extractor: YOLO11m-pose, COCO 17 kpts. In-house v2 stores **top-5 tracks per frame** over a 48-frame window. Track picker = movement score (vertical velocity + AR change), not “biggest AR range” (that heuristic picked staff).

---

## Misc that bit us once

- `cap.set(cv2.CAP_PROP_POS_FRAMES)` is slow on these files; use `cap.grab()` when skipping.
- Folder is literally named `FlasePositives`.
- First `events.csv` under `_fall_scan/` was **not** the deployed detector log. Do not treat it as fire times.
- Multi-person scenes need K=5 pose tracks; K=1 is wrong.
- Lighting QA grids: `verifier/outputs/fp_check/` on the Mac workspace.
- Git identity used for this repo: `waleedshoaib2` / `87224462+waleedshoaib2@users.noreply.github.com`.
- `gh` has been authed as both `DeveloperAeyron` and `waleedshoaib2`; repo is under `waleedshoaib2`.
