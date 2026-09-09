# fall-neuravue

Two-stage fall-detection **verifier** for the Neuravue NVR system.
Reduces false alarms from the deployed detector by scoring each candidate
window with a fine-tuned VideoMAE model.

## Result on the labeled test set

**Model B v2** (VideoMAE-base pretrained on Kinetics-400, head fine-tuned on
Le2i + HB hard negatives) evaluated on 43 candidate windows
(4 human-annotated true-positive falls + 39 hard-mined FP-clip peaks).

| Threshold | Recall | FP window reduction | FP clip reduction |
|---|---|---|---|
| **0.24** | **4/4 (100%)** | 37/39 (95%) | 12/13 (92%) |
| 0.30 | 3/4 (75%) | 39/39 (100%) | 13/13 (100%) |

Full clip-level scan on all 42 available clips (`scripts/organize_by_detection.py`)
detected 4 of 5 known TP clips and 1 of 13 FP clips, correctly rejecting
all 24 unlabeled HB (background) clips. See `data/no_fall_detected.csv` and
`fall-detected/_detections.csv`.

## What's in this repo

```
scripts/    Training and inference scripts
models/     Trained heads (VideoMAE head + LightGBM booster).
            The 87M VideoMAE backbone loads from Hugging Face at runtime.
data/       Manifests and human-annotated trigger times.
outputs/    Per-model evaluation reports (JSON).
fall-detected/
            Detected windows extracted from the source clips as short mp4s.
            _detections.csv summarizes what was detected and where.
```

Videos (`all_video_data/*.mp4`) are excluded — they contain identifiable
people and are kept off Git.

## Pipeline overview

1. **Pose extraction** (`scripts/extract_pose_*.py`)
   YOLO11m-pose (COCO 17 keypoints) run on labeled snippets and mined peak
   windows.
2. **Hard-negative mining** (`scripts/mine_hard_negatives.py`)
   For each FP / background clip, run pose at 5 fps end-to-end and locate the
   top-K most fall-like 3-second windows.
3. **Model C — LightGBM** (`scripts/train_model_c_*.py`)
   Gradient-boosting on hand-crafted skeleton features. Interpretable baseline.
4. **Model B — VideoMAE** (`scripts/train_model_b_videomae.py`,
   `scripts/train_model_b_v2.py`)
   Freeze VideoMAE-base backbone, fine-tune a small MLP head. This is the
   shipping model.
5. **Full-clip detection & organization** (`scripts/organize_by_detection.py`)
   Score every clip's candidate windows with Model B v2. Extract detections to
   `fall-detected/`, write `no_fall_detected.csv` for the rest.

## Environment

- Python 3.11
- PyTorch 2.11 + CUDA 12.8 (Blackwell / RTX 5070)
- `ultralytics`, `transformers`, `decord`, `lightgbm`, `scikit-learn`,
  `opencv-python`

## Known limitations

- Only 4 unique fall events in the labeled TP set (2 falls × 2 cameras). Recall
  numbers are conditional on this small sample.
- `TP_Ch44_1_..._065511.mp4` has no annotated trigger yet, so the full-clip
  scan misses it — the model was never told where to look.
- One FP clip (`FP_Ch58_2_..._191724`) still fires above threshold; it contains
  a person hunched at a bed edge under IR night vision that Le2i's training
  data doesn't cover.
