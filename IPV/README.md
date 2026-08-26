# IPV

Image-Patch Voting (IPV) landmark localisation with repeated k-fold model training, comparison-ready validation outputs, and standalone full-image inference.

IPV trains a multi-scale patch classifier with one distance head and one angle head per landmark. Validation grid predictions are converted into endpoint vote maps, so the reported `validation_error_px` is directly comparable with Heatmaps validation error when both methods use the same repeated-fold collection and source images. Classification losses and accuracies remain IPV-specific and must not be compared with heatmap-regression losses.

## Installation

From the outer `IPV` directory:

```bash
pip install -e .
```

This installs:

```text
ipv-train
ipv-infer
```

PyTorch and torchvision must be installed with builds suitable for the selected CPU/CUDA environment.

## Input data

A run requires:

1. a source-image directory;
2. a mark-list file;
3. repeated k-fold training and validation lists;
4. patch sampling and model settings.

Each mark-list row contains an image name followed by exactly `--num-points` coordinates in `(x, y)` order:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

Selected annotations, image paths, coordinate bounds, channels and all generated patch groups are validated before existing generated data or training outputs for the selected repetition/fold are removed.

## Repeated k-fold lists

IPV uses the same list structure as Heatmaps:

```text
folds/
  repeated_kfold_summary.csv
  repeated_kfold_membership.csv
  test_cases.xlsx                 # only for a configured fixed test cohort
  repetition_1/
    training_f1.txt
    val_f1.txt
    training_f2.txt
    val_f2.txt
  repetition_2/
    training_f1.txt
    val_f1.txt
```

The complete collection is checked for contiguous repetition and fold numbers, paired files, duplicates, split overlap, complete dataset coverage, complementary training/validation membership, validation exactly once per sample per repetition, and the same eligible dataset in every repetition.

Generate lists by editing the settings at the top of `IPV/utils/generate_folds.py` and running:

```bash
python -m IPV.utils.generate_folds
```

`TEST_SAMPLE_IDS` reserves one fixed external test cohort before cross-validation. Those cases are recorded in `test_cases.xlsx`; they are not written into fold-specific test lists. Use `ipv-infer` later to evaluate the trained model on that cohort.

## Training command

The command format is:

```text
ipv-train REPETITION FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Example:

```bash
ipv-train 1 1 prostate_transverse true true false false \
    --run-dir "$HOME/IPV_TRAINING" \
    --num-points 4 \
    --fold-lists-path "$HOME/DATA/folds" \
    --mark-list-file "$HOME/DATA/transverse_points_list.txt" \
    --image-data-dir "$HOME/DATA/transverse" \
    --data-creation-workers 8 \
    --train-workers 8 \
    --random-seed 42 \
    --keep-part-csvs false \
    --patches-per-training-sample 200 \
    --val-grid-spacing 10 \
    --batch-size 64 \
    --max-training-epochs 80 \
    --learning-rate 0.001 \
    --optimiser-name adamw \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --lr-schedule plateau \
    --lr-step-size 20 \
    --lr-gamma 0.5 \
    --early-stop-patience 15 \
    --early-stop-min-delta 0.0001 \
    --early-stop-warmup-epochs 10 \
    --use-amp false \
    --network-name small_cnn \
    --branch-features 128 \
    --frozen-stages 0 \
    --small-input-stem false
```

The supplied `run_pipeline.ps1` and `run_pipeline.sh` expose every commonly changed option as a top-level variable.

## Training behaviour

IPV performs one complete training pass and one complete validation pass per epoch. The validation pass reports:

- mean cross-entropy loss across distance/angle heads;
- mean classification accuracy across heads;
- mean endpoint localisation error in original-image pixels after grid voting and smoothing.

The endpoint error is the appropriate shared outcome for comparison with Heatmaps. Raw loss is internal to one model because IPV classification and heatmap regression optimise different objectives.

Python, NumPy, PyTorch, CUDA and DataLoader workers are deterministically seeded. Deterministic PyTorch algorithms are enabled, cuDNN benchmarking and TF32 are disabled, and runtime versions are recorded for audit.

### Optimisers and schedules

```text
--optimiser-name adamw|sgd
--weight-decay
--momentum
--lr-schedule none|step|plateau
--lr-step-size
--lr-gamma
```

The plateau scheduler and early stopping use validation classification loss. Best-checkpoint selection uses the lowest validation loss; endpoint error is recorded alongside it.

### Resuming

Set:

```text
--resume-training true
```

with `CREATE_DATA=false` and `TRAIN_MODEL=true`. Resumption uses only:

```text
TRAINING_RESULTS/TASK_NAME/RUN_NAME/repetition_N/fold_N/model_last_epoch.pth
```

The checkpoint restores model, optimiser, scheduler, AMP scaler, history, best/early-stopping state, random-number states and DataLoader generator states. A compatibility signature covers generated training/validation CSVs, fold membership, all IPV Python sources, configuration and relevant runtime versions. Incompatible continuation is rejected.

## Output layout

Generated patch data is isolated beneath:

```text
RUN_DIR/TRAINING_DATA/TASK_NAME/REPETITIONS_Repetitions_FOLDS_Folds/
  PATCH_CONFIGURATION/repetition_N/fold_N/
```

Training outputs match the Heatmaps repetition/fold hierarchy:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/repetition_N/fold_N/
  model_best_validation_loss.pth
  model_last_epoch.pth
  validation_checkpoint_summary.json
  training_validation_log.csv
  training_validation_plot.png
  run_info.json
  validation_results/
    validation_summary.xlsx
    validation_image_summary.csv
    validation_endpoints.csv
    validation_predictions.csv
    validation_heatmap_overlays/
    validation_point_overlays/
    validation_vote_maps/
    validation_logs/
      validation_run_metadata.json
```

Every validation CSV row identifies `dataset_split`, `repetition` and `fold`. Common endpoint columns are `target_x`, `target_y`, `pred_x`, `pred_y` and `error_px`. IPV-specific aliases and diagnostics such as classification accuracy, vote peak, grid spacing and number of centres are retained.

The training log contains one row per completed epoch:

```text
epoch
epoch_started_at
epoch_completed_at
lr
training_loss
training_accuracy
validation_loss
validation_accuracy
validation_error_px
training_duration_seconds
validation_duration_seconds
epoch_duration_seconds
```

Automatic run names contain the repetition/fold counts, data and model labels, plus a configuration fingerprint. The fingerprint includes the SHA-256 digest of every active split list and every result-affecting training, validation and model option.

## Standalone inference

Edit the paths and switches in `IPV/infer_landmarks.py`, then run:

```bash
ipv-infer
```

Standalone inference remains independent of fold-list generation. It reconstructs the model from checkpoint metadata, accepts one image or an image directory, optionally reads ground truth from a mark list, and saves vote maps, heatmap overlays, endpoint overlays, an Excel summary and run metadata.

## Tests

From the outer `IPV` directory:

```bash
python -m unittest discover -s tests -v
```

Tests cover deterministic repeated k-fold membership, held-out test exclusion, fold-collection fingerprints, isolated output leaves, common validation schemas, epoch-history fields and atomic checkpoint replacement.
