# Heatmaps

Heatmap-regression landmark localisation package for the `LandmarkIdentificationFrameworks/Heatmaps` subdirectory.

This package is intended to sit alongside the IPV and Detection packages. It uses the same fold-list and mark-list idea as the IPV package, but it does not create patch CSVs or patch image folders. Images are loaded directly during training and converted into Gaussian landmark heatmaps on demand.

## Package layout

```text
Heatmaps/
  pyproject.toml
  README.md
  run_pipeline.ps1
  run_pipeline.sh
  Heatmaps/
    __init__.py
    custom_dataset.py
    heatmap_training_pipeline.py
    model_registry.py
    models.py
    parameters.py
    train_model.py
    utils/
      __init__.py
      io_utils.py
      progress_bar.py
      visualisation_utils.py
```

## Install

From inside this `Heatmaps` directory:

```bash
pip install -e .
```

## Expected input files

Fold lists should use the same naming style as the IPV package:

```text
folds/
  train_f1.txt
  val_f1.txt
  train_f2.txt
  val_f2.txt
  ...
```

The mark-list file should contain one image and its landmark coordinates per line, for example:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

The fold-list entries can be stems such as `A1` or filenames such as `A1.jpg`.

## Train from the command line

Example transverse prostate run:

```bash
heatmaps-train 1 prostate_transverse true false false \
    --run-dir "$HOME/HEATMAP_TRAINING" \
    --run-name "unet_basic" \
    --num-points 4 \
    --fold-lists-path "$HOME/DATA/folds" \
    --mark-list-file "$HOME/DATA/doctors_resampled_transverseMarkList.txt" \
    --image-data-dir "$HOME/DATA/TRANSVERSE" \
    --image-size 512 512 \
    --heatmap-sigma 8 \
    --input-channels 1 \
    --batch-size 4 \
    --learning-rate 0.001 \
    --max-training-epochs 80
```

For sagittal prostate images, change `--num-points 2` and use the sagittal mark list and image directory.

## Outputs

Outputs are written to:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/
```

The core outputs are:

```text
model_f1_best.pth
model_f1_last.pth
checkpoint_summary_f1.json
train_log_f1.csv
train_plot_f1.png
validation_predictions_f1.csv
```

If `--save-validation-overlays true` is used, validation endpoint and heatmap overlay images are also saved.

## Model registry

The current registry contains one model:

```text
unet_basic
```

Additional models can be added later in `model_registry.py` and `models.py` without changing the training pipeline.
