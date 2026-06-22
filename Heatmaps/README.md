# Heatmaps

Heatmap-regression landmark localisation package for the `LandmarkIdentificationFrameworks/Heatmaps` subdirectory.

This package is intended to sit alongside the IPV and Detection packages. It trains convolutional neural networks to predict one 
heatmap per landmark from complete input images. Source images are loaded directly during training,
and Gaussian target heatmaps are generated from the supplied landmark coordinates when each sample is requested.

**Package version:** `0.1.0`

## Current scope

The package currently provides:

- fold-based model training and validation;
- configurable U-Net heatmap regression;
- landmark-preserving image augmentation;
- automatic input-channel detection;
- best and last model checkpoints;
- validation metrics, predictions, heatmap overlays, and point overlays.

Standalone held-out test evaluation and general-purpose inference are not yet included.

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
    heatmap_transforms.py
    model_registry.py
    models.py
    parameters.py
    train_model.py
    utils/
      __init__.py
      calculate_image_size.py
      generate_folds.py
      io_utils.py
      progress_bar.py
      verify_transforms.py
      visualisation_utils.py
```

## Installation

From inside the repository's `Heatmaps` directory:

```bash
pip install -e .
```

This installs the training command:

```text
heatmaps-train
```

Check the available arguments with:

```bash
heatmaps-train --help
```

## Input data

A training run requires:

1. a source-image directory;
2. a landmark mark-list file;
3. train, validation, and test sample lists for each fold.

### Fold lists

Fold files must use the following names:

```text
folds/
  train_f1.txt
  val_f1.txt
  test_f1.txt
  train_f2.txt
  val_f2.txt
  test_f2.txt
  ...
```

Each file contains one sample identifier per line. Entries may be image stems such as `A1` or filenames such as `A1.jpg`.

All three files are required for each fold so that the fold definition is complete and can be validated. The current training workflow uses the training and validation lists; the test lists are reserved for later held-out evaluation.

### Landmark mark list

The mark-list file contains an image name followed by its landmark coordinates:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

Coordinates use `(x, y)` order, where `x` is the horizontal image coordinate and `y` is the vertical image coordinate. Every sample must contain at least the number of points specified by `--num-points`.

Before training, every landmark is checked against its resolved source image. For an image with width `W` and height `H`, each point must satisfy:

```text
0 <= x < W
0 <= y < H
```

The run stops if a point is outside the image. The error identifies the sample, point index, coordinate, and valid image bounds.

### Source images

The source-image directory may contain greyscale, RGB, or RGBA images. Images can be found recursively by setting:

```text
--recursive-image-search true
```

All images used by one task must have a consistent source channel count.

## Creating fold lists

A fold-generation utility is provided at:

```text
Heatmaps/utils/generate_folds.py
```

Edit its top-level paths and switches, then run:

```bash
python -m Heatmaps.utils.generate_folds
```

The utility creates deterministic train, validation, and test lists together with optional summary and membership CSV files.

## Choosing an image size

`--image-size HEIGHT WIDTH` is required. Every image and landmark set is resized into this common training coordinate system. Each dimension must be at least `2 ** depth` so that every U-Net downsampling level remains valid.

A helper utility can calculate the average dimensions of a source-image directory. Edit the top-level `IMAGE_DATA_DIR` in:

```text
Heatmaps/utils/calculate_image_size.py
```

Then run:

```bash
python -m Heatmaps.utils.calculate_image_size
```

The utility prints the average and rounded image dimensions and a ready-to-use `--image-size HEIGHT WIDTH` argument.

## Training

The command format is:

```text
heatmaps-train FOLD TASK_NAME TRAIN_MODEL COPY_FILES [OPTIONS]
```

A transverse prostate example is:

```bash
heatmaps-train 1 prostate_transverse true false \
    --run-dir "$HOME/HEATMAP_TRAINING" \
    --save-dir "$HOME/HEATMAP_SAVING" \
    --num-points 4 \
    --fold-lists-path "$HOME/DATA/folds" \
    --mark-list-file "$HOME/DATA/doctors_resampled_transverseMarkList.txt" \
    --image-data-dir "$HOME/DATA/TRANSVERSE" \
    --image-size 512 512 \
    --heatmap-sigma 8 \
    --oversampling-factor 1 \
    --batch-size 4 \
    --learning-rate 0.001 \
    --max-training-epochs 80
```

For sagittal prostate images, use `--num-points 2` together with the corresponding sagittal mark list and image directory.

The supplied `run_pipeline.sh` and `run_pipeline.ps1` files provide editable Linux/macOS/HPC and Windows PowerShell examples. Paths and training settings are exposed as top-level variables.

## Data and heatmap settings

The main data options are:

```text
--num-points
--fold-lists-path
--mark-list-file
--image-data-dir
--image-size HEIGHT WIDTH
--heatmap-sigma
--oversampling-factor
--recursive-image-search
```

`--heatmap-sigma` controls the Gaussian spread of each target landmark heatmap in the resized training image.

`--oversampling-factor` affects the training split only:

- `1` uses each training sample once without augmentation;
- values greater than `1` add augmented passes through the training samples;
- validation samples are never oversampled or augmented.

## Augmentation

The default augmentation policy is defined in:

```text
Heatmaps/heatmap_transforms.py
```

The transform chain is:

```text
RandomAffine
GaussianNoise
GaussianBlur
```

The affine transform is applied to both the image and its landmark coordinates. Intensity transforms do not move landmarks.

`RandomErasing` is available for optional experiments but is not enabled by default.

The transform settings are stored in checkpoint metadata so that the training configuration can be reviewed later.

## Input channels

Input channels are detected automatically from the selected fold's training and validation images. There is no input-channel command-line argument.

| Source images | Model input channels |
|---|---:|
| Greyscale | 1 |
| RGB | 3 |
| RGBA | 4 |

The resolved channel count configures the first U-Net layer and is written to the final run metadata and checkpoints. A run stops if the training and validation images do not use a consistent channel count.

## Optimisation settings

The main optimisation options are:

```text
--batch-size
--learning-rate
--max-training-epochs
--train-workers
--random-seed
--optimiser-name adamw|sgd
--loss-name mse|weighted_mse|smooth_l1|bce_logits
--positive-weight
--weight-decay
--momentum
--lr-schedule none|step|plateau
--lr-step-size
--lr-gamma
--early-stop-patience
--early-stop-min-delta
--early-stop-warmup-epochs
--use-amp
```

The default loss is `weighted_mse`. `--positive-weight` increases the contribution of the landmark peak regions relative to the heatmap background.

Random seeds are applied to Python, NumPy, PyTorch, and DataLoader workers.

## Model settings

The current model registry contains:

```text
unet_basic
```

The main model options are:

```text
--network-name unet_basic
--base-channels
--depth
--channel-multiplier
--max-channels
--normalisation batch|instance|group|none
--activation relu|leaky_relu|elu|gelu
--dropout
--upsampling bilinear|transpose
--output-activation none|sigmoid|softplus
--padding-mode zeros|reflect|replicate|circular
--final-kernel-size 1|3
```

The network returns one output heatmap for each configured landmark. Additional architectures can be registered in `model_registry.py` and implemented in `models.py` without changing the external training command.

## Validation outputs

Validation export is enabled by default:

```text
--save-validation-predictions true
```

When enabled, validation predictions, metrics, heatmap overlays, and point overlays are all saved automatically. Set it to `false` to skip the complete validation export. After training, the selected checkpoint is reloaded and evaluated on the validation images. Predictions are converted from heatmap peaks into original-image coordinates before endpoint errors are calculated.

## Output structure

Before training a fold, existing outputs for that same fold are removed from the run directory and the cleanup is printed to the terminal. Outputs belonging to other folds are retained.

Training outputs are written to:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/
```

Typical outputs include:

```text
model_f1_best.pth
model_f1_last.pth
checkpoint_summary_f1.json
train_log_f1.csv
train_plot_f1.png
run_info_TASK_NAME_f1.json
validation_results_F1/
  validation_summary.xlsx
  validation_image_summary.csv
  validation_endpoints.csv
  validation_predictions_f1.csv
  logs/
    validation_run_metadata.json
```

The validation directory also contains:

```text
heatmap_overlays/
point_overlays/
```

The training plot contains loss curves and mean endpoint-error curves. The validation workbook contains `image_summary` and `endpoints` sheets. Point overlays label the ground-truth and predicted endpoints, while heatmap overlays show the model response over the source image.

If `COPY_FILES` is `true`, the completed run directory is copied to:

```text
SAVE_DIR/TASK_NAME/RUN_NAME/
```

## Checkpoints and metadata

Both best and last checkpoints are saved. Checkpoint metadata includes:

```text
schema
schema_version
checkpoint
task
model
data
preprocessing
inference
augmentation
training
raw_configs
```

The metadata records the model reconstruction arguments, landmark count, input channels, target image size, heatmap sigma, preprocessing information, augmentation policy, training settings, and checkpoint metrics.

`run_info_TASK_NAME_fN.json` contains the resolved run, data, model, and training configuration. It is rewritten after input-channel detection so that the stored channel count reflects the model that was actually trained.

## Run names

`--network-name` selects the architecture. `--run-name` optionally overrides the output folder name.

When `--run-name` is omitted, a deterministic name is built from the fold count, point count, model, image size, heatmap sigma, U-Net settings, loss settings, oversampling factor, batch size, learning rate, and epoch count.

## Transform inspection

Individual transforms can be inspected visually with:

```bash
python -m Heatmaps.utils.verify_transforms /path/to/images /path/to/points.txt default
```

This is useful for confirming that an augmentation policy remains appropriate for a new anatomy, imaging modality, or landmark definition.

