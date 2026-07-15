# Heatmaps

Full-image heatmap-regression landmark localisation for the `LandmarkIdentificationFrameworks/Heatmaps` package.

The package trains a convolutional neural network to produce one heatmap per landmark. Source images are loaded directly, resized into a common training coordinate system, and paired with Gaussian target heatmaps generated from the supplied landmark coordinates.

**Package version:** `0.2.0`

## Current scope

The package currently provides:

- fold-based training and validation;
- a configurable U-Net heatmap regressor;
- landmark-preserving image augmentation;
- automatic greyscale, RGB, or RGBA input-channel detection;
- deterministic seeding for Python, NumPy, PyTorch, and DataLoader workers;
- best and last checkpoints with reconstruction metadata;
- validation predictions, endpoint metrics, heatmap overlays, and point overlays;
- optional copying of a completed run to a separate save directory.

Standalone held-out test evaluation, checkpoint resumption, and general-purpose inference are not yet included. Fold test lists are still required so that each fold definition can be checked for completeness and overlap.

The code targets Python 3.10 or later and PyTorch 2.4 or later. It does not contain legacy PyTorch-loading fallbacks or older command aliases.

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

From the repository's outer `Heatmaps` directory:

```bash
pip install -e .
```

This installs:

```text
heatmaps-train
```

Display the complete command-line interface with:

```bash
heatmaps-train --help
```

## Input data

A training run requires:

1. a source-image directory;
2. a landmark mark-list file;
3. train, validation, and test sample lists for each fold;
4. a common training image size.

### Fold lists

Fold files must use these names:

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

Each file contains one sample identifier per line. Entries may be stems such as `A1` or filenames such as `A1.jpg`.

Before training, the selected fold is checked for:

- duplicate sample identifiers within a split;
- overlap between training, validation, and test splits;
- the presence of all three split files.

The training workflow currently reads the training and validation lists. Test lists are reserved for later held-out evaluation.

### Landmark mark list

Each mark-list row contains an image name followed by landmark coordinates:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

Coordinates use `(x, y)` order, where `x` is horizontal and `y` is vertical. Every row must contain at least the number of points specified by `--num-points`. Additional points on a row are ignored.

Every selected landmark is checked against the resolved source image. For an image with width `W` and height `H`, each point must satisfy:

```text
0 <= x < W
0 <= y < H
```

The run stops with the sample, point number, coordinate, image path, and valid bounds when a landmark is invalid.

### Source images

Supported image suffixes are:

```text
.png .jpg .jpeg .bmp .tif .tiff
```

Images may be greyscale, RGB, or RGBA. The selected fold's training and validation images must all have the same source channel count.

Integer images are converted to `float32` in the `0` to `1` range. Floating-point source images must already contain finite values within that range; the run stops if NaN, infinity, or out-of-range values are found.

Images are searched directly beneath `--image-data-dir` by default. Enable recursive searching with:

```text
--recursive-image-search true
```

An exact mark-list filename is preferred. If fallback stem matching finds more than one possible image, the run stops instead of selecting one silently.

## Creating fold lists

The utility at:

```text
Heatmaps/utils/generate_folds.py
```

creates deterministic five-fold train, validation, and test lists with approximate 80/10/10 splits. Edit the top-level paths and switches, then run:

```bash
python -m Heatmaps.utils.generate_folds
```

It can also write:

```text
fold_summary.csv
fold_membership.csv
```

## Choosing an image size

`--image-size HEIGHT WIDTH` is required. Every image and landmark set is resized into this common coordinate system.

Each image dimension must be at least:

```text
2 ** depth
```

This ensures that every U-Net downsampling level remains valid. Odd or non-divisible dimensions are supported; decoder outputs are aligned to the corresponding skip-connection sizes.

For `batch` or `instance` normalisation, the deepest feature map must contain at least two spatial values. Group normalisation is checked against the number of values available per group. Reflect padding additionally requires both deepest feature-map dimensions to be at least `2`. Invalid combinations are rejected before training.

A helper utility can calculate average source-image dimensions. Edit `IMAGE_DATA_DIR` in:

```text
Heatmaps/utils/calculate_image_size.py
```

Then run:

```bash
python -m Heatmaps.utils.calculate_image_size
```

The utility prints average and rounded dimensions together with a ready-to-use argument:

```text
--image-size HEIGHT WIDTH
```

## Training command

The command format is:

```text
heatmaps-train FOLD TASK_NAME TRAIN_MODEL COPY_FILES [OPTIONS]
```

At least one of `TRAIN_MODEL` or `COPY_FILES` must be `true`.

A transverse prostate example is:

```bash
heatmaps-train 1 prostate_transverse true false \
    --run-dir "$HOME/HEATMAP_TRAINING" \
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

For sagittal prostate images, use `--num-points 2` with the sagittal mark list and image directory.

The supplied `run_pipeline.sh` and `run_pipeline.ps1` files expose paths, actions, data settings, optimisation settings, and model settings as top-level variables.

## Training and copying actions

`TRAIN_MODEL=true` trains the selected fold. Existing outputs belonging to that fold and run are cleared before training. Outputs from other folds in the same run directory are retained.

`COPY_FILES=true` copies the complete run directory to:

```text
SAVE_DIR/TASK_NAME/RUN_NAME/
```

When both actions are enabled, copying occurs after successful training.

A copy-only invocation uses:

```text
TRAIN_MODEL=false
COPY_FILES=true
```

The matching run directory must already exist. Copy-only operation does not create an empty run directory or rewrite the original run metadata. The resolved copy source and destination must be separate paths and must not contain one another.

`--save-dir` is required only when `COPY_FILES=true`.

## Data and target-heatmap settings

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

`--heatmap-sigma` controls the Gaussian spread of each target landmark heatmap in resized-image pixels. Each target heatmap is normalised so that its maximum value is `1.0`.

`--oversampling-factor` affects the training split only:

- `1` uses every training sample once without augmentation;
- `2` uses one original and one independently augmented pass per sample;
- larger values add further independently augmented passes;
- validation samples are never oversampled or augmented.

## Augmentation

The default oversampling policy is defined in:

```text
Heatmaps/heatmap_transforms.py
```

The default sequence is:

```text
RandomAffine
GaussianNoise
GaussianBlur
```

`RandomAffine` applies the same spatial transform to the image and landmark coordinates. A sampled transform is accepted only when every landmark remains within the image.

`GaussianNoise` and `GaussianBlur` do not move landmarks. For RGBA input, the alpha channel is preserved by these intensity transforms. `RandomErasing` remains available for experiments but is not enabled by default.

The complete augmentation policy is stored in checkpoint metadata.

Inspect transforms interactively with:

```bash
python -m Heatmaps.utils.verify_transforms /path/to/images /path/to/points.txt default
```

Available transform names are:

```text
erasing affine noise blur default
```

Press the space bar to select another marked image and resample the chosen transform.

## Input channels

Input channels are detected from all training and validation images in the selected fold. There is no public input-channel argument.

| Source image type | Model input channels |
|---|---:|
| Greyscale | 1 |
| RGB | 3 |
| RGBA | 4 |

The resolved channel count configures the first network layer and is written to run metadata and checkpoints.

## Optimisation settings

The main options are:

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

The default loss is `weighted_mse`:

```text
weight = 1 + target_heatmap * positive_weight
```

This increases the contribution of landmark peak regions relative to the background. `--positive-weight` is used only by `weighted_mse`.

`bce_logits` requires `--output-activation none` because `BCEWithLogitsLoss` expects raw logits.

The training CSV records the learning rate used during each epoch, not the rate prepared for the following epoch.

Training stops immediately with a clear error when a reported loss or endpoint-error metric becomes NaN or infinite.

## Model settings

The current model registry contains:

```text
unet_basic
```

Its options are:

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

The network returns one output heatmap per configured landmark. The registry stores the implementation module and class name so that checkpoint metadata is not hard-coded to a particular architecture.

Additional models can be added to `models.py` and registered in `model_registry.py`. Model-specific configuration fields will also need to be added to `HeatmapModelConfig`, the command-line parser, and the model-building logic.

## Validation outputs

Validation export is enabled by default:

```text
--save-validation-predictions true
```

After training, the best checkpoint is reloaded when available; otherwise the last checkpoint is used. Heatmap maxima are converted to resized-image coordinates and then scaled back into original-image pixels before endpoint errors are calculated.

Set the option to `false` to skip the complete validation export.

## Output structure

Training outputs are written to:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/
```

Typical outputs are:

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
  heatmap_overlays/
  point_overlays/
  logs/
    validation_run_metadata.json
```

The validation workbook contains `image_summary` and `endpoints` sheets. Ground-truth points are shown in green and predicted points in red on point overlays. Heatmap overlays show the combined model response and predicted endpoint labels.

## Checkpoints and metadata

Best and last checkpoints contain:

```text
format_version
schema
schema_version
created_at
epoch
checkpoint_type
state_dict
optimiser_state_dict
metrics
metadata
```

The metadata contains these sections:

```text
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

It records the model registry entry, implementation module and class, reconstruction arguments, landmark count, input channels, image size, heatmap sigma, coordinate conventions, augmentation policy, training settings, and checkpoint metrics.

`run_info_TASK_NAME_fN.json` contains the resolved run, data, training, and model configurations. It is rewritten after automatic channel detection so that the final metadata reflects the model that was actually trained.

## Run and task names

`TASK_NAME` and `--run-name` are used as directory components. Unsupported characters are replaced with underscores, and empty cleaned names are rejected.

When `--run-name` is omitted, the package creates a readable name containing the fold count, point count, model, image size, heatmap sigma, loss, oversampling factor, batch size, and learning rate. A 12-character SHA-256 configuration fingerprint is appended.

The fingerprint includes all data-processing, optimisation, early-stopping, AMP, and model options that can affect the trained result. This prevents two materially different configurations from silently using the same automatically generated output directory.

Dataset paths and file contents are not included in the fingerprint. Use a distinct `TASK_NAME` or explicit `--run-name` when training different datasets with otherwise identical settings.
