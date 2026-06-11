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

## Install

From inside this `Heatmaps` directory:

```bash
pip install -e .
```

The command-line entry points installed by `pyproject.toml` are:

```text
heatmaps-train
heatmaps-generate-folds
heatmaps-calculate-image-size
```

## Expected input files

Fold lists use the same naming style as the IPV package. The current code validates that every fold has train, validation, and test list files, even though only the train and validation splits are used during heatmap training:

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

The mark-list file should contain one image and its landmark coordinates per line, for example:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

Fold-list entries can be stems such as `A1` or filenames such as `A1.jpg`.

## Generate fold lists

A fold-generation utility is included under:

```text
Heatmaps/Heatmaps/utils/generate_folds.py
```

Edit the top-level `MARK_LIST_PATH`, `OUTPUT_DIR`, `NUM_FOLDS`, and switches, then run:

```bash
python -m Heatmaps.utils.generate_folds
```

or use the installed entry point:

```bash
heatmaps-generate-folds
```

The utility follows the same 80/10/10 train/test/validation structure as the IPV package and writes `train_fN.txt`, `test_fN.txt`, `val_fN.txt`, plus optional summary CSVs.

## Train from the command line

The command-line positional arguments are:

```text
heatmaps-train FOLD TASK_NAME TRAIN_MODEL COPY_FILES [OPTIONS]
```

There is no delete-data/delete-files stage in the heatmap package. The heatmap workflow does not create patch datasets, so there is nothing equivalent to the IPV generated training-data clean-up step.

Example transverse prostate run:

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

For sagittal prostate images, change `--num-points 2` and use the sagittal mark list and image directory.

## Image size utility

`--image-size HEIGHT WIDTH` is required. The training pipeline no longer infers image size automatically.

To estimate a sensible value from a folder of images, edit the top-level `IMAGE_DATA_DIR` in:

```text
Heatmaps/Heatmaps/utils/calculate_image_size.py
```

Then run:

```bash
python -m Heatmaps.utils.calculate_image_size
```

or:

```bash
heatmaps-calculate-image-size
```

The utility prints the average width, average height, rounded width, rounded height, and the corresponding `--image-size HEIGHT WIDTH` argument.

## Oversampling and transforms

Use `--oversampling-factor` to increase only the training dataset size. The default is `1`, which keeps the original training set unchanged.

For example, `--oversampling-factor 4` makes the training split four times larger. Indices in the first original dataset pass are returned unchanged; additional passes apply a random transform to the image and landmark points before target heatmaps are generated. Validation data is never oversampled or augmented.

The default augmentation policy is stored in:

```text
Heatmaps/Heatmaps/heatmap_transforms.py
```

The current default transform chain is task-agnostic:

```text
RandomAffine
RandomHorizontalFlip
GaussianNoise
GaussianBlur
```

`RandomHorizontalFlip` mirrors the x-coordinate of every landmark, but it does not reorder landmark channels by default. This avoids hidden prostate-specific assumptions. If a particular dataset has symmetric landmark labels that need to swap after a horizontal flip, pass `point_index_swaps` explicitly when constructing `RandomHorizontalFlip` in `heatmap_transforms.py`.

`RandomErasing` is implemented in `heatmap_transforms.py` for optional local experimentation, but it is not part of the default transform chain.

Intensity transforms preserve greyscale RGB ultrasound images by applying noise consistently across RGB channels.

## Input channels

Input channels are detected automatically from the train and validation images for the selected fold. There is no command-line option for this.

The package assumes that every image for a given task has the same number of source channels. The detected source channel count is used directly to configure the first U-Net layer:

| Source images | Model input channels |
|---------------|----------------------|
| All greyscale | 1                    |
| All RGB       | 3                    |
| All RGBA      | 4                    |

If any image has a different number of channels from the rest of the train/validation images, the run stops with a clear error. The loader does not silently convert greyscale, RGB, or RGBA images to another channel count.

## Key training options

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
run_info_TASK_NAME_f1.json
validation_results_F1/validation_summary.xlsx
validation_results_F1/validation_image_summary.csv
validation_results_F1/validation_endpoints.csv
validation_results_F1/validation_predictions_f1.csv
validation_results_F1/logs/validation_run_metadata.json
```

`train_plot_f1.png` contains both loss curves and mean endpoint-error curves. After training, the selected checkpoint is loaded and IPV-style validation outputs are written when `--save-validation-predictions true` is used. The Excel workbook contains `image_summary` and `endpoints` sheets.

If `--save-validation-overlays true` is used, validation images are saved under:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/validation_results_F1/heatmap_overlays/
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/validation_results_F1/point_overlays/
```

Point overlays use labelled ground-truth and predicted endpoints.

## Checkpoint metadata

Saved `.pth` checkpoints now include IPV-style metadata sections:

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

The `model` section stores the registry name, model class, module, and U-Net reconstruction arguments. The `preprocessing` section stores the fixed image size, heatmap sigma, channel order, value range, and resize method. The `augmentation` section records whether oversampling was enabled and the default augmentation policy from `heatmap_transforms.py`.

## Model registry

The current registry contains one heatmap model:

```text
unet_basic
```

Additional models can be added later in `model_registry.py` and `models.py` without changing the training pipeline.

## Run names

`--network-name` selects the model architecture. `--run-name` is only an optional output-folder override.

If `--run-name` is omitted, the package builds a deterministic run folder from the fold count, point count, selected network, image size, heatmap sigma, U-Net settings, loss settings, oversampling factor, batch size, learning rate, and epoch count.

## Helper scripts

`run_pipeline.sh` and `run_pipeline.ps1` are editable examples for Linux/macOS/HPC shells and Windows PowerShell. They expose the same heatmap training options as top-level variables and then call `heatmaps-train`.

`verify_transforms.py` can be used to visually inspect individual augmentation transforms:

```bash
python -m Heatmaps.utils.verify_transforms /path/to/images /path/to/points.txt default
```
