# Heatmaps

Full-image heatmap-regression landmark localisation for the `LandmarkIdentificationFrameworks/Heatmaps` package.

The package trains a convolutional neural network to produce one heatmap per landmark. Source images are loaded directly, resized into a common training coordinate
system, and paired with Gaussian target heatmaps generated from the supplied landmark coordinates.

**Package version:** `0.1.0`

## Current scope

The package currently provides:

- repeated k-fold training and validation;
- configurable U-Net, HRNet, stacked-hourglass, and ViTPose heatmap regressors;
- landmark-preserving image augmentation;
- automatic greyscale, RGB, or RGBA input-channel detection;
- deterministic seeding for Python, NumPy, PyTorch, and DataLoader workers;
- best and last checkpoints with reconstruction metadata;
- validation predictions, endpoint metrics, heatmap overlays, and point overlays;
- optional copying of a completed run to a separate save directory.

Checkpoint resumption and general-purpose inference are not yet included.

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
      annotation_utils.py
      calculate_image_size.py
      generate_folds.py
      io_utils.py
      progress_bar.py
      verify_transforms.py
      visualisation_utils.py
  tests/
    test_annotation_utils.py
    test_repeated_kfold.py
    test_runtime_integration.py
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
3. repeated k-fold training and validation sample lists;
4. a common training image size.

### Fold lists

Fold files are grouped by repetition and must use these names:

```text
folds/
  repeated_kfold_summary.csv
  repeated_kfold_membership.csv
  repetition_1/
    training_f1.txt
    val_f1.txt
    training_f2.txt
    val_f2.txt
    ...
  repetition_2/
    training_f1.txt
    val_f1.txt
    ...
```

Each file contains one sample identifier per line. Entries may be stems such as `A1` or filenames such as `A1.jpg`.

Before training, the complete repeated k-fold collection is checked for:

- contiguous `repetition_N` directories;
- identical contiguous fold numbers in every repetition;
- the presence of `training_fN.txt` and `val_fN.txt` for every fold;
- duplicate sample identifiers within a split;
- overlap between the training and validation lists;
- complete dataset coverage in every fold;
- use of every sample as validation exactly once per repetition;
- use of the same full dataset in every repetition.

### Landmark mark list

Each mark-list row contains an image name followed by landmark coordinates:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
```

Coordinates use `(x, y)` order, where `x` is horizontal and `y` is vertical. Every row used by the selected repetition and fold must contain exactly the number of points
specified by `--num-points`. Missing and additional points both cancel training; points are never silently discarded.

Every selected landmark is checked against the resolved source image. For an image with width `W` and height `H`, each point must satisfy:

```text
0 <= x < W
0 <= y < H
```

The run stops with the sample, point number, coordinate, image path, and valid bounds when a landmark is invalid. Landmark-count errors identify the repetition, fold,
split, patient/sample, annotation file, and source line. Complete annotation and image validation occurs before earlier outputs for that repetition and fold are removed.

### Source images

Supported image suffixes are:

```text
.png .jpg .jpeg .bmp .tif .tiff
```

Images may be greyscale, RGB, or RGBA. The selected fold's training and validation images must all have the same source channel count.

Integer images are converted to `float32` in the `0` to `1` range. Floating-point source images must already contain finite values within that range; the run stops if
NaN, infinity, or out-of-range values are found.

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

creates deterministic repeated k-fold training and validation lists. Edit these top-level settings before running it:

```text
NUM_REPETITIONS
NUM_FOLDS_PER_REPETITION
BASE_SEED
MARK_LIST_PATH
OUTPUT_DIR
CLEAN_OUTPUT_DIR
```

Each repetition starts from the full dataset, uses `BASE_SEED + repetition - 1`, builds balanced validation folds, and uses all remaining samples for training. Run:

```bash
python -m Heatmaps.utils.generate_folds
```

Regeneration replaces managed `repetition_N` directories and summary files. It also removes legacy flat `*_fN.txt`, `fold_summary.csv`, and `fold_membership.csv`
artefacts from the selected output root; unrelated files are retained unless `CLEAN_OUTPUT_DIR` is enabled.

It can also write:

```text
repeated_kfold_summary.csv
repeated_kfold_membership.csv
```

## Choosing an image size

`--image-size HEIGHT WIDTH` is required. Every image and landmark set is resized into this common coordinate system, and every model returns heatmaps at exactly that
size.

Architecture-specific minimum sizes are checked before training:

| Model               | Minimum size rule                                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------------------------------|
| `unet_basic`        | Each dimension must be at least `2 ** depth`; normalisation and reflect padding can require a larger deepest feature map |
| `hrnet`             | Each dimension must be at least `64` pixels                                                                              |
| `stacked_hourglass` | Each dimension must be at least `8 * (2 ** hourglass_depth)`                                                             |
| `vitpose`           | Each dimension must be at least `vit_patch_size`                                                                         |

Odd and non-divisible dimensions are supported. CNN decoder outputs are aligned to the requested image size, while ViTPose pads internally to a complete patch grid and
crops the result back to the requested size.

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
heatmaps-train REPETITION FOLD TASK_NAME TRAIN_MODEL COPY_FILES [OPTIONS]
```

At least one of `TRAIN_MODEL` or `COPY_FILES` must be `true`.

A transverse prostate example is:

```bash
heatmaps-train 1 1 prostate_transverse true false \
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

`TRAIN_MODEL=true` trains the selected repetition and fold. Existing outputs belonging to that repetition/fold leaf are cleared only after complete training and
validation annotation, image, channel, preprocessing, and target-heatmap validation. Outputs from every other repetition and fold are retained. A validation failure
leaves existing results untouched.

`COPY_FILES=true` copies the selected repetition/fold output leaf to:

```text
SAVE_DIR/TASK_NAME/RUN_NAME/repetition_N/fold_N/
```

When both actions are enabled, copying occurs after successful training.

A copy-only invocation uses:

```text
TRAIN_MODEL=false
COPY_FILES=true
```

The matching run directory must already exist. Copy-only operation does not create an empty run directory or rewrite the original run metadata. The resolved copy source
and destination must be separate paths and must not contain one another.

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

`--heatmap-sigma` controls the Gaussian spread of each target landmark heatmap in resized-image pixels. Each target heatmap is normalised so that its maximum value is
`1.0`.

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

`RandomAffine` applies the same spatial transform to the image and landmark coordinates. A sampled transform is accepted only when every landmark remains within the
image.

`GaussianNoise` and `GaussianBlur` do not move landmarks. For RGBA input, the alpha channel is preserved by these intensity transforms. `RandomErasing` remains available
for experiments but is not enabled by default.

The complete augmentation policy is stored in checkpoint metadata.

Inspect transforms interactively with:

```bash
python -m Heatmaps.utils.verify_transforms /path/to/images /path/to/points.txt default --num-points 4
```

Available transform names are:

```text
erasing affine noise blur default
```

Press the space bar to select another marked image and resample the chosen transform.

## Input channels

Input channels are detected from all training and validation images in the selected fold. There is no public input-channel argument.

| Source image type | Model input channels |
|-------------------|---------------------:|
| Greyscale         |                    1 |
| RGB               |                    3 |
| RGBA              |                    4 |

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

Validation loss is primarily an internal control signal for learning-rate scheduling, early stopping, and best-checkpoint selection within one run. Cross-model validation
losses must not be compared. Architecture-specific objectives can differ even when the exported final-heatmap endpoint errors use the same calculation.

## Model settings

The model registry contains:

```text
unet_basic
hrnet
stacked_hourglass
vitpose
```

Every architecture produces one full-resolution heatmap per configured landmark and can be selected through `--network-name` without changing the training, validation,
checkpoint, or export workflow.

These are native PyTorch implementations for this package. They preserve the main design of the cited architectures but do not copy the authors' official repositories or
bundle pretrained weights.

### U-Net

`unet_basic` uses a contracting encoder to collect wider anatomical context and a symmetric decoder with skip connections to recover fine spatial detail. The
implementation is based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

```text
--network-name unet_basic
--base-channels
--depth
--channel-multiplier
--max-channels
--upsampling bilinear|transpose
```

### HRNet

`hrnet` maintains a high-resolution stream while processing lower-resolution streams in parallel. Repeated fusion moves contextual information between the streams before
their features are combined for heatmap prediction. The implementation is based
on [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212).

```text
--network-name hrnet
--hrnet-width
--hrnet-modules
--hrnet-blocks
```

### Stacked Hourglass

`stacked_hourglass` repeatedly applies bottom-up and top-down processing so that local landmark evidence and whole-image anatomical relationships can refine one another.
Heatmaps from earlier stacks are fed into later stacks, and intermediate heatmaps receive auxiliary supervision during training. The implementation is based
on [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937).

```text
--network-name stacked_hourglass
--hourglass-features
--hourglass-stacks
--hourglass-depth
--hourglass-blocks
--auxiliary-loss-weight
```

`--auxiliary-loss-weight` multiplies the mean loss from all non-final stacks before it is added to the final heatmap loss. Set it to `0` to disable intermediate
supervision while retaining stack-to-stack feature feedback.

For `stacked_hourglass`, both training loss and validation loss are the final-stack heatmap loss plus `auxiliary_loss_weight` multiplied by the mean loss from all non-final
stacks. Other architectures report only their final-output loss. This is why validation-loss values must not be compared across architectures. The combined validation
loss is used only within that stacked-hourglass run for the plateau scheduler, early stopping, and best-checkpoint selection; exported predictions and endpoint errors use
the final stack only.

### ViTPose

`vitpose` divides the image into patches, embeds them as tokens, applies a plain Vision Transformer to model long-range relationships, and uses a lightweight
transposed-convolution decoder to reconstruct landmark heatmaps. The implementation is based
on [ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484).

```text
--network-name vitpose
--vit-patch-size
--vit-embed-dim
--vit-depth
--vit-heads
--vit-mlp-ratio
--vit-dropout
--vit-decoder-channels
```

`--vit-patch-size` must be a power of two. `--vit-heads` must divide `--vit-embed-dim` exactly. ViTPose is normally the most memory-intensive option and is particularly
dependent on dataset size or suitable pretraining; this package currently trains it from scratch.

### Shared CNN and output settings

The CNN architectures use:

```text
--normalisation batch|instance|group|none
--activation relu|leaky_relu|elu|gelu
--dropout
--padding-mode zeros|reflect|replicate|circular
```

All architectures use:

```text
--output-activation none|sigmoid|softplus
--final-kernel-size 1|3
```

The registry stores the implementation module, class name, paper link, and architecture-specific constructor fields. Checkpoint metadata contains only the constructor
fields used by the selected model, together with the resolved image size and input-channel count.

## Validation outputs

Validation export is enabled by default:

```text
--save-validation-predictions true
```

After training, the best checkpoint is reloaded when available; otherwise the last checkpoint is used. Heatmap maxima are converted to resized-image coordinates and then
scaled back into original-image pixels before endpoint errors are calculated.

Set the option to `false` to skip the complete validation export.

## Output structure

Training outputs are written to:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/repetition_N/fold_N/
```

Typical outputs are:

```text
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
  validation_logs/
    validation_run_metadata.json
```

Every exported validation CSV row includes `dataset_split=validation`, repetition, and fold. The validation workbook contains `validation_image_summary` and
`validation_endpoints` sheets. Ground-truth points are shown in green and predicted points in red on point overlays. Heatmap overlays show the combined model response and
predicted endpoint labels.

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
validation_metrics
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

It records the model registry entry, implementation module and class, reconstruction arguments, repetition, fold, landmark count, input channels, image size, heatmap
sigma, coordinate conventions, augmentation policy, training settings, and explicitly named validation checkpoint metrics.

`run_info.json` contains the resolved run, repetition, fold, data, training, and model configurations. It is rewritten after automatic channel detection so that the final
metadata reflects the model that was actually trained.

## Run and task names

`TASK_NAME` and `--run-name` are used as directory components. Unsupported characters are replaced with underscores, and empty cleaned names are rejected.

When `--run-name` is omitted, the package creates a readable name containing the repetition count, folds per repetition, point count, model, image size, heatmap sigma,
loss, oversampling factor, batch size, and learning rate. A 12-character SHA-256 configuration fingerprint is appended.

The fingerprint includes all data-processing, optimisation, early-stopping, AMP, and model options that can affect the trained result. It also includes a SHA-256 digest of
every active training and validation list, so regenerated fold memberships do not silently reuse an earlier output directory. The complete fold-collection digest is
stored in `run_info.json`.

The annotation and image contents are not included in the fingerprint. Use a distinct `TASK_NAME` or explicit `--run-name` when training different datasets that use the
same fold membership and otherwise identical settings.
