# Heatmaps

Full-image heatmap-regression landmark localisation for the `LandmarkIdentificationFrameworks/Heatmaps` package.

The package trains a convolutional or transformer network to produce one heatmap per landmark. Source images are loaded directly, resized into a common training coordinate
system, and paired with Gaussian target heatmaps generated from the supplied landmark coordinates.

**Package version:** `0.1`

## Current scope

The package currently provides:

- repeated k-fold training and validation;
- configurable U-Net, HRNet, stacked-hourglass, pretrained ViTPose-B and MedSAM ViT-B heatmap regressors;
- landmark-preserving image augmentation;
- automatic greyscale, RGB, or RGBA input-channel detection;
- deterministic seeding for Python, NumPy, PyTorch, and DataLoader workers;
- best checkpoints plus atomically committed last-epoch checkpoints that can continue interrupted training;
- validation predictions, endpoint metrics, heatmap overlays, and point overlays;
- standalone inference for one image or an image directory using any registered trained model;
- optional copying of a completed run to a separate save directory.

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
    normalisation.py
    heatmap_training_pipeline.py
    heatmap_transforms.py
    infer_landmarks.py
    model_registry.py
    heatmap_losses.py
    validation_progress.py
    models/
      common.py
      unet.py
      hrnet.py
      stacked_hourglass.py
      pretrained.py
      vitpose.py
      vit_medsam.py
      sam/  # Bundled encoder, licence and provenance
    parameters.py
    train_model.py
    utils/
      __init__.py
      annotation_utils.py
      calculate_image_size.py
      generate_folds.py
      heatmap_inference_utils.py
      io_utils.py
      progress_bar.py
      verify_transforms.py
      visualisation_utils.py
  tests/
    test_annotation_utils.py
    test_inference.py
    test_normalisation.py
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
heatmaps-infer
```

Display the complete command-line interface with:

```bash
heatmaps-train --help
```

## Standalone inference

Edit the paths and switches near the top of `Heatmaps/infer_landmarks.py`, then run either:

```bash
heatmaps-infer
```

or:

```bash
python -m Heatmaps.infer_landmarks
```

Set `MODEL_PATH` to `model_best_validation_loss.pth` (normally the checkpoint to use for prediction), `INPUT_PATH` to one supported image or a directory, and
`OUTPUT_DIR` to the result directory. `GROUND_TRUTH_MARK_LIST_PATH` is optional; when supplied, matching image stems receive pixel-error metrics and ground-truth points
on their overlays. Directory searches can be made recursive with `RECURSIVE_IMAGE_SEARCH`.

Inference reconstructs the selected U-Net, HRNet, stacked-hourglass, or ViTPose architecture directly from checkpoint metadata. It also restores the training image size,
channel count and optional three-channel normalisation constants, applies the same image loading, resize and normalisation path used during training, decodes every heatmap by argmax within the valid image area, removes padding offsets and scales predictions back into original-image
pixel coordinates. When a checkpoint expects three input channels, a single-channel greyscale inference image is replicated across all three channels automatically. Other
channel mismatches remain errors. No architecture settings need to be copied into the script.

`BATCH_SIZE` controls the number of full resized images passed through the model together. Keep it at `1` for large models or images and increase it only when device
memory allows. `SAVE_RAW_HEATMAPS` optionally writes the model-resolution arrays as `.npy` files.

The output directory contains:

```text
inference_summary.xlsx
inference_image_summary.csv
inference_endpoints.csv
inference_predictions.csv
inference_heatmap_overlays/
inference_point_overlays/
inference_raw_heatmaps/       # only when SAVE_RAW_HEATMAPS is true
inference_logs/
  inference_run_metadata.json
```

The Excel workbook uses `image_summary` and `endpoints` sheets. CSV output includes both long endpoint rows and one wide, comparison-ready prediction row per image.
Error fields are blank when no matching ground truth is available.

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
  test_cases.xlsx                 # only when fixed test cases are configured
  repetition_1/
    training_fall.txt
    val_fall.txt
    training_f1.txt
    val_f1.txt
    training_f2.txt
    val_f2.txt
    ...
  repetition_2/
    training_fall.txt
    val_fall.txt
    training_f1.txt
    val_f1.txt
    ...
```

Each training and validation text file contains one sample identifier per line. Entries may be stems such as `A1` or filenames such as `A1.jpg`. The special
`training_fall.txt` and `val_fall.txt` files intentionally contain the same complete set of non-test samples so that `fold=all` retains the ordinary validation,
checkpoint-selection, early-stopping, and validation-export workflow.

Before training, the complete repeated k-fold collection is checked for:

- contiguous `repetition_N` directories;
- identical contiguous fold numbers in every repetition;
- the presence of `training_fN.txt` and `val_fN.txt` for every fold;
- duplicate sample identifiers within a split;
- no overlap between numbered training and validation lists;
- identical `training_fall.txt` and `val_fall.txt` membership, when fold-all files are present;
- complete cross-validation-eligible dataset coverage in every fold;
- use of every sample as validation exactly once per repetition;
- use of the same cross-validation-eligible dataset in every repetition.

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
split, patient/sample, annotation file, and source line. Complete annotation and image validation for the selected training and validation samples occurs before earlier
outputs for that repetition and fold are removed. Annotation-only cases reserved by `TEST_SAMPLE_IDS` are parsed for sample identity and duplicate-stem checks, but are
not included in either dataset; their images, landmark counts, coordinates, and preprocessing are not validated during training.

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
TEST_SAMPLE_IDS
MARK_LIST_PATH
OUTPUT_DIR
CLEAN_OUTPUT_DIR
```

`TEST_SAMPLE_IDS` reserves a fixed external test cohort before any folds are generated. Enter mark-list stems such as:

```python
TEST_SAMPLE_IDS = ['A4', 'A50', 'A8']
```

Use `TEST_SAMPLE_IDS = []` or `TEST_SAMPLE_IDS = None` to include every mark-list sample in repeated k-fold cross-validation. Configured IDs are checked for blanks,
duplicates, and values missing from the mark list. Invalid configuration or too few remaining samples cancels generation before existing fold outputs are removed.

Each repetition starts from the remaining cross-validation-eligible dataset, uses `BASE_SEED + repetition - 1`, builds balanced validation folds, and uses all other
eligible samples for training. It also writes `training_fall.txt` and `val_fall.txt`; both contain every sample not reserved by `TEST_SAMPLE_IDS`. Run:

```bash
python -m Heatmaps.utils.generate_folds
```

Regeneration replaces managed `repetition_N` directories, summary files, and any earlier `test_cases.xlsx` manifest only after the complete in-memory split collection is
validated. It also removes legacy flat `*_fN.txt`, `fold_summary.csv`, and `fold_membership.csv` artefacts from the selected output root; unrelated files are retained
unless `CLEAN_OUTPUT_DIR` is enabled.

It can also write:

```text
repeated_kfold_summary.csv
repeated_kfold_membership.csv
test_cases.xlsx
```

`test_cases.xlsx` is created only for a non-empty `TEST_SAMPLE_IDS` list and records the held-out sample IDs in the output root beside the `repetition_N` directories.
It is a manifest for later external MRI-reference evaluation only. Heatmaps training does not read it, and the generator does not create `test_fN.txt` files, a test
loader, or test results.

## Choosing an image size

`--image-size SIZE` is required and accepts one positive integer only. The longest
image edge is resized to SIZE, preserving aspect ratio up to integer rounding.
The shorter dimension is rounded to the nearest integer (half upwards, minimum
one pixel), then centred on a SIZE x SIZE canvas. Odd padding puts the extra pixel
on the bottom or right. All models return heatmaps on this square canvas.

Landmark pixel centres follow the actual rounded resize dimensions:
`x_canvas = (x_original + 0.5) * resized_width / original_width - 0.5 + left_padding`
(and equivalently for y). Gaussian targets are generated at these transformed
locations and set to zero outside the image content. Sigma is measured in canvas
pixels. Predictions exclude padding before argmax; offsets and scaling are then
inverted, and reported coordinates are bounded to original pixel centres.

All four losses (MSE, weighted MSE, Smooth L1 and BCE logits) ignore padding. Loss
is averaged over valid pixels and landmarks separately for each image, then over
images in the batch. Hourglass intermediate outputs use exactly the same rule.
Padding is zero in model-input space: normalisation is applied only to resized
content, and training-only channel statistics exclude padding. Augmentation occurs
before letterboxing; validation and inference use the same resize and normalisation.

Visual overlays crop out padding before resizing responses to the source image.
Raw inference heatmaps are also cropped to resized-content dimensions, without
resampling their values. Original image dimensions and the exact resize policy
are available in the summaries/checkpoint metadata for coordinate reconstruction.

Version remains 0.1. The new preprocessing policy and scalar size are required
in checkpoints; old stretch-resize checkpoints and two-value size arguments are
unsupported, including for resumption. Train new models with this contract.

Architecture-specific minimum sizes are checked before training:

| Model               | Minimum size rule                                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------------------------------|
| `unet_basic`        | Each dimension must be at least `2 ** depth`; normalisation and reflect padding can require a larger deepest feature map |
| `hrnet`             | Each dimension must be at least `64` pixels                                                                              |
| `stacked_hourglass` | Each dimension must be at least `8 * (2 ** hourglass_depth)`                                                             |
| `vitpose`           | Canvas must be at least 16 pixels                                                                         |

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
--image-size SIZE
```

## Training command

The command format is:

```text
heatmaps-train REPETITION FOLD TASK_NAME TRAIN_MODEL COPY_FILES [OPTIONS]
```

At least one of `TRAIN_MODEL` or `COPY_FILES` must be `true`.

`FOLD` may be a numbered fold such as `1`, or `all`. The latter reads `training_fall.txt` and `val_fall.txt` and otherwise follows exactly the same training and
validation process as a numbered fold.

A transverse prostate example is:

```bash
heatmaps-train 1 1 prostate_transverse true false \
    --run-dir "$HOME/HEATMAP_TRAINING" \
    --num-points 4 \
    --fold-lists-path "$HOME/DATA/folds" \
    --mark-list-file "$HOME/DATA/doctors_resampled_transverseMarkList.txt" \
    --image-data-dir "$HOME/DATA/TRANSVERSE" \
    --image-size 512 \
    --heatmap-sigma 8 \
    --oversampling-factor 1 \
    --normalise-inputs true \
    --batch-size 4 \
    --learning-rate 0.001 \
    --max-training-epochs 80
```

For sagittal prostate images, use `--num-points 2` with the sagittal mark list and image directory.

The supplied `run_pipeline.sh` and `run_pipeline.ps1` files expose paths, actions, data settings, optimisation settings, and model settings as top-level variables.

## Training and copying actions

`TRAIN_MODEL=true` trains the selected repetition and fold. For a fresh run (`RESUME_TRAINING=false`), existing outputs belonging to that repetition/fold leaf are cleared only after complete training and
validation annotation, image, channel, preprocessing, and target-heatmap validation. Outputs from every other repetition and fold are retained. A validation failure
leaves existing results untouched.

`RESUME_TRAINING=true` explicitly continues the same run from its `model_last_epoch.pth`. Resume mode never clears the fold output leaf. It first completes dataset
validation, then validates the checkpoint schema and the complete compatibility signature before changing any saved output. `TRAIN_MODEL` must also be `true`.

`COPY_FILES=true` copies the selected repetition/fold output leaf to:

```text
SAVE_DIR/TASK_NAME/RUN_NAME/repetition_N/fold_N/
```

For `FOLD=all`, the final path component is `fold_all`.

When both actions are enabled, copying occurs after successful training.

A copy-only invocation uses:

```text
TRAIN_MODEL=false
COPY_FILES=true
```

The matching run directory must already exist. Copy-only operation does not create an empty run directory or rewrite the original run metadata. The resolved copy source
and destination must be separate paths and must not contain one another.

`--save-dir` is required only when `COPY_FILES=true`.

### Resuming interrupted training

Use the same repetition, fold, task, run name, data and training/model settings as the original command, and set:

```text
RESUME_TRAINING=true
```

or pass:

```text
--resume-training true
```

The option does not change the automatically generated run name. The pipeline resolves only the isolated checkpoint at:

```text
RUN_DIR/TRAINING_RESULTS/TASK_NAME/RUN_NAME/repetition_N/fold_N/model_last_epoch.pth
```

Resumption continues from the epoch after the last atomically committed epoch. If interruption occurs part-way through an epoch, that incomplete epoch is repeated. The
checkpoint compatibility signature covers the selected training/validation lists and image contents, annotation file, repetition/fold, landmark and preprocessing settings, model
constructor, optimiser, scheduler, early stopping, AMP, seed, workers, batch size, oversampling, active Heatmaps source code and relevant Python/PyTorch/CUDA runtime
versions. A mismatch cancels resumption without deleting or rewriting existing
outputs.

If the final or early-stopping epoch was already committed but interruption occurred while writing the CSV, validation export, summary or run metadata, resume mode skips
the optimiser loop and rebuilds those final outputs from the committed last/best state.

The continuation checkpoint restores the model, optimiser, learning-rate scheduler, AMP scaler, best-loss state, early-stopping reference and counter, complete history,
Python/NumPy/PyTorch/CUDA random-number states, and training/validation DataLoader generators. State-complete continuation assumes a compatible device and software environment;
those versions are stored for audit. Deterministic PyTorch algorithms are enabled, cuDNN benchmarking and TF32 are disabled, and a deterministic cuBLAS workspace is
configured; training stops with a PyTorch error if the selected environment has no deterministic implementation for a required operation. The best-validation-loss
checkpoint is intended for prediction and reconstruction. Always resume from `model_last_epoch.pth`.

## Data and target-heatmap settings

The main data options are:

```text
--num-points
--fold-lists-path
--mark-list-file
--image-data-dir
--image-size SIZE
--heatmap-sigma
--oversampling-factor
--recursive-image-search
--normalise-inputs true|false
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

When input normalisation is enabled, statistics are calculated before training augmentation. Augmented copies are normalised only after their intensity/spatial transforms and resize have been applied.

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

### Input-value normalisation

`--normalise-inputs true` calculates a distinct population mean and standard deviation for each of the three channels using only the original images in the selected training split, after aspect-preserving resize and per-image content min-max scaling to float32 `[0, 1]`, excluding padding. Validation images and oversampled/augmented copies do not contribute to the statistics.

The current ultrasound data may be greyscale stored as RGB, but the calculation intentionally remains three-channel so future colour RGB images follow the same contract. Enabling input normalisation requires exactly three source channels; one- and four-channel training data are rejected rather than collapsed to a single statistic. This option is separate from `--normalisation`, which selects internal CNN normalisation layers such as batch or group normalisation.

The enabled flag, three means and three standard deviations are stored under `metadata.preprocessing.normalisation` in each checkpoint. The same values are applied to training, validation and standalone inference inputs. `--normalise-inputs false` uses per-image min-max scaled float32 `[0, 1]` content, with zero padding. This scaling is now shared by every model.

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
--lr-schedule none|step|plateau|cosine|linear|exponential
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

It also records UTC start/completion timestamps and training, validation and epoch-processing durations in seconds. The epoch duration runs from epoch start through
validation and scheduler/control updates; checkpoint, plot and CSV output writes are excluded. Run metadata stores dataset-validation, model-setup/resume,
validation-export and cumulative epoch timings, together with the termination reason and individual fresh/resumed execution sessions.

Training stops immediately with a clear error when a reported loss or endpoint-error metric becomes NaN or infinite.

Validation loss is primarily an internal control signal for early stopping and best-loss checkpoint selection within one run. Cross-model validation
losses must not be compared. Architecture-specific objectives can differ even when the exported final-heatmap endpoint errors use the same calculation.

## Model settings

The model registry contains:

```text
unet_basic
hrnet
stacked_hourglass
vitpose
vit-medsam
```

Every architecture produces one full-resolution heatmap per configured landmark and can be selected through `--network-name` without changing the training, validation,
checkpoint, or export workflow.

The CNNs are native PyTorch implementations. ViTPose uses a checkpoint-compatible ViT-B backbone; MedSAM bundles its original image encoder with its licence and ViTDet attribution. Pretrained weights are supplied separately. Neither model requires installing ViTPose, timm, mmcv, MedSAM or segment_anything.

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
loss is used within that stacked-hourglass run for early stopping and best-loss checkpoint selection; the plateau scheduler uses pixel error; exported predictions and endpoint errors use
the final stack only.

### ViTPose and MedSAM

`vitpose` loads the backbone from the **standard ViTPose-B COCO 256x192 pose checkpoint** supplied via `--pretrained-checkpoint`. Its 12-block, 768-wide, 12-head encoder uses 16-pixel patches. The original human-pose head is discarded; a new projection and full-resolution landmark decoder are trained. The official backbone layout, patch padding, positional-token addition and stochastic-depth schedule are retained. The 16x12 spatial position grid is interpolated onto the configured square canvas; incompatible checkpoint keys/shapes fail explicitly. MAE-only, ViTPose+, other sizes and older custom ViTPose checkpoints are not accepted by this loader.

`vit-medsam` uses the bundled MedSAM ViT-B image encoder and the same landmark decoder. Supply original `medsam_vit_b.pth` through `--pretrained-checkpoint`. No prompts or segmentation decoder are used. See [MEDSAM.md](MEDSAM.md) for encoder provenance and positional adaptation.

Both models accept one `--image-size SIZE >=16`, fixed for the run. Internal patch-alignment padding is cropped from the output. Their launchers default to **512**, with **sigma 8**. Larger canvases substantially increase ViTPose attention memory. Greyscale is replicated to three channels; content is min-max scaled to [0,1]. ViTPose additionally applies the fixed ImageNet mean/std from its source configuration inside the model; MedSAM receives min-max values. Dataset standardisation is disabled for both.

Use `run_vitpose.ps1` / `.sh` or `run_vit_medsam.ps1` / `.sh` after editing paths and activating your Python environment. The scripts run the package from its source directory. No pretrained weights are downloaded automatically; inference/resume use the self-contained landmark checkpoint.

### Shared staged training and schedulers

Every model uses the same trainer, data pipeline, optional constraints, checkpoint selection and export code. CNNs train all parameters from epoch one. The two pretrained ViTs first train their landmark decoder (including ViTPose's newly initialised projection). The backbone stays frozen until both conditions hold:

- At least `--freeze-encoder-epochs` epochs have run (default 5).
- Validation pixel error has plateaued for `--decoder-plateau-patience` epochs (10), using `--decoder-plateau-min-delta` pixels (0.1).

The next epoch restores the best pixel checkpoint and starts joint training, clearing optimiser state and restarting the scheduler. The decoder LR is multiplied by `--finetune-decoder-lr-factor` (0.5); encoder LR defaults to 1e-5. `--finetune-last-blocks` defaults to 4 in the CLI; 12 includes patch/position embeddings. MedSAM's pretrained neck is fine-tuned with its encoder. Early stopping monitors loss and starts in the joint stage. The maximum epoch budget still applies if the decoder never plateaus; joint training is not forced.

ViT CLI defaults: AdamW, decoder LR 1e-4, weight decay 1e-4, maximum 300 epochs, early-stop patience 30, gradient checkpointing enabled. Both require supplied pretrained weights for fresh training.

All models support `none`, `step`, `plateau`, `cosine`, `linear` and `exponential` schedulers. Plateau monitors **validation pixel error**, with `--lr-plateau-patience`, `--lr-gamma` and `--lr-min-factor`. Cosine/linear use `--lr-decay-epochs` and `--lr-min-factor`; exponential uses gamma and the same floor; step uses `--lr-step-size` and gamma. CNNs support AdamW or SGD. Full training state, stage transitions and RNG state are restored on resume.

### Optional anatomical losses

Constraints are disabled unless `--landmark-constraint-loss TYPE` is supplied. All architectures support:

- `prostate_taus`: four points ordered top, right, bottom, left. Penalises angles outside 90 +/-20 degrees between P1-P3 and P4-P2, and points falling on the wrong sides of those axes.
- `prostate-saus`: two points, encouraging P1 above P2 only.

`prostate-taus` and `prostate_saus` are accepted spellings. Differentiable softmax coordinates exclude padding, reverse letterboxing, and apply the exact inverse augmentation matrix before geometry is evaluated. This accounts for rotation, shear and scaling. `--constraint-angle-weight` and `--constraint-side-weight` default to 0.01, `--constraint-temperature` to 0.05, and `--constraint-margin` to 0.02 of the original image diagonal. The total reported loss includes the constraints; hourglass auxiliary heatmaps also receive their usual heatmap supervision. These are soft penalties, not guarantees on final argmax coordinates.

Add base objectives to `HEATMAP_LOSSES` in `Heatmaps/heatmap_losses.py`: a factory accepts the training config and returns an elementwise criterion. Padding masking/reduction is shared. Add geometry functions to `LANDMARK_CONSTRAINTS` as `(landmark_count, callable)` returning angle and side terms from restored, diagonal-normalised coordinates.

### Validation progress images

```text
--visualise-validation-progress-images 5
--visualise-validation-progress-epochs 5
```

Both values must be positive. If either is omitted or zero, no progress images are generated. A local seeded sampler chooses up to the requested number of validation images at the start; their IDs are saved in the resume state and checked on continuation. Every requested interval, those same images run through both best checkpoints. Outputs are saved under `validation_progress/epoch_NNNN/best_validation_loss/` and `best_validation_pixel_error/`, with heatmap overlays, point overlays and `selection.json` identifying the observed epoch, checkpoint epoch and sample IDs. Rendering restores model and RNG state and does not alter subsequent training. It adds inference and image-writing time.

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

After training, both the best-loss and best-pixel checkpoints are reloaded and exported separately, even if their best epochs coincide. Heatmap maxima are selected only within image content, padding offsets are removed, and coordinates are
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
model_best_validation_pixel_error.pth
model_last_epoch.pth
validation_checkpoint_summary.json
training_validation_log.csv
training_validation_plot.png
run_info.json
validation_best_loss/  # validation_best_pixel_error/ has the same contents
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
next_epoch
checkpoint_type
resume_capable
state_dict
optimiser_state_dict
validation_metrics
metadata
```

`model_last_epoch.pth` additionally contains:

```text
scheduler_state_dict
grad_scaler_state_dict
training_state
rng_state
data_loader_generator_states
best_model_state_dict
resume_signature
```

The last checkpoint is written through a same-directory temporary file and atomic replacement. It is the commit record for the last completed epoch and embeds the
corresponding best-model weights so the best checkpoint can be recovered if interruption occurs between the two checkpoint writes.

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
runtime_environment
timing
raw_configs
```

It records the model registry entry, implementation module and class, reconstruction arguments, repetition, fold, landmark count, input channels, image size, heatmap
sigma, three-channel input-normalisation constants and their training-only source, coordinate conventions, augmentation policy, training settings, and explicitly named validation checkpoint metrics. A concrete checkpoint metadata block is written
only when a concrete checkpoint is being described; run-level summary metadata does not contain null checkpoint placeholders.

Runtime metadata includes framework version `0.1`, Python and operating-system information, dependency versions, the resolved compute device, AMP state, CUDA availability,
PyTorch CUDA build, cuDNN version/settings, selected GPU name/index/compute capability/memory, NVIDIA driver when available, and Git commit/branch/dirty state when the source
is inside a Git worktree.

`run_info.json` contains the resolved run, repetition, fold, data, training, and model configurations. It is rewritten after automatic channel detection so that the final
metadata reflects the model that was actually trained. It also contains the current training status, termination reason, runtime environment, session history and timings.

## Run and task names

`TASK_NAME` and `--run-name` are used as directory components. Unsupported characters are replaced with underscores, and empty cleaned names are rejected.

When `--run-name` is omitted, the package creates a readable name containing the repetition count, folds per repetition, point count, model, image size, heatmap sigma,
loss, oversampling factor, batch size, and learning rate. A 12-character SHA-256 configuration fingerprint is appended.

The fingerprint includes all data-processing, optimisation, early-stopping, AMP, and model options that can affect the trained result. It also includes a SHA-256 digest of
every active training and validation list, so regenerated fold memberships do not silently reuse an earlier output directory. The complete fold-collection digest is
stored in `run_info.json`.

The annotation and image contents are not included in the fingerprint. Use a distinct `TASK_NAME` or explicit `--run-name` when training different datasets that use the
same fold membership and otherwise identical settings.

## Standardised result presentation (v0.1)

Validation workbooks use `validation_image_summary` and `validation_endpoints`;
standalone inference workbooks use `image_summary` and `endpoints`. All image,
endpoint and wide prediction CSVs include `network_name`.

Both frameworks cycle landmark response colours in this order: red, blue, yellow,
green, cyan, magenta, orange and purple, repeating after eight landmarks. Each
response is scaled by its own positive maximum, with negative values clipped to
zero; overlapping colours are added and clipped. Heatmap endpoint labels use the
same landmark colours. Point-only overlays retain green ground truth and red
predictions. Colours identify landmarks, not comparable confidence scores.

Training plots use a single loss panel and a second axis for mean endpoint error
in original-image pixels. Heatmaps also plots training endpoint error; IPV retains
classification accuracy in its CSV log. Losses remain framework-specific.

From the repository root, run the cross-framework output checks with:
`python -m unittest discover -s tests -v`.

Standalone inference model, input, output and ground-truth paths are empty strings.
Set the three required paths before running; leave the ground-truth path empty
for inference without annotations.

### Gain, contrast and gamma augmentation

The default augmented-training-copy sequence is RandomAffine, RandomGlobalGain,
RandomContrast, RandomGamma, GaussianNoise, GaussianBlur. Original training copies,
validation and inference do not receive these stochastic augmentations.

Each new intensity transform has an independent 50% probability. Gain multiplies
intensities by a factor in [0.8, 1.2]. Contrast scales deviations from each channel's
nonblack-content mean by a shared factor in [0.8, 1.2]. Gamma uses `image ** gamma`
with gamma in [0.8, 1.25] (below one brightens; above one darkens).
These are mild image-domain approximations, not calibrated scanner controls.

The same sampled factor is used across colour channels; replicated greyscale RGB
channels remain equal. Alpha and landmark coordinates are unchanged. The new
transforms preserve all-zero background pixels and clip colour values to [0, 1].
They run before normalisation and letterboxing, so canvas padding is unaffected.
Existing noise and blur retain their previous behaviour. Settings are recorded in
augmentation metadata and sampled factors in the transform preview logs.

Edit the ranges and probability in `Heatmaps/heatmap_transforms.py`. The preview
utility also accepts `gain`, `contrast` and `gamma` individually.

### Enforced greyscale

Use `--enforce-greyscale true` to convert source images to three identical
luminance channels before preprocessing (default: `false`). RGB uses
0.299 R + 0.587 G + 0.114 B; single-channel images are replicated and RGBA
uses its RGB channels, ignoring alpha. This removes colour differences,
not image annotations or watermarks.

With `--normalise-inputs true`, statistics are calculated from the converted
training inputs, giving identical means and standard deviations across channels.
The setting is saved in checkpoints and applied automatically at inference.
Checkpoints must declare this policy; earlier checkpoints are unsupported.


## MedSAM landmark model

The `vit-medsam` encoder is documented in [MEDSAM.md](MEDSAM.md). All architectures use the shared advanced training workflow above. Version remains `0.1`; earlier checkpoint layouts and retired custom ViTPose constructor flags are not supported. Start fresh runs after this refactor.

### Training diagnostic plots

Heatmap training updates three PNGs after each epoch and rebuilds them when resuming:

- `training_validation_plot.png`: the existing combined total-loss and pixel-error plot.
- `individual_loss_plot.png`: each recorded loss contribution, with matching colours for training (solid) and validation (dashed). Values include configured weights; auxiliary heads each include their share of the auxiliary average. Disabled contributions and the sagittal constraint's unused angle term are omitted. Enabled terms remain visible even when their value is zero.
- `learning_rate_plot.png`: optimiser learning rates at the start of each epoch, before the scheduler advances. Single-rate models use the left axis. Pretrained models use the left axis for the decoder and the right axis for the encoder; the encoder rate is its configured rate, including while its parameters are frozen during warm-up.

Component values and the second learning rate are also saved in the training CSV and checkpoint history. History formats without these optional columns are supported: unrecorded values are blank in the CSV and gaps in the plots, rather than reconstructed estimates. The existing strict resume signature still requires identical source code and run settings, so checkpoints created before this code update cannot be resumed with the updated implementation. An individual-loss plot is generated once component history is available.
