# Image-Patch Voting (IPV) Landmark Identification

Dataset creation, model training, validation inference, and standalone inference utilities for Image-Patch Voting (IPV) landmark localisation in 2D medical images.

The pipeline creates multi-scale image patches from annotated images, trains a quadruplet network to predict distance and angle classes for each ordered landmark, saves
checkpoints with self-describing metadata, and can run full-image landmark inference using voting maps. The original use case is prostate volume endpoint localisation
from transabdominal ultrasound, but the code can be reused for other 2D medical-image landmark-identification tasks that fit the same patch-voting formulation.

## Scope

This repository is designed for:

- 2D image files such as PNG, JPG, BMP, and TIFF;
- ordered landmark localisation with between 1 and 30 points per image;
- patch-based IPV training and inference;
- distance-class and angle-class prediction for each landmark;
- validation overlays and endpoint-error summaries.

This is not a general-purpose medical-image framework. DICOM/NIfTI loading, physical pixel-spacing handling, and task-specific volume calculations are outside the current
implementation.

The current model is a quadruplet model, so every training sample uses exactly four scaled sub-patches. Each landmark point adds two output heads to the model: one
distance head and one angle head.

## Repository structure

Expected layout:

```text
.
├── IPV/
│   ├── __init__.py
│   ├── custom_dataset.py
│   ├── data_creator.py
│   ├── infer_landmarks.py
│   ├── ipv_training_pipeline.py
│   ├── model_registry.py
│   ├── parameters.py
│   ├── quadruplet.py
│   ├── train_model.py
│   └── utils/
│       ├── generate_folds.py
│       ├── landmark_inference_utils.py
│       ├── patch_utils.py
│       └── progress_bar.py
├── run_pipeline.ps1
├── run_pipeline.sh
├── pyproject.toml
└── README.md
```

`pyproject.toml` exposes these entry points:

```toml
ipv-train = "IPV.ipv_training_pipeline:main"
ipv-infer = "IPV.infer_landmarks:main"
```

`ipv-train` is a configurable command-line tool. `ipv-infer` currently runs the editable example script in `IPV/infer_landmarks.py`, so inference paths and switches must
be edited in that file before use.

## Installation

### 1. Create a Python environment

Ubuntu/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch and torchvision

Install PyTorch and torchvision for your machine before installing/running this project. The correct command depends on whether you need CPU-only, CUDA, or another
accelerator build.

Torch is intentionally not pinned in `pyproject.toml`, because CUDA-specific installs should be chosen explicitly for the target machine. ResNet backbones require a
working torchvision installation. The `small_cnn` model still requires PyTorch, but it does not require torchvision model weights.

### 3. Install the package

From the directory that contains `pyproject.toml`:

```bash
pip install -e .
```

Then check the training CLI:

```bash
ipv-train --help
```

Modules can also be run directly:

```bash
python -m IPV.ipv_training_pipeline --help
python -m IPV.infer_landmarks
```

## Data layout

Keep image data and generated run outputs outside the Git repository. A typical prostate IPV data layout is:

```text
DATA/
├── folds/
│   ├── train_f1.txt
│   ├── val_f1.txt
│   ├── test_f1.txt
│   ├── train_f2.txt
│   ├── val_f2.txt
│   ├── test_f2.txt
│   └── ...
├── transverse/
│   ├── A1.jpg
│   ├── A2.jpg
│   └── ...
├── sagittal/
│   ├── A1.jpg
│   ├── A2.jpg
│   └── ...
├── transverse_points_list.txt
└── sagittal_points_list.txt
```

`TASK_NAME` is used in output paths and metadata. For the original prostate endpoint tasks, common task names are `prostate_transverse` with `--num-points 4` and
`prostate_sagittal` with `--num-points 2`.

## Mark-list format

Each mark-list row must start with the image filename followed by coordinate pairs. Sample names are taken from the image filename stem, so `A1.jpg` becomes sample `A1`.

Transverse prostate example with four endpoints:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
A10.jpg (252, 180) (362, 215) (263, 280) (163, 239)
```

Sagittal prostate example with two endpoints:

```text
A1.jpg (205, 237) (223, 355)
A2.jpg (198, 354) (238, 466)
```

The number of coordinate pairs must be at least `--num-points`. If a row contains more coordinate pairs than requested, only the first `--num-points` points are used.
Duplicate sample names are rejected, and requested landmark points must lie inside the matching image dimensions.

## Fold lists

Each fold-list file contains one sample name per line, without the image extension:

```text
A1
A5
A7
```

The training pipeline discovers folds from files named `train_fN.txt`. Fold numbers must be contiguous from `train_f1.txt`. For every fold number `N`, the fold directory
must contain:

```text
train_fN.txt
val_fN.txt
test_fN.txt
```

Train, validation, and test lists for a fold must not overlap.

### Generating fold lists

`IPV/utils/generate_folds.py` creates deterministic 5-fold train/test/validation lists from one mark-list file. It is configured with top-level variables and switches.
Edit these values first:

```python
NUM_FOLDS = 5
SEED = 42
MARK_LIST_PATH = Path(r'D:\path\to\points_list.txt')
OUTPUT_DIR = Path(r'D:\path\to\folds')
```

Then run:

```bash
python -m IPV.utils.generate_folds
```

The script writes `train_fN.txt`, `test_fN.txt`, and `val_fN.txt` files for each fold. The current implementation is intentionally fixed to 5 folds, giving an approximate
80/10/10 train/test/validation split.

## Image handling

Training patches preserve the source image channel count:

- greyscale images create one-channel patches;
- RGB images create three-channel patches;
- RGBA images create four-channel patches.

During dataset loading, patches are converted to channel-first tensors with shape:

```text
[num_sub_patches, channels, height, width]
```

The model detects the channel count from generated training and validation patches before construction. ResNet and small-CNN backbones are built with a matching first
convolution. For pretrained ResNet backbones, first-layer weights are adapted when the patch channel count is not three.

All images within one generated training run should have the same channel count. Saved PNG patches support greyscale, RGB, and RGBA data. Overlay/debug images are loaded
separately from the original source image for visualisation.

## Task configuration

Static task settings are stored in `IPV/parameters.py`:

- `sub_patch_scales`: four patch sizes used by the quadruplet model;
- `sampling_variances`: training-centre sampling variances around each landmark;
- `distance_intervals`: class boundaries for distance labels;
- `angle_intervals`: class boundaries for angle labels.

Run-specific values are supplied through command-line arguments, including fold number, task name, point count, paths, worker counts, training schedule, patch count, grid
spacing, and model backbone.

## Running the training pipeline

Command structure:

```bash
ipv-train FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Equivalent direct module usage:

```bash
python -m IPV.ipv_training_pipeline FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Boolean arguments accept values such as `true`, `false`, `yes`, `no`, `1`, and `0`.

Minimal example:

```bash
python -m IPV.ipv_training_pipeline \
    1 prostate_transverse true true true false \
    --run-dir "$HOME/IPV_TRAINING" \
    --save-dir "$HOME/IPV_SAVING" \
    --num-points 4 \
    --fold-lists-path "$HOME/DATA/folds" \
    --mark-list-file "$HOME/DATA/transverse_points_list.txt" \
    --image-data-dir "$HOME/DATA/transverse" \
    --data-creation-workers 8 \
    --train-workers 8 \
    --random-seed 42 \
    --keep-part-csvs false \
    --generate-test-data false \
    --batch-size 64 \
    --max-training-epochs 15 \
    --learning-rate 0.01 \
    --lr-schedule true \
    --loss-print-samples 3200 \
    --patches-per-training-sample 200 \
    --val-grid-spacing 10 \
    --network-name small_cnn \
    --branch-features 128 \
    --frozen-stages 0 \
    --small-input-stem false
```

The supplied `run_pipeline.sh` and `run_pipeline.ps1` files are editable examples. Review all path variables and switches before running them, especially `DELETE_FILES`.
Leave `DELETE_FILES=false` until you are confident that the generated training-data artefacts can be removed.

### Important training options

| Option                              | Description                                                                                            |
|-------------------------------------|--------------------------------------------------------------------------------------------------------|
| `--run-dir`                         | Required root directory for generated training data and model results.                                 |
| `--save-dir`                        | Required only when `COPY_FILES=true`; ignored when `COPY_FILES=false`.                                 |
| `--num-points`                      | Number of ordered landmark points per image. Must be between 1 and 30.                                 |
| `--fold-lists-path`                 | Directory containing `train_fN.txt`, `val_fN.txt`, and `test_fN.txt`.                                  |
| `--mark-list-file`                  | Text file containing image filenames and point coordinates.                                            |
| `--image-data-dir`                  | Directory containing the source images.                                                                |
| `--data-creation-workers`           | Number of worker processes for patch/data creation.                                                    |
| `--train-workers`                   | Number of PyTorch DataLoader workers. Use `0` for single-process loading.                              |
| `--random-seed`                     | Seed used for deterministic sampled training centres.                                                  |
| `--keep-part-csvs`                  | Keep temporary per-sample CSV files after merging.                                                     |
| `--generate-test-data`              | Generate test CSVs, patches, and overlay images when true. Test fold-list checks still run when false. |
| `--batch-size`                      | Training batch size.                                                                                   |
| `--max-training-epochs`             | Maximum training epochs.                                                                               |
| `--learning-rate`                   | Initial SGD learning rate.                                                                             |
| `--lr-schedule`                     | Enable the validation-accuracy-triggered StepLR scheduler.                                             |
| `--lr-step-size`                    | StepLR step size. Default: `1`.                                                                        |
| `--lr-gamma`                        | StepLR multiplicative decay factor. Default: `0.1`.                                                    |
| `--early-stop-patience`             | Number of validation epochs without sufficient loss improvement before early stopping.                 |
| `--early-stop-min-delta`            | Minimum validation-loss improvement needed to reset patience.                                          |
| `--early-stop-warmup-epochs`        | Initial epochs before early stopping is allowed.                                                       |
| `--loss-print-samples`              | Approximate sample interval used to derive validation/logging batch interval.                          |
| `--save-validation-results`         | Run full validation-image inference after training. Default: `true`.                                   |
| `--validation-inference-batch-size` | Batch size for validation-image inference. Default: `2048`.                                            |
| `--validation-vote-smoothing-sigma` | Gaussian smoothing sigma used before selecting vote-map peaks. Default: `7.0`.                         |
| `--validation-save-raw-vote-maps`   | Save raw per-image vote maps as `.npy` files. These can be large.                                      |
| `--patches-per-training-sample`     | Sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`.     |
| `--val-grid-spacing`                | Pixel stride for validation and optional test grid-centre creation.                                    |
| `--network-name`                    | Backbone name from `IPV/model_registry.py`.                                                            |
| `--branch-features`                 | Feature count output by each branch before concatenation.                                              |
| `--frozen-stages`                   | Number of pretrained ResNet stages to freeze. Use `0` for untrained models and `small_cnn`.            |
| `--small-input-stem`                | Use the small-input ResNet stem. Use `false` for `small_cnn`.                                          |
| `--run-name`                        | Optional custom run name. If omitted, a deterministic name is generated.                               |

## Supported models

Available model names are defined in `IPV/model_registry.py`. The `Quadruplet` class uses four copies of the selected branch, one for each patch scale.

| Model name            | Notes                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------|
| `small_cnn`           | Lightweight CNN for 64 × 64 patch inputs. Useful for fast local testing.              |
| `resnet10_untrained`  | Small custom ResNet trained from scratch.                                             |
| `resnet14_untrained`  | Small custom ResNet trained from scratch.                                             |
| `resnet18_untrained`  | ResNet-18 trained from scratch.                                                       |
| `resnet18_pretrained` | ResNet-18 with torchvision ImageNet weights. Requires cached or downloadable weights. |
| `resnet34_untrained`  | ResNet-34 trained from scratch.                                                       |
| `resnet34_pretrained` | ResNet-34 with torchvision ImageNet weights. Requires cached or downloadable weights. |

For untrained models, use `--frozen-stages 0`. For `small_cnn`, use `--small-input-stem false` and `--frozen-stages 0`.

## Run and save directories

Every run uses a required `--run-dir`. The pipeline creates these high-level directories inside it:

```text
<RUN_DIR>/
├── TRAINING_DATA/
│   └── <TASK_NAME>/
│       └── <NUM_FOLDS>_Folds/
│           └── <SCALES>_<NUM_POINTS>points_<PATCHES_PER_TRAINING_SAMPLE>pertrainingsample/
└── TRAINING_RESULTS/
    └── <TASK_NAME>/
        └── <RUN_NAME>/
```

Generated fold data is shared by compatible runs. Model checkpoints, logs, plots, metadata, and validation-inference outputs are written under the run results directory.

When `COPY_FILES=true`, result files and subdirectories from:

```text
<RUN_DIR>/TRAINING_RESULTS/<TASK_NAME>/<RUN_NAME>/
```

are copied to:

```text
<SAVE_DIR>/<TASK_NAME>/<RUN_NAME>/
```

When `DELETE_FILES=true`, only generated training-data artefacts for the current fold/task/sample configuration are deleted from `<RUN_DIR>/TRAINING_DATA`. Files in
`<RUN_DIR>/TRAINING_RESULTS` and the optional save directory are not deleted.

## Generated outputs

Generated fold data is written under:

```text
<RUN_DIR>/TRAINING_DATA/<TASK_NAME>/<NUM_FOLDS>_Folds/<SCALES>_<NUM_POINTS>points_<PATCHES_PER_TRAINING_SAMPLE>pertrainingsample/
```

Common generated training-data files include:

```text
Train_f<FOLD>.csv
Val_f<FOLD>.csv
Test_f<FOLD>.csv                 # only when test data generation is enabled
Train_Patches_F<FOLD>/
Val_Patches_F<FOLD>/
Test_Patches_F<FOLD>/            # only when test data generation is enabled
Train_Images_F<FOLD>/
Val_Images_F<FOLD>/
Test_Images_F<FOLD>/             # only when test data generation is enabled
data_info_f<FOLD>.csv
run_info_<TASK_NAME>_f<FOLD>.json
```

If `--keep-part-csvs true` is used, temporary per-sample CSV directories may also be retained:

```text
Train_csv_parts_F<FOLD>/
Val_csv_parts_F<FOLD>/
Test_csv_parts_F<FOLD>/
```

Training results are written under:

```text
<RUN_DIR>/TRAINING_RESULTS/<TASK_NAME>/<RUN_NAME>/
```

Common result files include:

```text
run_info_<TASK_NAME>_f<FOLD>.json
train_log_f<FOLD>.csv
train_plot_f<FOLD>.png
model_f<FOLD>_best.pth
model_f<FOLD>_last.pth
checkpoint_summary_f<FOLD>.json
validation_inference_f<FOLD>/     # when validation-image inference is enabled
```

Validation-image inference outputs are written under `validation_inference_f<FOLD>/`:

```text
validation_inference_f<FOLD>/
├── heatmap_overlays/
├── point_overlays/
├── vote_maps/
├── raw_vote_maps/                       # only when enabled
├── logs/
│   └── validation_inference_run_metadata.json
└── validation_inference_summary.xlsx
```

The Excel workbook contains:

- `image_summary`: one row per image, including point-count and mean/max endpoint-error fields when ground truth is available;
- `endpoints`: one row per predicted endpoint, including predicted coordinates, optional ground-truth coordinates, optional pixel error, peak vote value, and
  run/checkpoint metadata.

## Running standalone inference

`IPV/infer_landmarks.py` is currently an editable example script rather than a fully parameterised CLI. Edit the path variables and switches at the top of the file before
running it:

```python
MODEL_PATH = Path(r'D:\path\to\model_f1_best.pth')
INPUT_PATH = Path(r'D:\path\to\images')
OUTPUT_DIR = Path(r'D:\path\to\inference_outputs')
GROUND_TRUTH_MARK_LIST_PATH = None
```

Then run:

```bash
python -m IPV.infer_landmarks
```

or, after installing the package:

```bash
ipv-infer
```

The script loads checkpoint metadata saved during training, so `num_points`, patch scales, class intervals, image channel count, grid spacing, and model constructor
arguments are inferred automatically. Runtime overrides currently available in the script include:

```python
DEVICE = 'auto'
BATCH_SIZE = 4096
GRID_SPACING_OVERRIDE = 10
VOTE_SMOOTH_SIGMA_OVERRIDE = None
USE_PROBABILITY_WEIGHTS = True
SAVE_RAW_VOTE_MAPS = False
RECURSIVE_IMAGE_SEARCH = False
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
RUN_LABEL = 'inference'
```

Reusable inference utilities live in `IPV/utils/landmark_inference_utils.py`.

### Standalone inference outputs

Standalone inference outputs are written under `OUTPUT_DIR`:

```text
OUTPUT_DIR/
├── heatmap_overlays/
├── point_overlays/
├── vote_maps/
├── raw_vote_maps/              # only when enabled
├── logs/
│   └── inference_run_metadata.json
└── inference_summary.xlsx
```

The summary workbook contains `image_summary` and `endpoints` sheets. If `GROUND_TRUTH_MARK_LIST_PATH` is supplied, endpoint pixel-error metrics are included. Without
ground truth, predicted endpoint coordinates and peak vote values are still saved.

## Checkpoint metadata

Training checkpoints include metadata that describes the task, model constructor arguments, preprocessing settings, input channel count, and inference defaults.
Standalone inference uses this metadata to rebuild the correct model and preprocessing configuration.

Current inference requires checkpoints saved with the current metadata schema. Older checkpoint formats may need conversion or retraining.

## Development and repository hygiene

Recommended before committing or publishing:

- run `python -m compileall IPV` after laying the files out under the expected `IPV/` package directory;
- run `ipv-train --help` after `pip install -e .`;
- run a tiny smoke test with `small_cnn`, `--train-workers 0`, and a very small toy dataset;
- keep raw datasets, generated patches, checkpoints, and run outputs outside Git;
- add or maintain a `.gitignore` for local environments, caches, generated data, and model artefacts.

Suggested `.gitignore` entries include:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
build/
dist/
*.pth
*.npy
TRAINING_DATA/
TRAINING_RESULTS/
IPV_TRAINING/
IPV_SAVING/
DATA/
```

## Notes and limitations

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- Duplicate mark-list sample names are rejected.
- Requested mark-list points must lie inside the matching image dimensions.
- `--num-points` must be between 1 and 30.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Static task settings currently live in `IPV/parameters.py`; edit this file before running a new task configuration.
- Standalone inference currently uses editable constants rather than command-line flags.
- Generated metadata and CSV label counts are checked before training starts.
- Training from existing data checks for partial fold data before model training starts.
- This code operates on standard 2D image files. DICOM/NIfTI loading and physical pixel spacing are not currently handled.
- Dimension-summary outputs are not currently implemented in the reusable inference pipeline.
