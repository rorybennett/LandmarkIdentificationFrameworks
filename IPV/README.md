# Image-Patch Voting (IPV) Landmark Identification

Dataset creation, training, validation inference, and standalone inference tools for 
Image-Patch Voting (IPV) landmark localisation in 2D medical images.

The pipeline creates multi-scale image patches from annotated images, 
trains a quadruplet network to predict distance and angle classes for each 
ordered landmark, saves checkpoints with self-describing metadata, and can run 
full-image landmark inference using voting maps. The original use case is prostate
volume estimation from transabdominal ultrasound, but the code is intended to be usable 
for other 2D medical image landmark-identification tasks that fit the same IPV formulation.

## Scope

The package supports between 1 and 30 ordered landmark points per image. Each point adds two 
output heads to the model: one distance-class head and one angle-class head. The current model 
is a quadruplet model, so each sample uses exactly four scaled sub-patches.

This repository is not a general-purpose medical image framework. It is designed for 2D image 
files, patch-based IPV landmark localisation, and endpoint/dimension summary workflows.

## Repository structure

The package should be laid out as follows:

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

The expected command-line entry points in `pyproject.toml` are:

```toml
ipv-train = "IPV.ipv_training_pipeline:main"
ipv-infer = "IPV.infer_landmarks:main"
```

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

Install a PyTorch and torchvision build that matches your CUDA or CPU environment before 
installing or running the project. The correct command depends on your machine and should 
be selected from the PyTorch installation guide.

### 3. Install the package

Install in editable mode from the directory that contains `pyproject.toml`:

```bash
pip install -e .
```

After installation, the command-line tools should be available:

```bash
ipv-train --help
ipv-infer
```

The modules can also be run directly:

```bash
python -m IPV.ipv_training_pipeline --help
python -m IPV.infer_landmarks
```

## Data layout

Keep image data and generated run outputs outside the Git repository. A typical prostate 
IPV data layout is:

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

`TASK_NAME` is used in output paths and metadata. For the original prostate endpoint tasks, 
common task names are `prostate_transverse` with `--num-points 4` and `prostate_sagittal` 
with `--num-points 2`.

## Mark-list format

Each mark-list row must start with the image filename followed by coordinate pairs. Sample names 
are taken from the image filename stem, so `A1.jpg` becomes sample `A1`.

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

The number of coordinate pairs must be at least `--num-points`. If a row contains more 
coordinate pairs than requested, only the first `--num-points` points are used. Duplicate 
sample names are rejected, and the requested landmark points must lie inside the matching 
image dimensions.

## Fold lists

Each fold-list file contains one sample name per line, without the image extension:

```text
A1
A5
A7
```

The training pipeline discovers folds from files named `train_fN.txt`. Fold numbers must be 
contiguous from `train_f1.txt`. For every fold number `N`, the fold directory must contain:

```text
train_fN.txt
val_fN.txt
test_fN.txt
```

The train, validation, and test lists for each fold must not overlap.

### Generating fold lists

`IPV/utils/generate_folds.py` creates deterministic 5-fold train/test/validation splits from 
one mark-list file. It is configured with top-level path variables and switches. Edit these 
values first:

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

The script writes `train_fN.txt`, `test_fN.txt`, and `val_fN.txt` files for each fold. 
The current implementation is intentionally fixed to 5 folds, giving an approximate 
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

The model detects the channel count from generated training and validation patches before 
construction. ResNet and small-CNN backbones are built with a matching first convolution.
For pretrained ResNet backbones, first-layer weights are adapted when the patch channel 
count is not three.

All images within one generated training run should have the same channel count. Saved PNG 
patches support greyscale, RGB, and RGBA data. Overlay/debug images are loaded separately from
the original source image for visualisation.

## Task configuration

Static task settings are stored in `IPV/parameters.py`:

- `sub_patch_scales`: four patch sizes used by the quadruplet model;
- `sampling_variances`: training-centre sampling variances around each landmark;
- `distance_intervals`: class boundaries for distance labels;
- `angle_intervals`: class boundaries for angle labels.

Run-specific values are supplied through command-line arguments, including fold number, 
task name, point count, paths, workers, training schedule, patch count, grid spacing, and 
model backbone.

## Running the training pipeline

The command structure is:

```bash
ipv-train FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Equivalent direct module usage:

```bash
python -m IPV.ipv_training_pipeline FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Boolean arguments accept values such as `true`, `false`, `yes`, `no`, `1`, and `0`.

A minimal example is:

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

The supplied `run_pipeline.sh` and `run_pipeline.ps1` scripts are intended as editable examples. 

### Important training options

| Option | Description |
|---|---|
| `--run-dir` | Required root directory for generated training data and model results. |
| `--save-dir` | Required only when `COPY_FILES=true`; used for copied result files. |
| `--num-points` | Number of ordered landmark points per image. Must be between 1 and 30. |
| `--fold-lists-path` | Directory containing `train_fN.txt`, `val_fN.txt`, and `test_fN.txt`. |
| `--mark-list-file` | Text file containing image filenames and point coordinates. |
| `--image-data-dir` | Directory containing the source images. |
| `--data-creation-workers` | Number of worker processes for patch/data creation. |
| `--train-workers` | Number of PyTorch DataLoader workers. Use 0 for single-process loading. |
| `--random-seed` | Seed used for deterministic sampled training centres. |
| `--keep-part-csvs` | Keep temporary per-sample CSV files after merging. |
| `--generate-test-data` | Generate test CSVs, patches, and overlay images when true. Fold-list checks still run when false. |
| `--batch-size` | Training batch size. |
| `--max-training-epochs` | Maximum training epochs. |
| `--learning-rate` | Initial SGD learning rate. |
| `--lr-schedule` | Enable the validation-accuracy-triggered StepLR scheduler. |
| `--lr-step-size` | StepLR step size. Default: 1. |
| `--lr-gamma` | StepLR multiplicative decay factor. Default: 0.1. |
| `--early-stop-patience` | Number of validation epochs without sufficient loss improvement before early stopping. |
| `--early-stop-min-delta` | Minimum validation-loss improvement needed to reset patience. |
| `--early-stop-warmup-epochs` | Initial epochs before early stopping is allowed. |
| `--loss-print-samples` | Approximate sample interval used to derive validation/logging batch interval. |
| `--save-validation-results` | Run full validation-image inference after training. Default: true. |
| `--validation-inference-batch-size` | Batch size for validation-image inference. Default: 2048. |
| `--validation-vote-smoothing-sigma` | Gaussian smoothing sigma used before selecting vote-map peaks. Default: 7.0. |
| `--validation-save-raw-vote-maps` | Save raw per-image vote maps as `.npy` files. These can be large. |
| `--patches-per-training-sample` | Sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`. |
| `--val-grid-spacing` | Pixel stride for validation and optional test grid-centre creation. |
| `--network-name` | Backbone name from `IPV/model_registry.py`. |
| `--branch-features` | Feature count output by each branch before concatenation. |
| `--frozen-stages` | Number of pretrained ResNet stages to freeze. Use 0 for untrained models and `small_cnn`. |
| `--small-input-stem` | Use the small-input ResNet stem. Use `false` for `small_cnn`. |
| `--run-name` | Optional custom run name. If omitted, a deterministic name is generated. |

## Supported models

Available model names are defined in `IPV/model_registry.py`. The `Quadruplet` class 
uses four copies of the selected branch, one for each patch scale.

| Model name | Notes |
|---|---|
| `small_cnn` | Lightweight CNN for 64 × 64 patch inputs. Useful for fast local testing. |
| `resnet10_untrained` | Small custom ResNet trained from scratch. |
| `resnet14_untrained` | Small custom ResNet trained from scratch. |
| `resnet18_untrained` | ResNet-18 trained from scratch. |
| `resnet18_pretrained` | ResNet-18 with torchvision ImageNet weights. Requires cached or downloadable weights. |
| `resnet34_untrained` | ResNet-34 trained from scratch. |
| `resnet34_pretrained` | ResNet-34 with torchvision ImageNet weights. Requires cached or downloadable weights. |

For untrained models, use `--frozen-stages 0`. For `small_cnn`, use `--small-input-stem false` 
and `--frozen-stages 0`.

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

Generated fold data is shared by compatible runs. Model checkpoints, logs, plots, metadata,
and validation-inference outputs are written under the run results directory.

When `COPY_FILES=true`, result files and subdirectories from:

```text
<RUN_DIR>/TRAINING_RESULTS/<TASK_NAME>/<RUN_NAME>/
```

are copied to:

```text
<SAVE_DIR>/<TASK_NAME>/<RUN_NAME>/
```

When `DELETE_FILES=true`, only generated training-data artefacts for the current
fold/task/sample configuration are deleted from `<RUN_DIR>/TRAINING_DATA`. Files in 
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

Validation-image inference outputs include combined heatmap overlays, predicted-versus-ground-truth
endpoint overlays, per-endpoint vote-map overlays, one Excel metrics workbook per validation image,
combined CSV/XLSX summaries, and run metadata.

## Running standalone inference

`IPV/infer_landmarks.py` is an editable example script for applying a trained checkpoint to 
one image or a directory of images. Edit the path variables and switches at the top of the file:

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

The script loads checkpoint metadata saved during training, so `num_points`, patch scales, 
class intervals, image channel count, grid spacing, and model constructor arguments are inferred
automatically. Runtime overrides include `BATCH_SIZE`, `GRID_SPACING_OVERRIDE`, 
`VOTE_SMOOTH_SIGMA_OVERRIDE`, `USE_PROBABILITY_WEIGHTS`, `SAVE_RAW_VOTE_MAPS`, 
`RECURSIVE_IMAGE_SEARCH`, and `DIMENSION_POINT_MAP`.

Reusable inference utilities live in `IPV/utils/landmark_inference_utils.py`.

For non-prostate landmark tasks, leave `DIMENSION_POINT_MAP = None` or provide a custom 
one-indexed endpoint pairing dictionary:

```python
DIMENSION_POINT_MAP = {'vertical': (1, 3), 'horizontal': (2, 4)}
```

Without a custom map, prostate-style dimension summaries are produced only when the 
task name contains `transverse` or `sagittal`. Endpoint predictions are always saved.

Standalone inference outputs are written under `OUTPUT_DIR`:

```text
OUTPUT_DIR/
├── heatmap_overlays/
├── point_overlays/
├── vote_maps/
├── raw_vote_maps/        # only when enabled
├── metrics/
├── logs/
├── inference_summary.xlsx
├── inference_image_summary.csv
├── inference_endpoint_predictions.csv
└── inference_dimension_predictions.csv
```

## Notes and limitations

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- Duplicate mark-list sample names are rejected.
- The requested mark-list points must lie inside the matching image dimensions.
- `--num-points` must be between 1 and 30.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Static task settings currently live in `IPV/parameters.py`; edit this file before running a new task configuration.
- Standalone inference requires checkpoints saved with the current self-describing metadata format.
- Generic tasks can use endpoint predictions directly; dimension summaries require a recognised prostate task name or a custom `DIMENSION_POINT_MAP`.
- Generated metadata and CSV label counts are checked before training starts.
- Training from existing data checks for partial fold data before model training starts.
- This code operates on standard 2D image files. DICOM/NIfTI loading and physical pixel spacing are not currently handled.
- Dimension calculations are only valid for images that are resampled to a known pixel density. 
Other dimension calculations would require custom functions.