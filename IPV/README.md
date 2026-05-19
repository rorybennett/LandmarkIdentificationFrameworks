# Image-Patch Voting (IPV) Landmark Identification

Dataset creation and quadruplet model training tools for Image-Patch Voting
(IPV) landmark identification.

The pipeline creates multi-scale grayscale patch data from annotated images,
trains a quadruplet network, writes checkpoints, logs, and metadata to a run
directory, optionally copies selected result files to a separate save directory,
and can safely delete generated training-data artefacts.

The package is task-agnostic for 2D image landmark localisation. Each task can
use between 1 and 30 ordered landmark points per image. The original prostate
IPV use case uses 4 endpoints on transverse transabdominal ultrasound images
and 2 endpoints on sagittal transabdominal ultrasound images.

## Repository structure

The package is expected to be laid out as follows:

```text
.
├── IPV/
│   ├── __init__.py
│   ├── create_dataset_and_train_model.py
│   ├── custom_dataset.py
│   ├── data_creator.py
│   ├── gpu_utils.py
│   ├── model_registry.py
│   ├── parameters.py
│   ├── quadruplet.py
│   └── train_model.py
├── pyproject.toml
└── README.md
```

`pyproject.toml` exposes the command-line entry point `ipv-train`.

## Installation

### 1. Create an environment

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

Install a PyTorch and torchvision build that matches your CUDA or CPU
environment before installing/running the project. This project imports both
`torch` and `torchvision`, but they are not pinned in `pyproject.toml` because
the correct build depends on your machine.

Use the selector at the PyTorch website to choose the correct command for your
environment.

### 3. Install the IPV package

Install in editable mode from the directory that contains `pyproject.toml`:

```bash
git clone https://github.com/rorybennett/LandmarkIdentificationFrameworks
cd LandmarkIdentificationFrameworks/IPV
pip install -e .
```

After installation, the CLI entry point should be available as:

```bash
ipv-train --help
```

You can also run the module directly:

```bash
python -m IPV.create_dataset_and_train_model --help
```

## Image handling

Training patches are intentionally grayscale.

Colour source images are accepted, but the patch-creation stage converts each
source image to grayscale before creating training and validation patches. Saved
patch PNGs are grayscale. During dataset loading, each grayscale patch is loaded
and repeated across three channels so ResNet/torchvision-style models still
receive tensors shaped like 3-channel images.

In other words, the model receives:

```text
R = grayscale
G = grayscale
B = grayscale
```

It does not receive true RGB colour information. This is appropriate for the
original ultrasound use case. For colour-dependent tasks, update both patch
creation and dataset loading before training.

The saved overlay/debug images are separate from the training patches. They are
loaded from the original source image for visualisation and may preserve colour
where applicable.

## Data layout

Keep image data and generated run outputs outside the Git repository. A typical
prostate IPV data layout is:

```text
DATA/
├── folds/
│   ├── train_f1.txt
│   ├── train_f2.txt
│   ├── train_f3.txt
│   ├── train_f4.txt
│   ├── train_f5.txt
│   └── val.txt
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

`TASK_NAME` is used in output paths and metadata. For the original prostate
endpoint tasks, use `transverse` with `--num-points 4` and `sagittal` with
`--num-points 2`.

### Fold lists

Each fold-list file should contain one sample name per line, without the image
extension:

```text
A1
A5
A7
```

The pipeline discovers folds from files named `train_fN.txt`. Fold numbers must
be contiguous from `train_f1.txt`, and `val.txt` must exist for training-time
validation.

### Mark-list files

Each mark-list row must start with the image filename followed by coordinate
pairs. The number of coordinate pairs must match `--num-points`, unless the mark
list contains additional points and you deliberately want the first
`--num-points` points to be used. The maximum supported number of points per
image is 30.

Prostate transverse example with four endpoints:

```text
A1.jpg (236, 214) (342, 271) (245, 354) (134, 291)
A10.jpg (252, 180) (362, 215) (263, 280) (163, 239)
```

Prostate sagittal example with two endpoints:

```text
A1.jpg (205, 237) (223, 355)
A2.jpg (198, 354) (238, 466)
```

The generated CSV files contain five metadata columns followed by two labels per
point:

```text
patch_id, patch_path, sample_name, centre_x, centre_y, point_1_distance,
point_1_angle, point_2_distance, point_2_angle, ...
```

Training checks that `Train_f<FOLD>.csv` and `Val_f<FOLD>.csv` contain the same
number of points as requested by `--num-points`.

## Configuration

Static task settings are stored in `IPV/parameters.py`:

- `sub_patch_scales`: four patch sizes used by the quadruplet model.
- `sampling_variances`: training-centre sampling variances around each landmark.
- `distance_intervals`: class boundaries for distance labels.
- `angle_intervals`: class boundaries for angle labels.

Run-specific settings are supplied through the command line, including fold
number, task name, number of points, batch size, learning rate, number of epochs,
data workers, training workers, patch count, validation grid spacing, run
directory, optional save directory, and model backbone.

## Run and save directories

Every run uses a required `--run-dir`. The current code creates these two
high-level directories inside it:

```text
<RUN_DIR>/
├── TRAINING_DATA/
│   └── <NUM_FOLDS>_Folds/
│       └── <TASK_NAME>/
│           └── <SCALES>_<NUM_POINTS>points_<PATCHES_PER_TRAINING_SAMPLE>pertrainingsample/
│               └── ... generated CSVs, patch images, overlay images, and data metadata
└── TRAINING_RESULTS/
    └── <RUN_NAME>/
        └── <TASK_NAME>/
            └── ... checkpoints, logs, plots, and run metadata
```

`RUN_NAME` is generated deterministically from the run configuration, including
fold count, patch scales, sampling variances, point count, patch count, grid
spacing, network, branch features, frozen stages, stem setting, batch size,
learning rate, epoch count, and random seed.

The CLI currently accepts `--run-name`, but `build_configs()` generates the run
name from the configuration and does not use the provided `--run-name` value. If
you want manual run names, update `build_configs()` before relying on that
argument.

Use `--save-dir` only when you want a copy of selected result files outside the
run directory. When `COPY_FILES=true`, `--save-dir` is required. When
`COPY_FILES=false`, any supplied `--save-dir` is ignored.

When `DELETE_FILES=true`, only generated training-data artefacts for the current
fold/task/sample configuration are deleted from `<RUN_DIR>/TRAINING_DATA`. Files
in `<RUN_DIR>/TRAINING_RESULTS` and the optional save directory are not deleted.

## Supported models

Available network names are defined in `IPV/model_registry.py`.

| Model name | Notes |
|---|---|
| `small_cnn` | Lightweight CNN for 64 × 64 patch inputs. Useful for fast local testing. |
| `resnet10_untrained` | Small custom ResNet trained from scratch. |
| `resnet14_untrained` | Small custom ResNet trained from scratch. |
| `resnet18_untrained` | ResNet-18 trained from scratch. |
| `resnet18_pretrained` | ResNet-18 with torchvision ImageNet weights. Requires cached/downloadable weights. |
| `resnet34_untrained` | ResNet-34 trained from scratch. |
| `resnet34_pretrained` | ResNet-34 with torchvision ImageNet weights. Requires cached/downloadable weights. |

For untrained models, use `--frozen-stages 0`. For `small_cnn`, use
`--small-input-stem false` and `--frozen-stages 0`.

## Running the pipeline

The command has the following structure:

```bash
ipv-train FOLD TASK_NAME CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Positional arguments:

| Argument | Values | Description |
|---|---|---|
| `FOLD` | integer | Fold number to run. |
| `TASK_NAME` | string | Name of the landmark task. For prostate IPV, use `transverse` or `sagittal`. |
| `CREATE_DATA` | boolean | Create patch data before training. |
| `TRAIN_MODEL` | boolean | Train the model. |
| `COPY_FILES` | boolean | Copy selected result files to `--save-dir`. If true, `--save-dir` is required. |
| `DELETE_FILES` | boolean | Delete generated fold training data after completion. |

Boolean values can be supplied as `true`, `false`, `yes`, `no`, `1`, or `0`.

Important options:

| Option | Description |
|---|---|
| `--run-dir` | Required directory used for generated training data and run results. |
| `--save-dir` | Required only when `COPY_FILES=true`; used for copies of selected result files after training. |
| `--num-points` | Number of ordered landmark points per image. Must be between 1 and 30. |
| `--fold-lists-path` | Directory containing `train_fN.txt` files and `val.txt`. |
| `--mark-list-file` | Text file containing image filenames and point coordinates. |
| `--image-data-dir` | Directory containing the source images. |
| `--data-creation-workers` | Number of worker processes used for patch/data creation. |
| `--train-workers` | Number of PyTorch DataLoader workers used during training. |
| `--random-seed` | Seed used for deterministic sampled training centres. |
| `--keep-part-csvs` | Keep per-sample temporary CSV part files if true. |
| `--batch-size` | Training batch size. |
| `--max-training-epochs` | Maximum training epochs. |
| `--learning-rate` | SGD learning rate. |
| `--lr-schedule` | Enable the validation-accuracy-triggered learning-rate scheduler. |
| `--loss-print-samples` | Approximate sample interval used to derive the validation/logging batch interval. |
| `--patches-per-training-sample` | Number of sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`. |
| `--grid-spacing` | Grid stride for validation patch-centre creation. |
| `--network-name` | Model backbone name from the model registry. |
| `--branch-features` | Number of features output by each branch before concatenation. |
| `--frozen-stages` | Number of pretrained ResNet stages to freeze. Use `0` for untrained models and `small_cnn`. |
| `--small-input-stem` | Use the small-input ResNet stem when true. Use `false` for `small_cnn`. |
| `--run-name` | Currently accepted by the parser but not applied by `build_configs()`. |

## Ubuntu example script

Save as `run_prostate_transverse_ubuntu.sh`, edit the paths at the top, then run
with `bash run_prostate_transverse_ubuntu.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/LandmarkIdentificationFrameworks/IPV"
RUN_DIR="$HOME/IPV_TRAINING"
SAVE_DIR="$HOME/IPV_SAVING"

FOLD_LISTS_DIR="$HOME/DATA/folds"
MARK_LIST_FILE="$HOME/DATA/transverse_points_list.txt"
IMAGE_DATA_DIR="$HOME/DATA/transverse"

FOLD=1
TASK_NAME="transverse"
NUM_POINTS=4

CREATE_DATA="true"
TRAIN_MODEL="true"
COPY_FILES="true"
DELETE_FILES="false"
KEEP_PART_CSVS="false"

DATA_CREATION_WORKERS=8
PATCHES_PER_TRAINING_SAMPLE=200
GRID_SPACING=10
RANDOM_SEED=42

TRAIN_WORKERS=8
BATCH_SIZE=64
MAX_TRAINING_EPOCHS=15
LEARNING_RATE=0.01
LR_SCHEDULE="true"
LOSS_PRINT_SAMPLES=1600

NETWORK_NAME="small_cnn"
BRANCH_FEATURES=128
FROZEN_STAGES=0
SMALL_INPUT_STEM="false"

cd "$PROJECT_DIR"

python -m IPV.create_dataset_and_train_model \
    "$FOLD" "$TASK_NAME" "$CREATE_DATA" "$TRAIN_MODEL" "$COPY_FILES" "$DELETE_FILES" \
    --run-dir "$RUN_DIR" \
    --save-dir "$SAVE_DIR" \
    --num-points "$NUM_POINTS" \
    --fold-lists-path "$FOLD_LISTS_DIR" \
    --mark-list-file "$MARK_LIST_FILE" \
    --image-data-dir "$IMAGE_DATA_DIR" \
    --data-creation-workers "$DATA_CREATION_WORKERS" \
    --train-workers "$TRAIN_WORKERS" \
    --random-seed "$RANDOM_SEED" \
    --keep-part-csvs "$KEEP_PART_CSVS" \
    --batch-size "$BATCH_SIZE" \
    --max-training-epochs "$MAX_TRAINING_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --lr-schedule "$LR_SCHEDULE" \
    --loss-print-samples "$LOSS_PRINT_SAMPLES" \
    --patches-per-training-sample "$PATCHES_PER_TRAINING_SAMPLE" \
    --grid-spacing "$GRID_SPACING" \
    --network-name "$NETWORK_NAME" \
    --branch-features "$BRANCH_FEATURES" \
    --frozen-stages "$FROZEN_STAGES" \
    --small-input-stem "$SMALL_INPUT_STEM"
```

If `COPY_FILES="false"`, remove the `--save-dir "$SAVE_DIR"` line or leave it
in place knowing the current code will ignore it after normalising arguments.

## Windows PowerShell example script

Save as `run_prostate_transverse_windows.ps1`, edit the paths at the top, then
run from PowerShell.

```powershell
$PROJECT_DIR = "D:\LandmarkIdentificationFrameworks\IPV"
$RUN_DIR = "D:\IPV_TRAINING"
$SAVE_DIR = "D:\IPV_SAVING"

$FOLD_LISTS_DIR = "D:\DATA\folds"
$MARK_LIST_FILE = "D:\DATA\transverse_points_list.txt"
$IMAGE_DATA_DIR = "D:\DATA\transverse"

$FOLD = 1
$TASK_NAME = "transverse"
$NUM_POINTS = 4

$CREATE_DATA = "true"
$TRAIN_MODEL = "true"
$COPY_FILES = "true"
$DELETE_FILES = "false"
$KEEP_PART_CSVS = "false"

$DATA_CREATION_WORKERS = 8
$PATCHES_PER_TRAINING_SAMPLE = 200
$GRID_SPACING = 10
$RANDOM_SEED = 42

$TRAIN_WORKERS = 8
$BATCH_SIZE = 64
$MAX_TRAINING_EPOCHS = 15
$LEARNING_RATE = 0.01
$LR_SCHEDULE = "true"
$LOSS_PRINT_SAMPLES = 1600

$NETWORK_NAME = "small_cnn"
$BRANCH_FEATURES = 128
$FROZEN_STAGES = 0
$SMALL_INPUT_STEM = "false"

Set-Location $PROJECT_DIR

ipv-train $FOLD $TASK_NAME $CREATE_DATA $TRAIN_MODEL $COPY_FILES $DELETE_FILES `
    --run-dir $RUN_DIR `
    --save-dir $SAVE_DIR `
    --num-points $NUM_POINTS `
    --fold-lists-path $FOLD_LISTS_DIR `
    --mark-list-file $MARK_LIST_FILE `
    --image-data-dir $IMAGE_DATA_DIR `
    --data-creation-workers $DATA_CREATION_WORKERS `
    --train-workers $TRAIN_WORKERS `
    --random-seed $RANDOM_SEED `
    --keep-part-csvs $KEEP_PART_CSVS `
    --batch-size $BATCH_SIZE `
    --max-training-epochs $MAX_TRAINING_EPOCHS `
    --learning-rate $LEARNING_RATE `
    --lr-schedule $LR_SCHEDULE `
    --loss-print-samples $LOSS_PRINT_SAMPLES `
    --patches-per-training-sample $PATCHES_PER_TRAINING_SAMPLE `
    --grid-spacing $GRID_SPACING `
    --network-name $NETWORK_NAME `
    --branch-features $BRANCH_FEATURES `
    --frozen-stages $FROZEN_STAGES `
    --small-input-stem $SMALL_INPUT_STEM
```

## Outputs

Generated fold data is written under `<RUN_DIR>/TRAINING_DATA`. Models and
training outputs are written under `<RUN_DIR>/TRAINING_RESULTS`.

Common generated training-data files include:

- `Train_f<FOLD>.csv`
- `Val_f<FOLD>.csv`
- `Train_Patches_F<FOLD>/`
- `Val_Patches_F<FOLD>/`
- `Train_Images_F<FOLD>/`
- `Val_Images_F<FOLD>/`
- `data_info_f<FOLD>.csv`
- `run_info_<TASK_NAME>_f<FOLD>.json`

If `--keep-part-csvs true` is used, temporary per-sample CSV directories may
also be retained:

- `Train_csv_parts_F<FOLD>/`
- `Val_csv_parts_F<FOLD>/`

Common result files include:

- `run_info_<TASK_NAME>_f<FOLD>.json`
- `train_log_f<FOLD>_<TRAINING_NAME>.csv`
- `train_plot_f<FOLD>_<TRAINING_NAME>.png`
- `model_f<FOLD>_<TRAINING_NAME>_best.pth`
- `model_f<FOLD>_<TRAINING_NAME>_last.pth`
- `checkpoint_summary_f<FOLD>_<TRAINING_NAME>.json`

When `COPY_FILES=true`, files from:

```text
<RUN_DIR>/TRAINING_RESULTS/<RUN_NAME>/<TASK_NAME>/
```

are copied to:

```text
<SAVE_DIR>/<RUN_NAME>/<TASK_NAME>/
```

## Notes and limits

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- `--num-points` must be between 1 and 30.
- Training patches are grayscale. Colour images are converted to grayscale before patch extraction.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Generated metadata and CSV label counts are checked before training starts.
- Keep generated data, checkpoints, logs, and copied outputs out of Git.
