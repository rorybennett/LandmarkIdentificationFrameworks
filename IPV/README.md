# Image-Patch Voting (IPV) Landmark Identification

Dataset creation and quadruplet model training tools for Image-Patch
Voting (IPV) landmark identification.

The pipeline creates multi-scale patch data from annotated images, 
trains a quadruplet network, writes checkpoints, logs, and metadata 
to a run directory, optionally copies selected result files to a separate 
save directory, and can safely delete generated training data.

The package is task-agnostic for 2D image landmark localisation. 
Each task can use between 1 and 30 ordered landmark points per image. 
The original prostate IPV use case uses 4 endpoints on transverse 
transabdominal ultrasound images and 2 endpoints on sagittal 
transabdominal ultrasound images.

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

### 2. Install the package

Install in editable mode from the repository root:

```bash
git clone https://github.com/rorybennett/LandmarkIdentificationFrameworks
cd LandmarkIdentificationFrameworks/IPV
pip install -e .
```

The project depends on [PyTorch](https://pytorch.org/get-started/locally/) 
and torchvision. Install the PyTorch build that matches your CUDA or CPU 
environment before long training runs.

## Data layout

Keep image data and generated run outputs outside the Git repository. 
A typical prostate IPV data layout is:

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

Each fold-list file should contain one sample name per line, without the 
image extension:

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
pairs. The number of coordinate pairs must match `--num-points`, unless the 
mark list contains additional points and you deliberately want the first 
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
data workers, training workers, patch count, grid step, run directory, optional
save directory, and model backbone.

## Run and save directories

Every run uses a required `--run-dir`. The pipeline creates two high-level 
directories inside it:

```text
<RUN_DIR>/
├── TRAINING_DATA/
│   └── Data_F<FOLD>_<TASK_NAME>/
│       └── ... generated fold CSVs, patch images, validation images, and data metadata
└── RESULTS/
    └── <RUN_NAME>/
        └── <TASK_NAME>/
            └── ... checkpoints, logs, plots, and run metadata
```

Use `--save-dir` only when you want a copy of selected result files outside 
the run directory. When `COPY_FILES=true` and `--save-dir` is not supplied, 
nothing is copied and files remain in `<RUN_DIR>/RESULTS`.

When `DELETE_FILES=true`, only the current fold/task folder inside 
`<RUN_DIR>/TRAINING_DATA` is deleted. Files in `<RUN_DIR>/RESULTS` 
and the optional save directory are not deleted.

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
ipv-train FOLD TASK_NAME PHASE CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Positional arguments:

| Argument | Values | Description |
|---|---|---|
| `FOLD` | integer | Fold number to run. |
| `TASK_NAME` | string | Name of the landmark task. For prostate IPV, use `transverse` or `sagittal`. |
| `PHASE` | `Train`, `Val`, `both` | Data phase to create. Use `both` for training runs. |
| `CREATE_DATA` | boolean | Create patch data before training. |
| `TRAIN_MODEL` | boolean | Train the model. |
| `COPY_FILES` | boolean | Copy selected result files to `--save-dir`, if provided. |
| `DELETE_FILES` | boolean | Delete generated fold training data after completion. |

Boolean values can be supplied as `true`, `false`, `yes`, `no`, `1`, or `0`.

Important options:

| Option | Description |
|---|---|
| `--run-dir` | Required directory used for generated training data and run results. |
| `--save-dir` | Optional directory used for copies of selected result files after training. |
| `--num-points` | Number of ordered landmark points per image. Must be between 1 and 30. |
| `--fold-lists-path` | Directory containing `train_fN.txt` files and `val.txt`. |
| `--mark-list-file` | Text file containing image filenames and point coordinates. |
| `--image-data-dir` | Directory containing the source images. |
| `--patches-per-training-sample` | Number of sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`. |
| `--test-data-step` | Grid stride for validation patch-centre creation. |

### Prostate transverse example

```bash
ipv-train 1 transverse both true true true false \
    --run-dir "$HOME/IPV_RUNS/prostate_ipv" \
    --save-dir "$HOME/IPV_SAVED/prostate_ipv" \
    --num-points 4 \
    --fold-lists-path "$HOME/IPV_DATA/folds" \
    --mark-list-file "$HOME/IPV_DATA/doctors_resampled_transverseMarkList.txt" \
    --image-data-dir "$HOME/IPV_DATA/transverse" \
    --data-creation-workers 8 \
    --train-workers 8 \
    --random-seed 42 \
    --keep-part-csvs false \
    --batch-size 64 \
    --max-training-epochs 15 \
    --learning-rate 0.01 \
    --lr-schedule true \
    --loss-print-samples 1600 \
    --patches-per-training-sample 200 \
    --test-data-step 10 \
    --run-name prostate_transverse_fold1 \
    --network-name small_cnn \
    --branch-features 128 \
    --frozen-stages 0 \
    --small-input-stem false
```

### Prostate sagittal example

Change the task, point count, mark list and image directory:

```bash
ipv-train 1 sagittal both true true true false \
    --run-dir "$HOME/IPV_RUNS/prostate_ipv" \
    --save-dir "$HOME/IPV_SAVED/prostate_ipv" \
    --num-points 2 \
    --fold-lists-path "$HOME/IPV_DATA/folds" \
    --mark-list-file "$HOME/IPV_DATA/doctors_resampled_sagittalMarkList.txt" \
    --image-data-dir "$HOME/IPV_DATA/sagittal" \
    --data-creation-workers 8 \
    --train-workers 8 \
    --random-seed 42 \
    --keep-part-csvs false \
    --batch-size 64 \
    --max-training-epochs 15 \
    --learning-rate 0.01 \
    --lr-schedule true \
    --loss-print-samples 1600 \
    --patches-per-training-sample 200 \
    --test-data-step 10 \
    --run-name prostate_sagittal_fold1 \
    --network-name small_cnn \
    --branch-features 128 \
    --frozen-stages 0 \
    --small-input-stem false
```

You can also run the module directly:

```bash
python -m IPV.create_dataset_and_train_model 1 transverse both true true true false [OPTIONS]
```

## Ubuntu example script

Save as `run_prostate_transverse_ubuntu.sh`, edit the paths at the top, then run with `bash run_prostate_transverse_ubuntu.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Coding/Python/IPV"
RUN_DIR="$HOME/IPV_RUNS/prostate_ipv"
SAVE_DIR="$HOME/IPV_SAVED/prostate_ipv"
DATA_DIR="$HOME/IPV_DATA"
FOLD_LISTS_PATH="$DATA_DIR/folds"

FOLD=1
TASK_NAME="transverse"
PHASE="both"
NUM_POINTS=4

MARK_LIST_FILE="$DATA_DIR/doctors_resampled_transverseMarkList.txt"
IMAGE_DATA_DIR="$DATA_DIR/transverse"

CREATE_DATA="true"
TRAIN_MODEL="true"
COPY_FILES="true"
DELETE_FILES="false"
KEEP_PART_CSVS="false"

DATA_CREATION_WORKERS=8
PATCHES_PER_TRAINING_SAMPLE=200
TEST_DATA_STEP=10
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

RUN_NAME="prostate_${TASK_NAME}_fold${FOLD}_points${NUM_POINTS}_${NETWORK_NAME}_bs${BATCH_SIZE}_epochs${MAX_TRAINING_EPOCHS}_seed${RANDOM_SEED}"

cd "$PROJECT_DIR"

python -m IPV.create_dataset_and_train_model \
    "$FOLD" "$TASK_NAME" "$PHASE" "$CREATE_DATA" "$TRAIN_MODEL" "$COPY_FILES" "$DELETE_FILES" \
    --run-dir "$RUN_DIR" \
    --save-dir "$SAVE_DIR" \
    --num-points "$NUM_POINTS" \
    --fold-lists-path "$FOLD_LISTS_PATH" \
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
    --test-data-step "$TEST_DATA_STEP" \
    --run-name "$RUN_NAME" \
    --network-name "$NETWORK_NAME" \
    --branch-features "$BRANCH_FEATURES" \
    --frozen-stages "$FROZEN_STAGES" \
    --small-input-stem "$SMALL_INPUT_STEM"
```

## Windows PowerShell example script

Save as `run_prostate_transverse_windows.ps1`, edit the paths at the top, then run from PowerShell.

```powershell
$PROJECT_DIR = "D:\Coding\Python\IPV"
$RUN_DIR = "D:\IPV_RUNS\prostate_ipv"
$SAVE_DIR = "D:\IPV_SAVED\prostate_ipv"
$DATA_DIR = "D:\IPV_DATA"

$FOLD = 1
$TASK_NAME = "transverse"
$PHASE = "both"
$NUM_POINTS = 4

$FOLD_LISTS_PATH = "$DATA_DIR\folds"
$MARK_LIST_FILE = "$DATA_DIR\doctors_resampled_transverseMarkList.txt"
$IMAGE_DATA_DIR = "$DATA_DIR\transverse"

$CREATE_DATA = "true"
$TRAIN_MODEL = "true"
$COPY_FILES = "true"
$DELETE_FILES = "false"
$KEEP_PART_CSVS = "false"

$DATA_CREATION_WORKERS = 8
$PATCHES_PER_TRAINING_SAMPLE = 200
$TEST_DATA_STEP = 10
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

$RUN_NAME = "prostate_${TASK_NAME}_fold${FOLD}_points${NUM_POINTS}_${NETWORK_NAME}_bs${BATCH_SIZE}_epochs${MAX_TRAINING_EPOCHS}_seed${RANDOM_SEED}"

Set-Location $PROJECT_DIR

ipv-train $FOLD $TASK_NAME $PHASE $CREATE_DATA $TRAIN_MODEL $COPY_FILES $DELETE_FILES `
    --run-dir $RUN_DIR `
    --save-dir $SAVE_DIR `
    --num-points $NUM_POINTS `
    --fold-lists-path $FOLD_LISTS_PATH `
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
    --test-data-step $TEST_DATA_STEP `
    --run-name $RUN_NAME `
    --network-name $NETWORK_NAME `
    --branch-features $BRANCH_FEATURES `
    --frozen-stages $FROZEN_STAGES `
    --small-input-stem $SMALL_INPUT_STEM
```

## Outputs

Generated fold data is written under `<RUN_DIR>/TRAINING_DATA`. Models and training outputs are written under `<RUN_DIR>/RESULTS`.

Common generated training-data files include:

- `Train_f<FOLD>.csv`
- `Val_f<FOLD>.csv`
- `data_info_<PHASE>_f<FOLD>.csv`
- `run_info_<TASK_NAME>_<PHASE>_f<FOLD>.json`

Common result files include:

- `run_info_<TASK_NAME>_<PHASE>_f<FOLD>.json`
- `train_log_f<FOLD>_<RUN>.csv`
- `train_plot_f<FOLD>_<RUN>.png`
- `model_f<FOLD>_<RUN>_best.pth`
- `model_f<FOLD>_<RUN>_last.pth`
- `checkpoint_summary_f<FOLD>_<RUN>.json`

When `COPY_FILES=true` and `--save-dir` is provided, files from:

```text
<RUN_DIR>/RESULTS/<RUN_NAME>/<TASK_NAME>/
```

are copied to:

```text
<SAVE_DIR>/<RUN_NAME>/<TASK_NAME>/
```

## Notes and limits

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- `--num-points` must be between 1 and 30.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Generated metadata and CSV label counts are checked before training starts.
