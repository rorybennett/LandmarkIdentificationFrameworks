# Image Patch Voting (IPV) Landmark Identification

Dataset creation and quadruplet model training tools for Image-Patch Voting (IPV) landmark identification.

The pipeline creates multi-scale patch data from annotated images, trains a quadruplet network, saves checkpoints and logs, optionally copies outputs into a results directory, and can safely delete temporary fold data.

This repository is task-agnostic for 2D image landmark localisation. Each task can use between 1 and 30 ordered landmark points per image. The model predicts distance and angle classes for each landmark from multi-scale image patches.

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

### 2. Download and install the package

Install in editable mode from the repository root:

```bash
git clone rorybennett/LandmarkIdentificationFrameworks
cd LandmarkIdentificationFrameworks/IPV
pip install -e .
```

The project depends on PyTorch and torchvision.

## Data layout

Keep image data and generated outputs outside the Git repository. A typical data layout is:

```text
DATA/
├── folds/
│   ├── train_f1.txt
│   ├── train_f2.txt
│   ├── train_f3.txt
│   ├── train_f4.txt
│   ├── train_f5.txt
│   └── val.txt
├── LANDMARK_TASK/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── landmark_task_points.txt
```

`LANDMARK_TASK` can be any task name you choose, for example `hand_keypoints`, `organ_boundaries`, or `fiducials`. Use the same task name as the second positional command-line argument.

### Fold lists

Each fold-list file should contain one sample name per line, without the image extension:

```text
image_001
image_005
image_007
```

The pipeline discovers folds from files named `train_fN.txt`. Fold numbers must be contiguous from `train_f1.txt`, and `val.txt` must exist for training-time validation.

### Mark-list files

Each mark-list row must start with the image filename followed by coordinate pairs. The number of coordinate pairs must match `--num-points`, unless the mark list contains additional points and you deliberately want the first `--num-points` points to be used. The maximum supported number of points per image is 30.

Ensure point order is consistent across all images. For example, a 3-point task should look like this:

```text
image_001.png (120, 80) (200, 90) (180, 160)
image_002.png (115, 76) (198, 88) (176, 155)
```

The generated CSV files contain five metadata columns followed by two labels per point:

```text
patch_id, patch_path, sample_name, centre_x, centre_y, point_1_distance, point_1_angle, point_2_distance, point_2_angle, ...
```

Training checks that `Train_f<FOLD>.csv` and `Val_f<FOLD>.csv` contain the same number of points as requested by `--num-points`.

## Configuration

Static task settings are stored in `IPV/parameters.py`:

- `sub_patch_scales`: four patch sizes used by the quadruplet model.
- `sampling_variances`: training-centre sampling variances around each landmark.
- `distance_intervals`: class boundaries for distance labels.
- `angle_intervals`: class boundaries for angle labels.

Run-specific settings are supplied through the command line, including fold number, task name, number of points, batch size, learning rate, number of epochs, data workers, training workers, patch count, grid step, and model backbone.

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

For untrained models, use `--frozen-stages 0`. For `small_cnn`, use `--small-input-stem false` and `--frozen-stages 0`.

## Running the pipeline

The command has the following structure:

```bash
ipv-train FOLD TASK_NAME PHASE CREATE_DATA TRAIN_MODEL COPY_FILES DELETE_FILES [OPTIONS]
```

Positional arguments:

| Argument | Values | Description |
|---|---|---|
| `FOLD` | integer | Fold number to run. |
| `TASK_NAME` | string | Name of the landmark task. This is used in output paths and should match the intended image/mark-list task. |
| `PHASE` | `Train`, `Val`, `both` | Data phase to create. Use `both` for training runs. |
| `CREATE_DATA` | boolean | Create patch data before training. |
| `TRAIN_MODEL` | boolean | Train the model. |
| `COPY_FILES` | boolean | Copy output files to the results directory. |
| `DELETE_FILES` | boolean | Delete temporary fold data after completion. |

Boolean values can be supplied as `true`, `false`, `yes`, `no`, `1`, or `0`.

Important options:

| Option | Description |
|---|---|
| `--num-points` | Number of ordered landmark points per image. Must be between 1 and 30. |
| `--mark-list-file` | Text file containing image filenames and point coordinates. |
| `--image-data-dir` | Directory containing the source images. |
| `--patches-per-training-sample` | Number of sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`. |
| `--test-data-step` | Grid stride for validation patch-centre creation. |

### Minimal example

```bash
ipv-train 1 hand_keypoints both true true true false \
    --scratch-dir "$HOME/Scratch/IPV" \
    --results-dir "$HOME/Scratch/IPV/Results" \
    --num-points 3 \
    --fold-lists-path "$HOME/Scratch/IPV/DATA/folds" \
    --mark-list-file "$HOME/Scratch/IPV/DATA/hand_keypoints_points.txt" \
    --image-data-dir "$HOME/Scratch/IPV/DATA/HAND_KEYPOINTS" \
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
    --run-name ipv_hand_keypoints_example \
    --network-name small_cnn \
    --branch-features 128 \
    --frozen-stages 0 \
    --small-input-stem false
```

You can also run the module directly:

```bash
python -m IPV.create_dataset_and_train_model 1 hand_keypoints both true true true false [OPTIONS]
```

## Ubuntu example script

Save as `run_ubuntu.sh`, edit the paths at the top, then run with `bash run_ubuntu.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Coding/Python/IPV"
SCRATCH_DIR="$HOME/Scratch/IPV"
RESULTS_DIR="$SCRATCH_DIR/Results"
DATA_DIR="$SCRATCH_DIR/DATA"
FOLD_LISTS_PATH="$DATA_DIR/folds"

FOLD=1
TASK_NAME="hand_keypoints"
PHASE="both"
NUM_POINTS=3

MARK_LIST_FILE="$DATA_DIR/${TASK_NAME}_points.txt"
IMAGE_DATA_DIR="$DATA_DIR/HAND_KEYPOINTS"

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

RUN_NAME="ipv_${TASK_NAME}_${PHASE}_points${NUM_POINTS}_${NETWORK_NAME}_bf${BRANCH_FEATURES}_bs${BATCH_SIZE}_epochs${MAX_TRAINING_EPOCHS}_lr${LEARNING_RATE}_ppts${PATCHES_PER_TRAINING_SAMPLE}_step${TEST_DATA_STEP}_seed${RANDOM_SEED}"

cd "$PROJECT_DIR"

python -m IPV.create_dataset_and_train_model \
    "$FOLD" "$TASK_NAME" "$PHASE" "$CREATE_DATA" "$TRAIN_MODEL" "$COPY_FILES" "$DELETE_FILES" \
    --scratch-dir "$SCRATCH_DIR" \
    --results-dir "$RESULTS_DIR" \
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

Save as `run_windows.ps1`, edit the paths at the top, then run from PowerShell.

```powershell
$PROJECT_DIR = "D:\Coding\Python\IPV"
$SCRATCH_DIR = "D:\Scratch\IPV"
$RESULTS_DIR = "$SCRATCH_DIR\Results"
$DATA_DIR = "$SCRATCH_DIR\DATA"

$FOLD = 1
$TASK_NAME = "hand_keypoints"
$PHASE = "both"
$NUM_POINTS = 3

$FOLD_LISTS_PATH = "$DATA_DIR\folds"
$MARK_LIST_FILE = "$DATA_DIR\${TASK_NAME}_points.txt"
$IMAGE_DATA_DIR = "$DATA_DIR\HAND_KEYPOINTS"

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
$BATCH_SIZE = 32
$MAX_TRAINING_EPOCHS = 30
$LEARNING_RATE = 0.01
$LR_SCHEDULE = "true"
$LOSS_PRINT_SAMPLES = 1600

$NETWORK_NAME = "small_cnn"
$BRANCH_FEATURES = 128
$FROZEN_STAGES = 0
$SMALL_INPUT_STEM = "false"

$RUN_NAME = "ipv_${TASK_NAME}_${PHASE}_points${NUM_POINTS}_${NETWORK_NAME}_bf${BRANCH_FEATURES}_bs${BATCH_SIZE}_epochs${MAX_TRAINING_EPOCHS}_lr${LEARNING_RATE}_ppts${PATCHES_PER_TRAINING_SAMPLE}_step${TEST_DATA_STEP}_seed${RANDOM_SEED}"

Set-Location $PROJECT_DIR

ipv-train $FOLD $TASK_NAME $PHASE $CREATE_DATA $TRAIN_MODEL $COPY_FILES $DELETE_FILES `
    --scratch-dir $SCRATCH_DIR `
    --results-dir $RESULTS_DIR `
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

Generated fold data is written under the scratch directory. The internal path includes the fold, task name, sub-patch scales, number of landmark points, and number of patches per training sample.

Common output files include:

- `Train_f<FOLD>.csv`
- `Val_f<FOLD>.csv`
- `data_info_<PHASE>_f<FOLD>.csv`
- `run_info_<TASK_NAME>_<PHASE>_f<FOLD>.json`
- `train_log_f<FOLD>_<RUN>.csv`
- `train_plot_f<FOLD>_<RUN>.png`
- `model_f<FOLD>_<RUN>_best.pth`
- `model_f<FOLD>_<RUN>_last.pth`
- `checkpoint_summary_f<FOLD>_<RUN>.json`

When `COPY_FILES=true`, top-level output files from the scratch data directory are copied to:

```text
<RESULTS_DIR>/<RUN_NAME>/<TASK_NAME>/
```

When `DELETE_FILES=true`, temporary fold data is removed only if it is inside the configured `SCRATCH_DIR`.

## Notes and limits

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- `--num-points` must be between 1 and 30.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Generated metadata and CSV label counts are checked before training starts.
