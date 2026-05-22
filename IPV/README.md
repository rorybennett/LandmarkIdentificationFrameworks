# Image-Patch Voting (IPV) Landmark Identification

Dataset creation and quadruplet model training tools for Image-Patch Voting
(IPV) landmark identification.

The pipeline creates multi-scale patch data from annotated 2D images, trains a
quadruplet network, writes checkpoints, logs, plots, and metadata to a run
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

Install a [PyTorch and torchvision](https://pytorch.org/get-started/locally/) build that matches your CUDA or CPU
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

Training patches preserve the source image channel count. Greyscale images create
one-channel patches, RGB images create three-channel patches, and RGBA images
create four-channel patches. During dataset loading, patches are converted to
channel-first tensors shaped like:

```text
[num_sub_patches, channels, height, width]
```

The model detects the channel count from the generated training and validation
patches before construction. ResNet and small-CNN backbones are built with a
matching first convolution. For pretrained ResNet backbones, the first-layer
weights are adapted when the patch channel count is not three channels.

All images within one generated training run should have the same channel count.
Saved PNG patches support greyscale, RGB, and RGBA data.

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
pairs. The number of coordinate pairs must be at least `--num-points`. If a row
contains more coordinate pairs than requested, only the first `--num-points`
points are used. The maximum supported number of points per image is 30. This is an
arbitrary upper limit which can be changed manually.

Sample names are taken from the image filename stem. Duplicate sample names in a
mark-list file are rejected, and the requested landmark points must lie within
the corresponding image bounds.

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

Every run uses a required `--run-dir`. The code creates these two high-level
directories inside it:

```text
<RUN_DIR>/
├── TRAINING_DATA/
│   └── <TASK_NAME>/
│       └── <NUM_FOLDS>_Folds/
│           └── <SCALES>_<NUM_POINTS>points_<PATCHES_PER_TRAINING_SAMPLE>pertrainingsample/
│               └── ... generated CSVs, patch images, overlay images, and data metadata
└── TRAINING_RESULTS/
    └── <TASK_NAME>/
        └── <RUN_NAME>/
            └── ... checkpoints, logs, plots, and run metadata
```

`RUN_NAME` is generated deterministically from the run configuration, including
fold count, patch scales, sampling variances, point count, patch count, validation
grid spacing, network, branch features, frozen stages, stem setting, batch size,
learning rate, scheduler, early-stopping, epoch count, and random seed. This 
creates quite a cumbersome file name, but almost all of the details are there.

By default, `RUN_NAME` is generated from the configuration. Supplying
`--run-name` overrides the generated name after path-safe cleaning.

Use `--save-dir` only when you want a copy of selected result files outside the
run directory. When `COPY_FILES=true`, `--save-dir` is required. When
`COPY_FILES=false`, any supplied `--save-dir` is ignored.

When `COPY_FILES=true`, files from:

```text
<RUN_DIR>/TRAINING_RESULTS/<TASK_NAME>/<RUN_NAME>/
```

are copied to:

```text
<SAVE_DIR>/<TASK_NAME>/<RUN_NAME>/
```

When `DELETE_FILES=true`, only generated training-data artefacts for the current
fold/task/sample configuration are deleted from `<RUN_DIR>/TRAINING_DATA`. Files
in `<RUN_DIR>/TRAINING_RESULTS` and the optional save directory are not deleted.

When `TRAIN_MODEL=true` and `CREATE_DATA=false`, the pipeline checks for partial
fold data before training. If some expected fold-specific files exist and others
are missing, training stops so stale or incomplete data are not accidentally
reused.

## Supported models

Available network names are defined in `IPV/model_registry.py`. The Quadruplet class makes use
of four of the chosen model, one for each patch scale.

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
| `FOLD` | integer | Fold number to run. Must match an available `train_fN.txt` file. |
| `TASK_NAME` | string | Name of the landmark task. For prostate IPV, use `transverse` or `sagittal`. |
| `CREATE_DATA` | boolean | Create patch data before training. |
| `TRAIN_MODEL` | boolean | Train the model. |
| `COPY_FILES` | boolean | Copy selected result files to `--save-dir`. If true, `--save-dir` is required. |
| `DELETE_FILES` | boolean | Delete generated fold training data after completion. |

Boolean values can be supplied as `true`, `false`, `yes`, `no`, `1`, or `0`.

Important options:

| Option                          | Description |
|---------------------------------|---|
| `--run-dir`                     | Required directory used for generated training data and run results. |
| `--save-dir`                    | Required only when `COPY_FILES=true`; used for copies of selected result files after training. |
| `--num-points`                  | Number of ordered landmark points per image. Must be between 1 and 30. |
| `--fold-lists-path`             | Directory containing `train_fN.txt` files and `val.txt`. |
| `--mark-list-file`              | Text file containing image filenames and point coordinates. |
| `--image-data-dir`              | Directory containing the source images. |
| `--data-creation-workers`       | Number of worker processes used for patch/data creation. Must be at least 1. |
| `--train-workers`               | Number of PyTorch DataLoader workers used during training. Use 0 for single-process loading. |
| `--random-seed`                 | Seed used for deterministic sampled training centres. |
| `--keep-part-csvs`              | Keep per-sample temporary CSV part files if true. |
| `--batch-size`                  | Training batch size. |
| `--max-training-epochs`         | Maximum training epochs. |
| `--learning-rate`               | Initial SGD learning rate. |
| `--lr-schedule`                 | Enable the validation-accuracy-triggered learning-rate scheduler. |
| `--lr-step-size`                | StepLR step size used when `--lr-schedule true`. Default: 1. |
| `--lr-gamma`                    | StepLR multiplicative decay factor used when `--lr-schedule true`. Default: 0.1. |
| `--early-stop-patience`         | Number of validation epochs without sufficient loss improvement before early stopping. Default: 5. |
| `--early-stop-min-delta`        | Minimum validation-loss improvement required to reset early-stopping patience. Default: 0.001. |
| `--early-stop-warmup-epochs`    | Number of initial epochs before early stopping is allowed. Default: 3. |
| `--loss-print-samples`          | Approximate sample interval used to derive the validation/logging batch interval. |
| `--patches-per-training-sample` | Number of sampled patch centres per training image. Must be at least `num_points * len(sampling_variances)`. |
| `--val-grid-spacing`            | Grid stride for validation patch-centre creation. |
| `--network-name`                | Model backbone name from the model registry. |
| `--branch-features`             | Number of features output by each branch before concatenation. |
| `--frozen-stages`               | Number of pretrained ResNet stages to freeze. Use `0` for untrained models and `small_cnn`. |
| `--small-input-stem`            | Use the small-input ResNet stem when true. Use `false` for `small_cnn`. |
| `--run-name`                    | Optional custom run name. When omitted, a deterministic name is generated from the run configuration. |


## Outputs

Generated fold data is written under:

```text
<RUN_DIR>/TRAINING_DATA/<TASK_NAME>/<NUM_FOLDS>_Folds/<SCALES>_<NUM_POINTS>points_<PATCHES_PER_TRAINING_SAMPLE>pertrainingsample/
```

Models and training outputs are written under:

```text
<RUN_DIR>/TRAINING_RESULTS/<TASK_NAME>/<RUN_NAME>/
```

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

## Notes and limits

- Each image must have at least `--num-points` coordinate pairs in the mark-list file.
- Duplicate mark-list sample names are rejected.
- The requested mark-list points must lie inside the matching image dimensions.
- Temporary per-sample CSV filenames and overlay filenames are derived from sanitised sample names.
- `--num-points` must be between 1 and 30.
- The current model expects exactly four sub-patch scales because each sample is represented as a quadruplet.
- The current label structure is distance plus angle per point, so each point adds two model output heads.
- Generated metadata and CSV label counts are checked before training starts.
- Training from existing data checks for partial fold data before model training starts.
- Keep generated data, checkpoints, logs, and copied outputs out of Git.
