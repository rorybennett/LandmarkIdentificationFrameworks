#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$HOME/HEATMAP_TRAINING"
SAVE_DIR="$HOME/HEATMAP_SAVING"
FOLD_LISTS_DIR="$HOME/DATA/folds"
MARK_LIST_FILE="$HOME/DATA/doctors_resampled_transverseMarkList.txt"
IMAGE_DATA_DIR="$HOME/DATA/TRANSVERSE"

REPETITION=1
FOLD=1  # Use "all" to train with training_fall.txt and validate with val_fall.txt.
TASK_NAME="prostate_transverse"
NUM_POINTS=4

TRAIN_MODEL="true"
COPY_FILES="false"
RESUME_TRAINING="false"
RUN_NAME=""

NETWORK_NAME="unet_basic" # unet_basic, hrnet, stacked_hourglass, vitpose, vit-medsam
PRETRAINED_CHECKPOINT="" # Required for fresh ViTPose / MedSAM training.

# The selected network supplies all recommended data, model and training defaults.
# Add only deliberate overrides, for example (--batch-size 16).
EXTRA_ARGS=()
ARGS=(
    "$REPETITION" "$FOLD" "$TASK_NAME" "$TRAIN_MODEL" "$COPY_FILES"
    --run-dir "$RUN_DIR" --num-points "$NUM_POINTS"
    --fold-lists-path "$FOLD_LISTS_DIR" --mark-list-file "$MARK_LIST_FILE"
    --image-data-dir "$IMAGE_DATA_DIR" --resume-training "$RESUME_TRAINING"
    --network-name "$NETWORK_NAME" --save-dir "$SAVE_DIR"
)
[[ -z "$RUN_NAME" ]] || ARGS+=(--run-name "$RUN_NAME")
[[ -z "$PRETRAINED_CHECKPOINT" ]] || ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
ARGS+=("${EXTRA_ARGS[@]}")
cd "$(dirname "${BASH_SOURCE[0]}")"
python -u -m Heatmaps.heatmap_training_pipeline "${ARGS[@]}"
