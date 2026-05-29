#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$HOME/HEATMAP_TRAINING"
SAVE_DIR="$HOME/HEATMAP_SAVING"
FOLD_LISTS_DIR="$HOME/DATA/folds"
MARK_LIST_FILE="$HOME/DATA/doctors_resampled_transverseMarkList.txt"
IMAGE_DATA_DIR="$HOME/DATA/TRANSVERSE"

FOLD=1
TASK_NAME="prostate_transverse"
NUM_POINTS=4

TRAIN_MODEL="true"
COPY_FILES="false"

RUN_NAME=""
NETWORK_NAME="unet_basic"

IMAGE_HEIGHT=512
IMAGE_WIDTH=512
HEATMAP_SIGMA=8
OVERSAMPLING_FACTOR=1

BATCH_SIZE=4
MAX_TRAINING_EPOCHS=80
LEARNING_RATE=0.001
TRAIN_WORKERS=8
LOSS_NAME="weighted_mse"
POSITIVE_WEIGHT=20
EARLY_STOP_PATIENCE=15
EARLY_STOP_WARMUP_EPOCHS=10
SAVE_VALIDATION_OVERLAYS="false"

BASE_CHANNELS=32
DEPTH=4
CHANNEL_MULTIPLIER=2
MAX_CHANNELS=512
NORMALISATION="batch"
ACTIVATION="relu"
DROPOUT=0
UPSAMPLING="bilinear"

ARGS=(
    "$FOLD" "$TASK_NAME" "$TRAIN_MODEL" "$COPY_FILES"
    --run-dir "$RUN_DIR"
    --save-dir "$SAVE_DIR"
    --num-points "$NUM_POINTS"
    --fold-lists-path "$FOLD_LISTS_DIR"
    --mark-list-file "$MARK_LIST_FILE"
    --image-data-dir "$IMAGE_DATA_DIR"
    --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH"
    --heatmap-sigma "$HEATMAP_SIGMA"
    --oversampling-factor "$OVERSAMPLING_FACTOR"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --max-training-epochs "$MAX_TRAINING_EPOCHS"
    --train-workers "$TRAIN_WORKERS"
    --loss-name "$LOSS_NAME"
    --positive-weight "$POSITIVE_WEIGHT"
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --early-stop-warmup-epochs "$EARLY_STOP_WARMUP_EPOCHS"
    --save-validation-overlays "$SAVE_VALIDATION_OVERLAYS"
    --network-name "$NETWORK_NAME"
    --base-channels "$BASE_CHANNELS"
    --depth "$DEPTH"
    --channel-multiplier "$CHANNEL_MULTIPLIER"
    --max-channels "$MAX_CHANNELS"
    --normalisation "$NORMALISATION"
    --activation "$ACTIVATION"
    --dropout "$DROPOUT"
    --upsampling "$UPSAMPLING"
)

if [[ -n "$RUN_NAME" ]]; then
    ARGS+=(--run-name "$RUN_NAME")
fi

heatmaps-train "${ARGS[@]}"
