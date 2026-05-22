#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$HOME/IPV_TRAINING"
SAVE_DIR="$HOME/IPV_SAVING"

FOLD_LISTS_DIR="$HOME/DATA/folds"
MARK_LIST_FILE="$HOME/DATA/transverse_points_list.txt"
IMAGE_DATA_DIR="$HOME/DATA/transverse"

FOLD=1
TASK_NAME="prostate_transverse"
NUM_POINTS=4

CREATE_DATA="true"
TRAIN_MODEL="true"
COPY_FILES="true"
DELETE_FILES="true"
KEEP_PART_CSVS="false"

DATA_CREATION_WORKERS=8
PATCHES_PER_TRAINING_SAMPLE=200
VAL_GRID_SPACING=10
RANDOM_SEED=42

TRAIN_WORKERS=8
BATCH_SIZE=64
MAX_TRAINING_EPOCHS=15
LEARNING_RATE=0.01
LR_SCHEDULE="true"
LR_STEP_SIZE=1
LR_GAMMA=0.1
EARLY_STOP_PATIENCE=5
EARLY_STOP_MIN_DELTA=0.001
EARLY_STOP_WARMUP_EPOCHS=3
LOSS_PRINT_SAMPLES=3200

NETWORK_NAME="small_cnn"
BRANCH_FEATURES=128
FROZEN_STAGES=0
SMALL_INPUT_STEM="false"

RUN_NAME="${NETWORK_NAME}_fs${FROZEN_STAGES}_stem${SMALL_INPUT_STEM}_ppts${PATCHES_PER_TRAINING_SAMPLE}"

python -m IPV.create_dataset_and_train_model \
    "$FOLD" "$TASK_NAME" "$CREATE_DATA" "$TRAIN_MODEL" "$COPY_FILES" "$DELETE_FILES" \
    --run-dir "$RUN_DIR" \
    --save-dir "$SAVE_DIR" \
    --run-name "$RUN_NAME" \
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
    --lr-step-size "$LR_STEP_SIZE" \
    --lr-gamma "$LR_GAMMA" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
    --early-stop-warmup-epochs "$EARLY_STOP_WARMUP_EPOCHS" \
    --loss-print-samples "$LOSS_PRINT_SAMPLES" \
    --patches-per-training-sample "$PATCHES_PER_TRAINING_SAMPLE" \
    --val-grid-spacing "$VAL_GRID_SPACING" \
    --network-name "$NETWORK_NAME" \
    --branch-features "$BRANCH_FEATURES" \
    --frozen-stages "$FROZEN_STAGES" \
    --small-input-stem "$SMALL_INPUT_STEM"