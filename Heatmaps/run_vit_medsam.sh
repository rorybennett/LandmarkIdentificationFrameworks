#!/usr/bin/env bash
# Activate your Python environment first. Positional/CLI arguments are forwarded to the pipeline.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
if (( $# > 0 )); then
    exec "$PYTHON" -m Heatmaps.heatmap_training_pipeline "$@"
fi
# Edit these input paths. No installation or checkpoint download is performed here.
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/outputs}"
MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-$PROJECT_DIR/checkpoints/medsam_vit_b.pth}"
FOLD_LISTS_DIR="${FOLD_LISTS_DIR:-$PROJECT_DIR/data/folds}"
MARK_LIST_FILE="${MARK_LIST_FILE:-$PROJECT_DIR/data/doctors_resampled_transverseMarkList.txt}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-$PROJECT_DIR/data/TRANSVERSE}"
REPETITION="${REPETITION:-1}"
FOLD="${FOLD:-1}"
TASK_NAME="${TASK_NAME:-prostate_transverse}"
NUM_POINTS="${NUM_POINTS:-4}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-}"

# The selected network supplies all recommended data, model and training defaults.
# Add only deliberate overrides, for example (--batch-size 16).
EXTRA_ARGS=()
ARGS=(
    "$REPETITION" "$FOLD" "$TASK_NAME" true false
    --run-dir "$RUN_DIR" --num-points "$NUM_POINTS"
    --fold-lists-path "$FOLD_LISTS_DIR" --mark-list-file "$MARK_LIST_FILE"
    --image-data-dir "$IMAGE_DATA_DIR" --resume-training "$RESUME_TRAINING"
    --network-name vit-medsam --device "$DEVICE"
    --pretrained-checkpoint "$MEDSAM_CHECKPOINT"
)
[[ -z "$RUN_NAME" ]] || ARGS+=(--run-name "$RUN_NAME")
ARGS+=("${EXTRA_ARGS[@]}")
exec "$PYTHON" -u -m Heatmaps.heatmap_training_pipeline "${ARGS[@]}"
