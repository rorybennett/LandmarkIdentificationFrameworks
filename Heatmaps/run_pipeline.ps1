$RUN_DIR = "$HOME\HEATMAP_TRAINING"
$SAVE_DIR = "$HOME\HEATMAP_SAVING"
$FOLD_LISTS_DIR = "$HOME\DATA\folds"
$MARK_LIST_FILE = "$HOME\DATA\doctors_resampled_transverseMarkList.txt"
$IMAGE_DATA_DIR = "$HOME\DATA\TRANSVERSE"

$FOLD = 1
$TASK_NAME = "prostate_transverse"
$NUM_POINTS = 4

$TRAIN_MODEL = "true"
$COPY_FILES = "false"
$RUN_NAME = ""

$IMAGE_HEIGHT = 512
$IMAGE_WIDTH = 512
$HEATMAP_SIGMA = 8
$OVERSAMPLING_FACTOR = 1
$RECURSIVE_IMAGE_SEARCH = "false"

$BATCH_SIZE = 4
$MAX_TRAINING_EPOCHS = 80
$LEARNING_RATE = 0.001
$TRAIN_WORKERS = 8
$RANDOM_SEED = 42
$OPTIMISER_NAME = "adamw"
$LOSS_NAME = "weighted_mse"
$POSITIVE_WEIGHT = 20
$WEIGHT_DECAY = 0.0001
$MOMENTUM = 0.9
$LR_SCHEDULE = "plateau"
$LR_STEP_SIZE = 20
$LR_GAMMA = 0.5
$EARLY_STOP_PATIENCE = 15
$EARLY_STOP_MIN_DELTA = 0.0001
$EARLY_STOP_WARMUP_EPOCHS = 10
$USE_AMP = "false"
$SAVE_VALIDATION_PREDICTIONS = "true"

$NETWORK_NAME = "unet_basic"
$BASE_CHANNELS = 32
$DEPTH = 4
$CHANNEL_MULTIPLIER = 2
$MAX_CHANNELS = 512
$NORMALISATION = "batch"
$ACTIVATION = "relu"
$DROPOUT = 0
$UPSAMPLING = "bilinear"
$OUTPUT_ACTIVATION = "none"
$PADDING_MODE = "zeros"
$FINAL_KERNEL_SIZE = 1

$Arguments = @(
    $FOLD, $TASK_NAME, $TRAIN_MODEL, $COPY_FILES,
    "--run-dir", $RUN_DIR,
    "--save-dir", $SAVE_DIR,
    "--num-points", $NUM_POINTS,
    "--fold-lists-path", $FOLD_LISTS_DIR,
    "--mark-list-file", $MARK_LIST_FILE,
    "--image-data-dir", $IMAGE_DATA_DIR,
    "--image-size", $IMAGE_HEIGHT, $IMAGE_WIDTH,
    "--heatmap-sigma", $HEATMAP_SIGMA,
    "--oversampling-factor", $OVERSAMPLING_FACTOR,
    "--recursive-image-search", $RECURSIVE_IMAGE_SEARCH,
    "--batch-size", $BATCH_SIZE,
    "--learning-rate", $LEARNING_RATE,
    "--max-training-epochs", $MAX_TRAINING_EPOCHS,
    "--train-workers", $TRAIN_WORKERS,
    "--random-seed", $RANDOM_SEED,
    "--optimiser-name", $OPTIMISER_NAME,
    "--loss-name", $LOSS_NAME,
    "--positive-weight", $POSITIVE_WEIGHT,
    "--weight-decay", $WEIGHT_DECAY,
    "--momentum", $MOMENTUM,
    "--lr-schedule", $LR_SCHEDULE,
    "--lr-step-size", $LR_STEP_SIZE,
    "--lr-gamma", $LR_GAMMA,
    "--early-stop-patience", $EARLY_STOP_PATIENCE,
    "--early-stop-min-delta", $EARLY_STOP_MIN_DELTA,
    "--early-stop-warmup-epochs", $EARLY_STOP_WARMUP_EPOCHS,
    "--use-amp", $USE_AMP,
    "--save-validation-predictions", $SAVE_VALIDATION_PREDICTIONS,
    "--network-name", $NETWORK_NAME,
    "--base-channels", $BASE_CHANNELS,
    "--depth", $DEPTH,
    "--channel-multiplier", $CHANNEL_MULTIPLIER,
    "--max-channels", $MAX_CHANNELS,
    "--normalisation", $NORMALISATION,
    "--activation", $ACTIVATION,
    "--dropout", $DROPOUT,
    "--upsampling", $UPSAMPLING,
    "--output-activation", $OUTPUT_ACTIVATION,
    "--padding-mode", $PADDING_MODE,
    "--final-kernel-size", $FINAL_KERNEL_SIZE
)

if ($RUN_NAME -ne "") {
    $Arguments += @("--run-name", $RUN_NAME)
}

& heatmaps-train @Arguments
