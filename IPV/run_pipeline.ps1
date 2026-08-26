$RUN_DIR = "D:\IPV_TRAINING"
$SAVE_DIR = "D:\IPV_SAVING"

$FOLD_LISTS_DIR = "D:\DATA\folds"
$MARK_LIST_FILE = "D:\DATA\transverse_points_list.txt"
$IMAGE_DATA_DIR = "D:\DATA\transverse"

$REPETITION = 1
$FOLD = 1
$TASK_NAME = "prostate_transverse"
$NUM_POINTS = 4

$CREATE_DATA = "true"
$TRAIN_MODEL = "true"
$COPY_FILES = "true"
$DELETE_FILES = "false"
$RESUME_TRAINING = "false"
$KEEP_PART_CSVS = "false"

$DATA_CREATION_WORKERS = 8
$PATCHES_PER_TRAINING_SAMPLE = 200
$VAL_GRID_SPACING = 10
$RANDOM_SEED = 42

$TRAIN_WORKERS = 8
$BATCH_SIZE = 64
$MAX_TRAINING_EPOCHS = 15
$LEARNING_RATE = 0.01
$OPTIMISER_NAME = "adamw"
$WEIGHT_DECAY = 0.0001
$MOMENTUM = 0.9
$LR_SCHEDULE = "plateau"
$LR_STEP_SIZE = 20
$LR_GAMMA = 0.5
$EARLY_STOP_PATIENCE = 15
$EARLY_STOP_MIN_DELTA = 0.0001
$EARLY_STOP_WARMUP_EPOCHS = 10
$USE_AMP = "false"
$SAVE_VALIDATION_RESULTS = "true"
$VALIDATION_INFERENCE_BATCH_SIZE = 2048
$VALIDATION_VOTE_SMOOTHING_SIGMA = 7.0
$VALIDATION_USE_PROBABILITY_WEIGHTS = "true"
$VALIDATION_SAVE_RAW_VOTE_MAPS = "false"

$NETWORK_NAME = "small_cnn"
$BRANCH_FEATURES = 128
$FROZEN_STAGES = 0
$SMALL_INPUT_STEM = "false"

$RUN_NAME = ""
$RunNameArguments = @()

if ($RUN_NAME -ne "") {
    $RunNameArguments = @("--run-name", $RUN_NAME)
}

ipv-train $REPETITION $FOLD $TASK_NAME $CREATE_DATA $TRAIN_MODEL $COPY_FILES $DELETE_FILES `
    --run-dir $RUN_DIR `
    --save-dir $SAVE_DIR `
    --resume-training $RESUME_TRAINING `
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
    --optimiser-name $OPTIMISER_NAME `
    --weight-decay $WEIGHT_DECAY `
    --momentum $MOMENTUM `
    --lr-schedule $LR_SCHEDULE `
    --lr-step-size $LR_STEP_SIZE `
    --lr-gamma $LR_GAMMA `
    --early-stop-patience $EARLY_STOP_PATIENCE `
    --early-stop-min-delta $EARLY_STOP_MIN_DELTA `
    --early-stop-warmup-epochs $EARLY_STOP_WARMUP_EPOCHS `
    --use-amp $USE_AMP `
    --save-validation-results $SAVE_VALIDATION_RESULTS `
    --validation-inference-batch-size $VALIDATION_INFERENCE_BATCH_SIZE `
    --validation-vote-smoothing-sigma $VALIDATION_VOTE_SMOOTHING_SIGMA `
    --validation-use-probability-weights $VALIDATION_USE_PROBABILITY_WEIGHTS `
    --validation-save-raw-vote-maps $VALIDATION_SAVE_RAW_VOTE_MAPS `
    --patches-per-training-sample $PATCHES_PER_TRAINING_SAMPLE `
    --val-grid-spacing $VAL_GRID_SPACING `
    --network-name $NETWORK_NAME `
    --branch-features $BRANCH_FEATURES `
    --frozen-stages $FROZEN_STAGES `
    --small-input-stem $SMALL_INPUT_STEM @RunNameArguments
