# Activate your Python environment first, then edit the paths below.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectDir = $PSScriptRoot
$Python = 'python'
if ($args.Count -gt 0) {
    Push-Location -LiteralPath $ProjectDir
    try { & $Python -m Heatmaps.heatmap_training_pipeline @args; $RunExitCode = $LASTEXITCODE }
    finally { Pop-Location }
    exit $RunExitCode
}
$RUN_DIR = 'C:\Storage\GeneratedFiles\ViTPose-MedSAM'
$MEDSAM_CHECKPOINT = 'C:\Storage\Coding\Python\ViTPose\MedSAM\work_dir\MedSAM\medsam_vit_b.pth'
$FOLD_LISTS_DIR = 'C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\folds_network_study'
$MARK_LIST_FILE = 'C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\doctors_resampled_transverseMarkList.txt'
$IMAGE_DATA_DIR =  'C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\TRANSVERSE'
$REPETITION = '1'
$FOLD = '1' # Or 'all', using the framework's all-data fold convention.
$TASK_NAME = 'prostate_transverse'
$NUM_POINTS = '4'
$RESUME_TRAINING = 'false'
$DEVICE = 'cuda'
$IMAGE_SIZE = '512'
$HEATMAP_SIGMA = '8'
$OVERSAMPLING_FACTOR = '8'
$BATCH_SIZE = '8'
$MAX_TRAINING_EPOCHS = '300'
$LEARNING_RATE = '0.0001'
$ENCODER_LEARNING_RATE = '0.00001'
$FREEZE_ENCODER_EPOCHS = '5' # Minimum only; the decoder must also plateau.
$FINETUNE_LAST_BLOCKS = '4'
$GRADIENT_CHECKPOINTING = 'true'
$TRAIN_WORKERS = '8'
$RANDOM_SEED = '42'
$LOSS_NAME = 'weighted_mse'
$LANDMARK_CONSTRAINT_LOSS = '' # Empty disables; use 'prostate_taus' for four transverse landmarks.
$POSITIVE_WEIGHT = '20'
$WEIGHT_DECAY = '0.0001'
$LR_SCHEDULE = 'plateau' # plateau, cosine, linear, exponential, step, none
$DECODER_PLATEAU_PATIENCE = '10'
$DECODER_PLATEAU_MIN_DELTA = '0.1'
$FINETUNE_DECODER_LR_FACTOR = '0.5'
$LR_PLATEAU_PATIENCE = '3'
$LR_DECAY_EPOCHS = '50'
$LR_MIN_FACTOR = '0.01'
$LR_STEP_SIZE = '20'
$LR_GAMMA = '0.5'
$EARLY_STOP_PATIENCE = '15'
$EARLY_STOP_MIN_DELTA = '0.0001'
$EARLY_STOP_WARMUP_EPOCHS = '10'
$USE_AMP = 'false'
$SAVE_VALIDATION_PREDICTIONS = 'true'
$DECODER_CHANNELS = '256'
$OUTPUT_ACTIVATION = 'none'
$FINAL_KERNEL_SIZE = '1'


$RUN_NAME = "vit-medsam_of$($OVERSAMPLING_FACTOR)_bs$($BATCH_SIZE)_lr$($LEARNING_RATE)_size$($IMAGE_SIZE)_plateau$($DECODER_PLATEAU_PATIENCE)_blocks$($FINETUNE_LAST_BLOCKS)_$($LR_SCHEDULE)"

$VISUALISE_VALIDATION_PROGRESS_IMAGES = 0
$VISUALISE_VALIDATION_PROGRESS_EPOCHS = 0
$Arguments = @(
    $REPETITION, $FOLD, $TASK_NAME, 'true', 'false',
    '--run-dir', $RUN_DIR, '--pretrained-checkpoint', $MEDSAM_CHECKPOINT,
    '--num-points', $NUM_POINTS, '--fold-lists-path', $FOLD_LISTS_DIR,
    '--mark-list-file', $MARK_LIST_FILE, '--image-data-dir', $IMAGE_DATA_DIR,
    '--resume-training', $RESUME_TRAINING, '--device', $DEVICE,
    '--network-name', 'vit-medsam', '--optimiser-name', 'adamw',
    '--enforce-greyscale', 'true', '--normalise-inputs', 'false', '--recursive-image-search', 'false',
    '--visualise-validation-progress-images', $VISUALISE_VALIDATION_PROGRESS_IMAGES,
    '--visualise-validation-progress-epochs', $VISUALISE_VALIDATION_PROGRESS_EPOCHS,
    '--image-size', $IMAGE_SIZE,
    '--heatmap-sigma', $HEATMAP_SIGMA,
    '--oversampling-factor', $OVERSAMPLING_FACTOR,
    '--batch-size', $BATCH_SIZE,
    '--max-training-epochs', $MAX_TRAINING_EPOCHS,
    '--learning-rate', $LEARNING_RATE,
    '--encoder-learning-rate', $ENCODER_LEARNING_RATE,
    '--freeze-encoder-epochs', $FREEZE_ENCODER_EPOCHS,
    '--finetune-last-blocks', $FINETUNE_LAST_BLOCKS,
    '--gradient-checkpointing', $GRADIENT_CHECKPOINTING,
    '--train-workers', $TRAIN_WORKERS,
    '--random-seed', $RANDOM_SEED,
    '--loss-name', $LOSS_NAME,
    '--positive-weight', $POSITIVE_WEIGHT,
    '--weight-decay', $WEIGHT_DECAY,
    '--lr-schedule', $LR_SCHEDULE,
    '--decoder-plateau-patience', $DECODER_PLATEAU_PATIENCE,
    '--decoder-plateau-min-delta', $DECODER_PLATEAU_MIN_DELTA,
    '--finetune-decoder-lr-factor', $FINETUNE_DECODER_LR_FACTOR,
    '--lr-plateau-patience', $LR_PLATEAU_PATIENCE,
    '--lr-decay-epochs', $LR_DECAY_EPOCHS,
    '--lr-min-factor', $LR_MIN_FACTOR,
    '--lr-step-size', $LR_STEP_SIZE,
    '--lr-gamma', $LR_GAMMA,
    '--early-stop-patience', $EARLY_STOP_PATIENCE,
    '--early-stop-min-delta', $EARLY_STOP_MIN_DELTA,
    '--early-stop-warmup-epochs', $EARLY_STOP_WARMUP_EPOCHS,
    '--use-amp', $USE_AMP,
    '--save-validation-predictions', $SAVE_VALIDATION_PREDICTIONS,
    '--decoder-channels', $DECODER_CHANNELS,
    '--output-activation', $OUTPUT_ACTIVATION,
    '--final-kernel-size', $FINAL_KERNEL_SIZE
 )
if ($LANDMARK_CONSTRAINT_LOSS) { $Arguments += @('--landmark-constraint-loss', $LANDMARK_CONSTRAINT_LOSS); $RUN_NAME += '_constraint_' + $LANDMARK_CONSTRAINT_LOSS }
if ($RUN_NAME) { $Arguments += @('--run-name', $RUN_NAME) }
Push-Location -LiteralPath $ProjectDir
try { & $Python -u -m Heatmaps.heatmap_training_pipeline @Arguments; $RunExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $RunExitCode
