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
$RUN_NAME = '' # Empty uses a name generated from the resolved configuration.

# The selected network supplies all recommended data, model and training defaults.
# Add only deliberate overrides, for example @('--batch-size', '16').
$ExtraArguments = @()
$Arguments = @(
    $REPETITION, $FOLD, $TASK_NAME, 'true', 'false',
    '--run-dir', $RUN_DIR, '--num-points', $NUM_POINTS,
    '--fold-lists-path', $FOLD_LISTS_DIR, '--mark-list-file', $MARK_LIST_FILE,
    '--image-data-dir', $IMAGE_DATA_DIR, '--resume-training', $RESUME_TRAINING,
    '--network-name', 'vit-medsam', '--device', $DEVICE,
    '--pretrained-checkpoint', $MEDSAM_CHECKPOINT
)
if ($RUN_NAME) { $Arguments += @('--run-name', $RUN_NAME) }
$Arguments += $ExtraArguments
Push-Location -LiteralPath $ProjectDir
try { & $Python -u -m Heatmaps.heatmap_training_pipeline @Arguments; $RunExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $RunExitCode
