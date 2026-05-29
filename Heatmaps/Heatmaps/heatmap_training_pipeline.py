"""
Train heatmap landmark models using fold lists and mark-list annotations.
"""

import argparse
import datetime as dt
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from . import parameters as pms
from .model_registry import get_available_model_names
from .train_model import HeatmapDataConfig, HeatmapModelConfig, TrainConfig, TrainModel
from .utils.io_utils import discover_fold_numbers, str_to_bool, validate_fold_split_overlaps

RESULTS_DIR_NAME = 'TRAINING_RESULTS'
MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30


@dataclass
class RunConfig:
    fold: int
    task_name: str
    num_of_points: int
    train_model: bool
    copy_files: bool
    run_dir: Path
    save_dir: Path | None
    run_name: str


class HeatmapTrainingPipeline:
    """Run heatmap training and optional result copying for one fold."""

    def __init__(self, run_config, data_config, train_config, model_config):
        self.run_config = run_config
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.run_results_root = self.build_run_results_root()
        self.run_results_path = self.build_run_results_path()

    def run(self):
        """Run the requested pipeline stages."""
        total_start_time = dt.datetime.now()
        self.prepare_run_directories()
        self.print_inputs()
        self.write_run_info()

        if self.run_config.train_model:
            self.train_model()

        if self.run_config.copy_files:
            self.copy_files()

        total_end_time = dt.datetime.now()
        self.print_section_start('Heatmap workflow complete')
        print(f'\tTotal runtime: {self.format_runtime(total_start_time, total_end_time)}.', flush=True)
        print(f'\tRaw total elapsed time: {total_end_time - total_start_time}', flush=True)
        self.print_section_end()

    def train_model(self):
        """Train one heatmap model for the configured fold."""
        self.print_section_start(f'Fold {self.run_config.fold} {self.run_config.task_name} training')
        start_time = dt.datetime.now()
        trainer = TrainModel(data_config=self.data_config, train_config=self.train_config, model_config=self.model_config, output_save_path=self.run_results_path)
        trainer.train()
        end_time = dt.datetime.now()
        print(f'\tFold {self.run_config.fold} {self.run_config.task_name} training complete in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def copy_files(self):
        """Copy run outputs to the optional save directory."""
        self.print_section_start(f'Fold {self.run_config.fold} {self.run_config.task_name} copying outputs')
        start_time = dt.datetime.now()
        save_path = self.get_save_copy_path()

        if save_path is None:
            print(f'\tNo save dir supplied. Outputs remain in {self.run_results_path}.', flush=True)
            self.print_section_end()
            return

        if not self.run_results_path.is_dir():
            raise ValueError(f'Run results path does not exist: {self.run_results_path}')

        save_path.mkdir(exist_ok=True, parents=True)
        entries = list(self.run_results_path.iterdir())
        print(f'\tCopying {len(entries)} result entries from {self.run_results_path} to {save_path}...', flush=True)

        for entry_path in entries:
            destination_path = save_path / entry_path.name
            if entry_path.is_dir():
                if destination_path.exists():
                    shutil.rmtree(destination_path)
                shutil.copytree(entry_path, destination_path)
            else:
                shutil.copy2(entry_path, destination_path)

        end_time = dt.datetime.now()
        print(f'\tFold {self.run_config.fold} outputs copied in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def prepare_run_directories(self):
        """Create output directories."""
        self.run_results_root.mkdir(exist_ok=True, parents=True)
        self.run_results_path.mkdir(exist_ok=True, parents=True)

    def write_run_info(self):
        """Write full run metadata."""
        run_info = {'created_at': dt.datetime.now().isoformat(), 'run_results_root': self.run_results_root, 'run_results_path': self.run_results_path,
                    'save_copy_path': self.get_save_copy_path(), 'run_config': asdict(self.run_config), 'data_config': asdict(self.data_config),
                    'train_config': asdict(self.train_config), 'model_config': asdict(self.model_config)}
        run_info_path = self.run_results_path / f'run_info_{self.run_config.task_name}_f{self.run_config.fold}.json'

        with open(run_info_path, 'w', encoding='utf-8') as run_info_file:
            json.dump(run_info, run_info_file, indent=4, default=str)

    def get_save_copy_path(self):
        """Return the external save path if copying is enabled."""
        if not self.run_config.copy_files:
            return None

        if self.run_config.save_dir is None:
            raise ValueError('save_dir must be supplied when copy_files is True.')

        return self.run_config.save_dir / self.run_config.task_name / self.run_config.run_name

    def build_run_results_root(self):
        """Build the run-level results root."""
        return self.run_config.run_dir / RESULTS_DIR_NAME

    def build_run_results_path(self):
        """Build the folder for this task and run name."""
        return self.run_results_root / self.run_config.task_name / self.run_config.run_name

    def print_inputs(self):
        """Print the resolved pipeline settings."""
        self.print_section_start('Input arguments')
        print(f'\tFold: {self.run_config.fold}', flush=True)
        print(f'\tTask name: {self.run_config.task_name}', flush=True)
        print(f'\tNumber of landmark points: {self.run_config.num_of_points}', flush=True)
        print(f'\tTrain model: {self.run_config.train_model}', flush=True)
        print(f'\tCopy files: {self.run_config.copy_files}', flush=True)
        print(f'\tNumber of folds: {len(discover_fold_numbers(self.data_config.fold_lists_path))}', flush=True)
        print(f'\tImage size: {self.data_config.image_size}', flush=True)
        print(f'\tHeatmap sigma: {self.data_config.heatmap_sigma}', flush=True)
        print(f'\tOversampling factor: {self.data_config.oversampling_factor}', flush=True)
        print('\tInput channels: automatic', flush=True)
        print(f'\tRecursive image search: {self.data_config.recursive_image_search}', flush=True)
        print(f'\tTraining workers: {self.train_config.num_workers}', flush=True)
        print(f'\tRandom seed: {self.train_config.random_seed}', flush=True)
        print(f'\tBatch size: {self.train_config.batch_size}', flush=True)
        print(f'\tLearning rate: {self.train_config.learning_rate}', flush=True)
        print(f'\tOptimiser: {self.train_config.optimiser_name}', flush=True)
        print(f'\tLoss: {self.train_config.loss_name}', flush=True)
        print(f'\tPositive weight: {self.train_config.positive_weight}', flush=True)
        print(f'\tWeight decay: {self.train_config.weight_decay}', flush=True)
        print(f'\tMomentum: {self.train_config.momentum}', flush=True)
        print(f'\tLR schedule: {self.train_config.lr_schedule}', flush=True)
        print(f'\tLR step size: {self.train_config.lr_step_size}', flush=True)
        print(f'\tLR gamma: {self.train_config.lr_gamma}', flush=True)
        print(f'\tEarly stop patience: {self.train_config.early_stop_patience}', flush=True)
        print(f'\tEarly stop min delta: {self.train_config.early_stop_min_delta}', flush=True)
        print(f'\tEarly stop warmup epochs: {self.train_config.early_stop_warmup_epochs}', flush=True)
        print(f'\tUse AMP: {self.train_config.use_amp}', flush=True)
        print(f'\tSave validation predictions: {self.train_config.save_validation_predictions}', flush=True)
        print(f'\tSave validation overlays: {self.train_config.save_validation_overlays}', flush=True)
        print(f'\tNetwork: {self.model_config.network_name}', flush=True)
        print(f'\tBase channels: {self.model_config.base_channels}', flush=True)
        print(f'\tDepth: {self.model_config.depth}', flush=True)
        print(f'\tChannel multiplier: {self.model_config.channel_multiplier}', flush=True)
        print(f'\tMax channels: {self.model_config.max_channels}', flush=True)
        print(f'\tNormalisation: {self.model_config.normalisation}', flush=True)
        print(f'\tActivation: {self.model_config.activation}', flush=True)
        print(f'\tDropout: {self.model_config.dropout}', flush=True)
        print(f'\tUpsampling: {self.model_config.upsampling}', flush=True)
        print(f'\tOutput activation: {self.model_config.output_activation}', flush=True)
        print(f'\tPadding mode: {self.model_config.padding_mode}', flush=True)
        print(f'\tFinal kernel size: {self.model_config.final_kernel_size}', flush=True)
        print(f'\tRun dir: {self.run_config.run_dir}', flush=True)
        print(f'\tSave dir: {self.run_config.save_dir}', flush=True)
        print(f'\tTraining results dir: {self.run_results_root}', flush=True)
        print(f'\tRun name: {self.run_config.run_name}', flush=True)
        print(f'\tRun results path: {self.run_results_path}', flush=True)
        print(f'\tSave copy path: {self.get_save_copy_path()}', flush=True)
        print(f'\tFold lists path: {self.data_config.fold_lists_path}', flush=True)
        print(f'\tMark list file: {self.data_config.mark_list_file}', flush=True)
        print(f'\tImage data dir: {self.data_config.image_data_dir}', flush=True)
        self.print_section_end()

    @staticmethod
    def print_section_start(message):
        """Print a section heading."""
        print('======================================================================================', flush=True)
        print(f"\t{dt.datetime.now().strftime('%d %m %Y %H:%M:%S')} - {message}...", flush=True)

    @staticmethod
    def print_section_end():
        """Print a section divider."""
        print('======================================================================================', flush=True)

    @staticmethod
    def format_runtime(start_time, end_time):
        """Format a runtime duration."""
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'



def normalise_save_dir(args):
    """Enforce save-dir behaviour from the copy-files switch."""
    if not args.copy_files:
        args.save_dir = None
        return

    if args.save_dir is None:
        raise ValueError('--save-dir must be supplied when COPY_FILES is true.')


def validate_args(args, num_of_folds):
    """Validate numeric, path, split, training, and model terminal arguments."""
    normalise_save_dir(args)

    if num_of_folds < 2:
        raise ValueError(f'At least 2 fold files are required. Found {num_of_folds}.')

    if args.fold < 1 or args.fold > num_of_folds:
        raise ValueError(f'fold must be between 1 and {num_of_folds}. Got fold={args.fold}.')

    if args.run_dir.exists() and not args.run_dir.is_dir():
        raise ValueError(f'--run-dir exists but is not a directory: {args.run_dir}')

    if args.save_dir is not None and args.save_dir.exists() and not args.save_dir.is_dir():
        raise ValueError(f'--save-dir exists but is not a directory: {args.save_dir}')

    if not args.mark_list_file.is_file():
        raise ValueError(f'--mark-list-file does not exist or is not a file: {args.mark_list_file}')

    if not args.image_data_dir.is_dir():
        raise ValueError(f'--image-data-dir does not exist or is not a directory: {args.image_data_dir}')

    if args.image_size is None or len(args.image_size) != 2:
        raise ValueError('--image-size must contain exactly two values: HEIGHT WIDTH.')

    image_height, image_width = args.image_size

    if image_height < 1 or image_width < 1:
        raise ValueError('--image-size values must be at least 1.')

    if args.heatmap_sigma <= 0:
        raise ValueError('--heatmap-sigma must be greater than 0.')

    if args.oversampling_factor < 1:
        raise ValueError('--oversampling-factor must be at least 1.')

    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1.')

    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be greater than 0.')

    if args.max_training_epochs < 1:
        raise ValueError('--max-training-epochs must be at least 1.')

    if args.train_workers < 0:
        raise ValueError('--train-workers must be at least 0.')

    if args.random_seed < 0:
        raise ValueError('--random-seed must be at least 0.')

    if args.positive_weight < 0:
        raise ValueError('--positive-weight must be at least 0.')

    if args.weight_decay < 0:
        raise ValueError('--weight-decay must be at least 0.')

    if args.momentum < 0:
        raise ValueError('--momentum must be at least 0.')

    if args.lr_step_size < 1:
        raise ValueError('--lr-step-size must be at least 1.')

    if args.lr_gamma <= 0:
        raise ValueError('--lr-gamma must be greater than 0.')

    if args.early_stop_patience < 1:
        raise ValueError('--early-stop-patience must be at least 1.')

    if args.early_stop_min_delta < 0:
        raise ValueError('--early-stop-min-delta must be at least 0.')

    if args.early_stop_warmup_epochs < 0:
        raise ValueError('--early-stop-warmup-epochs must be at least 0.')

    if args.base_channels < 1:
        raise ValueError('--base-channels must be at least 1.')

    if args.depth < 1:
        raise ValueError('--depth must be at least 1.')

    if args.channel_multiplier < 1:
        raise ValueError('--channel-multiplier must be at least 1.')

    if args.max_channels < args.base_channels:
        raise ValueError('--max-channels must be greater than or equal to --base-channels.')

    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError('--dropout must be in the range [0, 1).')

    if args.loss_name == 'bce_logits' and args.output_activation != 'none':
        raise ValueError('--loss-name bce_logits requires --output-activation none because BCEWithLogitsLoss expects raw logits.')

    if args.normalisation == 'group' and args.base_channels % 8 != 0:
        raise ValueError('--base-channels must be divisible by 8 when --normalisation group is used.')

    validate_fold_split_overlaps(fold_lists_path=args.fold_lists_path, fold=args.fold)

def validate_num_points(value):
    """Validate the number of landmark points for each image."""
    value = int(value)

    if value < MIN_POINTS_PER_IMAGE or value > MAX_POINTS_PER_IMAGE:
        raise argparse.ArgumentTypeError(f'num-points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}.')

    return value


def format_number(value):
    """Format numeric values safely for run folder names."""
    return f'{value:g}'.replace('-', 'm').replace('.', 'p')


def build_run_name(args, num_of_folds):
    """Build a deterministic folder name shared by every fold in the same experiment."""
    height, width = args.image_size
    parts = ['heatmap', f'{num_of_folds}fold', f'{args.num_points}points', args.network_name, f'im{height}x{width}', f'sigma{format_number(args.heatmap_sigma)}',
             f'bc{args.base_channels}', f'depth{args.depth}', f'cm{args.channel_multiplier}', f'mc{args.max_channels}', f'norm{args.normalisation}',
             f'act{args.activation}', f'drop{format_number(args.dropout)}', f'up{args.upsampling}', f'loss{args.loss_name}', f'pw{format_number(args.positive_weight)}',
             f'of{args.oversampling_factor}', f'seed{args.random_seed}', f'bs{args.batch_size}', f'lr{format_number(args.learning_rate)}', f'ep{args.max_training_epochs}']
    return clean_run_name('_'.join(parts))


def clean_run_name(run_name):
    """Return a safe run-name string."""
    run_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(run_name)).strip('._-')

    if not run_name:
        raise ValueError('--run-name cannot be empty after cleaning.')

    return run_name


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a heatmap landmark model using fold lists and mark-list annotations.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fold', type=int, help='Fold number to train.')
    parser.add_argument('task_name', type=str, help='Task/output name, for example prostate_transverse.')
    parser.add_argument('train_model', type=str_to_bool, nargs='?', default=True, help='Train the model.')
    parser.add_argument('copy_files', type=str_to_bool, nargs='?', default=False, help='Copy results to save-dir after training.')

    parser.add_argument('--run-dir', type=Path, required=True, help='Working directory for training outputs.')
    parser.add_argument('--save-dir', type=Path, default=None, help='Optional directory for copied final outputs.')
    parser.add_argument('--run-name', type=str, default=None, help='Optional output-folder override. If omitted, a name is generated from settings.')
    parser.add_argument('--num-points', type=validate_num_points, required=True,
                        help=f'Number of landmarks per image, from {MIN_POINTS_PER_IMAGE} to {MAX_POINTS_PER_IMAGE}.')
    parser.add_argument('--fold-lists-path', type=Path, required=True, help='Directory containing train_fN.txt, val_fN.txt, and test_fN.txt files.')
    parser.add_argument('--mark-list-file', type=Path, required=True, help='Landmark mark-list file.')
    parser.add_argument('--image-data-dir', type=Path, required=True, help='Directory containing source images.')
    parser.add_argument('--image-size', type=int, nargs=2, default=list(pms.image_size), metavar=('HEIGHT', 'WIDTH'), help='Training image size.')
    parser.add_argument('--heatmap-sigma', type=float, default=pms.heatmap_sigma, help='Gaussian sigma for target heatmaps.')
    parser.add_argument('--oversampling-factor', type=int, default=1, help='Training-set multiplier. A value of 1 uses each image once; values above 1 add augmented copies using Heatmaps/Heatmaps/heatmap_transforms.py.')
    parser.add_argument('--recursive-image-search', type=str_to_bool, default=False, help='Search image-data-dir recursively.')

    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--max-training-epochs', type=int, default=80, help='Maximum training epochs.')
    parser.add_argument('--train-workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed used for Python, NumPy, PyTorch, and DataLoader workers.')
    parser.add_argument('--optimiser-name', choices=['adamw', 'sgd'], default='adamw', help='Optimiser.')
    parser.add_argument('--loss-name', choices=['mse', 'weighted_mse', 'smooth_l1', 'bce_logits'], default='weighted_mse', help='Training loss.')
    parser.add_argument('--positive-weight', type=float, default=20.0, help='Peak-region weight for weighted_mse.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--lr-schedule', choices=['none', 'step', 'plateau'], default='plateau', help='Learning-rate schedule.')
    parser.add_argument('--lr-step-size', type=int, default=20, help='StepLR epoch interval.')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='Learning-rate reduction factor.')
    parser.add_argument('--early-stop-patience', type=int, default=15, help='Epochs without improvement before stopping.')
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4, help='Minimum validation-loss improvement.')
    parser.add_argument('--early-stop-warmup-epochs', type=int, default=10, help='Epochs before early stopping is enabled.')
    parser.add_argument('--use-amp', type=str_to_bool, default=False, help='Use automatic mixed precision.')

    parser.add_argument('--save-validation-predictions', type=str_to_bool, default=True, help='Save validation prediction CSV.')
    parser.add_argument('--save-validation-overlays', type=str_to_bool, default=False, help='Save validation heatmap and point overlays.')

    parser.add_argument('--network-name', choices=get_available_model_names(), default='unet_basic', help='Model architecture.')
    parser.add_argument('--base-channels', type=int, default=32, help='First U-Net channel width.')
    parser.add_argument('--depth', type=int, default=4, help='Number of U-Net downsampling levels.')
    parser.add_argument('--channel-multiplier', type=int, default=2, help='Channel multiplier per level.')
    parser.add_argument('--max-channels', type=int, default=512, help='Maximum U-Net channel width.')
    parser.add_argument('--normalisation', choices=['batch', 'instance', 'group', 'none'], default='batch', help='Normalisation layer.')
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu', 'gelu'], default='relu', help='Activation function.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability.')
    parser.add_argument('--upsampling', choices=['bilinear', 'transpose'], default='bilinear', help='Decoder upsampling method.')
    parser.add_argument('--output-activation', choices=['none', 'sigmoid', 'softplus'], default='none', help='Final heatmap activation.')
    parser.add_argument('--padding-mode', choices=['zeros', 'reflect', 'replicate', 'circular'], default='zeros', help='Convolution padding mode.')
    parser.add_argument('--final-kernel-size', type=int, choices=[1, 3], default=1, help='Final convolution kernel size.')

    return parser.parse_args()


def build_configs(args):
    """Build dataclass configurations from terminal arguments."""
    fold_numbers = discover_fold_numbers(args.fold_lists_path)
    num_of_folds = len(fold_numbers)

    if args.fold not in fold_numbers:
        raise ValueError(f'Fold {args.fold} was requested, but available folds are {fold_numbers}.')

    validate_args(args=args, num_of_folds=num_of_folds)

    run_name = clean_run_name(args.run_name) if args.run_name else build_run_name(args=args, num_of_folds=num_of_folds)
    run_config = RunConfig(fold=args.fold, task_name=args.task_name, num_of_points=args.num_points, train_model=args.train_model, copy_files=args.copy_files,
                           run_dir=args.run_dir, save_dir=args.save_dir, run_name=run_name)
    data_config = HeatmapDataConfig(fold=args.fold, task_name=args.task_name, num_of_points=args.num_points, fold_lists_path=args.fold_lists_path,
                                    mark_list_file=args.mark_list_file, image_data_dir=args.image_data_dir, image_size=tuple(args.image_size),
                                    heatmap_sigma=args.heatmap_sigma, input_channels=None, recursive_image_search=args.recursive_image_search,
                                    oversampling_factor=args.oversampling_factor)
    train_config = TrainConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_training_epochs=args.max_training_epochs, num_workers=args.train_workers,
                               random_seed=args.random_seed, optimiser_name=args.optimiser_name, loss_name=args.loss_name, positive_weight=args.positive_weight, weight_decay=args.weight_decay,
                               momentum=args.momentum, lr_schedule=args.lr_schedule, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
                               early_stop_patience=args.early_stop_patience, early_stop_min_delta=args.early_stop_min_delta,
                               early_stop_warmup_epochs=args.early_stop_warmup_epochs, use_amp=args.use_amp, save_validation_predictions=args.save_validation_predictions,
                               save_validation_overlays=args.save_validation_overlays)
    model_config = HeatmapModelConfig(network_name=args.network_name, base_channels=args.base_channels, depth=args.depth, channel_multiplier=args.channel_multiplier,
                                      max_channels=args.max_channels, normalisation=None if args.normalisation == 'none' else args.normalisation,
                                      activation=args.activation, dropout=args.dropout, upsampling=args.upsampling, output_activation=args.output_activation,
                                      padding_mode=args.padding_mode, final_kernel_size=args.final_kernel_size)
    return run_config, data_config, train_config, model_config


def main():
    """Run the command-line training workflow."""
    args = parse_args()
    run_config, data_config, train_config, model_config = build_configs(args)
    pipeline = HeatmapTrainingPipeline(run_config=run_config, data_config=data_config, train_config=train_config, model_config=model_config)
    pipeline.run()


if __name__ == '__main__':
    main()
