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
from .utils.io_utils import discover_fold_numbers, str_to_bool

RESULTS_DIR_NAME = 'TRAINING_RESULTS'


@dataclass
class RunConfig:
    fold: int
    task_name: str
    num_of_points: int
    train_model: bool
    copy_files: bool
    delete_files: bool
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

        if self.run_config.delete_files:
            self.delete_files()

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

    def delete_files(self):
        """Report that no generated patch data exists for heatmap training."""
        self.print_section_start(f'Fold {self.run_config.fold} delete requested')
        print('\tNo generated patch dataset is created by the heatmap pipeline, so there is no training-data folder to delete.', flush=True)
        self.print_section_end()

    def prepare_run_directories(self):
        """Create output directories."""
        self.run_results_root.mkdir(exist_ok=True, parents=True)
        self.run_results_path.mkdir(exist_ok=True, parents=True)

    def write_run_info(self):
        """Write full run metadata."""
        run_info = {'created_at': dt.datetime.now().isoformat(), 'run_results_root': self.run_results_root, 'run_results_path': self.run_results_path, 'save_copy_path': self.get_save_copy_path(), 'run_config': asdict(self.run_config), 'data_config': asdict(self.data_config), 'train_config': asdict(self.train_config), 'model_config': asdict(self.model_config)}
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
        self.print_section_start('Heatmap training inputs')
        print(f'\tFold: {self.run_config.fold}', flush=True)
        print(f'\tTask name: {self.run_config.task_name}', flush=True)
        print(f'\tNumber of points: {self.run_config.num_of_points}', flush=True)
        print(f'\tRun directory: {self.run_config.run_dir}', flush=True)
        print(f'\tResults path: {self.run_results_path}', flush=True)
        print(f'\tFold lists: {self.data_config.fold_lists_path}', flush=True)
        print(f'\tMark list: {self.data_config.mark_list_file}', flush=True)
        print(f'\tImages: {self.data_config.image_data_dir}', flush=True)
        print(f'\tImage size: {self.data_config.image_size}', flush=True)
        print('\tInput channels: automatic', flush=True)
        print(f'\tModel: {self.model_config.network_name}', flush=True)
        self.print_section_end()

    @staticmethod
    def print_section_start(title):
        """Print a section header."""
        print('\n' + '=' * 100, flush=True)
        print(title, flush=True)
        print('=' * 100, flush=True)

    @staticmethod
    def print_section_end():
        """Print a section footer."""
        print('=' * 100 + '\n', flush=True)

    @staticmethod
    def format_runtime(start_time, end_time):
        """Format a runtime duration."""
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def clean_run_name(run_name):
    """Return a safe run-name string."""
    run_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(run_name)).strip('._-')

    if not run_name:
        raise ValueError('--run-name cannot be empty after cleaning.')

    return run_name


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a heatmap landmark model using fold lists and mark-list annotations.')
    parser.add_argument('fold', type=int)
    parser.add_argument('task_name', type=str)
    parser.add_argument('train_model', type=str_to_bool, nargs='?', default=True)
    parser.add_argument('copy_files', type=str_to_bool, nargs='?', default=False)
    parser.add_argument('delete_files', type=str_to_bool, nargs='?', default=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, default=None)
    parser.add_argument('--run-name', type=str, default='unet_basic')
    parser.add_argument('--num-points', type=int, required=True)
    parser.add_argument('--fold-lists-path', type=Path, required=True)
    parser.add_argument('--mark-list-file', type=Path, required=True)
    parser.add_argument('--image-data-dir', type=Path, required=True)
    parser.add_argument('--image-size', type=int, nargs=2, default=list(pms.image_size), metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--heatmap-sigma', type=float, default=pms.heatmap_sigma)
    parser.add_argument('--pixels-per-cm', type=float, default=pms.pixels_per_cm)
    parser.add_argument('--recursive-image-search', type=str_to_bool, default=False)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-training-epochs', type=int, default=80)
    parser.add_argument('--train-workers', type=int, default=8)
    parser.add_argument('--optimiser-name', choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--loss-name', choices=['mse', 'weighted_mse', 'smooth_l1', 'bce_logits'], default='weighted_mse')
    parser.add_argument('--positive-weight', type=float, default=20.0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr-schedule', choices=['none', 'step', 'plateau'], default='plateau')
    parser.add_argument('--lr-step-size', type=int, default=20)
    parser.add_argument('--lr-gamma', type=float, default=0.5)
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser.add_argument('--early-stop-warmup-epochs', type=int, default=10)
    parser.add_argument('--use-amp', type=str_to_bool, default=False)
    parser.add_argument('--save-validation-predictions', type=str_to_bool, default=True)
    parser.add_argument('--save-validation-overlays', type=str_to_bool, default=False)
    parser.add_argument('--network-name', choices=get_available_model_names(), default='unet_basic')
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--channel-multiplier', type=int, default=2)
    parser.add_argument('--max-channels', type=int, default=512)
    parser.add_argument('--normalisation', choices=['batch', 'instance', 'group', 'none'], default='batch')
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu', 'gelu'], default='relu')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsampling', choices=['bilinear', 'transpose'], default='bilinear')
    parser.add_argument('--output-activation', choices=['none', 'sigmoid', 'softplus'], default='none')
    parser.add_argument('--padding-mode', choices=['zeros', 'reflect', 'replicate', 'circular'], default='zeros')
    parser.add_argument('--final-kernel-size', type=int, choices=[1, 3], default=1)
    return parser.parse_args()


def build_configs(args):
    """Build dataclass configurations from terminal arguments."""
    discover_fold_numbers(args.fold_lists_path)
    run_name = clean_run_name(args.run_name)
    run_config = RunConfig(fold=args.fold, task_name=args.task_name, num_of_points=args.num_points, train_model=args.train_model, copy_files=args.copy_files, delete_files=args.delete_files, run_dir=args.run_dir, save_dir=args.save_dir, run_name=run_name)
    data_config = HeatmapDataConfig(fold=args.fold, task_name=args.task_name, num_of_points=args.num_points, fold_lists_path=args.fold_lists_path, mark_list_file=args.mark_list_file, image_data_dir=args.image_data_dir, image_size=tuple(args.image_size), heatmap_sigma=args.heatmap_sigma, pixels_per_cm=args.pixels_per_cm, input_channels=args.input_channels, recursive_image_search=args.recursive_image_search)
    train_config = TrainConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_training_epochs=args.max_training_epochs, num_workers=args.train_workers, optimiser_name=args.optimiser_name, loss_name=args.loss_name, positive_weight=args.positive_weight, weight_decay=args.weight_decay, momentum=args.momentum, lr_schedule=args.lr_schedule, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma, early_stop_patience=args.early_stop_patience, early_stop_min_delta=args.early_stop_min_delta, early_stop_warmup_epochs=args.early_stop_warmup_epochs, use_amp=args.use_amp, save_validation_predictions=args.save_validation_predictions, save_validation_overlays=args.save_validation_overlays)
    model_config = HeatmapModelConfig(network_name=args.network_name, base_channels=args.base_channels, depth=args.depth, channel_multiplier=args.channel_multiplier, max_channels=args.max_channels, normalisation=None if args.normalisation == 'none' else args.normalisation, activation=args.activation, dropout=args.dropout, upsampling=args.upsampling, output_activation=args.output_activation, padding_mode=args.padding_mode, final_kernel_size=args.final_kernel_size)
    return run_config, data_config, train_config, model_config


def main():
    """Run the command-line training workflow."""
    args = parse_args()
    run_config, data_config, train_config, model_config = build_configs(args)
    pipeline = HeatmapTrainingPipeline(run_config=run_config, data_config=data_config, train_config=train_config, model_config=model_config)
    pipeline.run()


if __name__ == '__main__':
    main()
