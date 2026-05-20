"""
Create train/validation fold data, train a model, optionally copy outputs, and safely delete generated training data.
"""
import argparse
import csv
import datetime as dt
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from . import parameters as pms
from .data_creator import DataCreator, discover_fold_numbers
from .model_registry import get_available_model_names
from .train_model import TrainModel, TrainConfig, QuadrupletConfig

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
BASE_CSV_COLUMNS = 5
TRAINING_DATA_DIR_NAME = 'TRAINING_DATA'
RESULTS_DIR_NAME = 'TRAINING_RESULTS'


def format_fold_collection_name(num_of_folds):
    """Return the shared fold collection directory name."""
    return f'{num_of_folds}_Folds'


def format_training_sample_dir_name(sub_patch_scales, num_of_points, patches_per_training_sample):
    """Return the shared data directory name for one task and sampling setup."""
    scale_label = '-'.join(str(scale) for scale in sub_patch_scales)

    return f'{scale_label}_{num_of_points}points_{patches_per_training_sample}pertrainingsample'


@dataclass
class DataCreationConfig:
    distance_intervals: list
    angle_intervals: list
    num_of_folds: int
    sub_patch_scales: list
    patches_per_training_sample: int
    grid_spacing: int
    fold_lists_path: Path
    mark_list_file: Path
    image_data_dir: Path
    sampling_variances: tuple
    num_workers: int
    random_seed: int
    keep_part_csvs: bool

    @property
    def tasks_classes(self):
        """Return task classes in the order expected by the model."""
        return [self.distance_intervals, self.angle_intervals]


@dataclass
class RunConfig:
    fold: int
    task_name: str
    num_of_points: int
    create_data: bool
    train_model: bool
    copy_files: bool
    delete_files: bool
    run_dir: Path
    save_dir: Path | None
    run_name: str


class CreateTrain:
    def __init__(self, run_config, data_config, train_config, quadruplet_config):
        self.run_config = run_config
        self.data_config = data_config
        self.train_config = train_config
        self.quadruplet_config = quadruplet_config

        self.fold = run_config.fold
        self.task_name = run_config.task_name
        self.num_of_points = run_config.num_of_points

        self.run_training_root = self.build_run_training_root()
        self.run_results_root = self.build_run_results_root()
        self.fold_training_data_dir = self.build_fold_training_data_dir()
        self.data_save_path = self.build_data_save_path()
        self.run_results_path = self.build_run_results_path()

    def create_data(self):
        """Create fold data for the requested task after removing existing artefacts for this fold."""
        self.print_section_start(f'Fold {self.fold} {self.task_name} data creation')
        start_time = dt.datetime.now()

        self.delete_existing_fold_data()

        data_creator = DataCreator(
            distance_intervals=self.data_config.distance_intervals,
            angle_intervals=self.data_config.angle_intervals,
            subpatch_scales=self.data_config.sub_patch_scales,
            task_name=self.task_name,
            data_save_path=self.data_save_path,
            num_of_points=self.num_of_points,
            patches_per_training_sample=self.data_config.patches_per_training_sample,
            fold_lists_path=self.data_config.fold_lists_path,
            mark_list_path=self.data_config.mark_list_file,
            image_data_path=self.data_config.image_data_dir,
            sampling_variances=self.data_config.sampling_variances,
            num_workers=self.data_config.num_workers,
            random_seed=self.data_config.random_seed,
            keep_part_csvs=self.data_config.keep_part_csvs
        )

        data_creator.create(grid_spacing=self.data_config.grid_spacing, current_fold=self.fold)
        self.update_quadruplet_input_channels_from_generated_data()
        self.write_metadata()

        end_time = dt.datetime.now()
        print(f'\tFold {self.fold} {self.task_name} data created in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def update_quadruplet_input_channels_from_generated_data(self):
        """Infer generated patch channels and store them in the model config."""
        train_channels = self.infer_patch_input_channels_from_csv(self.get_phase_csv_path('Train'))
        val_channels = self.infer_patch_input_channels_from_csv(self.get_phase_csv_path('Val'))

        if train_channels != val_channels:
            raise ValueError(f'Train patches have {train_channels} channels, but validation patches have {val_channels} channels.')

        configured_channels = self.quadruplet_config.input_channels

        if configured_channels is not None and int(configured_channels) != train_channels:
            raise ValueError(f'QuadrupletConfig requested {configured_channels} input channels, but generated patches contain {train_channels}.')

        self.quadruplet_config.input_channels = int(train_channels)
        print(f'\tDetected {train_channels} input channel(s) per patch from generated data.', flush=True)

    @staticmethod
    def infer_patch_input_channels_from_csv(csv_path):
        """Infer patch channel count from the first valid patch path in a generated CSV."""
        from .custom_dataset import load_patch_image

        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise ValueError(f'Generated data CSV does not exist: {csv_path}')

        with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)

            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                if len(row) < 2:
                    raise ValueError(f'CSV row {row_number} in {csv_path} does not contain a patch path column.')

                patch_path = Path(row[1])

                if not patch_path.is_file():
                    candidate_path = csv_path.parent / patch_path

                    if candidate_path.is_file():
                        patch_path = candidate_path

                if not patch_path.is_file():
                    raise FileNotFoundError(f'Patch path from row {row_number} in {csv_path} was not found: {row[1]}')

                patch_image = load_patch_image(patch_path)

                return int(patch_image.shape[0])

        raise ValueError(f'{csv_path} is empty.')

    def delete_existing_fold_data(self):
        """Delete only the current fold artefacts before recreating that fold."""
        if not self.data_save_path.exists():
            return

        deleted_count = 0
        training_root = self.run_training_root.resolve()

        for target in self.get_fold_data_paths(include_metadata=True):
            resolved_target = target.resolve()

            if not target.exists():
                continue

            if resolved_target == training_root or training_root not in resolved_target.parents:
                raise ValueError(f'Refusing to delete {resolved_target}; it is not inside run training root={training_root}.')

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            deleted_count += 1
            print(f'\tDeleted existing fold artefact before regeneration: {target}', flush=True)

        if deleted_count == 0:
            print(f'\tExisting data directory found at {self.data_save_path}, but no artefacts for fold {self.fold} were found.', flush=True)

    def train_model(self):
        """Validate the generated data and train the model."""
        self.print_section_start(f'Fold {self.fold} {self.task_name} training')
        start_time = dt.datetime.now()

        self.validate_training_data_point_count()
        self.update_quadruplet_input_channels_from_generated_data()
        self.run_results_path.mkdir(exist_ok=True, parents=True)

        trainer = TrainModel(
            current_fold=self.fold,
            data_save_path=self.data_save_path,
            output_save_path=self.run_results_path,
            num_of_points=self.num_of_points,
            tasks_classes=self.data_config.tasks_classes,
            train_config=self.train_config,
            quadruplet_config=self.quadruplet_config
        )

        trainer.train()

        if trainer.input_channels is not None:
            self.quadruplet_config.input_channels = int(trainer.input_channels)

        self.write_run_info(write_to_data_dir=False)

        end_time = dt.datetime.now()
        print(f'\tFold {self.fold} {self.task_name} training complete in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def validate_training_data_point_count(self):
        """Check that train and validation CSV files match the requested point count."""
        tasks_per_point = len(self.data_config.tasks_classes)
        train_csv_path = self.get_phase_csv_path('Train')
        val_csv_path = self.get_phase_csv_path('Val')

        train_points = infer_num_points_from_csv(csv_path=train_csv_path, tasks_per_point=tasks_per_point)
        val_points = infer_num_points_from_csv(csv_path=val_csv_path, tasks_per_point=tasks_per_point)

        if train_points != val_points:
            raise ValueError(f'Train data has {train_points} points but validation data has {val_points} points.')

        if train_points != self.num_of_points:
            raise ValueError(f'Model requested {self.num_of_points} points but generated data contains {train_points} points.')

        self.validate_run_metadata_point_count()

    def validate_run_metadata_point_count(self):
        """Check data creation metadata when it is available."""
        metadata_path = self.data_save_path / f'run_info_{self.task_name}_f{self.fold}.json'

        if not metadata_path.is_file():
            return

        with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        created_points = int(metadata.get('num_of_points', self.num_of_points))

        if created_points != self.num_of_points:
            raise ValueError(f'Model requested {self.num_of_points} points but metadata says data was created with {created_points} points.')

    def copy_files(self):
        """Copy selected run results to the optional external save directory."""
        self.print_section_start(f'Fold {self.fold} {self.task_name} copying outputs')
        start_time = dt.datetime.now()

        save_path = self.get_save_copy_path()

        if save_path is None:
            print(f'\tNo save dir supplied. Outputs remain in {self.run_results_path}.', flush=True)
            self.print_section_end()
            return

        if not self.run_results_path.is_dir():
            raise ValueError(f'Run results path does not exist: {self.run_results_path}')

        save_path.mkdir(exist_ok=True, parents=True)
        files = [file_path for file_path in self.run_results_path.iterdir() if file_path.is_file()]
        print(f'\tCopying {len(files)} files from {self.run_results_path} to {save_path}...', flush=True)

        for file_path in files:
            shutil.copy(file_path, save_path / file_path.name)

        end_time = dt.datetime.now()
        print(f'\tFold {self.fold} {self.task_name} outputs copied in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def delete_files(self):
        """Delete only generated training-data artefacts for this fold."""
        self.print_section_start(f'Fold {self.fold} {self.task_name} deleting training data')
        start_time = dt.datetime.now()

        training_root = self.run_training_root.resolve()
        deleted_count = 0

        for target in self.get_fold_data_paths(include_metadata=True):
            resolved_target = target.resolve()

            if not target.exists():
                continue

            if resolved_target == training_root or training_root not in resolved_target.parents:
                raise ValueError(f'Refusing to delete {resolved_target}; it is not inside run training root={training_root}.')

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            deleted_count += 1
            print(f'\tDeleted {target}', flush=True)

        if deleted_count == 0:
            print(f'\tNothing to delete for fold {self.fold} in {self.data_save_path}', flush=True)

        end_time = dt.datetime.now()
        print(f'\tFold {self.fold} {self.task_name} training data deleted in {self.format_runtime(start_time, end_time)}.', flush=True)
        print(f'\tRaw elapsed time: {end_time - start_time}', flush=True)
        self.print_section_end()

    def run(self):
        """Run the requested pipeline stages."""
        total_start_time = dt.datetime.now()

        self.prepare_run_directories()
        self.print_inputs()
        self.write_run_info()

        if self.run_config.create_data:
            self.create_data()

        if self.run_config.train_model and not self.run_config.create_data:
            self.validate_no_partial_fold_data()

        if self.run_config.train_model:
            self.train_model()

        if self.run_config.copy_files:
            self.copy_files()

        if self.run_config.delete_files:
            self.delete_files()

        total_end_time = dt.datetime.now()
        self.print_section_start('Create and train workflow complete')
        print(f'\tTotal runtime: {self.format_runtime(total_start_time, total_end_time)}.', flush=True)
        print(f'\tRaw total elapsed time: {total_end_time - total_start_time}', flush=True)
        self.print_section_end()

    def fold_data_exists(self):
        """Return True when all required fold-specific generated data exists."""
        return all(path.exists() for path in self.get_fold_data_paths(include_metadata=False))

    def validate_no_partial_fold_data(self):
        """Prevent accidental reuse of partially generated fold data."""
        expected_paths = self.get_fold_data_paths(include_metadata=False)
        existing_paths = [path for path in expected_paths if path.exists()]

        if existing_paths and len(existing_paths) != len(expected_paths):
            missing_paths = [path for path in expected_paths if not path.exists()]
            missing_text = '\n'.join(str(path) for path in missing_paths)
            existing_text = '\n'.join(str(path) for path in existing_paths)

            raise ValueError(
                f'Partial fold data found for fold {self.fold} in {self.data_save_path}.\n'
                f'Existing paths:\n{existing_text}\n'
                f'Missing paths:\n{missing_text}\n'
                'Delete the partial fold data or regenerate it before training.'
            )

    def get_fold_data_paths(self, include_metadata=False):
        """Return fold-specific generated data paths inside the shared data directory."""
        paths = [
            self.get_phase_csv_path('Train'),
            self.get_phase_csv_path('Val'),
            self.get_phase_patch_dir('Train'),
            self.get_phase_patch_dir('Val'),
            self.get_phase_image_dir('Train'),
            self.get_phase_image_dir('Val')
        ]

        if include_metadata:
            paths.extend([
                self.data_save_path / f'data_info_f{self.fold}.csv',
                self.data_save_path / f'run_info_{self.task_name}_f{self.fold}.json',
                self.data_save_path / f'Train_csv_parts_F{self.fold}',
                self.data_save_path / f'Val_csv_parts_F{self.fold}'
            ])

        return paths

    def get_phase_csv_path(self, phase):
        """Return the generated phase CSV path for this fold."""
        return self.data_save_path / f'{phase}_f{self.fold}.csv'

    def get_phase_patch_dir(self, phase):
        """Return the generated phase patch directory for this fold."""
        return self.data_save_path / f'{phase}_Patches_F{self.fold}'

    def get_phase_image_dir(self, phase):
        """Return the generated phase image-overlay directory for this fold."""
        return self.data_save_path / f'{phase}_Images_F{self.fold}'

    def prepare_run_directories(self):
        """Create the two required high-level run directories."""
        self.run_training_root.mkdir(exist_ok=True, parents=True)
        self.run_results_root.mkdir(exist_ok=True, parents=True)
        self.run_results_path.mkdir(exist_ok=True, parents=True)

    def write_data_info(self):
        """Write fold data creation metadata."""
        self.data_save_path.mkdir(exist_ok=True, parents=True)
        info_path = self.data_save_path / f'data_info_f{self.fold}.csv'

        with open(info_path, 'w', newline='', encoding='utf-8') as info_csv:
            writer = csv.writer(info_csv)
            writer.writerow([
                'DISTANCE_INTERVALS',
                'ANGLE_INTERVALS',
                'FOLD_NUMBER',
                'TASK_NAME',
                'NUM_OF_POINTS',
                'SUB_PATCH_SCALES',
                'PATCH_SIZE',
                'PATCHES_PER_TRAINING_SAMPLE',
                'GRID_DATA_STEP',
                'SAMPLING_VARIANCES',
                'NUM_WORKERS',
                'RANDOM_SEED',
                'MARK_LIST_FILE',
                'IMAGE_DATA_DIR'
            ])
            writer.writerow([
                str(self.data_config.distance_intervals),
                str(self.data_config.angle_intervals),
                self.fold,
                self.task_name,
                self.num_of_points,
                self.data_config.sub_patch_scales,
                self.data_config.sub_patch_scales[0],
                self.data_config.patches_per_training_sample,
                self.data_config.grid_spacing,
                self.data_config.sampling_variances,
                self.data_config.num_workers,
                self.data_config.random_seed,
                self.data_config.mark_list_file,
                self.data_config.image_data_dir
            ])

    def write_run_info(self, write_to_data_dir=False):
        """Write full run, data, training, and model metadata."""
        run_info_path_name = f'run_info_{self.task_name}_f{self.fold}.json'
        save_copy_path = self.get_save_copy_path()

        run_info = {
            'created_at': dt.datetime.now().isoformat(),
            'fold': self.fold,
            'task_name': self.task_name,
            'num_of_points': self.num_of_points,
            'run_dir': self.run_config.run_dir,
            'run_training_root': self.run_training_root,
            'run_results_root': self.run_results_root,
            'fold_training_data_dir': self.fold_training_data_dir,
            'data_save_path': self.data_save_path,
            'run_results_path': self.run_results_path,
            'save_copy_path': save_copy_path,
            'mark_list_file': self.data_config.mark_list_file,
            'image_data_dir': self.data_config.image_data_dir,
            'run_config': asdict(self.run_config),
            'data_config': asdict(self.data_config),
            'train_config': asdict(self.train_config),
            'quadruplet_config': asdict(self.quadruplet_config)
        }

        output_dirs = [self.run_results_path]

        if write_to_data_dir:
            output_dirs.append(self.data_save_path)

        for output_dir in output_dirs:
            output_dir.mkdir(exist_ok=True, parents=True)

            with open(output_dir / run_info_path_name, 'w', encoding='utf-8') as run_info_file:
                json.dump(run_info, run_info_file, indent=4, default=str)

    def write_metadata(self):
        """Write compact CSV metadata and full JSON metadata."""
        self.write_data_info()
        self.write_run_info(write_to_data_dir=True)

    def get_save_copy_path(self):
        """Return the optional external save path for copied result files."""
        if not self.run_config.copy_files:
            return None

        if self.run_config.save_dir is None:
            raise ValueError('save_dir must be supplied when copy_files is True.')

        return self.run_config.save_dir / self.task_name / self.run_config.run_name

    def build_run_training_root(self):
        """Build the run-level training-data root."""
        return self.run_config.run_dir / TRAINING_DATA_DIR_NAME

    def build_run_results_root(self):
        """Build the run-level results root."""
        return self.run_config.run_dir / RESULTS_DIR_NAME

    def build_fold_training_data_dir(self):
        """Build the shared task-level training-data directory."""
        return self.run_training_root / self.task_name / format_fold_collection_name(self.data_config.num_of_folds)

    def build_run_results_path(self):
        """Build the folder containing model checkpoints, logs, plots, and metadata."""
        return self.run_results_root / self.task_name / self.run_config.run_name

    def build_data_save_path(self):
        """Build the shared folder containing generated fold CSVs, images, and patches."""
        sample_dir_name = format_training_sample_dir_name(
            sub_patch_scales=self.data_config.sub_patch_scales,
            num_of_points=self.num_of_points,
            patches_per_training_sample=self.data_config.patches_per_training_sample
        )

        return self.fold_training_data_dir / sample_dir_name

    def print_inputs(self):
        """Print the resolved pipeline settings."""
        self.print_section_start('Input arguments')
        print(f'\t\tFold: {self.fold}', flush=True)
        print(f'\t\tTask name: {self.task_name}', flush=True)
        print(f'\t\tNumber of landmark points: {self.run_config.num_of_points}', flush=True)
        print(f'\t\tCreate data: {self.run_config.create_data}', flush=True)
        print(f'\t\tTrain model: {self.run_config.train_model}', flush=True)
        print(f'\t\tCopy files: {self.run_config.copy_files}', flush=True)
        print(f'\t\tDelete files: {self.run_config.delete_files}', flush=True)
        print(f'\t\tNumber of folds: {self.data_config.num_of_folds}', flush=True)
        print(f'\t\tSub-patch scales: {self.data_config.sub_patch_scales}', flush=True)
        print(f'\t\tPatches per training sample: {self.data_config.patches_per_training_sample}', flush=True)
        print(f'\t\tGrid data step: {self.data_config.grid_spacing}', flush=True)
        print(f'\t\tSampling variances: {self.data_config.sampling_variances}', flush=True)
        print(f'\t\tData workers: {self.data_config.num_workers}', flush=True)
        print(f'\t\tTraining workers: {self.train_config.num_workers}', flush=True)
        print(f'\t\tBatch size: {self.train_config.batch_size}', flush=True)
        print(f'\t\tValidation/logging interval: {self.train_config.loss_print_interval} batches', flush=True)
        print(f'\t\tRandom seed: {self.data_config.random_seed}', flush=True)
        print(f'\t\tNetwork: {self.quadruplet_config.network_name}', flush=True)
        print(f'\t\tBranch features: {self.quadruplet_config.branch_features}', flush=True)
        print(f'\t\tFrozen stages: {self.quadruplet_config.frozen_stages}', flush=True)
        print(f'\t\tSmall input stem: {self.quadruplet_config.small_input_stem}', flush=True)
        print(f'\t\tRun dir: {self.run_config.run_dir}', flush=True)
        print(f'\t\tSave dir: {self.run_config.save_dir}', flush=True)
        print(f'\t\tTraining root dir: {self.run_training_root}', flush=True)
        print(f'\t\tTask training data dir: {self.fold_training_data_dir}', flush=True)
        print(f'\t\tTraining results dir: {self.run_results_root}', flush=True)
        print(f'\t\tRun name: {self.run_config.run_name}', flush=True)
        print(f'\t\tRun results path: {self.run_results_path}', flush=True)
        print(f'\t\tSave copy path: {self.get_save_copy_path()}', flush=True)
        print(f'\t\tFold lists path: {self.data_config.fold_lists_path}', flush=True)
        print(f'\t\tMark list file: {self.data_config.mark_list_file}', flush=True)
        print(f'\t\tImage data dir: {self.data_config.image_data_dir}', flush=True)
        print(f'\t\tData save path: {self.data_save_path}', flush=True)
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
        """Format elapsed runtime as hours, minutes, and seconds."""
        elapsed = end_time - start_time
        total_seconds = int(elapsed.total_seconds())

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def str_to_bool(value):
    """Convert command-line strings to booleans."""
    value = str(value).lower().strip()

    if value in ('true', 't', 'yes', 'y', '1'):
        return True

    if value in ('false', 'f', 'no', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def optional_path(value):
    """Convert an optional command-line path, treating empty strings as not supplied."""
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return Path(value)


def normalise_save_dir(args):
    """Enforce save-dir behaviour from the copy-files switch."""
    if not args.copy_files:
        args.save_dir = None
        return

    if args.save_dir is None:
        raise ValueError('--save-dir must be supplied when COPY_FILES is true.')


def parse_args():
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description='Create train/validation fold data, train a model, copy selected outputs, and delete generated training data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fold', type=int, metavar='FOLD', help='Fold number to run. Must match an available train_fN.txt file in --fold-lists-path.')
    parser.add_argument('task_name', type=validate_task_name, metavar='TASK_NAME',
                        help='Task name used in output paths and metadata, for example transverse or sagittal for prostate imaging.')
    parser.add_argument('create_data', type=str_to_bool, metavar='CREATE_DATA',
                        help='Whether to create patch data before training. Accepted values: true/false, yes/no, 1/0.')
    parser.add_argument('train_model', type=str_to_bool, metavar='TRAIN_MODEL', help='Whether to train the model using the generated or existing fold data.')
    parser.add_argument('copy_files', type=str_to_bool, metavar='COPY_FILES', help='Whether to copy selected result files to --save-dir after training.')
    parser.add_argument('delete_files', type=str_to_bool, metavar='DELETE_FILES',
                        help='Whether to delete generated training-data artefacts for this fold after completion.')

    parser.add_argument('--run-dir', type=Path, required=True, help='Root directory used for generated training data and model results.')
    parser.add_argument('--save-dir', type=optional_path, default=None,
                        help='Optional external directory for copied result files. Required when COPY_FILES is true and ignored when COPY_FILES is false.')
    parser.add_argument('--num-points', type=validate_num_points, required=True,
                        help=f'Number of ordered landmark points per image. Must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}.')
    parser.add_argument('--fold-lists-path', type=Path, required=True, help='Directory containing train_fN.txt files and val.txt.')
    parser.add_argument('--mark-list-file', type=Path, required=True, help='Text file containing image filenames followed by landmark coordinate pairs.')
    parser.add_argument('--image-data-dir', type=Path, required=True, help='Directory containing the source image files referenced by the mark-list file.')
    parser.add_argument('--data-creation-workers', type=int, required=True, help='Number of worker processes used during patch/data creation.')
    parser.add_argument('--train-workers', type=int, required=True, help='Number of PyTorch DataLoader workers used during training. Use 0 for single-process loading.')
    parser.add_argument('--random-seed', type=int, required=True, help='Random seed used for deterministic training-centre sampling.')
    parser.add_argument('--keep-part-csvs', type=str_to_bool, required=True,
                        help='Whether to keep temporary per-sample CSV part files after merging. Accepted values: true/false, yes/no, 1/0.')
    parser.add_argument('--batch-size', type=int, required=True, help='Training batch size.')
    parser.add_argument('--max-training-epochs', type=int, required=True, help='Maximum number of training epochs.')
    parser.add_argument('--learning-rate', type=float, required=True, help='Initial SGD learning rate.')
    parser.add_argument('--lr-schedule', type=str_to_bool, required=True, help='Whether to enable the validation-accuracy-triggered learning-rate scheduler.')
    parser.add_argument('--lr-step-size', type=int, default=1, help='StepLR step size used when --lr-schedule is true.')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='StepLR multiplicative learning-rate decay factor used when --lr-schedule is true.')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Number of validation epochs without sufficient loss improvement before early stopping.')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.001, help='Minimum validation-loss improvement required to reset early-stopping patience.')
    parser.add_argument('--early-stop-warmup-epochs', type=int, default=3, help='Number of initial epochs before early stopping is allowed.')
    parser.add_argument('--loss-print-samples', type=int, required=True,
                        help='Approximate number of training samples between validation/logging events. Converted internally to a batch interval.')
    parser.add_argument('--patches-per-training-sample', type=int, required=True,
                        help='Total number of sampled patch centres created per training image, distributed across all landmarks and sampling variances.')
    parser.add_argument('--grid-spacing', type=int, required=True, help='Pixel stride used to create grid patch centres for validation images.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional custom run name. When omitted, a deterministic name is generated from the run configuration.')

    parser.add_argument('--network-name', type=str, choices=get_available_model_names(), required=True, help='Model backbone to train.')
    parser.add_argument('--branch-features', type=int, required=True, help='Number of features output by each quadruplet branch before concatenation.')
    parser.add_argument('--frozen-stages', type=int, required=True,
                        help='Number of pretrained ResNet stages to freeze. Use 0 for untrained networks, small_cnn, and non-conventional ResNet networks.')
    parser.add_argument('--small-input-stem', type=str_to_bool, required=True, help='Whether to use the small-input ResNet stem. Use false for small_cnn.')

    return parser.parse_args()


def validate_args(args, num_of_folds):
    """Validate numeric, path, training, and model terminal arguments."""
    normalise_save_dir(args)

    if args.data_creation_workers < 1:
        raise ValueError('--data-creation-workers must be at least 1.')

    if args.train_workers < 0:
        raise ValueError('--train-workers must be at least 0.')

    if args.random_seed < 0:
        raise ValueError('--random-seed must be at least 0.')

    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1.')

    if args.max_training_epochs < 1:
        raise ValueError('--max-training-epochs must be at least 1.')

    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be greater than 0.')

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

    if args.loss_print_samples < 1:
        raise ValueError('--loss-print-samples must be at least 1.')

    if args.patches_per_training_sample < 1:
        raise ValueError('--patches-per-training-sample must be at least 1.')

    if args.grid_spacing < 1:
        raise ValueError('--grid-spacing must be at least 1.')

    if args.branch_features < 1:
        raise ValueError('--branch-features must be at least 1.')

    if args.frozen_stages < 0 or args.frozen_stages > 5:
        raise ValueError('--frozen-stages must be between 0 and 5.')

    if args.network_name.endswith('_untrained') and args.frozen_stages != 0:
        raise ValueError('--frozen-stages must be 0 for untrained networks.')

    if args.network_name == 'small_cnn' and args.frozen_stages != 0:
        raise ValueError('--frozen-stages must be 0 for small_cnn.')

    if args.network_name == 'small_cnn' and args.small_input_stem:
        raise ValueError('--small-input-stem must be false for small_cnn.')

    if num_of_folds < 2:
        raise ValueError(f'At least 2 fold files are required. Found {num_of_folds}.')

    if args.fold < 1 or args.fold > num_of_folds:
        raise ValueError(f'fold must be between 1 and {num_of_folds}. Got fold={args.fold}.')

    if args.run_dir.exists() and not args.run_dir.is_dir():
        raise ValueError(f'--run-dir exists but is not a directory: {args.run_dir}')

    if args.save_dir is not None and args.save_dir.exists() and not args.save_dir.is_dir():
        raise ValueError(f'--save-dir exists but is not a directory: {args.save_dir}')

    if len(pms.sub_patch_scales) != 4:
        raise ValueError(f'Quadruplet requires exactly 4 sub-patch scales, got {len(pms.sub_patch_scales)}: {pms.sub_patch_scales}')

    if any(scale <= 0 for scale in pms.sub_patch_scales):
        raise ValueError(f'sub_patch_scales must contain positive integers. Got: {pms.sub_patch_scales}')

    if sorted(pms.sub_patch_scales) != list(pms.sub_patch_scales):
        raise ValueError(f'sub_patch_scales should be in ascending order because the first value is used as the output patch size. Got: {pms.sub_patch_scales}')

    validate_intervals(name='distance_intervals', intervals=pms.distance_intervals, expected_start=0)
    validate_intervals(name='angle_intervals', intervals=pms.angle_intervals, expected_start=0, expected_end=360)
    validate_sampling_variances(pms.sampling_variances)

    minimum_patches = args.num_points * len(pms.sampling_variances)

    if args.patches_per_training_sample < minimum_patches:
        raise ValueError(
            f'--patches-per-training-sample must be at least {minimum_patches} '
            f'for {args.num_points} points and {len(pms.sampling_variances)} sampling variances.'
        )

    if not args.mark_list_file.is_file():
        raise ValueError(f'--mark-list-file does not exist or is not a file: {args.mark_list_file}')

    if not args.image_data_dir.is_dir():
        raise ValueError(f'--image-data-dir does not exist or is not a directory: {args.image_data_dir}')


def validate_num_points(value):
    """Validate the number of landmark points for each image."""
    value = int(value)

    if value < MIN_POINTS_PER_IMAGE or value > MAX_POINTS_PER_IMAGE:
        raise argparse.ArgumentTypeError(f'num-points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}.')

    return value


def validate_task_name(value):
    """Validate and clean the task name used in output paths."""
    value = clean_run_name(value)

    if not value:
        raise argparse.ArgumentTypeError('task_name cannot be empty after cleaning.')

    if value in ('.', '..'):
        raise argparse.ArgumentTypeError('task_name cannot be "." or "..".')

    return value


def validate_sampling_variances(values):
    """Validate sampling variances used for training-centre generation."""
    if not values:
        raise ValueError('sampling_variances must contain at least one value.')

    for index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise ValueError(f'sampling_variances[{index}] must be numeric. Got: {value}')

        if value <= 0:
            raise ValueError(f'sampling_variances[{index}] must be greater than 0. Got: {value}')


def validate_intervals(name, intervals, expected_start=None, expected_end=None):
    """Validate class interval boundaries."""
    if not intervals:
        raise ValueError(f'{name} must contain at least one interval.')

    previous_upper = None

    for index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(f'{name}[{index}] must contain exactly two values: (lower_bound, upper_bound). Got: {interval}')

        lower_bound, upper_bound = interval

        if lower_bound >= upper_bound:
            raise ValueError(f'{name}[{index}] has lower_bound >= upper_bound. Got: {interval}')

        if previous_upper is not None and lower_bound != previous_upper:
            raise ValueError(
                f'{name} intervals must be contiguous with no gaps or overlaps. Previous upper bound was {previous_upper}, but interval {index} starts at {lower_bound}.')

        previous_upper = upper_bound

    if expected_start is not None and intervals[0][0] != expected_start:
        raise ValueError(f'{name} must start at {expected_start}. Got first interval: {intervals[0]}')

    if expected_end is not None and intervals[-1][1] != expected_end:
        raise ValueError(f'{name} must end at {expected_end}. Got final interval: {intervals[-1]}')


def infer_num_points_from_csv(csv_path, tasks_per_point):
    """Infer the point count from the first valid row in a generated patch CSV."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise ValueError(f'Generated data CSV does not exist: {csv_path}')

    with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            if not row:
                continue

            label_count = len(row) - BASE_CSV_COLUMNS

            if label_count < tasks_per_point:
                raise ValueError(f'{csv_path} has {label_count} label columns, expected at least {tasks_per_point}.')

            if label_count % tasks_per_point != 0:
                raise ValueError(f'{csv_path} has {label_count} label columns, which is incompatible with {tasks_per_point} tasks per point.')

            return label_count // tasks_per_point

    raise ValueError(f'{csv_path} is empty.')


def build_run_name(args, num_of_folds):
    """Build a deterministic folder name shared by every fold in the same experiment."""
    parts = [
        'ipv',
        f'{num_of_folds}fold',
        f'patch{format_scales(pms.sub_patch_scales)}',
        f'sv{format_scales(pms.sampling_variances)}',
        f'points{args.num_points}',
        f'ppts{args.patches_per_training_sample}',
        f'gs{args.grid_spacing}',
        args.network_name,
        f'bf{args.branch_features}',
        f'fs{args.frozen_stages}',
        f'stem{int(args.small_input_stem)}',
        f'bs{args.batch_size}',
        f'lr{format_number(args.learning_rate)}',
        f'ep{args.max_training_epochs}',
        f'sched{int(args.lr_schedule)}',
        f'lrs{args.lr_step_size}',
        f'lrg{format_number(args.lr_gamma)}',
        f'esp{args.early_stop_patience}',
        f'esd{format_number(args.early_stop_min_delta)}',
        f'esw{args.early_stop_warmup_epochs}',
        f'seed{args.random_seed}'
    ]

    return clean_run_name('_'.join(parts))


def clean_run_name(value):
    """Remove characters that are awkward in shared paths."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_')


def format_scales(values):
    """Format numeric lists for run folders."""
    return '-'.join(str(value) for value in values)


def format_number(value):
    """Format numeric values safely for file names."""
    return f'{value:g}'.replace('-', 'm').replace('.', 'p')


def build_configs(args):
    """Build run, data, training, and model configs."""
    fold_numbers = discover_fold_numbers(args.fold_lists_path)
    num_of_folds = len(fold_numbers)

    validate_args(args, num_of_folds)
    loss_print_interval = max(1, args.loss_print_samples // args.batch_size)
    run_name = clean_run_name(args.run_name) if args.run_name is not None else build_run_name(args, num_of_folds)

    if not run_name:
        raise ValueError('--run-name cannot be empty after cleaning.')

    run_config = RunConfig(
        fold=args.fold,
        task_name=args.task_name,
        num_of_points=args.num_points,
        create_data=args.create_data,
        train_model=args.train_model,
        copy_files=args.copy_files,
        delete_files=args.delete_files,
        run_dir=args.run_dir,
        save_dir=args.save_dir,
        run_name=run_name
    )

    data_config = DataCreationConfig(
        distance_intervals=pms.distance_intervals,
        angle_intervals=pms.angle_intervals,
        num_of_folds=num_of_folds,
        sub_patch_scales=pms.sub_patch_scales,
        patches_per_training_sample=args.patches_per_training_sample,
        grid_spacing=args.grid_spacing,
        fold_lists_path=args.fold_lists_path,
        mark_list_file=args.mark_list_file,
        image_data_dir=args.image_data_dir,
        sampling_variances=pms.sampling_variances,
        num_workers=args.data_creation_workers,
        random_seed=args.random_seed,
        keep_part_csvs=args.keep_part_csvs
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_training_epochs=args.max_training_epochs,
        loss_print_interval=loss_print_interval,
        num_workers=args.train_workers,
        lr_schedule=args.lr_schedule,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup_epochs=args.early_stop_warmup_epochs
    )

    quadruplet_config = QuadrupletConfig(
        network_name=args.network_name,
        branch_features=args.branch_features,
        frozen_stages=args.frozen_stages,
        small_input_stem=args.small_input_stem,
        num_sub_patches=len(pms.sub_patch_scales)
    )

    return run_config, data_config, train_config, quadruplet_config


def main():
    """Run the fold creation and training workflow."""
    args = parse_args()
    run_config, data_config, train_config, quadruplet_config = build_configs(args)
    create_train = CreateTrain(run_config, data_config, train_config, quadruplet_config)
    create_train.run()


if __name__ == '__main__':
    main()
