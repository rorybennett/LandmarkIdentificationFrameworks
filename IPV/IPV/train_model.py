import ast
import csv
import copy
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .custom_dataset import CustomDataset, ToTensor
from .model_registry import is_pretrained_model
from .normalisation import (ChannelStatistics, EXPECTED_NORMALISATION_CHANNELS, IMAGENET_RGB_MEAN,
                            IMAGENET_RGB_STD, validate_normalisation_constants)
from .runtime_metadata import collect_runtime_metadata, utc_now_iso
from .utils.landmark_inference_utils import (LandmarkInferenceConfig, accumulate_votes, detect_points, load_input_image,
                                             read_mark_list, run_validation_inference_for_trained_model)
from .utils.fold_utils import get_split_file_path, normalise_fold

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
CSV_METADATA_COLUMNS = 5
CHECKPOINT_FORMAT_VERSION = '0.1'
CHECKPOINT_SCHEMA_VERSION = '0.1'
CHECKPOINT_SCHEMA_NAME = 'ipv_checkpoint_metadata'
HISTORY_FIELDS = (
    'epoch', 'epoch_started_at', 'epoch_completed_at', 'lr', 'training_loss', 'training_accuracy',
    'validation_loss', 'validation_accuracy', 'validation_error_px', 'training_duration_seconds',
    'validation_duration_seconds', 'epoch_duration_seconds',
)


def seed_worker(_worker_id):
    """Seed NumPy and Python RNGs inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class QuadrupletConfig:
    network_name: str = 'resnet18_pretrained'
    branch_features: int = 128
    frozen_stages: int = 0
    small_input_stem: bool = True
    num_sub_patches: int = 4
    input_channels: int | None = None


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    max_training_epochs: int
    num_workers: int = 8
    random_seed: int = 42
    optimiser_name: str = 'adamw'
    weight_decay: float = 1e-4
    momentum: float = 0.9
    lr_schedule: str = 'plateau'
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4
    early_stop_warmup_epochs: int = 10
    use_amp: bool = False
    save_validation_results: bool = True
    validation_inference_batch_size: int = 2048
    validation_vote_smoothing_sigma: float = 7.0
    validation_use_probability_weights: bool = True
    validation_save_raw_vote_maps: bool = False
    normalise_inputs: bool = False


class TrainModel:
    def __init__(self, repetition, current_fold, num_of_points, data_save_path, tasks_classes, train_config, quadruplet_config,
                 output_save_path=None, device=None, fold_collection_sha256=None, resume_training=False):
        self.repetition = int(repetition)
        self.fold = normalise_fold(current_fold)
        self.num_of_points = int(num_of_points)
        self.train_path = Path(data_save_path)
        self.output_path = Path(output_save_path) if output_save_path is not None else self.train_path
        self.tasks_classes = tasks_classes
        self.train_config = train_config
        self.quadruplet_config = quadruplet_config
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fold_collection_sha256 = fold_collection_sha256
        self.resume_training = bool(resume_training)
        self.validate_num_of_points(self.num_of_points)
        self.validate_tasks_classes_structure(self.tasks_classes)
        self.tasks_per_point = len(self.tasks_classes)
        self.expected_label_count = self.num_of_points * self.tasks_per_point
        self.input_channels = None
        self.normalisation_mean = None
        self.normalisation_std = None
        self.normalisation_source = 'disabled'
        self.num_of_classes = [len(task_classes) for _ in range(self.num_of_points) for task_classes in self.tasks_classes]
        self.runtime_metadata = None
        self.training_status = 'initialising'
        self.termination_reason = None
        self.failure = None
        self.workflow_started_at = None
        self.workflow_completed_at = None
        self.workflow_start_perf = None
        self.workflow_duration_seconds = None
        self.dataset_validation_duration_seconds = 0.0
        self.model_setup_duration_seconds = 0.0
        self.validation_export_duration_seconds = 0.0
        self.resume_state_validated = False
        self.training_sessions = []
        self.current_session_index = None
        self.current_session_start_perf = None
        self.training_generator = None
        self.validation_generator = None
        self.history = self.empty_history()
        self.validate_configs()

    def train(self, on_dataset_validated=None, on_training_state_ready=None):
        """Run or explicitly resume deterministic training for one repetition and fold."""
        self.workflow_started_at = utc_now_iso()
        self.workflow_start_perf = time.perf_counter()
        self.set_random_seed(self.train_config.random_seed)
        self.runtime_metadata = collect_runtime_metadata(self.device, self.train_config.use_amp)

        try:
            self.output_path.mkdir(exist_ok=True, parents=True)
            validation_start = time.perf_counter()
            self.validate_training_inputs()
            train_loader, val_loader = self.build_data_loaders()
            self.input_channels = self.resolve_input_channels(train_loader.dataset, val_loader.dataset)
            self.configure_input_normalisation(train_loader.dataset, val_loader.dataset)
            self.dataset_validation_duration_seconds = time.perf_counter() - validation_start
            self.training_status = 'dataset_validated'

            if on_dataset_validated is not None:
                on_dataset_validated()

            setup_start = time.perf_counter()
            model = self.build_model(input_channels=self.input_channels)
            criterion = nn.CrossEntropyLoss()
            optimiser = self.build_optimiser(model)
            scheduler = self.build_scheduler(optimiser)
            scaler = torch.amp.GradScaler('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda')
            resume_signature = self.build_resume_signature(train_loader, val_loader)
            history = self.empty_history()
            best_epoch = None
            best_metrics = None
            last_epoch = 0
            last_metrics = None
            best_checkpoint_path = None
            last_checkpoint_path = None
            best_model_state_dict = None
            early_stop_best_validation_loss = float('inf')
            bad_epochs = 0
            start_epoch = 1
            saved_termination_reason = None

            if self.resume_training:
                state = self.load_training_checkpoint(model, optimiser, scheduler, scaler, resume_signature)
                history = state['history']
                best_epoch = state['best_epoch']
                best_metrics = state['best_validation_metrics']
                last_epoch = state['completed_epoch']
                last_metrics = state['last_validation_metrics']
                best_model_state_dict = state['best_model_state_dict']
                early_stop_best_validation_loss = state['early_stop_best_validation_loss']
                bad_epochs = state['bad_epochs']
                start_epoch = last_epoch + 1
                saved_termination_reason = state['termination_reason']
                best_checkpoint_path = self.get_checkpoint_path('best_validation_loss')
                last_checkpoint_path = self.get_checkpoint_path('last_epoch')
                self.ensure_best_checkpoint(best_checkpoint_path, best_epoch, best_metrics, best_model_state_dict)

            self.begin_training_session(resumed=self.resume_training, resumed_from_epoch=last_epoch)

            self.model_setup_duration_seconds = time.perf_counter() - setup_start
            self.training_status = 'running'
            self.history = history
            self.write_history_log(history)
            self.save_history_plot(history)

            if on_training_state_ready is not None:
                on_training_state_ready()

            print('\tData loaded...', flush=True)
            print(f'\tNetwork loaded on {self.device}. Training network...', flush=True)

            if self.resume_training:
                print(f'\tResuming after completed epoch {last_epoch}.', flush=True)

            epochs = (() if saved_termination_reason in ('early_stopping', 'max_epochs_reached')
                      else range(start_epoch, self.train_config.max_training_epochs + 1))

            for epoch in epochs:
                epoch_started_at = utc_now_iso()
                epoch_start = time.perf_counter()
                epoch_lr = self.get_current_lr(optimiser)
                print(f"\t{dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Epoch {epoch}/{self.train_config.max_training_epochs}", flush=True)

                self.synchronise_device()
                training_start = time.perf_counter()
                training_metrics = self.train_epoch(model, train_loader, criterion, optimiser, scaler)
                self.synchronise_device()
                training_duration = time.perf_counter() - training_start

                validation_start = time.perf_counter()
                validation_metrics = self.validate(model, val_loader, criterion)
                self.synchronise_device()
                validation_duration = time.perf_counter() - validation_start
                self.validate_finite_metrics('training', training_metrics)
                self.validate_finite_metrics('validation', validation_metrics)

                if scheduler is not None:
                    scheduler.step(validation_metrics['loss']) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

                is_new_best = best_metrics is None or validation_metrics['loss'] < best_metrics['loss']
                is_early_stop_improvement = validation_metrics['loss'] < early_stop_best_validation_loss - self.train_config.early_stop_min_delta

                if is_new_best:
                    best_epoch = epoch
                    best_metrics = dict(validation_metrics)
                    best_model_state_dict = self.clone_state_dict_to_cpu(model.state_dict())

                if is_early_stop_improvement:
                    early_stop_best_validation_loss = validation_metrics['loss']

                if epoch >= self.train_config.early_stop_warmup_epochs:
                    bad_epochs = 0 if is_early_stop_improvement else bad_epochs + 1

                should_stop = epoch >= self.train_config.early_stop_warmup_epochs and bad_epochs >= self.train_config.early_stop_patience
                epoch_reason = 'early_stopping' if should_stop else ('max_epochs_reached' if epoch >= self.train_config.max_training_epochs else 'in_progress')
                last_epoch = epoch
                last_metrics = dict(validation_metrics)
                epoch_completed_at = utc_now_iso()
                epoch_duration = time.perf_counter() - epoch_start
                self.update_history(history, epoch, epoch_started_at, epoch_completed_at, epoch_lr, training_metrics,
                                    validation_metrics, training_duration, validation_duration, epoch_duration)
                self.history = history
                self.update_current_training_session(status=epoch_reason, completed_epoch=last_epoch)
                training_state = self.build_training_state(last_epoch, history, best_epoch, best_metrics,
                                                           early_stop_best_validation_loss, bad_epochs, last_metrics, epoch_reason)

                if is_new_best:
                    best_checkpoint_path = self.save_checkpoint(model, optimiser, scheduler, scaler, 'best_validation_loss',
                                                                epoch, validation_metrics, None, resume_signature, None)

                last_checkpoint_path = self.save_checkpoint(model, optimiser, scheduler, scaler, 'last_epoch', epoch,
                                                            validation_metrics, training_state, resume_signature,
                                                            best_model_state_dict)
                self.write_history_log(history)
                self.save_history_plot(history)

                if is_new_best:
                    print(f"\tNew best model saved from epoch {epoch} with validation_loss={validation_metrics['loss']:.6f} "
                          f"and validation_error={validation_metrics['error_px']:.2f}px", flush=True)

                if should_stop:
                    print(f'\tEarly stop: validation loss stopped improving by at least {self.train_config.early_stop_min_delta:g}.', flush=True)
                    break

            self.termination_reason = (saved_termination_reason if saved_termination_reason in ('early_stopping', 'max_epochs_reached') else
                                       ('early_stopping' if bad_epochs >= self.train_config.early_stop_patience
                                        and last_epoch >= self.train_config.early_stop_warmup_epochs else 'max_epochs_reached'))
            validation_results_path = None

            if self.train_config.save_validation_results:
                export_start = time.perf_counter()
                validation_results_path = self.run_validation_inference(model, best_checkpoint_path, last_checkpoint_path)
                self.validation_export_duration_seconds = time.perf_counter() - export_start

            self.training_status = 'completed'
            self.finish_workflow()
            self.write_checkpoint_summary(best_epoch, last_epoch, best_metrics, last_metrics, best_checkpoint_path,
                                          last_checkpoint_path, validation_results_path)
            plt.clf()
            return best_checkpoint_path or last_checkpoint_path
        except BaseException as error:
            self.training_status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            self.termination_reason = 'keyboard_interrupt' if isinstance(error, KeyboardInterrupt) else 'exception'
            self.failure = {'type': type(error).__name__, 'message': str(error)}
            self.finish_workflow()
            plt.clf()
            raise

    def run_validation_inference(self, model, best_checkpoint_path=None, last_checkpoint_path=None):
        """Run full-image inference on validation images and save overlays and Excel metrics."""
        checkpoint_path = best_checkpoint_path or last_checkpoint_path
        checkpoint_type = 'best_validation_loss' if best_checkpoint_path is not None else 'last_epoch'
        loaded_checkpoint = None

        if checkpoint_path is not None:
            loaded_checkpoint = self.load_checkpoint_state(model=model, checkpoint_path=checkpoint_path)

        data_metadata = self.read_data_creation_metadata()
        validation_output_path = self.get_validation_staging_path()
        self.prepare_validation_staging_path()
        config = LandmarkInferenceConfig(
            repetition=int(self.repetition),
            fold=self.fold,
            task_name=str(data_metadata.get('task_name') or ''),
            data_save_path=self.train_path,
            output_dir=validation_output_path,
            mark_list_file=Path(self.require_metadata_value(data_metadata, 'mark_list_file')),
            image_data_dir=Path(self.require_metadata_value(data_metadata, 'image_data_dir')),
            num_points=int(self.num_of_points),
            sub_patch_scales=self.require_metadata_value(data_metadata, 'sub_patch_scales'),
            distance_intervals=self.tasks_classes[0],
            angle_intervals=self.tasks_classes[1],
            grid_spacing=int(self.require_metadata_value(data_metadata, 'grid_spacing')),
            input_channels=int(self.input_channels),
            batch_size=int(self.train_config.validation_inference_batch_size),
            smoothing_sigma=float(self.train_config.validation_vote_smoothing_sigma),
            use_probability_weights=bool(self.train_config.validation_use_probability_weights),
            save_raw_vote_maps=bool(self.train_config.validation_save_raw_vote_maps),
            checkpoint_path=checkpoint_path,
            checkpoint_type=checkpoint_type,
            network_name=self.quadruplet_config.network_name,
            normalisation_mean=self.normalisation_mean,
            normalisation_std=self.normalisation_std,
            checkpoint_metadata=None if loaded_checkpoint is None else loaded_checkpoint.get('metadata')
        )

        print(f'	Running validation image inference with {checkpoint_type} checkpoint...', flush=True)
        run_validation_inference_for_trained_model(model=model, config=config, device=self.device)
        self.commit_validation_output()
        print(f'	Validation image inference outputs saved to {self.get_validation_output_path()}', flush=True)
        return self.get_validation_output_path()

    @staticmethod
    def load_checkpoint_state(model, checkpoint_path):
        """Load checkpoint weights into an existing model."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None

        if state_dict is None:
            raise ValueError(f'Checkpoint does not contain a state_dict: {checkpoint_path}')

        model.load_state_dict(state_dict)
        model.eval()
        return checkpoint

    def validate_training_inputs(self):
        """Validate generated fold data before model construction."""
        train_csv_path = self.get_train_csv_path()
        val_csv_path = self.get_val_csv_path()

        self.validate_csv_exists(train_csv_path)
        self.validate_csv_exists(val_csv_path)
        self.validate_metadata_point_count()

        train_points = self.validate_csv_point_count(csv_path=train_csv_path, phase='Train')
        val_points = self.validate_csv_point_count(csv_path=val_csv_path, phase='Val')

        if train_points != val_points:
            raise ValueError(f'Train data has {train_points} points, but validation data has {val_points} points.')

        if train_points != self.num_of_points:
            raise ValueError(f'Model requested {self.num_of_points} points, but generated data contains {train_points} points.')

        print(f'\tTraining data validated: {self.num_of_points} points, {self.tasks_per_point} tasks per point, {self.expected_label_count} label columns.', flush=True)

    def validate_metadata_point_count(self):
        """Validate data-creation metadata when run_info JSON files are available."""
        metadata_paths = [self.train_path / 'run_info.json'] if (self.train_path / 'run_info.json').is_file() else []

        if not metadata_paths:
            print(f'\tNo run_info metadata found in {self.train_path}; CSV label-count validation will be used.', flush=True)
            return

        metadata_point_counts = []

        for metadata_path in metadata_paths:
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                metadata = json.load(metadata_file)

            if 'num_of_points' not in metadata:
                raise ValueError(f'Metadata file {metadata_path} does not contain num_of_points.')

            created_points = int(metadata['num_of_points'])
            metadata_point_counts.append(created_points)

            if created_points != self.num_of_points:
                raise ValueError(f'Model requested {self.num_of_points} points, but {metadata_path} says the data was created with {created_points} points.')

        if len(set(metadata_point_counts)) != 1:
            raise ValueError(f'Conflicting num_of_points values found in metadata files: {metadata_point_counts}')

    def validate_csv_point_count(self, csv_path, phase):
        """Validate label columns in a generated patch CSV and return its point count."""
        detected_label_count = None
        row_count = 0

        with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)

            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                row_count += 1

                if len(row) <= CSV_METADATA_COLUMNS:
                    raise ValueError(f'{phase} CSV row {row_number} in {csv_path} has {len(row)} columns; expected metadata columns plus labels.')

                label_count = len(row) - CSV_METADATA_COLUMNS

                if detected_label_count is None:
                    detected_label_count = label_count
                elif label_count != detected_label_count:
                    raise ValueError(f'{phase} CSV row {row_number} in {csv_path} has {label_count} label columns; expected {detected_label_count}.')

                if label_count % self.tasks_per_point != 0:
                    raise ValueError(
                        f'{phase} CSV row {row_number} in {csv_path} has {label_count} label columns, which is not divisible by {self.tasks_per_point} tasks per point.')

                if label_count != self.expected_label_count:
                    detected_points = label_count // self.tasks_per_point
                    raise ValueError(
                        f'{phase} CSV row {row_number} in {csv_path} has {detected_points} points and {label_count} labels; model expects {self.num_of_points} points and {self.expected_label_count} labels.')

        if row_count == 0 or detected_label_count is None:
            raise ValueError(f'{phase} CSV is empty: {csv_path}')

        return detected_label_count // self.tasks_per_point

    @staticmethod
    def validate_csv_exists(csv_path):
        """Validate that a generated fold CSV exists."""
        if not csv_path.is_file():
            raise ValueError(f'Generated CSV file does not exist: {csv_path}')

    @staticmethod
    def validate_num_of_points(num_of_points):
        """Validate configured landmark count."""
        if num_of_points < MIN_POINTS_PER_IMAGE or num_of_points > MAX_POINTS_PER_IMAGE:
            raise ValueError(f'num_of_points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}. Got: {num_of_points}')

    @staticmethod
    def validate_tasks_classes_structure(tasks_classes):
        """Validate task class definitions used to build output heads."""
        if not tasks_classes:
            raise ValueError('tasks_classes must contain at least one task.')

        for task_index, task_classes in enumerate(tasks_classes):
            if not task_classes:
                raise ValueError(f'tasks_classes[{task_index}] must contain at least one class interval.')

    def build_data_loaders(self):
        """Create train and validation data loaders."""
        train_csv_path = self.get_train_csv_path()
        val_csv_path = self.get_val_csv_path()

        train_dataset = CustomDataset(train_csv_path, num_sub_patches=self.quadruplet_config.num_sub_patches)
        val_dataset = CustomDataset(val_csv_path, num_sub_patches=self.quadruplet_config.num_sub_patches)
        self.training_generator = torch.Generator()
        self.validation_generator = torch.Generator()
        self.training_generator.manual_seed(int(self.train_config.random_seed))
        self.validation_generator.manual_seed(int(self.train_config.random_seed))

        train_loader = DataLoader(train_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers,
                                  pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=self.training_generator)
        val_loader = DataLoader(val_dataset, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.num_workers,
                                pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=self.validation_generator)

        return train_loader, val_loader

    def configure_input_normalisation(self, train_dataset, validation_dataset):
        """Resolve constants from pretrained weights or the training split and attach tensor transforms."""
        if not self.train_config.normalise_inputs:
            train_dataset.transform = ToTensor()
            validation_dataset.transform = ToTensor()
            self.normalisation_mean = None
            self.normalisation_std = None
            self.normalisation_source = 'disabled'
            print('\tInput normalisation disabled.', flush=True)
            return

        if int(self.input_channels) != EXPECTED_NORMALISATION_CHANNELS:
            raise ValueError(
                f'Input normalisation requires exactly {EXPECTED_NORMALISATION_CHANNELS} channels so each RGB channel remains distinct; '
                f'the training data contains {self.input_channels} channel(s).'
            )

        if is_pretrained_model(self.quadruplet_config.network_name):
            mean, standard_deviation = IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
            source = 'torchvision_imagenet_pretrained_weights'
        else:
            statistics = ChannelStatistics()

            for sample_index in range(len(train_dataset)):
                statistics.update(train_dataset[sample_index]['image'])

            mean, standard_deviation = statistics.finalise()
            source = 'training_split_patches'

        mean, standard_deviation = validate_normalisation_constants(mean, standard_deviation)
        self.normalisation_mean = list(mean)
        self.normalisation_std = list(standard_deviation)
        self.normalisation_source = source
        train_dataset.transform = ToTensor(self.normalisation_mean, self.normalisation_std)
        validation_dataset.transform = ToTensor(self.normalisation_mean, self.normalisation_std)
        print(f'\tInput normalisation enabled from {source}: mean={self.normalisation_mean}, std={self.normalisation_std}.', flush=True)

    def resolve_input_channels(self, train_dataset, val_dataset):
        """Resolve the input channel count used by the model."""
        train_channels = int(train_dataset.input_channels)
        val_channels = int(val_dataset.input_channels)

        if train_channels != val_channels:
            raise ValueError(f'Train patches have {train_channels} channels, but validation patches have {val_channels} channels.')

        configured_channels = self.quadruplet_config.input_channels

        if configured_channels is not None and int(configured_channels) != train_channels:
            raise ValueError(f'QuadrupletConfig requested {configured_channels} input channels, but generated patches contain {train_channels}.')

        print(f'	Detected {train_channels} input channel(s) per patch.', flush=True)

        return train_channels

    def build_model(self, input_channels):
        """Create the Quadruplet model."""
        from .quadruplet import Quadruplet

        model = Quadruplet(
            num_of_points=self.num_of_points,
            tasks_classes=self.tasks_classes,
            network_name=self.quadruplet_config.network_name,
            branch_features=self.quadruplet_config.branch_features,
            frozen_stages=self.quadruplet_config.frozen_stages,
            small_input_stem=self.quadruplet_config.small_input_stem,
            input_channels=input_channels
        )

        return model.to(self.device)

    def build_optimiser(self, model):
        """Build the configured optimiser."""
        name = str(self.train_config.optimiser_name).lower()

        if name == 'adamw':
            return AdamW(model.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)
        if name == 'sgd':
            return SGD(model.parameters(), lr=self.train_config.learning_rate, momentum=self.train_config.momentum,
                       weight_decay=self.train_config.weight_decay)

        raise ValueError(f'Unknown optimiser_name: {self.train_config.optimiser_name}')

    def build_scheduler(self, optimiser):
        """Build the configured epoch-level scheduler."""
        schedule = str(self.train_config.lr_schedule).lower()

        if schedule == 'none':
            return None
        if schedule == 'step':
            return StepLR(optimiser, step_size=self.train_config.lr_step_size, gamma=self.train_config.lr_gamma)
        if schedule == 'plateau':
            return ReduceLROnPlateau(optimiser, mode='min', factor=self.train_config.lr_gamma, patience=5)

        raise ValueError(f'Unknown lr_schedule: {self.train_config.lr_schedule}')

    def train_epoch(self, model, train_loader, criterion, optimiser, scaler):
        """Train one complete epoch and return sample-weighted classification metrics."""
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_predictions = 0

        for data in train_loader:
            images = data['image'].to(self.device, non_blocking=True)
            labels = data['labels'].to(self.device, non_blocking=True).long()
            batch_size = labels.shape[0]
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda'):
                outputs = model(images)
                train_loss = self.calculate_loss(outputs, labels, criterion)

            scaler.scale(train_loss).backward()
            scaler.step(optimiser)
            scaler.update()

            batch_correct = self.count_correct(outputs, labels)
            batch_predictions = batch_size * len(outputs)
            epoch_loss += train_loss.item() * batch_size
            epoch_correct += batch_correct
            epoch_predictions += batch_predictions

        return {'loss': epoch_loss / max(len(train_loader.dataset), 1),
                'accuracy': epoch_correct / max(epoch_predictions, 1)}

    def validate(self, model, val_loader, criterion):
        """Evaluate classification metrics and reconstruct validation endpoints by voting."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_predictions = 0
        vote_records = {}

        with torch.inference_mode():
            for data in val_loader:
                images = data['image'].to(self.device, non_blocking=True)
                labels = data['labels'].to(self.device, non_blocking=True).long()
                batch_size = labels.shape[0]

                with torch.amp.autocast('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda'):
                    outputs = model(images)
                    loss = self.calculate_loss(outputs, labels, criterion)

                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_correct += self.count_correct(outputs, labels)
                total_predictions += batch_size * len(outputs)
                probabilities = [torch.softmax(output, dim=1) for output in outputs]
                predictions = [torch.argmax(probability, dim=1).detach().cpu().numpy() for probability in probabilities]
                confidences = [torch.max(probability, dim=1).values.detach().cpu().numpy() for probability in probabilities]
                coordinates = data['coordinates'].detach().cpu().numpy()

                for batch_index, sample_name in enumerate(data['sample_name']):
                    sample_name = str(sample_name)
                    record = vote_records.setdefault(sample_name, {
                        'centres': [],
                        'vote_inputs': [{'distance_classes': [], 'angle_classes': [], 'scores': []}
                                        for _ in range(self.num_of_points)],
                    })
                    record['centres'].append((int(coordinates[batch_index][0]), int(coordinates[batch_index][1])))

                    for point_index in range(self.num_of_points):
                        distance_index = point_index * 2
                        angle_index = distance_index + 1
                        score = (float(confidences[distance_index][batch_index] * confidences[angle_index][batch_index])
                                 if self.train_config.validation_use_probability_weights else 1.0)
                        point_input = record['vote_inputs'][point_index]
                        point_input['distance_classes'].append(int(predictions[distance_index][batch_index]))
                        point_input['angle_classes'].append(int(predictions[angle_index][batch_index]))
                        point_input['scores'].append(score)

        validation_error = self.calculate_validation_endpoint_error(vote_records)
        return {'loss': total_loss / max(total_samples, 1),
                'accuracy': total_correct / max(total_predictions, 1),
                'error_px': validation_error}

    def calculate_validation_endpoint_error(self, vote_records):
        """Convert one validation pass into a mean original-image endpoint error."""
        metadata = self.read_data_creation_metadata()
        mark_list_path = Path(self.require_metadata_value(metadata, 'mark_list_file'))
        image_data_dir = Path(self.require_metadata_value(metadata, 'image_data_dir'))
        mark_records = read_mark_list(mark_list_path, expected_points=self.num_of_points,
                                      selected_sample_names=vote_records.keys())
        errors = []

        for sample_name, record in vote_records.items():
            if sample_name not in mark_records:
                raise ValueError(f'Validation sample {sample_name} is missing from {mark_list_path}.')

            image_name, target_points = mark_records[sample_name]
            image_path = image_data_dir / image_name
            image = load_input_image(image_path, input_channels=self.input_channels)
            vote_inputs = []

            for point_input in record['vote_inputs']:
                vote_inputs.append({name: np.asarray(values) for name, values in point_input.items()})

            vote_maps, _, _ = accumulate_votes(centres=record['centres'], vote_inputs=vote_inputs,
                                                image_shape=image.shape[:2], distance_intervals=self.tasks_classes[0],
                                                angle_intervals=self.tasks_classes[1], num_points=self.num_of_points)
            predicted_points, _, _ = detect_points(vote_maps, self.train_config.validation_vote_smoothing_sigma)
            errors.extend(np.linalg.norm(np.asarray(predicted_points, dtype=np.float32) -
                                         np.asarray(target_points, dtype=np.float32), axis=1).tolist())

        if not errors:
            raise ValueError('Validation endpoint voting produced no endpoint errors.')

        return float(np.mean(errors))

    def calculate_loss(self, outputs, labels, criterion):
        """Calculate average loss across all output heads."""
        loss = 0.0

        for output_index, output in enumerate(outputs):
            loss += criterion(output, labels[:, output_index])

        return loss / len(outputs)

    def count_correct(self, outputs, labels):
        """Count correct predictions across all output heads."""
        correct = 0

        for output_index, output in enumerate(outputs):
            predictions = torch.argmax(output, dim=1)
            correct += torch.eq(predictions, labels[:, output_index]).sum().item()

        return correct

    @staticmethod
    def update_history(history, epoch, epoch_started_at, epoch_completed_at, epoch_lr, training_metrics, validation_metrics,
                       training_duration_seconds, validation_duration_seconds, epoch_duration_seconds):
        """Append one complete epoch to the training history."""
        values = {
            'epoch': int(epoch), 'epoch_started_at': epoch_started_at, 'epoch_completed_at': epoch_completed_at,
            'lr': float(epoch_lr), 'training_loss': float(training_metrics['loss']),
            'training_accuracy': float(training_metrics['accuracy']), 'validation_loss': float(validation_metrics['loss']),
            'validation_accuracy': float(validation_metrics['accuracy']), 'validation_error_px': float(validation_metrics['error_px']),
            'training_duration_seconds': float(training_duration_seconds),
            'validation_duration_seconds': float(validation_duration_seconds), 'epoch_duration_seconds': float(epoch_duration_seconds),
        }

        for field_name in HISTORY_FIELDS:
            history[field_name].append(values[field_name])

    def write_history_log(self, history):
        """Atomically rebuild the epoch-level CSV."""
        log_path = self.get_log_path()
        temporary_path = log_path.with_name(f'.{log_path.name}.tmp')

        try:
            with open(temporary_path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=list(HISTORY_FIELDS))
                writer.writeheader()
                for row_index in range(len(history['epoch'])):
                    writer.writerow({field_name: history[field_name][row_index] for field_name in HISTORY_FIELDS})
            os.replace(temporary_path, log_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def save_checkpoint(self, model, optimiser, scheduler, scaler, checkpoint_type, epoch, validation_metrics,
                        training_state, resume_signature, best_model_state_dict):
        """Atomically save an inference checkpoint and complete continuation state when applicable."""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type)
        created_at = utc_now_iso()
        labelled_metrics = self.label_validation_metrics(validation_metrics)
        checkpoint = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'schema': 'ipv_training_checkpoint',
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': created_at,
            'checkpoint_type': checkpoint_type,
            'epoch': int(epoch),
            'next_epoch': int(epoch) + 1,
            'resume_capable': checkpoint_type == 'last_epoch',
            'state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'validation_metrics': labelled_metrics,
            'metadata': self.build_checkpoint_metadata(checkpoint_type=checkpoint_type, epoch=epoch,
                                                       metrics=validation_metrics, created_at=created_at),
        }

        if checkpoint_type == 'last_epoch':
            if training_state is None or best_model_state_dict is None:
                raise ValueError('A last-epoch checkpoint requires complete training state and the best-model snapshot.')

            checkpoint.update({
                'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                'grad_scaler_state_dict': scaler.state_dict(),
                'training_state': training_state,
                'rng_state': self.capture_rng_state(),
                'data_loader_generator_states': self.capture_data_loader_generator_states(),
                'best_model_state_dict': best_model_state_dict,
                'resume_signature': resume_signature,
            })

        self.atomic_torch_save(checkpoint, checkpoint_path)
        return checkpoint_path

    def save_history_plot(self, history):
        """Save epoch-level loss, accuracy, and endpoint-error traces."""
        if not history['epoch']:
            return

        figure, (loss_axis, metric_axis) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        error_axis = metric_axis.twinx()
        loss_axis.plot(history['epoch'], history['training_loss'], label='training_loss')
        loss_axis.plot(history['epoch'], history['validation_loss'], label='validation_loss')
        metric_axis.plot(history['epoch'], history['training_accuracy'], label='training_accuracy')
        metric_axis.plot(history['epoch'], history['validation_accuracy'], label='validation_accuracy')
        error_axis.plot(history['epoch'], history['validation_error_px'], linestyle='--', color='tab:red', label='validation_error_px')
        loss_axis.set_ylabel('Classification loss')
        metric_axis.set_xlabel('Epoch')
        metric_axis.set_ylabel('Classification accuracy')
        error_axis.set_ylabel('Mean endpoint error (px)')
        loss_axis.legend(loc='best')
        metric_lines, metric_labels = metric_axis.get_legend_handles_labels()
        error_lines, error_labels = error_axis.get_legend_handles_labels()
        metric_axis.legend(metric_lines + error_lines, metric_labels + error_labels, loc='best')
        figure.tight_layout()
        figure.savefig(self.get_plot_path())
        plt.close(figure)

    def build_checkpoint_metadata(self, checkpoint_type=None, epoch=None, metrics=None, created_at=None):
        """Build the single metadata structure saved in every checkpoint."""
        data_metadata = self.read_data_creation_metadata()

        return {
            'schema': CHECKPOINT_SCHEMA_NAME,
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': created_at or utc_now_iso(),
            'checkpoint': {
                'format_version': CHECKPOINT_FORMAT_VERSION,
                'type': checkpoint_type,
                'epoch': epoch,
                'validation_metrics': self.label_validation_metrics(metrics)
            },
            'task': self.build_task_metadata(data_metadata),
            'model': self.build_model_metadata(),
            'data': self.build_data_metadata(data_metadata),
            'preprocessing': self.build_preprocessing_metadata(data_metadata),
            'inference': self.build_inference_metadata(data_metadata),
            'training': self.build_training_metadata(),
            'runtime_environment': self.runtime_metadata,
        }

    def build_task_metadata(self, data_metadata):
        """Build task and output-head metadata without repeating interval definitions per head."""
        task_names = self.get_task_names()
        output_heads = []

        for point_index in range(1, self.num_of_points + 1):
            for task_name in task_names:
                output_heads.append({'head_index': len(output_heads), 'point_index': point_index, 'task': task_name})

        return {
            'name': data_metadata.get('task_name'),
            'num_points': int(self.num_of_points),
            'task_names': task_names,
            'output_heads': output_heads,
            'num_output_heads': int(self.expected_label_count),
            'num_classes_per_head': [int(value) for value in self.num_of_classes]
        }

    def build_model_metadata(self):
        """Build constructor metadata for the Quadruplet model."""
        return {
            'module': 'IPV.quadruplet',
            'class_name': 'Quadruplet',
            'init_args': self.build_model_init_args()
        }

    def build_model_init_args(self):
        """Return the exact arguments needed to rebuild the model."""
        return {
            'num_of_points': int(self.num_of_points),
            'tasks_classes': self.serialise_tasks_classes(self.tasks_classes),
            'network_name': self.quadruplet_config.network_name,
            'branch_features': int(self.quadruplet_config.branch_features),
            'frozen_stages': int(self.quadruplet_config.frozen_stages),
            'small_input_stem': bool(self.quadruplet_config.small_input_stem),
            'input_channels': int(self.input_channels)
        }

    def build_data_metadata(self, data_metadata):
        """Build compact data-source metadata."""
        return {
            'repetition': int(self.repetition),
            'fold': self.fold,
            'fold_collection_sha256': self.fold_collection_sha256,
            'data_save_path': str(self.train_path),
            'output_save_path': str(self.output_path),
            'fold_lists_path': self.path_metadata_to_string(data_metadata.get('fold_lists_path')),
            'mark_list_file': self.path_metadata_to_string(data_metadata.get('mark_list_file')),
            'image_data_dir': self.path_metadata_to_string(data_metadata.get('image_data_dir')),
            'patches_per_training_sample': self.optional_int(data_metadata.get('patches_per_training_sample')),
            'sampling_variances': self.optional_number_list(data_metadata.get('sampling_variances')),
            'random_seed': self.optional_int(data_metadata.get('random_seed'))
        }

    def build_preprocessing_metadata(self, data_metadata):
        """Build image and patch preprocessing metadata used during training and inference."""
        sub_patch_scales = self.require_metadata_value(data_metadata, 'sub_patch_scales')
        patch_size = int(data_metadata.get('patch_size', sub_patch_scales[0]))

        return {
            'sub_patch_scales': [int(scale) for scale in sub_patch_scales],
            'patch_size': int(patch_size),
            'num_sub_patches': int(self.quadruplet_config.num_sub_patches),
            'input_channels': int(self.input_channels),
            'tensor_shape': '[batch, num_sub_patches, channels, patch_size, patch_size]',
            'channel_order': 'channels_first',
            'loaded_image_value_range': 'float32_0_to_1',
            'model_input_values': ('three_channel_standardised' if self.train_config.normalise_inputs else 'float32_0_to_1'),
            'normalisation': self.build_normalisation_metadata(),
            'patch_resize': {
                'library': 'skimage.transform.resize',
                'preserve_range': True,
                'anti_aliasing': True
            }
        }

    def build_normalisation_metadata(self):
        """Return the exact three-channel input normalisation contract."""
        return {
            'enabled': bool(self.train_config.normalise_inputs),
            'channels': EXPECTED_NORMALISATION_CHANNELS,
            'mean': None if self.normalisation_mean is None else list(self.normalisation_mean),
            'standard_deviation': None if self.normalisation_std is None else list(self.normalisation_std),
            'source': self.normalisation_source,
            'statistic': 'population',
            'calculated_from': ('pretrained_weight_recipe' if is_pretrained_model(self.quadruplet_config.network_name)
                                and self.train_config.normalise_inputs else
                                ('training_split_only' if self.train_config.normalise_inputs else None)),
        }

    def build_inference_metadata(self, data_metadata=None):
        """Build inference-specific metadata."""
        data_metadata = data_metadata or self.read_data_creation_metadata()
        grid_spacing = int(self.require_metadata_value(data_metadata, 'grid_spacing'))
        smoothing_sigma = float(self.train_config.validation_vote_smoothing_sigma)
        batch_size = int(self.train_config.validation_inference_batch_size)

        return {
            'grid_spacing': grid_spacing,
            'centre_grid': {
                'x_start': 0,
                'y_start': 0,
                'x_step': grid_spacing,
                'y_step': grid_spacing,
                'loop_order': 'x_outer_y_inner'
            },
            'vote_accumulation': {
                'class_prediction': 'top_1_softmax_class',
                'use_probability_weights': bool(self.train_config.validation_use_probability_weights),
                'smoothing_sigma': smoothing_sigma,
                'batch_size': batch_size
            }
        }

    def build_training_metadata(self):
        """Build training configuration metadata."""
        return {
            'train_config': asdict(self.train_config),
            'quadruplet_config': asdict(self.quadruplet_config)
        }

    def read_data_creation_metadata(self):
        """Read data-creation metadata from generated fold metadata files."""
        data_info_path = self.train_path / 'data_info.csv'

        if data_info_path.is_file():
            return self.read_data_info_csv(data_info_path)

        run_info_metadata = self.read_run_info_metadata()

        if run_info_metadata:
            return run_info_metadata

        raise ValueError(
            f'Cannot save inference metadata because no data metadata was found for fold {self.fold}. '
            f'Expected {data_info_path} or run_info.json in {self.train_path}.'
        )

    def read_data_info_csv(self, data_info_path):
        """Read compact data_info CSV metadata."""
        with open(data_info_path, 'r', newline='', encoding='utf-8') as data_info_file:
            reader = csv.reader(data_info_file)
            rows = list(reader)

        if len(rows) < 2:
            raise ValueError(f'Data metadata file is incomplete: {data_info_path}')

        raw_metadata = dict(zip(rows[0], rows[1]))

        return {
            'task_name': raw_metadata.get('TASK_NAME'),
            'num_of_points': int(raw_metadata.get('NUM_OF_POINTS')),
            'sub_patch_scales': self.parse_metadata_value(raw_metadata.get('SUB_PATCH_SCALES')),
            'patch_size': int(raw_metadata.get('PATCH_SIZE')),
            'patches_per_training_sample': int(raw_metadata.get('PATCHES_PER_TRAINING_SAMPLE')),
            'grid_spacing': int(raw_metadata.get('GRID_DATA_STEP')),
            'sampling_variances': self.parse_metadata_value(raw_metadata.get('SAMPLING_VARIANCES')),
            'random_seed': int(raw_metadata.get('RANDOM_SEED')),
            'mark_list_file': raw_metadata.get('MARK_LIST_FILE'),
            'image_data_dir': raw_metadata.get('IMAGE_DATA_DIR'),
            'fold_lists_path': raw_metadata.get('FOLD_LISTS_PATH')
        }

    def read_run_info_metadata(self):
        """Read full run_info JSON metadata when compact data_info CSV is unavailable."""
        metadata_paths = [self.train_path / 'run_info.json'] if (self.train_path / 'run_info.json').is_file() else []

        if not metadata_paths:
            return None

        with open(metadata_paths[0], 'r', encoding='utf-8') as metadata_file:
            run_info = json.load(metadata_file)

        data_config = run_info.get('data_config', {})

        if not data_config:
            return None

        sub_patch_scales = data_config.get('sub_patch_scales') or []

        return {
            'task_name': run_info.get('task_name'),
            'num_of_points': run_info.get('num_of_points'),
            'sub_patch_scales': sub_patch_scales,
            'patch_size': sub_patch_scales[0] if sub_patch_scales else None,
            'patches_per_training_sample': data_config.get('patches_per_training_sample'),
            'grid_spacing': data_config.get('val_grid_spacing', data_config.get('grid_spacing')),
            'sampling_variances': data_config.get('sampling_variances'),
            'random_seed': data_config.get('random_seed'),
            'mark_list_file': data_config.get('mark_list_file'),
            'image_data_dir': data_config.get('image_data_dir'),
            'fold_lists_path': data_config.get('fold_lists_path')
        }

    @staticmethod
    def parse_metadata_value(value):
        """Parse a value written into compact CSV metadata."""
        if value is None or not isinstance(value, str):
            return value

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    @staticmethod
    def require_metadata_value(metadata, key):
        """Return required metadata or raise a clear error."""
        value = metadata.get(key)

        if value is None:
            raise ValueError(f'Cannot save checkpoint metadata because {key} is missing from data metadata.')

        return value

    def write_checkpoint_summary(self, best_epoch, last_epoch, best_metrics, last_metrics, best_checkpoint_path,
                                 last_checkpoint_path, validation_results_path=None):
        """Write a compact run-level checkpoint summary."""
        summary_path = self.get_checkpoint_summary_path()
        metadata = self.build_checkpoint_metadata(checkpoint_type=None, epoch=None, metrics={})
        summary = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'schema': 'ipv_validation_checkpoint_summary',
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': utc_now_iso(),
            'run_name': self.get_run_name(),
            'repetition': int(self.repetition),
            'fold': self.fold,
            'training_status': self.training_status,
            'termination_reason': self.termination_reason,
            'task': metadata['task'],
            'model': metadata['model'],
            'data': metadata['data'],
            'preprocessing': metadata['preprocessing'],
            'inference': metadata['inference'],
            'training': metadata['training'],
            'checkpoints': {
                'best_validation_loss': {
                    'epoch': best_epoch,
                    'validation_loss': None if best_metrics is None else best_metrics.get('loss'),
                    'validation_accuracy': None if best_metrics is None else best_metrics.get('accuracy'),
                    'validation_error_px': None if best_metrics is None else best_metrics.get('error_px'),
                    'path': str(best_checkpoint_path) if best_checkpoint_path is not None else None
                },
                'last_epoch': {
                    'epoch': last_epoch,
                    'validation_loss': None if last_metrics is None else last_metrics.get('loss'),
                    'validation_accuracy': None if last_metrics is None else last_metrics.get('accuracy'),
                    'validation_error_px': None if last_metrics is None else last_metrics.get('error_px'),
                    'path': str(last_checkpoint_path) if last_checkpoint_path is not None else None
                }
            },
            'validation_inference': {
                'enabled': bool(self.train_config.save_validation_results),
                'path': str(validation_results_path) if validation_results_path is not None else None
            },
            'runtime_environment': self.runtime_metadata,
            'timing': self.get_timing_summary(),
        }

        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=4, default=str)

    def build_resume_signature(self, training_loader, validation_loader):
        """Hash every trajectory-defining configuration and selected data input."""
        data_metadata = self.read_data_creation_metadata()
        fold_lists_path = Path(self.require_metadata_value(data_metadata, 'fold_lists_path'))
        training_list = get_split_file_path(fold_lists_path, self.repetition, 'training', self.fold)
        validation_list = get_split_file_path(fold_lists_path, self.repetition, 'validation', self.fold)
        payload = {
            'task': {
                'repetition': self.repetition,
                'fold': self.fold,
                'num_of_points': self.num_of_points,
                'task_name': data_metadata.get('task_name'),
                'tasks_classes': self.serialise_tasks_classes(self.tasks_classes),
            },
            'data': {
                'fold_collection_sha256': self.fold_collection_sha256,
                'training_list_path': str(training_list.resolve()),
                'training_list_sha256': self.sha256_file(training_list),
                'validation_list_path': str(validation_list.resolve()),
                'validation_list_sha256': self.sha256_file(validation_list),
                'mark_list_path': str(Path(self.require_metadata_value(data_metadata, 'mark_list_file')).resolve()),
                'mark_list_sha256': self.sha256_file(self.require_metadata_value(data_metadata, 'mark_list_file')),
                'image_data_dir': str(Path(self.require_metadata_value(data_metadata, 'image_data_dir')).resolve()),
                'training_csv_path': str(self.get_train_csv_path().resolve()),
                'training_csv_sha256': self.sha256_file(self.get_train_csv_path()),
                'validation_csv_path': str(self.get_val_csv_path().resolve()),
                'validation_csv_sha256': self.sha256_file(self.get_val_csv_path()),
                'training_patches_sha256': self.sha256_dataset_patches(training_loader.dataset),
                'validation_patches_sha256': self.sha256_dataset_patches(validation_loader.dataset),
            },
            'training': self.serialise(asdict(self.train_config)),
            'model': self.serialise(asdict(self.quadruplet_config)),
            'normalisation': self.build_normalisation_metadata(),
            'implementation': {
                'ipv_source_sha256': self.sha256_python_sources(),
                'python_version': self.runtime_metadata['python']['version'],
                'torch_version': self.runtime_metadata['pytorch']['version'],
                'dependency_versions': self.runtime_metadata['dependencies'],
            },
            'compute': {
                'device_type': self.device.type,
                'selected_device': self.runtime_metadata['cuda']['selected_device'],
                'pytorch_cuda_build_version': self.runtime_metadata['cuda']['pytorch_cuda_build_version'],
                'nvidia_driver_version': self.runtime_metadata['cuda']['nvidia_driver_version'],
                'pytorch_runtime': self.runtime_metadata['pytorch'],
                'cudnn': self.runtime_metadata['cudnn'],
            },
        }
        serialised_payload = self.serialise(payload)
        canonical_json = json.dumps(serialised_payload, sort_keys=True, separators=(',', ':'))
        return {'algorithm': 'sha256', 'sha256': hashlib.sha256(canonical_json.encode('utf-8')).hexdigest(),
                'payload': serialised_payload}

    def load_training_checkpoint(self, model, optimiser, scheduler, scaler, resume_signature):
        """Validate and restore the last completed epoch checkpoint."""
        checkpoint_path = self.get_checkpoint_path('last_epoch')

        if not checkpoint_path.is_file():
            raise ValueError(f'Resume requested, but the last-epoch checkpoint does not exist: {checkpoint_path}')

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as error:
            raise ValueError(f'Resume checkpoint could not be loaded and existing outputs were left untouched: {checkpoint_path}') from error

        self.validate_resume_checkpoint(checkpoint, checkpoint_path, resume_signature)
        self.remove_stale_checkpoint_temps()

        training_state = checkpoint['training_state']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        saved_scheduler = checkpoint.get('scheduler_state_dict')
        if ((scheduler is None and saved_scheduler is not None) or
                (scheduler is not None and saved_scheduler is None)):
            raise ValueError('Resume scheduler state does not match the current scheduler configuration.')
        if scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        self.training_sessions = copy.deepcopy(training_state['training_sessions'])
        self.restore_data_loader_generator_states(checkpoint['data_loader_generator_states'])
        self.restore_rng_state(checkpoint['rng_state'])
        self.resume_state_validated = True

        return {
            'history': copy.deepcopy(training_state['history']),
            'best_epoch': int(training_state['best_epoch']),
            'best_validation_metrics': self.unlabel_validation_metrics(training_state['best_validation_metrics']),
            'completed_epoch': int(training_state['completed_epoch']),
            'last_validation_metrics': self.unlabel_validation_metrics(training_state['last_validation_metrics']),
            'best_model_state_dict': checkpoint['best_model_state_dict'],
            'early_stop_best_validation_loss': float(training_state['early_stop_best_validation_loss']),
            'bad_epochs': int(training_state['bad_epochs']),
            'termination_reason': training_state['termination_reason'],
        }

    def validate_resume_checkpoint(self, checkpoint, checkpoint_path, resume_signature):
        """Reject incomplete, completed, corrupt, or incompatible continuation state."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f'Resume checkpoint is not a structured IPV checkpoint: {checkpoint_path}')

        if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION or checkpoint.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f'Resume requires checkpoint format/schema version {CHECKPOINT_FORMAT_VERSION}. '
                f'Checkpoint {checkpoint_path} has format={checkpoint.get("format_version")}, schema={checkpoint.get("schema_version")}. '
                'Only checkpoints produced by the current 0.1 contract are supported.'
            )

        if checkpoint.get('schema') != 'ipv_training_checkpoint' or checkpoint.get('checkpoint_type') != 'last_epoch':
            raise ValueError(f'Resume requires model_last_epoch.pth from this IPV run: {checkpoint_path}')

        if not checkpoint.get('resume_capable'):
            raise ValueError(f'Checkpoint is not marked as resume-capable: {checkpoint_path}')

        required = {'epoch', 'next_epoch', 'validation_metrics', 'state_dict', 'optimiser_state_dict',
                    'scheduler_state_dict', 'grad_scaler_state_dict', 'training_state', 'rng_state',
                    'data_loader_generator_states', 'best_model_state_dict', 'resume_signature'}
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f'Resume checkpoint is incomplete; missing fields: {missing}. Existing outputs were left untouched.')

        saved_signature = checkpoint['resume_signature']
        if not isinstance(saved_signature, dict) or saved_signature.get('algorithm') != 'sha256' or not isinstance(saved_signature.get('payload'), dict):
            raise ValueError('Resume checkpoint has an invalid compatibility signature structure.')

        saved_json = json.dumps(saved_signature['payload'], sort_keys=True, separators=(',', ':'))
        if saved_signature.get('sha256') != hashlib.sha256(saved_json.encode('utf-8')).hexdigest():
            raise ValueError('Resume checkpoint compatibility signature is internally inconsistent or corrupt.')

        if saved_signature.get('sha256') != resume_signature.get('sha256'):
            raise ValueError('Resume checkpoint is incompatible with the current task, repetition/fold, data, code, runtime, model, or training settings. '
                             f'Saved signature={saved_signature.get("sha256")}; current signature={resume_signature.get("sha256")}. '
                             'Existing outputs were left untouched.')

        state = checkpoint['training_state']
        required_state = {'completed_epoch', 'next_epoch', 'history', 'best_epoch', 'best_validation_metrics',
                          'early_stop_best_validation_loss', 'bad_epochs', 'last_validation_metrics',
                          'termination_reason', 'training_sessions'}
        missing_state = sorted(required_state - set(state))
        if missing_state:
            raise ValueError(f'Resume checkpoint training state is incomplete; missing fields: {missing_state}.')

        completed_epoch = int(state['completed_epoch'])
        if int(checkpoint['epoch']) != completed_epoch or int(checkpoint['next_epoch']) != completed_epoch + 1 or int(state['next_epoch']) != completed_epoch + 1:
            raise ValueError('Resume checkpoint epoch and next_epoch fields are inconsistent.')
        if checkpoint['validation_metrics'] != state['last_validation_metrics']:
            raise ValueError('Resume checkpoint top-level validation metrics do not match its last-epoch training state.')

        best_epoch = int(state['best_epoch'])
        if best_epoch < 1 or best_epoch > completed_epoch:
            raise ValueError(f'Resume checkpoint best_epoch must be between 1 and {completed_epoch}; got {best_epoch}.')

        self.validate_state_dict_snapshot(checkpoint['state_dict'], checkpoint['best_model_state_dict'])
        allowed_reasons = {'in_progress', 'interrupted', 'exception', 'early_stopping', 'max_epochs_reached'}
        if state['termination_reason'] not in allowed_reasons:
            raise ValueError(f'Resume checkpoint has an unknown termination_reason: {state["termination_reason"]!r}.')
        if state['termination_reason'] not in ('early_stopping', 'max_epochs_reached') and completed_epoch >= self.train_config.max_training_epochs:
            raise ValueError(f'Resume checkpoint completed epoch {completed_epoch}, but max_training_epochs is {self.train_config.max_training_epochs}.')

        self.validate_history(state['history'], completed_epoch)

    def ensure_best_checkpoint(self, checkpoint_path, best_epoch, best_metrics, best_model_state_dict):
        """Recover the exact committed best checkpoint when its sibling is missing or stale."""
        checkpoint_is_committed = False
        if checkpoint_path.is_file():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_is_committed = (
                    checkpoint.get('checkpoint_type') == 'best_validation_loss'
                    and int(checkpoint.get('epoch', -1)) == int(best_epoch)
                    and checkpoint.get('validation_metrics') == self.label_validation_metrics(best_metrics)
                    and self.state_dicts_equal(checkpoint.get('state_dict'), best_model_state_dict)
                )
            except Exception:
                checkpoint_is_committed = False

        if checkpoint_is_committed:
            return

        payload = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'schema': 'ipv_training_checkpoint',
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': utc_now_iso(),
            'checkpoint_type': 'best_validation_loss', 'epoch': int(best_epoch), 'resume_capable': False,
            'next_epoch': int(best_epoch) + 1,
            'state_dict': best_model_state_dict, 'optimiser_state_dict': None,
            'validation_metrics': self.label_validation_metrics(best_metrics),
            'metadata': self.build_checkpoint_metadata('best_validation_loss', best_epoch, best_metrics),
            'recovered_from_last_epoch_checkpoint': True,
        }
        self.atomic_torch_save(payload, checkpoint_path)

    def build_training_state(self, completed_epoch, history, best_epoch, best_metrics, early_stop_best_validation_loss,
                             bad_epochs, last_metrics, termination_reason):
        """Build the complete epoch-boundary continuation state."""
        return {
            'completed_epoch': int(completed_epoch), 'next_epoch': int(completed_epoch) + 1,
            'history': copy.deepcopy(history),
            'best_epoch': int(best_epoch), 'best_validation_metrics': self.label_validation_metrics(best_metrics),
            'early_stop_best_validation_loss': float(early_stop_best_validation_loss), 'bad_epochs': int(bad_epochs),
            'last_validation_metrics': self.label_validation_metrics(last_metrics), 'termination_reason': termination_reason,
            'training_sessions': self.get_training_sessions_snapshot(),
        }

    @staticmethod
    def clone_state_dict_to_cpu(state_dict):
        """Clone model weights to an independent CPU snapshot."""
        return {name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for name, value in state_dict.items()}

    @staticmethod
    def validate_state_dict_snapshot(current_state_dict, best_state_dict):
        """Validate that the embedded best snapshot has the current model's tensor contract."""
        if not isinstance(current_state_dict, dict) or not isinstance(best_state_dict, dict):
            raise ValueError('Resume checkpoint model state and embedded best-model snapshot must be dictionaries.')
        if set(current_state_dict) != set(best_state_dict):
            raise ValueError('Resume checkpoint embedded best-model snapshot has different parameter keys.')

        for name, current_value in current_state_dict.items():
            best_value = best_state_dict[name]
            if torch.is_tensor(current_value) != torch.is_tensor(best_value):
                raise ValueError(f'Resume checkpoint best-model value type differs for {name}.')
            if torch.is_tensor(current_value) and (current_value.shape != best_value.shape or current_value.dtype != best_value.dtype):
                raise ValueError(f'Resume checkpoint best-model tensor contract differs for {name}.')

    @staticmethod
    def state_dicts_equal(first_state_dict, second_state_dict):
        """Return whether two model state dictionaries are exactly equal."""
        if not isinstance(first_state_dict, dict) or not isinstance(second_state_dict, dict) or set(first_state_dict) != set(second_state_dict):
            return False

        for name, first_value in first_state_dict.items():
            second_value = second_state_dict[name]
            if torch.is_tensor(first_value) != torch.is_tensor(second_value):
                return False
            if torch.is_tensor(first_value):
                if first_value.shape != second_value.shape or first_value.dtype != second_value.dtype or not torch.equal(first_value.cpu(), second_value.cpu()):
                    return False
            elif first_value != second_value:
                return False

        return True

    @staticmethod
    def sha256_file(path):
        """Hash one file without loading it all into memory."""
        digest = hashlib.sha256()
        with open(path, 'rb') as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def sha256_dataset_patches(dataset):
        """Hash generated patch identities and bytes in CSV order."""
        digest = hashlib.sha256()

        for raw_path in dataset.csv_data.iloc[:, 1].tolist():
            patch_path = Path(raw_path).resolve()
            digest.update(str(patch_path).encode('utf-8'))
            digest.update(b'\0')
            with open(patch_path, 'rb') as patch_file:
                for block in iter(lambda: patch_file.read(1024 * 1024), b''):
                    digest.update(block)
            digest.update(b'\0')

        return digest.hexdigest()

    @staticmethod
    def sha256_python_sources():
        """Hash active IPV Python source files for continuation compatibility."""
        package_root = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for source_path in sorted(package_root.rglob('*.py'), key=lambda path: path.as_posix()):
            digest.update(source_path.relative_to(package_root).as_posix().encode('utf-8'))
            digest.update(b'\0')
            digest.update(source_path.read_bytes())
            digest.update(b'\0')
        return digest.hexdigest()

    @staticmethod
    def atomic_torch_save(payload, checkpoint_path):
        """Atomically replace a checkpoint, preserving the previous file on failure."""
        checkpoint_path = Path(checkpoint_path)
        temporary_path = checkpoint_path.with_name(f'.{checkpoint_path.name}.tmp')
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def remove_stale_checkpoint_temps(self):
        """Discard uncommitted checkpoint siblings after the committed resume file validates."""
        for checkpoint_type in ('best_validation_loss', 'last_epoch'):
            checkpoint_path = self.get_checkpoint_path(checkpoint_type)
            temporary_path = checkpoint_path.with_name(f'.{checkpoint_path.name}.tmp')
            if temporary_path.exists():
                temporary_path.unlink()

    def capture_rng_state(self):
        """Capture Python, NumPy, PyTorch, and CUDA random-number states."""
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda_all': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None}

    def restore_rng_state(self, state):
        """Restore Python, NumPy, PyTorch, and CUDA random-number states."""
        required = {'python', 'numpy', 'torch_cpu', 'torch_cuda_all'}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f'Resume checkpoint RNG state is incomplete; missing fields: {missing}.')

        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'].cpu())
        cuda_states = state['torch_cuda_all']
        if cuda_states is not None:
            if self.device.type != 'cuda' or not torch.cuda.is_available():
                raise ValueError('Resume checkpoint contains CUDA RNG state, but CUDA is unavailable.')
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(f'Resume checkpoint contains RNG state for {len(cuda_states)} CUDA device(s), but {torch.cuda.device_count()} are available.')
            torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])

    def capture_data_loader_generator_states(self):
        """Capture training and validation DataLoader generator states."""
        if self.training_generator is None or self.validation_generator is None:
            raise RuntimeError('DataLoader generators have not been initialised.')
        return {'training': self.training_generator.get_state(), 'validation': self.validation_generator.get_state()}

    def restore_data_loader_generator_states(self, states):
        """Restore training and validation DataLoader generator states."""
        if self.training_generator is None or self.validation_generator is None:
            raise RuntimeError('DataLoader generators have not been initialised.')
        if set(states) != {'training', 'validation'}:
            raise ValueError('Resume checkpoint must contain training and validation DataLoader generator states.')
        self.training_generator.set_state(states['training'].cpu())
        self.validation_generator.set_state(states['validation'].cpu())

    @staticmethod
    def validate_history(history, completed_epoch):
        """Validate complete, finite, contiguous epoch history."""
        if not isinstance(history, dict):
            raise ValueError('Resume checkpoint history must be a dictionary of columns.')
        if set(HISTORY_FIELDS) - set(history):
            raise ValueError(f'Resume history is missing fields: {sorted(set(HISTORY_FIELDS) - set(history))}')
        lengths = {field: len(history[field]) for field in HISTORY_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'Resume history columns have inconsistent lengths: {lengths}')
        if list(history['epoch']) != list(range(1, int(completed_epoch) + 1)):
            raise ValueError('Resume history does not contain every completed epoch exactly once.')

        numeric_fields = set(HISTORY_FIELDS) - {'epoch_started_at', 'epoch_completed_at'}
        for field_name in numeric_fields:
            if not all(np.isfinite(value) for value in history[field_name]):
                raise ValueError(f'Resume history contains a non-finite value in {field_name}.')

    @staticmethod
    def validate_finite_metrics(phase, metrics):
        """Stop training when a reported metric is NaN or infinite."""
        invalid = {name: value for name, value in metrics.items() if not np.isfinite(value)}
        if invalid:
            raise FloatingPointError(f'Non-finite {phase} metric(s) detected: {invalid}')

    def validate_configs(self):
        """Validate training-control settings before any output mutation."""
        if self.repetition < 1:
            raise ValueError('repetition must be at least 1.')
        self.fold = normalise_fold(self.fold)
        if self.train_config.random_seed < 0:
            raise ValueError('random_seed must be at least 0.')
        if self.train_config.batch_size < 1 or self.train_config.max_training_epochs < 1 or self.train_config.num_workers < 0:
            raise ValueError('batch_size and max_training_epochs must be positive; num_workers must be non-negative.')
        if self.train_config.learning_rate <= 0 or self.train_config.weight_decay < 0 or self.train_config.momentum < 0:
            raise ValueError('learning_rate must be positive; weight_decay and momentum must be non-negative.')
        if str(self.train_config.optimiser_name).lower() not in ('adamw', 'sgd'):
            raise ValueError('optimiser_name must be adamw or sgd.')
        if str(self.train_config.lr_schedule).lower() not in ('none', 'step', 'plateau'):
            raise ValueError('lr_schedule must be none, step, or plateau.')
        if self.train_config.lr_step_size < 1 or self.train_config.lr_gamma <= 0:
            raise ValueError('lr_step_size must be at least 1 and lr_gamma must be positive.')
        if self.train_config.early_stop_patience < 1 or self.train_config.early_stop_min_delta < 0 or self.train_config.early_stop_warmup_epochs < 0:
            raise ValueError('Early-stopping patience must be positive; min_delta and warmup must be non-negative.')
        if self.train_config.validation_inference_batch_size < 1:
            raise ValueError('validation_inference_batch_size must be at least 1.')
        if self.train_config.validation_vote_smoothing_sigma < 0:
            raise ValueError('validation_vote_smoothing_sigma must be non-negative.')

    @staticmethod
    def set_random_seed(seed):
        """Enable deterministic Python, NumPy, PyTorch, CUDA, cuDNN, and cuBLAS behaviour."""
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    def synchronise_device(self):
        """Synchronise queued CUDA work before timing boundaries."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def finish_workflow(self):
        """Freeze workflow completion timestamps and duration."""
        if self.workflow_completed_at is None:
            self.workflow_completed_at = utc_now_iso()
        if self.workflow_duration_seconds is None and self.workflow_start_perf is not None:
            self.workflow_duration_seconds = float(time.perf_counter() - self.workflow_start_perf)
        self.update_current_training_session(status=self.training_status)
        self.current_session_start_perf = None

    def begin_training_session(self, resumed, resumed_from_epoch):
        """Append a timestamped fresh or resumed execution session."""
        if resumed and self.training_sessions and self.training_sessions[-1].get('status') == 'in_progress':
            self.training_sessions[-1]['status'] = 'interrupted'

        self.current_session_index = len(self.training_sessions)
        self.current_session_start_perf = time.perf_counter()
        self.training_sessions.append({
            'session_number': self.current_session_index + 1,
            'session_id': str(uuid.uuid4()),
            'started_at': utc_now_iso(),
            'last_updated_at': None,
            'duration_seconds': 0.0,
            'resumed': bool(resumed),
            'resumed_from_epoch': int(resumed_from_epoch),
            'completed_epoch': int(resumed_from_epoch),
            'status': 'in_progress',
            'runtime_environment': self.runtime_metadata,
        })

    def update_current_training_session(self, status=None, completed_epoch=None):
        """Refresh the current execution-session record."""
        if self.current_session_index is None or self.current_session_start_perf is None:
            return
        session = self.training_sessions[self.current_session_index]
        session['last_updated_at'] = utc_now_iso()
        session['duration_seconds'] = float(time.perf_counter() - self.current_session_start_perf)
        if status is not None:
            session['status'] = status
        if completed_epoch is not None:
            session['completed_epoch'] = int(completed_epoch)

    def get_training_sessions_snapshot(self):
        """Return an up-to-date copy of all execution sessions."""
        self.update_current_training_session()
        return copy.deepcopy(self.training_sessions)

    def get_timing_summary(self):
        """Return phase and cumulative epoch timing."""
        history = self.history
        workflow_duration = self.workflow_duration_seconds
        if workflow_duration is None and self.workflow_start_perf is not None:
            workflow_duration = float(time.perf_counter() - self.workflow_start_perf)
        return {
            'workflow_started_at': self.workflow_started_at, 'workflow_completed_at': self.workflow_completed_at,
            'workflow_duration_seconds': workflow_duration,
            'dataset_validation_duration_seconds': float(self.dataset_validation_duration_seconds),
            'model_setup_and_resume_duration_seconds': float(self.model_setup_duration_seconds),
            'validation_export_duration_seconds': float(self.validation_export_duration_seconds),
            'cumulative_training_duration_seconds': float(sum(history.get('training_duration_seconds', []))),
            'cumulative_validation_duration_seconds': float(sum(history.get('validation_duration_seconds', []))),
            'cumulative_epoch_duration_seconds': float(sum(history.get('epoch_duration_seconds', []))),
            'sessions': self.get_training_sessions_snapshot(),
        }

    def get_run_report(self):
        """Return pipeline-level status, timing, resume, and runtime metadata."""
        return {'training_status': self.training_status, 'termination_reason': self.termination_reason,
                'failure': self.failure, 'resume_training': self.resume_training,
                'resume_checkpoint_path': str(self.get_checkpoint_path('last_epoch')) if self.resume_training else None,
                'resume_state_validated': self.resume_state_validated,
                'normalisation': self.build_normalisation_metadata(),
                'runtime_environment': self.runtime_metadata, 'timing': self.get_timing_summary()}

    @staticmethod
    def label_validation_metrics(validation_metrics):
        """Return externally stored validation metrics with unambiguous names."""
        if validation_metrics is None:
            return None
        if not validation_metrics:
            return {}
        if 'validation_loss' in validation_metrics:
            return {
                'validation_loss': float(validation_metrics['validation_loss']),
                'validation_accuracy': float(validation_metrics['validation_accuracy']),
                'validation_error_px': float(validation_metrics['validation_error_px']),
            }
        return {
            'validation_loss': float(validation_metrics['loss']),
            'validation_accuracy': float(validation_metrics['accuracy']),
            'validation_error_px': float(validation_metrics['error_px']),
        }

    @staticmethod
    def unlabel_validation_metrics(validation_metrics):
        """Convert externally labelled metrics back to loop metric names."""
        return {
            'loss': float(validation_metrics['validation_loss']),
            'accuracy': float(validation_metrics['validation_accuracy']),
            'error_px': float(validation_metrics['validation_error_px']),
        }

    @staticmethod
    def get_current_lr(optimiser):
        """Return the current optimiser learning rate."""
        return optimiser.param_groups[0]['lr']

    @staticmethod
    def get_task_names():
        """Return model task names in output-head order."""
        return ['distance', 'angle']

    @staticmethod
    def serialise_tasks_classes(tasks_classes):
        """Convert interval tuples to lists for stable checkpoint metadata."""
        return [TrainModel.serialise_intervals(task_classes) for task_classes in tasks_classes]

    @staticmethod
    def serialise_intervals(intervals):
        """Convert interval pairs into plain serialisable lists."""
        return [[float(lower), float(upper)] for lower, upper in intervals]

    @staticmethod
    def serialise(value):
        """Convert paths and nested values to stable JSON-compatible objects."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: TrainModel.serialise(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [TrainModel.serialise(item) for item in value]
        return value

    @staticmethod
    def optional_int(value):
        """Return an int or None."""
        return None if value is None else int(value)

    @staticmethod
    def optional_number_list(values):
        """Return a list of numeric values or None."""
        if values is None:
            return None

        return [float(value) for value in values]

    @staticmethod
    def path_metadata_to_string(value):
        """Return path metadata as a string or None."""
        return None if value is None else str(value)

    def get_run_name(self):
        """Return the run-name path component."""
        return self.output_path.parent.parent.name

    def get_train_csv_path(self):
        """Return the generated training CSV path."""
        return self.train_path / f'Train_f{self.fold}.csv'

    def get_val_csv_path(self):
        """Return the generated validation CSV path."""
        return self.train_path / f'Val_f{self.fold}.csv'

    def get_log_path(self):
        """Return the log CSV path."""
        return self.output_path / 'training_validation_log.csv'

    def get_checkpoint_path(self, checkpoint_type):
        """Return a best or last checkpoint path."""
        return self.output_path / f'model_{checkpoint_type}.pth'

    def get_checkpoint_summary_path(self):
        """Return the checkpoint summary JSON path."""
        return self.output_path / 'validation_checkpoint_summary.json'

    def get_plot_path(self):
        """Return the plot path."""
        return self.output_path / 'training_validation_plot.png'

    def get_validation_output_path(self):
        """Return the validation-image inference output directory."""
        return self.output_path / 'validation_results'

    def get_validation_staging_path(self):
        """Return the same-volume staging path for validation outputs."""
        return self.output_path / '.validation_results.tmp'

    def get_validation_backup_path(self):
        """Return the temporary backup path used while committing validation outputs."""
        return self.output_path / '.validation_results.backup'

    def prepare_validation_staging_path(self):
        """Recover any committed backup and clear incomplete staging data."""
        final_path = self.get_validation_output_path()
        staging_path = self.get_validation_staging_path()
        backup_path = self.get_validation_backup_path()

        if backup_path.exists() and not final_path.exists():
            os.replace(backup_path, final_path)
        elif backup_path.exists():
            shutil.rmtree(backup_path)
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(exist_ok=False, parents=True)

    def commit_validation_output(self):
        """Atomically swap a complete validation export into place."""
        final_path = self.get_validation_output_path()
        staging_path = self.get_validation_staging_path()
        backup_path = self.get_validation_backup_path()

        if backup_path.exists():
            shutil.rmtree(backup_path)
        if final_path.exists():
            os.replace(final_path, backup_path)

        try:
            os.replace(staging_path, final_path)
        except BaseException:
            if backup_path.exists() and not final_path.exists():
                os.replace(backup_path, final_path)
            raise

        if backup_path.exists():
            shutil.rmtree(backup_path)

    @staticmethod
    def format_number(value):
        """Format numeric values safely for file names."""
        return f'{value:g}'.replace('-', 'm').replace('.', 'p')

    @staticmethod
    def empty_history():
        """Create the training history store."""
        return {field_name: [] for field_name in HISTORY_FIELDS}


def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load a Quadruplet model from a self-describing checkpoint."""
    from .quadruplet import Quadruplet

    device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' not in checkpoint:
        raise ValueError('This checkpoint only contains a state_dict. Recreate the model manually or convert the checkpoint.')

    metadata = checkpoint.get('metadata', {})
    model_metadata = metadata.get('model', checkpoint.get('model', {})) if isinstance(metadata, dict) else checkpoint.get('model', {})
    model_args = model_metadata.get('init_args')

    if not model_args:
        raise ValueError('Checkpoint does not contain model init_args.')

    model = Quadruplet(**model_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint
