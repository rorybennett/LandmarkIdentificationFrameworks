"""
Training and validation routines for heatmap landmark models.
"""

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

import matplotlib
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from openpyxl import Workbook
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from .heatmap_transforms import get_augmentation_policy
from .model_registry import build_heatmap_model, get_model_config_fields, get_model_kwargs, get_model_registry_entry
from .models import count_trainable_parameters, unpack_heatmap_output
from .normalisation import EXPECTED_NORMALISATION_CHANNELS, validate_normalisation_constants
from .runtime_metadata import collect_runtime_metadata, utc_now_iso
from .utils.io_utils import get_split_file_path, heatmaps_to_points, infer_image_channel_count, normalise_fold, safe_file_stem, scale_points_to_original
from .utils.progress_bar import ProgressBar
from .utils.visualisation_utils import save_validation_overlays

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def seed_worker(_worker_id):
    """Seed NumPy and Python RNGs inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


CHECKPOINT_FORMAT_VERSION = '0.1'
CHECKPOINT_SCHEMA_VERSION = '0.1'
CHECKPOINT_SCHEMA_NAME = 'heatmap_checkpoint_metadata'
MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
HISTORY_FIELDS = (
    'epoch',
    'epoch_started_at',
    'epoch_completed_at',
    'lr',
    'training_loss',
    'training_error_px',
    'validation_loss',
    'validation_error_px',
    'training_duration_seconds',
    'validation_duration_seconds',
    'epoch_duration_seconds',
)


@dataclass
class HeatmapDataConfig:
    repetition: int
    fold: int | str
    task_name: str
    num_of_points: int
    fold_lists_path: Path
    mark_list_file: Path
    image_data_dir: Path
    image_size: tuple[int, int]
    heatmap_sigma: float = 8.0
    input_channels: int | None = None
    recursive_image_search: bool = False
    oversampling_factor: int = 1
    fold_collection_sha256: str | None = None
    normalise_inputs: bool = False
    normalisation_mean: tuple[float, float, float] | None = None
    normalisation_std: tuple[float, float, float] | None = None


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    max_training_epochs: int
    num_workers: int = 8
    random_seed: int = 42
    optimiser_name: str = 'adamw'
    loss_name: str = 'weighted_mse'
    positive_weight: float = 20.0
    weight_decay: float = 1e-4
    momentum: float = 0.9
    lr_schedule: str = 'plateau'
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4
    early_stop_warmup_epochs: int = 10
    use_amp: bool = False
    save_validation_predictions: bool = True


@dataclass
class HeatmapModelConfig:
    network_name: str = 'unet_basic'
    base_channels: int = 32
    depth: int = 4
    channel_multiplier: int = 2
    max_channels: int = 512
    normalisation: str | None = 'batch'
    activation: str = 'relu'
    dropout: float = 0.0
    upsampling: str = 'bilinear'
    output_activation: str = 'none'
    padding_mode: str = 'zeros'
    final_kernel_size: int = 1
    hrnet_width: int = 32
    hrnet_modules: int = 3
    hrnet_blocks: int = 2
    hourglass_features: int = 128
    hourglass_stacks: int = 2
    hourglass_depth: int = 4
    hourglass_blocks: int = 1
    auxiliary_loss_weight: float = 1.0
    vit_patch_size: int = 16
    vit_embed_dim: int = 384
    vit_depth: int = 8
    vit_heads: int = 6
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    vit_decoder_channels: int = 256


class WeightedMSELoss(nn.Module):
    """Apply stronger loss near landmark heatmap peaks."""

    def __init__(self, positive_weight=20.0):
        super().__init__()
        self.positive_weight = float(positive_weight)

    def forward(self, outputs, targets):
        weights = 1.0 + targets * self.positive_weight
        return torch.mean(weights * (outputs - targets) ** 2)


class TrainModel:
    """Train, validate, and checkpoint one heatmap model for one fold."""

    def __init__(self, data_config, train_config, model_config, output_save_path, device=None, resume_training=False):
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.output_path = Path(output_save_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_training = bool(resume_training)
        self.runtime_metadata = None
        self.training_status = 'initialising'
        self.termination_reason = None
        self.failure = None
        self.resume_state_validated = False
        self.workflow_started_at = None
        self.workflow_completed_at = None
        self.workflow_start_perf = None
        self.workflow_duration_seconds = None
        self.dataset_validation_duration_seconds = 0.0
        self.model_setup_duration_seconds = 0.0
        self.validation_export_duration_seconds = 0.0
        self.validation_metadata_context = None
        self.training_sessions = []
        self.current_session_index = None
        self.current_session_start_perf = None
        self.training_generator = None
        self.validation_generator = None
        self.history = self.empty_history()
        self.validate_configs()

    def train(self, on_dataset_validated=None, on_training_state_ready=None):
        """Run or explicitly resume the fold training workflow."""
        self.workflow_started_at = utc_now_iso()
        self.workflow_start_perf = time.perf_counter()
        self.set_random_seed(self.train_config.random_seed)
        self.runtime_metadata = collect_runtime_metadata(device=self.device, use_amp=self.train_config.use_amp)

        try:
            self.output_path.mkdir(exist_ok=True, parents=True)
            dataset_validation_start = time.perf_counter()

            try:
                training_loader, validation_loader = self.build_data_loaders()
            finally:
                self.dataset_validation_duration_seconds = time.perf_counter() - dataset_validation_start

            self.training_status = 'dataset_validated'

            if on_dataset_validated is not None:
                on_dataset_validated()

            model_setup_start = time.perf_counter()
            model = self.build_model()
            criterion = self.build_criterion()
            optimiser = self.build_optimiser(model)
            scheduler = self.build_scheduler(optimiser)
            scaler = torch.amp.GradScaler('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda')
            resume_signature = self.build_resume_signature(training_loader=training_loader, validation_loader=validation_loader)
            history = self.empty_history()
            best_epoch = None
            best_validation_metrics = None
            early_stop_best_validation_loss = float('inf')
            last_epoch = 0
            last_validation_metrics = None
            best_checkpoint_path = None
            last_checkpoint_path = None
            best_model_state_dict = None
            bad_epochs = 0
            start_epoch = 1
            resumed_checkpoint_termination_reason = None

            if self.resume_training:
                resume_state = self.load_training_checkpoint(model=model, optimiser=optimiser, scheduler=scheduler, scaler=scaler,
                                                             resume_signature=resume_signature)
                history = resume_state['history']
                best_epoch = resume_state['best_epoch']
                best_validation_metrics = resume_state['best_validation_metrics']
                early_stop_best_validation_loss = resume_state['early_stop_best_validation_loss']
                last_epoch = resume_state['completed_epoch']
                last_validation_metrics = resume_state['last_validation_metrics']
                best_model_state_dict = resume_state['best_model_state_dict']
                bad_epochs = resume_state['bad_epochs']
                start_epoch = last_epoch + 1
                resumed_checkpoint_termination_reason = resume_state['termination_reason']
                best_checkpoint_path = self.get_checkpoint_path('best_validation_loss')
                last_checkpoint_path = self.get_checkpoint_path('last_epoch')
                self.ensure_best_checkpoint(best_checkpoint_path=best_checkpoint_path, best_epoch=best_epoch,
                                            best_validation_metrics=best_validation_metrics,
                                            best_model_state_dict=best_model_state_dict)

            self.begin_training_session(resumed=self.resume_training, resumed_from_epoch=last_epoch)
            self.model_setup_duration_seconds = time.perf_counter() - model_setup_start
            self.resume_state_validated = True
            self.training_status = 'running'
            self.history = history
            self.write_history_log(history)
            self.save_history_plot(history)

            if on_training_state_ready is not None:
                on_training_state_ready()

            print(f'\tNetwork loaded on {self.device}. Trainable parameters: {count_trainable_parameters(model):,}', flush=True)

            if self.resume_training:
                if resumed_checkpoint_termination_reason in ('early_stopping', 'max_epochs_reached'):
                    print(f'\tTraining epochs were already complete ({resumed_checkpoint_termination_reason}); rebuilding final outputs.', flush=True)
                else:
                    print(f'\tResuming from completed epoch {last_epoch}; next epoch: {start_epoch}.', flush=True)

            epochs_to_run = (() if resumed_checkpoint_termination_reason in ('early_stopping', 'max_epochs_reached')
                             else range(start_epoch, self.train_config.max_training_epochs + 1))

            for epoch in epochs_to_run:
                epoch_started_at = utc_now_iso()
                epoch_start = time.perf_counter()
                print(f"\t{dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Epoch {epoch}/{self.train_config.max_training_epochs}", flush=True)
                epoch_lr = self.get_current_lr(optimiser)

                self.synchronise_device()
                training_start = time.perf_counter()
                training_metrics = self.train_epoch(model=model, loader=training_loader, criterion=criterion, optimiser=optimiser, scaler=scaler)
                self.synchronise_device()
                training_duration_seconds = time.perf_counter() - training_start

                validation_start = time.perf_counter()
                validation_metrics = self.validate(model=model, loader=validation_loader, criterion=criterion)
                self.synchronise_device()
                validation_duration_seconds = time.perf_counter() - validation_start
                self.validate_finite_metrics(phase='training', metrics=training_metrics)
                self.validate_finite_metrics(phase='validation', metrics=validation_metrics)

                if scheduler is not None:
                    scheduler.step(validation_metrics['loss']) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

                is_new_best = best_validation_metrics is None or validation_metrics['loss'] < best_validation_metrics['loss']
                is_early_stop_improvement = validation_metrics['loss'] < early_stop_best_validation_loss - self.train_config.early_stop_min_delta

                if is_new_best:
                    best_epoch = epoch
                    best_validation_metrics = dict(validation_metrics)
                    best_model_state_dict = self.clone_state_dict_to_cpu(model.state_dict())

                if is_early_stop_improvement:
                    early_stop_best_validation_loss = validation_metrics['loss']

                if epoch >= self.train_config.early_stop_warmup_epochs:
                    bad_epochs = 0 if is_early_stop_improvement else bad_epochs + 1

                should_early_stop = epoch >= self.train_config.early_stop_warmup_epochs and bad_epochs >= self.train_config.early_stop_patience
                epoch_termination_reason = ('early_stopping' if should_early_stop else
                                            'max_epochs_reached' if epoch >= self.train_config.max_training_epochs else 'in_progress')
                last_epoch = epoch
                last_validation_metrics = dict(validation_metrics)
                epoch_completed_at = utc_now_iso()
                epoch_duration_seconds = time.perf_counter() - epoch_start
                self.update_history(history=history, epoch=epoch, epoch_started_at=epoch_started_at,
                                    epoch_completed_at=epoch_completed_at, epoch_lr=epoch_lr,
                                    training_metrics=training_metrics, validation_metrics=validation_metrics,
                                    training_duration_seconds=training_duration_seconds,
                                    validation_duration_seconds=validation_duration_seconds,
                                    epoch_duration_seconds=epoch_duration_seconds)
                self.history = history
                self.update_current_training_session(status=epoch_termination_reason, completed_epoch=epoch)
                training_state = self.build_training_state(completed_epoch=epoch, history=history, best_epoch=best_epoch,
                                                           best_validation_metrics=best_validation_metrics,
                                                           early_stop_best_validation_loss=early_stop_best_validation_loss,
                                                           bad_epochs=bad_epochs, last_validation_metrics=last_validation_metrics,
                                                           termination_reason=epoch_termination_reason)
                self.save_history_plot(history)

                if is_new_best:
                    best_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, scheduler=scheduler, scaler=scaler,
                                                                checkpoint_type='best_validation_loss', epoch=epoch,
                                                                validation_metrics=validation_metrics, training_state=None,
                                                                resume_signature=resume_signature, best_model_state_dict=None)

                last_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, scheduler=scheduler, scaler=scaler,
                                                            checkpoint_type='last_epoch', epoch=epoch,
                                                            validation_metrics=validation_metrics, training_state=training_state,
                                                            resume_signature=resume_signature,
                                                            best_model_state_dict=best_model_state_dict)
                self.write_history_log(history)

                if is_new_best:
                    print(f"\tNew best model saved from epoch {epoch} with validation_loss={best_validation_metrics['loss']:.6f} and "
                          f"validation_error={best_validation_metrics['error_px']:.2f}px", flush=True)

                if should_early_stop:
                    print(f'\tEarly stop: validation loss stopped improving by at least {self.train_config.early_stop_min_delta:g}. '
                          f'Best checkpoint epoch: {best_epoch}; early-stop reference loss: {early_stop_best_validation_loss:.6f}', flush=True)
                    break

            if resumed_checkpoint_termination_reason in ('early_stopping', 'max_epochs_reached'):
                self.termination_reason = resumed_checkpoint_termination_reason
            else:
                self.termination_reason = ('early_stopping' if bad_epochs >= self.train_config.early_stop_patience
                                           and last_epoch >= self.train_config.early_stop_warmup_epochs else 'max_epochs_reached')
            validation_output_paths = None

            if self.train_config.save_validation_predictions:
                validation_export_start = time.perf_counter()

                try:
                    validation_output_paths = self.save_validation_predictions(model=model, validation_loader=validation_loader,
                                                                               checkpoint_path=best_checkpoint_path or last_checkpoint_path)
                finally:
                    self.validation_export_duration_seconds = time.perf_counter() - validation_export_start

            self.training_status = 'completed'
            self.finish_workflow()

            if self.validation_metadata_context is not None:
                self.write_validation_run_metadata(**self.validation_metadata_context)
                self.commit_validation_output()

            self.write_checkpoint_summary(best_epoch=best_epoch, last_epoch=last_epoch,
                                          best_validation_metrics=best_validation_metrics,
                                          last_validation_metrics=last_validation_metrics,
                                          best_checkpoint_path=best_checkpoint_path, last_checkpoint_path=last_checkpoint_path,
                                          validation_output_paths=validation_output_paths)
            plt.clf()
            return best_checkpoint_path or last_checkpoint_path
        except BaseException as error:
            if 'model_setup_start' in locals() and self.model_setup_duration_seconds == 0.0:
                self.model_setup_duration_seconds = time.perf_counter() - model_setup_start

            self.training_status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            self.termination_reason = 'keyboard_interrupt' if isinstance(error, KeyboardInterrupt) else 'exception'
            self.failure = {'type': type(error).__name__, 'message': str(error)}
            self.update_current_training_session(status=self.training_status)
            self.finish_workflow()
            plt.clf()
            raise

    def train_epoch(self, model, loader, criterion, optimiser, scaler):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_error_px = 0.0
        total_points = 0

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            targets = batch['heatmaps'].to(self.device, non_blocking=True)
            points_original = batch['points_original'].to(self.device, non_blocking=True)
            original_size = batch['original_size'].to(self.device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda'):
                model_output = model(images)
                outputs, auxiliary_outputs = unpack_heatmap_output(model_output)
                loss = self.calculate_model_loss(outputs=outputs, auxiliary_outputs=auxiliary_outputs, targets=targets, criterion=criterion)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            batch_error = self.calculate_batch_error(outputs=outputs.detach(), points_original=points_original, original_size=original_size)
            total_loss += loss.item() * images.size(0)
            total_error_px += batch_error.sum().item()
            total_points += batch_error.numel()

        return self.format_metrics(loss=total_loss / max(len(loader.dataset), 1), error_px=total_error_px / max(total_points, 1))

    def validate(self, model, loader, criterion):
        """Evaluate on the validation split."""
        model.eval()
        total_loss = 0.0
        total_error_px = 0.0
        total_points = 0

        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['heatmaps'].to(self.device, non_blocking=True)
                points_original = batch['points_original'].to(self.device, non_blocking=True)
                original_size = batch['original_size'].to(self.device, non_blocking=True)
                model_output = model(images)
                outputs, auxiliary_outputs = unpack_heatmap_output(model_output)
                loss = self.calculate_model_loss(outputs=outputs, auxiliary_outputs=auxiliary_outputs, targets=targets, criterion=criterion)
                batch_error = self.calculate_batch_error(outputs=outputs, points_original=points_original, original_size=original_size)
                total_loss += loss.item() * images.size(0)
                total_error_px += batch_error.sum().item()
                total_points += batch_error.numel()

        return self.format_metrics(loss=total_loss / max(len(loader.dataset), 1), error_px=total_error_px / max(total_points, 1))

    def save_validation_predictions(self, model, validation_loader, checkpoint_path):
        """Save validation endpoint predictions using an IPV-like output layout."""
        checkpoint_payload = None

        if checkpoint_path is not None:
            checkpoint_payload = self.load_checkpoint_state(model=model, checkpoint_path=checkpoint_path)

        model.eval()
        final_validation_output_path = self.get_validation_output_path()
        validation_output_path = self.get_validation_staging_path()
        self.prepare_validation_staging_path()

        logs_path = validation_output_path / 'validation_logs'
        validation_output_path.mkdir(exist_ok=True, parents=True)
        logs_path.mkdir(exist_ok=True, parents=True)

        image_summary_rows = []
        endpoint_rows = []
        prediction_rows = []
        checkpoint_type = (checkpoint_payload.get('checkpoint_type') if isinstance(checkpoint_payload, dict)
                           else self.get_checkpoint_type_from_path(checkpoint_path))

        with torch.inference_mode(), ProgressBar(total=len(validation_loader), label='Validation predictions') as progress_bar:
            for batch in validation_loader:
                status = ', '.join(str(sample_name) for sample_name in list(batch['sample_name'])[:2])
                progress_bar.set_status(status)
                images = batch['image'].to(self.device, non_blocking=True)
                points_original = batch['points_original'].to(self.device, non_blocking=True)
                original_size = batch['original_size'].to(self.device, non_blocking=True)
                model_output = model(images)
                outputs, _ = unpack_heatmap_output(model_output)
                predicted_resized = heatmaps_to_points(outputs)
                predicted_original = scale_points_to_original(points=predicted_resized, original_sizes=original_size, image_size=self.data_config.image_size)
                errors = torch.linalg.norm(predicted_original - points_original, dim=2)

                for index, sample_name in enumerate(batch['sample_name']):
                    target_points = points_original[index].detach().cpu().numpy()
                    predicted_points = predicted_original[index].detach().cpu().numpy()
                    point_errors = errors[index].detach().cpu().numpy()
                    image_height, image_width = original_size[index].detach().cpu().numpy().astype(int).tolist()
                    image_path = str(batch['image_path'][index])
                    prediction_rows.append(self.create_prediction_row(sample_name=sample_name, target_points=target_points, predicted_points=predicted_points, point_errors=point_errors))
                    image_summary_rows.append(self.create_image_summary_row(sample_name=sample_name, image_path=image_path, image_height=image_height, image_width=image_width,
                                                                            point_errors=point_errors, checkpoint_type=checkpoint_type))
                    endpoint_rows.extend(self.create_endpoint_rows(sample_name=sample_name, image_path=image_path, target_points=target_points,
                                                                   predicted_points=predicted_points, point_errors=point_errors, checkpoint_type=checkpoint_type))

                    output_stem = safe_file_stem(sample_name)
                    predicted_heatmaps = outputs[index].detach().cpu().numpy()
                    save_validation_overlays(image_path=Path(image_path), output_dir=validation_output_path, output_stem=output_stem,
                                             target_points=target_points, predicted_points=predicted_points, predicted_heatmaps=predicted_heatmaps)

                progress_bar.update()

        staging_output_paths = self.write_validation_outputs(validation_output_path=validation_output_path, image_summary_rows=image_summary_rows,
                                                             endpoint_rows=endpoint_rows, prediction_rows=prediction_rows)
        output_paths = {name: final_validation_output_path / path.relative_to(validation_output_path)
                        for name, path in staging_output_paths.items()}
        self.validation_metadata_context = {
            'validation_output_path': validation_output_path,
            'checkpoint_path': checkpoint_path,
            'checkpoint_type': checkpoint_type,
            'checkpoint_payload': checkpoint_payload,
            'image_summary_rows': image_summary_rows,
            'endpoint_rows': endpoint_rows,
            'output_paths': output_paths,
        }
        print(f"\tValidation summary staged for {output_paths['validation_summary_xlsx']}", flush=True)
        return output_paths

    def build_data_loaders(self):
        """Build train and validation data loaders after resolving input channels."""
        training_dataset = HeatmapDataset(self.build_dataset_config(split_name='training'))
        validation_dataset = HeatmapDataset(self.build_dataset_config(split_name='validation'))
        self.validate_dataset_membership(training_dataset=training_dataset, validation_dataset=validation_dataset)
        self.resolve_input_channels(training_dataset=training_dataset, validation_dataset=validation_dataset)
        self.configure_input_normalisation(training_dataset=training_dataset, validation_dataset=validation_dataset)
        validated_training_records = training_dataset.validate_all_records()
        validated_validation_records = validation_dataset.validate_all_records()
        self.training_generator = torch.Generator()
        self.validation_generator = torch.Generator()
        self.training_generator.manual_seed(int(self.train_config.random_seed))
        self.validation_generator.manual_seed(int(self.train_config.random_seed))
        print(f'\tDataset validation complete: {validated_training_records} training and {validated_validation_records} validation records.', flush=True)
        print(f'\tTraining samples: {len(training_dataset)} ({len(training_dataset.records)} original, '
              f'oversampling_factor={training_dataset.oversampling_factor}).', flush=True)
        print(f'\tValidation samples: {len(validation_dataset)}.', flush=True)
        training_loader = DataLoader(training_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers,
                                     pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=self.training_generator)
        validation_loader = DataLoader(validation_dataset, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.num_workers,
                                       pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=self.validation_generator)
        return training_loader, validation_loader

    def validate_dataset_membership(self, training_dataset, validation_dataset):
        """Require every selected training and validation sample to have an annotation record."""
        annotation_samples = set(training_dataset.mark_records)
        listed_samples = {record['sample_name'] for record in training_dataset.records} | {record['sample_name'] for record in validation_dataset.records}
        unexpected_in_lists = sorted(listed_samples - annotation_samples)

        if not unexpected_in_lists:
            return

        raise ValueError(
            f'Dataset validation failed for repetition {self.data_config.repetition}, fold {self.data_config.fold}: training and validation lists contain '
            f'sample IDs that are not present in the annotation file: {unexpected_in_lists}. Annotation-only samples may be deliberately held out, but '
            f'every listed sample must have an annotation. Training cancelled; existing outputs were not removed.'
        )

    def resolve_input_channels(self, training_dataset, validation_dataset):
        """Automatically resolve and validate image input channels before model construction."""
        phase_counts = {'training': self.infer_dataset_channel_counts(training_dataset),
                        'validation': self.infer_dataset_channel_counts(validation_dataset)}
        unique_source_channels = sorted({channel_count for counts in phase_counts.values() for channel_count in counts})
        counts_text = ', '.join(f'{phase}: {counts}' for phase, counts in phase_counts.items())

        if not unique_source_channels:
            raise ValueError('No source images were available for input-channel detection.')

        unsupported_sources = [channel_count for channel_count in unique_source_channels if channel_count not in (1, 3, 4)]

        if unsupported_sources:
            raise ValueError(f'Unsupported source channel count(s): {unsupported_sources}. Supported source images are greyscale, RGB, and RGBA.')

        if len(unique_source_channels) != 1:
            raise ValueError(
                f'Input-channel mismatch detected across train/validation images: {counts_text}. All images for a task must have the same number of source channels.')

        resolved_channels = int(unique_source_channels[0])
        self.data_config.input_channels = resolved_channels
        training_dataset.config.input_channels = resolved_channels
        validation_dataset.config.input_channels = resolved_channels

        print(f'\tAutomatically detected {resolved_channels} input channel(s). Source channel counts: {counts_text}.', flush=True)
        return resolved_channels

    def configure_input_normalisation(self, training_dataset, validation_dataset):
        """Calculate and attach three-channel statistics from original training images only."""
        if not self.data_config.normalise_inputs:
            self.data_config.normalisation_mean = None
            self.data_config.normalisation_std = None
            training_dataset.config.normalisation_mean = None
            training_dataset.config.normalisation_std = None
            validation_dataset.config.normalisation_mean = None
            validation_dataset.config.normalisation_std = None
            print('\tInput normalisation disabled.', flush=True)
            return

        if int(self.data_config.input_channels) != EXPECTED_NORMALISATION_CHANNELS:
            raise ValueError(
                f'Input normalisation requires exactly {EXPECTED_NORMALISATION_CHANNELS} channels so each RGB channel remains distinct; '
                f'the training data contains {self.data_config.input_channels} channel(s).'
            )

        mean, standard_deviation = training_dataset.calculate_normalisation_statistics()
        mean, standard_deviation = validate_normalisation_constants(mean, standard_deviation)
        self.data_config.normalisation_mean = tuple(mean)
        self.data_config.normalisation_std = tuple(standard_deviation)

        for dataset in (training_dataset, validation_dataset):
            dataset.config.normalisation_mean = self.data_config.normalisation_mean
            dataset.config.normalisation_std = self.data_config.normalisation_std

        print(f'\tInput normalisation enabled from training split: mean={list(mean)}, std={list(standard_deviation)}.', flush=True)

    @staticmethod
    def infer_dataset_channel_counts(dataset):
        """Return a count of source image channel counts for a dataset split."""
        channel_counts = {}

        for record in dataset.records:
            channel_count = infer_image_channel_count(record['image_path'])
            channel_counts[channel_count] = channel_counts.get(channel_count, 0) + 1

        if not channel_counts:
            raise ValueError(f'No image records found for split {dataset.config.split_name}.')

        return channel_counts

    def build_dataset_config(self, split_name):
        """Build one dataset configuration."""
        return HeatmapDatasetConfig(repetition=self.data_config.repetition, fold=self.data_config.fold, split_name=split_name,
                                    num_of_points=self.data_config.num_of_points,
                                    fold_lists_path=self.data_config.fold_lists_path, mark_list_file=self.data_config.mark_list_file,
                                    image_data_dir=self.data_config.image_data_dir, image_size=self.data_config.image_size, heatmap_sigma=self.data_config.heatmap_sigma,
                                    input_channels=self.data_config.input_channels, recursive_image_search=self.data_config.recursive_image_search,
                                    oversampling_factor=self.data_config.oversampling_factor,
                                    normalisation_mean=self.data_config.normalisation_mean,
                                    normalisation_std=self.data_config.normalisation_std)

    def build_model(self):
        """Build the configured heatmap model."""
        if self.data_config.input_channels is None:
            raise ValueError('input_channels has not been resolved. Build data loaders before constructing the model.')

        model_kwargs = get_model_kwargs(self.model_config.network_name, self.model_config)
        model = build_heatmap_model(network_name=self.model_config.network_name, num_of_points=self.data_config.num_of_points,
                                    input_channels=int(self.data_config.input_channels), image_size=self.data_config.image_size, **model_kwargs)
        return model.to(self.device)

    def calculate_model_loss(self, outputs, auxiliary_outputs, targets, criterion):
        """Calculate final and optional intermediate-supervision losses."""
        loss = criterion(outputs, targets)

        if auxiliary_outputs and float(self.model_config.auxiliary_loss_weight) > 0:
            auxiliary_loss = torch.stack([criterion(auxiliary_output, targets) for auxiliary_output in auxiliary_outputs]).mean()
            loss = loss + float(self.model_config.auxiliary_loss_weight) * auxiliary_loss

        return loss

    def build_criterion(self):
        """Build the requested loss function."""
        loss_name = str(self.train_config.loss_name).lower()

        if loss_name == 'mse':
            return nn.MSELoss()

        if loss_name == 'weighted_mse':
            return WeightedMSELoss(positive_weight=self.train_config.positive_weight)

        if loss_name == 'smooth_l1':
            return nn.SmoothL1Loss()

        if loss_name == 'bce_logits':
            return nn.BCEWithLogitsLoss()

        raise ValueError(f'Unknown loss_name: {self.train_config.loss_name}')

    def build_optimiser(self, model):
        """Build the requested optimiser."""
        optimiser_name = str(self.train_config.optimiser_name).lower()

        if optimiser_name == 'adamw':
            return AdamW(model.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)

        if optimiser_name == 'sgd':
            return SGD(model.parameters(), lr=self.train_config.learning_rate, momentum=self.train_config.momentum, weight_decay=self.train_config.weight_decay)

        raise ValueError(f'Unknown optimiser_name: {self.train_config.optimiser_name}')

    def build_scheduler(self, optimiser):
        """Build the learning-rate scheduler."""
        schedule = str(self.train_config.lr_schedule).lower()

        if schedule == 'none':
            return None

        if schedule == 'step':
            return StepLR(optimiser, step_size=self.train_config.lr_step_size, gamma=self.train_config.lr_gamma)

        if schedule == 'plateau':
            return ReduceLROnPlateau(optimiser, mode='min', factor=self.train_config.lr_gamma, patience=5)

        raise ValueError(f'Unknown lr_schedule: {self.train_config.lr_schedule}')

    def calculate_batch_error(self, outputs, points_original, original_size):
        """Calculate endpoint error in original image pixels."""
        predicted_resized = heatmaps_to_points(outputs)
        predicted_original = scale_points_to_original(points=predicted_resized, original_sizes=original_size, image_size=self.data_config.image_size)
        return torch.linalg.norm(predicted_original - points_original, dim=2)

    def format_metrics(self, loss, error_px):
        """Return loss and pixel endpoint error."""
        return {'loss': float(loss), 'error_px': float(error_px)}

    @staticmethod
    def validate_finite_metrics(phase, metrics):
        """Stop training when a reported metric is NaN or infinite."""
        invalid_metrics = {name: value for name, value in metrics.items() if not np.isfinite(value)}

        if invalid_metrics:
            raise FloatingPointError(f'Non-finite {phase} metric(s) detected: {invalid_metrics}')

    def save_checkpoint(self, model, optimiser, scheduler, scaler, checkpoint_type, epoch, validation_metrics, training_state,
                        resume_signature, best_model_state_dict):
        """Atomically save a model checkpoint and, for ``last_epoch``, complete resume state."""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type)
        labelled_validation_metrics = self.label_validation_metrics(validation_metrics)
        payload = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'schema': 'heatmap_training_checkpoint',
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': utc_now_iso(),
            'epoch': int(epoch),
            'next_epoch': int(epoch) + 1,
            'checkpoint_type': checkpoint_type,
            'resume_capable': checkpoint_type == 'last_epoch',
            'state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'validation_metrics': labelled_validation_metrics,
            'metadata': self.build_checkpoint_metadata(checkpoint_type=checkpoint_type, epoch=epoch,
                                                       validation_metrics=validation_metrics),
        }

        if checkpoint_type == 'last_epoch':
            if training_state is None or best_model_state_dict is None:
                raise ValueError('A last-epoch checkpoint requires complete training state and the best-model snapshot.')

            payload.update({
                'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                'grad_scaler_state_dict': scaler.state_dict(),
                'training_state': training_state,
                'rng_state': self.capture_rng_state(),
                'data_loader_generator_states': self.capture_data_loader_generator_states(),
                'best_model_state_dict': best_model_state_dict,
                'resume_signature': resume_signature,
            })

        self.atomic_torch_save(payload=payload, checkpoint_path=checkpoint_path)
        return checkpoint_path

    def load_checkpoint_state(self, model, checkpoint_path):
        """Load checkpoint weights into a model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None

        if state_dict is None:
            raise ValueError(f'Checkpoint does not contain a state_dict: {checkpoint_path}')

        model.load_state_dict(state_dict)
        model.eval()
        return checkpoint

    def write_checkpoint_summary(self, best_epoch, last_epoch, best_validation_metrics, last_validation_metrics, best_checkpoint_path, last_checkpoint_path,
                                 validation_output_paths):
        """Write validation-based checkpoint-selection and run metadata."""
        validation_summary_path = None if validation_output_paths is None else validation_output_paths.get('validation_summary_xlsx')
        validation_predictions_path = None if validation_output_paths is None else validation_output_paths.get('validation_predictions_csv')
        summary = {'format_version': CHECKPOINT_FORMAT_VERSION, 'schema': 'heatmap_validation_checkpoint_summary',
                   'schema_version': CHECKPOINT_SCHEMA_VERSION, 'created_at': utc_now_iso(),
                   'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold), 'task_name': self.data_config.task_name,
                   'num_of_points': int(self.data_config.num_of_points), 'checkpoints': {
                'best_validation_loss': self.build_checkpoint_descriptor(path=best_checkpoint_path, checkpoint_type='best_validation_loss',
                                                                         epoch=best_epoch, validation_metrics=best_validation_metrics),
                'last_epoch': self.build_checkpoint_descriptor(path=last_checkpoint_path, checkpoint_type='last_epoch',
                                                                epoch=last_epoch, validation_metrics=last_validation_metrics)},
                   'termination_reason': self.termination_reason,
                   'timing': self.get_timing_summary(),
                   'runtime_environment': self.runtime_metadata,
                   'validation_summary_path': str(validation_summary_path) if validation_summary_path is not None else None,
                   'validation_predictions_path': str(validation_predictions_path) if validation_predictions_path is not None else None,
                   'metadata': self.build_run_metadata()}

        with open(self.get_checkpoint_summary_path(), 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=4, default=str)

    def build_run_metadata(self):
        """Build common run metadata without a misleading null checkpoint descriptor."""
        data_config = self.serialise(asdict(self.data_config))
        train_config = self.serialise(asdict(self.train_config))
        model_config = self.serialise(asdict(self.model_config))
        image_height, image_width = [int(value) for value in self.data_config.image_size]
        input_channels = None if self.data_config.input_channels is None else int(self.data_config.input_channels)
        model_init_config = self.serialise(get_model_kwargs(self.model_config.network_name, self.model_config))
        model_init_args = {'num_of_points': int(self.data_config.num_of_points), 'input_channels': input_channels, **model_init_config}

        if self.model_config.network_name == 'vitpose':
            model_init_args['image_size'] = [image_height, image_width]

        registry_entry = get_model_registry_entry(self.model_config.network_name)

        return {
            'schema': CHECKPOINT_SCHEMA_NAME,
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': utc_now_iso(),
            'task': {'name': self.data_config.task_name, 'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
                     'num_points': int(self.data_config.num_of_points),
                     'output_heads': int(self.data_config.num_of_points), 'prediction_type': 'landmark_heatmap_regression'},
            'model': {'registry_name': self.model_config.network_name, 'module': registry_entry['module'], 'class_name': registry_entry['class_name'],
                      'init_args': model_init_args},
            'data': {'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
                     'fold_lists_path': str(self.data_config.fold_lists_path),
                     'mark_list_file': str(self.data_config.mark_list_file), 'image_data_dir': str(self.data_config.image_data_dir),
                     'recursive_image_search': bool(self.data_config.recursive_image_search), 'input_channels': input_channels},
            'preprocessing': {'image_size': {'height': image_height, 'width': image_width}, 'heatmap_sigma': float(self.data_config.heatmap_sigma),
                              'input_channels': input_channels, 'tensor_shape': ['batch', input_channels, image_height, image_width],
                              'channel_order': 'channels_first', 'loaded_image_value_range': 'float32_0_to_1',
                              'model_input_values': ('three_channel_standardised' if self.data_config.normalise_inputs else 'float32_0_to_1'),
                              'normalisation': self.build_normalisation_metadata(),
                              'resize': {'library': 'cv2.resize', 'interpolation': 'INTER_AREA'},
                              'target_heatmaps': {'channels': int(self.data_config.num_of_points), 'generation': 'normalised_gaussian_per_landmark'}},
            'inference': {'heatmap_to_point': 'argmax', 'output_coordinate_space': 'original_image_pixels', 'scale_back_to_original': True,
                          'resized_coordinate_space': {'height': image_height, 'width': image_width}},
            'augmentation': self.build_augmentation_metadata(),
            'training': {'train_config': train_config, 'optimiser': self.train_config.optimiser_name, 'loss': self.train_config.loss_name,
                         'random_seed': int(self.train_config.random_seed),
                         'resume': {'supported_from': 'model_last_epoch.pth', 'epoch_boundary_only': True},
                         'auxiliary_loss_weight': float(self.model_config.auxiliary_loss_weight) if self.model_config.network_name == 'stacked_hourglass' else None},
            'runtime_environment': self.runtime_metadata,
            'timing': self.get_timing_summary(),
            'raw_configs': {'data_config': data_config, 'train_config': train_config, 'model_config': model_config}
        }

    def build_normalisation_metadata(self):
        """Return the exact three-channel input normalisation contract."""
        return {
            'enabled': bool(self.data_config.normalise_inputs),
            'channels': EXPECTED_NORMALISATION_CHANNELS,
            'mean': None if self.data_config.normalisation_mean is None else list(self.data_config.normalisation_mean),
            'standard_deviation': None if self.data_config.normalisation_std is None else list(self.data_config.normalisation_std),
            'source': 'training_split_images' if self.data_config.normalise_inputs else 'disabled',
            'statistic': 'population',
            'calculated_from': 'training_split_only' if self.data_config.normalise_inputs else None,
            'calculation_inputs': 'unaugmented_float32_0_to_1_training_images_after_resize',
        }

    def build_checkpoint_metadata(self, checkpoint_type, epoch, validation_metrics):
        """Build metadata for one concrete checkpoint; all checkpoint fields are required."""
        if checkpoint_type is None or epoch is None or validation_metrics is None:
            raise ValueError('Checkpoint metadata requires checkpoint_type, epoch, and validation_metrics.')

        metadata = self.build_run_metadata()
        metadata['checkpoint'] = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'type': checkpoint_type,
            'epoch': int(epoch),
            'validation_metrics': self.label_validation_metrics(validation_metrics),
        }
        return metadata

    @staticmethod
    def build_checkpoint_descriptor(path, checkpoint_type, epoch, validation_metrics):
        """Return one populated JSON checkpoint descriptor."""
        labelled_metrics = TrainModel.label_validation_metrics(validation_metrics)
        return {
            'path': str(path) if path is not None else None,
            'type': checkpoint_type,
            'epoch': None if epoch is None else int(epoch),
            **({} if labelled_metrics is None else labelled_metrics),
        }

    def load_training_checkpoint(self, model, optimiser, scheduler, scaler, resume_signature):
        """Validate and restore the last completed epoch without mutating saved outputs first."""
        checkpoint_path = self.get_checkpoint_path('last_epoch')

        if not checkpoint_path.is_file():
            raise ValueError(f'Resume requested, but the last-epoch checkpoint does not exist: {checkpoint_path}')

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as error:
            raise ValueError(f'Resume checkpoint could not be loaded and existing outputs were left untouched: {checkpoint_path}') from error

        self.validate_resume_checkpoint(checkpoint=checkpoint, checkpoint_path=checkpoint_path, resume_signature=resume_signature)
        self.remove_stale_checkpoint_temps()
        training_state = checkpoint['training_state']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        checkpoint_scheduler_state = checkpoint['scheduler_state_dict']

        if scheduler is None:
            if checkpoint_scheduler_state is not None:
                raise ValueError('Resume checkpoint contains scheduler state, but the current run has no scheduler.')
        elif checkpoint_scheduler_state is None:
            raise ValueError('Resume checkpoint has no scheduler state, but the current run requires a scheduler.')
        else:
            scheduler.load_state_dict(checkpoint_scheduler_state)

        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        self.training_sessions = copy.deepcopy(training_state['training_sessions'])
        self.restore_data_loader_generator_states(checkpoint['data_loader_generator_states'])
        self.restore_rng_state(checkpoint['rng_state'])
        return {
            'completed_epoch': int(training_state['completed_epoch']),
            'history': copy.deepcopy(training_state['history']),
            'best_epoch': int(training_state['best_epoch']),
            'best_validation_metrics': self.unlabel_validation_metrics(training_state['best_validation_metrics']),
            'early_stop_best_validation_loss': float(training_state['early_stop_best_validation_loss']),
            'bad_epochs': int(training_state['bad_epochs']),
            'last_validation_metrics': self.unlabel_validation_metrics(training_state['last_validation_metrics']),
            'best_model_state_dict': checkpoint['best_model_state_dict'],
            'termination_reason': training_state['termination_reason'],
        }

    def validate_resume_checkpoint(self, checkpoint, checkpoint_path, resume_signature):
        """Reject incomplete, completed, or configuration-incompatible resume checkpoints."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f'Resume checkpoint is not a structured Heatmaps checkpoint: {checkpoint_path}')

        if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION or checkpoint.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f'Resume requires checkpoint format/schema version {CHECKPOINT_FORMAT_VERSION}. '
                f'Checkpoint {checkpoint_path} has format={checkpoint.get("format_version")}, schema={checkpoint.get("schema_version")}. '
                'Only checkpoints produced by the current 0.1 contract are supported.'
            )

        if checkpoint.get('schema') != 'heatmap_training_checkpoint' or checkpoint.get('checkpoint_type') != 'last_epoch':
            raise ValueError(f'Resume requires model_last_epoch.pth from this run: {checkpoint_path}')

        if not checkpoint.get('resume_capable'):
            raise ValueError(f'Checkpoint is not marked as resume-capable: {checkpoint_path}')

        required_fields = {
            'epoch', 'next_epoch', 'validation_metrics',
            'state_dict', 'optimiser_state_dict', 'scheduler_state_dict', 'grad_scaler_state_dict', 'training_state',
            'rng_state', 'data_loader_generator_states', 'best_model_state_dict', 'resume_signature',
        }
        missing_fields = sorted(required_fields - set(checkpoint))

        if missing_fields:
            raise ValueError(f'Resume checkpoint is incomplete; missing fields: {missing_fields}. Existing outputs were left untouched.')

        saved_signature = checkpoint['resume_signature']

        if not isinstance(saved_signature, dict) or saved_signature.get('algorithm') != 'sha256' or not isinstance(saved_signature.get('payload'), dict):
            raise ValueError('Resume checkpoint has an invalid compatibility signature structure.')

        saved_canonical_json = json.dumps(saved_signature['payload'], sort_keys=True, separators=(',', ':'))
        recomputed_saved_digest = hashlib.sha256(saved_canonical_json.encode('utf-8')).hexdigest()

        if saved_signature.get('sha256') != recomputed_saved_digest:
            raise ValueError('Resume checkpoint compatibility signature is internally inconsistent or corrupted.')

        if saved_signature.get('sha256') != resume_signature.get('sha256'):
            raise ValueError(
                'Resume checkpoint is incompatible with the current task, repetition/fold, dataset lists, annotation file, model, or training settings. '
                f'Saved signature: {saved_signature.get("sha256")}; current signature: {resume_signature.get("sha256")}. '
                'Use exactly the original run settings. Existing outputs were left untouched.'
            )

        training_state = checkpoint['training_state']
        required_training_fields = {
            'completed_epoch', 'next_epoch', 'history', 'best_epoch', 'best_validation_metrics',
            'early_stop_best_validation_loss', 'bad_epochs', 'last_validation_metrics', 'termination_reason', 'training_sessions',
        }
        missing_training_fields = sorted(required_training_fields - set(training_state))

        if missing_training_fields:
            raise ValueError(f'Resume checkpoint training state is incomplete; missing fields: {missing_training_fields}.')

        completed_epoch = int(training_state['completed_epoch'])

        if int(checkpoint['epoch']) != completed_epoch:
            raise ValueError('Resume checkpoint epoch does not match training_state.completed_epoch.')

        if int(checkpoint['next_epoch']) != completed_epoch + 1:
            raise ValueError('Resume checkpoint top-level next_epoch is inconsistent with its completed epoch.')

        if int(training_state['next_epoch']) != completed_epoch + 1:
            raise ValueError('Resume checkpoint has an inconsistent next_epoch value.')

        if checkpoint['validation_metrics'] != training_state['last_validation_metrics']:
            raise ValueError('Resume checkpoint top-level validation metrics do not match the saved last-epoch training state.')

        best_epoch = int(training_state['best_epoch'])

        if best_epoch < 1 or best_epoch > completed_epoch:
            raise ValueError(f'Resume checkpoint best_epoch must be between 1 and {completed_epoch}; got {best_epoch}.')

        self.validate_state_dict_snapshot(checkpoint['state_dict'], checkpoint['best_model_state_dict'])

        allowed_termination_reasons = {'in_progress', 'interrupted', 'exception', 'early_stopping', 'max_epochs_reached'}

        if training_state['termination_reason'] not in allowed_termination_reasons:
            raise ValueError(f'Resume checkpoint has an unknown termination_reason: {training_state["termination_reason"]!r}.')

        if training_state['termination_reason'] not in ('early_stopping', 'max_epochs_reached') and completed_epoch >= int(self.train_config.max_training_epochs):
            raise ValueError(
                f'Resume checkpoint completed epoch {completed_epoch}, but max_training_epochs is {self.train_config.max_training_epochs}. '
                'There is no remaining epoch to run.'
            )

        self.validate_history(training_state['history'], completed_epoch=completed_epoch)

    def ensure_best_checkpoint(self, best_checkpoint_path, best_epoch, best_validation_metrics, best_model_state_dict):
        """Restore the committed best checkpoint if an interruption left its sibling file ahead or missing."""
        checkpoint_is_committed = False

        if best_checkpoint_path.is_file():
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_is_committed = (
                    checkpoint.get('checkpoint_type') == 'best_validation_loss'
                    and int(checkpoint.get('epoch', -1)) == int(best_epoch)
                    and checkpoint.get('validation_metrics') == self.label_validation_metrics(best_validation_metrics)
                    and self.state_dicts_equal(checkpoint.get('state_dict'), best_model_state_dict)
                )
            except Exception:
                checkpoint_is_committed = False

        if checkpoint_is_committed:
            return

        recovered_payload = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'schema': 'heatmap_training_checkpoint',
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'created_at': utc_now_iso(),
            'epoch': int(best_epoch),
            'next_epoch': int(best_epoch) + 1,
            'checkpoint_type': 'best_validation_loss',
            'resume_capable': False,
            'state_dict': best_model_state_dict,
            'optimiser_state_dict': None,
            'validation_metrics': self.label_validation_metrics(best_validation_metrics),
            'metadata': self.build_checkpoint_metadata(checkpoint_type='best_validation_loss', epoch=best_epoch,
                                                       validation_metrics=best_validation_metrics),
            'recovered_from_last_epoch_checkpoint': True,
        }
        self.atomic_torch_save(payload=recovered_payload, checkpoint_path=best_checkpoint_path)

    @staticmethod
    def validate_state_dict_snapshot(current_state_dict, best_state_dict):
        """Validate that the embedded best snapshot has the same tensor contract as the current model."""
        if not isinstance(current_state_dict, dict) or not isinstance(best_state_dict, dict):
            raise ValueError('Resume checkpoint model state and embedded best-model snapshot must be dictionaries.')

        if set(current_state_dict) != set(best_state_dict):
            raise ValueError('Resume checkpoint embedded best-model snapshot has different parameter keys from the current model state.')

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

            if torch.is_tensor(first_value) and torch.is_tensor(second_value):
                if first_value.shape != second_value.shape or first_value.dtype != second_value.dtype or not torch.equal(first_value.cpu(), second_value.cpu()):
                    return False
            elif first_value != second_value:
                return False

        return True

    def build_resume_signature(self, training_loader, validation_loader):
        """Hash every trajectory-defining configuration and selected annotation/list input."""
        training_list = get_split_file_path(self.data_config.fold_lists_path, self.data_config.repetition, 'training', self.data_config.fold)
        validation_list = get_split_file_path(self.data_config.fold_lists_path, self.data_config.repetition, 'validation', self.data_config.fold)
        payload = {
            'task': {
                'name': self.data_config.task_name,
                'repetition': int(self.data_config.repetition),
                'fold': normalise_fold(self.data_config.fold),
                'num_points': int(self.data_config.num_of_points),
            },
            'data': {
                'fold_collection_sha256': self.data_config.fold_collection_sha256,
                'training_list_path': str(Path(training_list).resolve()),
                'training_list_sha256': self.sha256_file(training_list),
                'validation_list_path': str(Path(validation_list).resolve()),
                'validation_list_sha256': self.sha256_file(validation_list),
                'mark_list_path': str(Path(self.data_config.mark_list_file).resolve()),
                'mark_list_sha256': self.sha256_file(self.data_config.mark_list_file),
                'image_data_dir': str(Path(self.data_config.image_data_dir).resolve()),
                'training_images_sha256': self.sha256_dataset_images(training_loader.dataset),
                'validation_images_sha256': self.sha256_dataset_images(validation_loader.dataset),
                'image_size': [int(value) for value in self.data_config.image_size],
                'heatmap_sigma': float(self.data_config.heatmap_sigma),
                'input_channels': int(self.data_config.input_channels),
                'recursive_image_search': bool(self.data_config.recursive_image_search),
                'oversampling_factor': int(self.data_config.oversampling_factor),
                'normalisation': self.build_normalisation_metadata(),
            },
            'training': self.serialise(asdict(self.train_config)),
            'model': {
                'network_name': self.model_config.network_name,
                'init_args': self.serialise(get_model_kwargs(self.model_config.network_name, self.model_config)),
                'auxiliary_loss_weight': float(self.model_config.auxiliary_loss_weight),
            },
            'implementation': {
                'heatmaps_source_sha256': self.sha256_python_sources(),
                'framework_version': self.runtime_metadata['framework']['version'],
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
        return {'algorithm': 'sha256', 'sha256': hashlib.sha256(canonical_json.encode('utf-8')).hexdigest(), 'payload': serialised_payload}

    @staticmethod
    def sha256_file(path):
        """Return the SHA-256 digest of one input file."""
        digest = hashlib.sha256()

        with open(path, 'rb') as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b''):
                digest.update(block)

        return digest.hexdigest()

    @staticmethod
    def sha256_dataset_images(dataset):
        """Hash selected image identities and bytes in split-list order."""
        digest = hashlib.sha256()

        for record in dataset.records:
            image_path = Path(record['image_path']).resolve()
            digest.update(str(record['sample_name']).encode('utf-8'))
            digest.update(b'\0')
            digest.update(str(image_path).encode('utf-8'))
            digest.update(b'\0')

            with open(image_path, 'rb') as image_file:
                for block in iter(lambda: image_file.read(1024 * 1024), b''):
                    digest.update(block)

            digest.update(b'\0')

        return digest.hexdigest()

    @staticmethod
    def sha256_python_sources():
        """Hash the active Heatmaps Python implementation, including uncommitted changes."""
        package_root = Path(__file__).resolve().parent
        digest = hashlib.sha256()

        for source_path in sorted(package_root.rglob('*.py'), key=lambda path: path.relative_to(package_root).as_posix()):
            digest.update(source_path.relative_to(package_root).as_posix().encode('utf-8'))
            digest.update(b'\0')
            digest.update(source_path.read_bytes())
            digest.update(b'\0')

        return digest.hexdigest()

    @staticmethod
    def atomic_torch_save(payload, checkpoint_path):
        """Replace a checkpoint atomically so a failed save preserves the previous committed file."""
        checkpoint_path = Path(checkpoint_path)
        temporary_path = checkpoint_path.with_name(f'.{checkpoint_path.name}.tmp')

        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def remove_stale_checkpoint_temps(self):
        """Discard uncommitted temporary siblings after the committed last checkpoint validates."""
        for checkpoint_type in ('best_validation_loss', 'last_epoch'):
            checkpoint_path = self.get_checkpoint_path(checkpoint_type)
            temporary_path = checkpoint_path.with_name(f'.{checkpoint_path.name}.tmp')

            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def clone_state_dict_to_cpu(state_dict):
        """Keep a self-contained CPU snapshot of the best model inside the last checkpoint."""
        return {name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value) for name, value in state_dict.items()}

    def build_training_state(self, completed_epoch, history, best_epoch, best_validation_metrics, early_stop_best_validation_loss,
                             bad_epochs, last_validation_metrics, termination_reason):
        """Build the loop/control state required to continue at the next epoch."""
        return {
            'completed_epoch': int(completed_epoch),
            'next_epoch': int(completed_epoch) + 1,
            'history': copy.deepcopy(history),
            'best_epoch': int(best_epoch),
            'best_validation_metrics': self.label_validation_metrics(best_validation_metrics),
            'early_stop_best_validation_loss': float(early_stop_best_validation_loss),
            'bad_epochs': int(bad_epochs),
            'last_validation_metrics': self.label_validation_metrics(last_validation_metrics),
            'termination_reason': termination_reason,
            'training_sessions': self.get_training_sessions_snapshot(),
        }

    @staticmethod
    def unlabel_validation_metrics(validation_metrics):
        """Convert externally labelled validation metrics back to loop metric names."""
        return {
            'loss': float(validation_metrics['validation_loss']),
            'error_px': float(validation_metrics['validation_error_px']),
        }

    def capture_rng_state(self):
        """Capture Python, NumPy, torch CPU, and every available CUDA RNG."""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda_all': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
        }

    def restore_rng_state(self, rng_state):
        """Restore all process RNGs after model and optimiser reconstruction."""
        required_fields = {'python', 'numpy', 'torch_cpu', 'torch_cuda_all'}
        missing_fields = sorted(required_fields - set(rng_state))

        if missing_fields:
            raise ValueError(f'Resume checkpoint RNG state is incomplete; missing fields: {missing_fields}.')

        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch_cpu'].cpu())
        cuda_states = rng_state['torch_cuda_all']

        if cuda_states is not None:
            if self.device.type != 'cuda' or not torch.cuda.is_available():
                raise ValueError('Resume checkpoint contains CUDA RNG state, but CUDA is unavailable.')

            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    f'Resume checkpoint contains RNG state for {len(cuda_states)} CUDA device(s), but {torch.cuda.device_count()} are available.'
                )

            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    def capture_data_loader_generator_states(self):
        """Capture shuffle and worker-seeding generators at the completed epoch boundary."""
        if self.training_generator is None or self.validation_generator is None:
            raise RuntimeError('DataLoader generators have not been initialised.')

        return {
            'training': self.training_generator.get_state(),
            'validation': self.validation_generator.get_state(),
        }

    def restore_data_loader_generator_states(self, generator_states):
        """Restore DataLoader shuffle and worker-seeding generators."""
        if self.training_generator is None or self.validation_generator is None:
            raise RuntimeError('DataLoader generators have not been initialised.')

        if set(generator_states) != {'training', 'validation'}:
            raise ValueError('Resume checkpoint must contain training and validation DataLoader generator states.')

        self.training_generator.set_state(generator_states['training'].cpu())
        self.validation_generator.set_state(generator_states['validation'].cpu())

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
        """Refresh the current session record for metadata and checkpointing."""
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

    def finish_workflow(self):
        """Freeze completion timestamps and wall-clock duration for the current invocation."""
        if self.workflow_completed_at is None:
            self.workflow_completed_at = utc_now_iso()

        if self.workflow_duration_seconds is None and self.workflow_start_perf is not None:
            self.workflow_duration_seconds = float(time.perf_counter() - self.workflow_start_perf)

        self.update_current_training_session(status=self.training_status)
        self.current_session_start_perf = None

    def get_timing_summary(self):
        """Return persisted run, session, and per-phase timing totals in seconds."""
        history = self.history or self.empty_history()
        workflow_duration = self.workflow_duration_seconds

        if workflow_duration is None and self.workflow_start_perf is not None:
            workflow_duration = float(time.perf_counter() - self.workflow_start_perf)

        return {
            'definitions': {
                'training_duration_seconds': 'Synchronised wall time spent in the training pass.',
                'validation_duration_seconds': 'Synchronised wall time spent in the validation pass.',
                'epoch_duration_seconds': 'Wall time from epoch start through validation and scheduler/control updates; checkpoint, plot, and CSV writes are excluded.',
                'workflow_duration_seconds': 'Wall time for this invocation through training and optional validation export; final summary/run-info writes are excluded.',
            },
            'workflow_started_at': self.workflow_started_at,
            'workflow_completed_at': self.workflow_completed_at,
            'workflow_duration_seconds': workflow_duration,
            'dataset_validation_duration_seconds': float(self.dataset_validation_duration_seconds),
            'model_setup_and_resume_duration_seconds': float(self.model_setup_duration_seconds),
            'validation_export_duration_seconds': float(self.validation_export_duration_seconds),
            'recorded_epoch_count': len(history.get('epoch', [])),
            'cumulative_training_duration_seconds': float(sum(history.get('training_duration_seconds', []))),
            'cumulative_validation_duration_seconds': float(sum(history.get('validation_duration_seconds', []))),
            'cumulative_epoch_duration_seconds': float(sum(history.get('epoch_duration_seconds', []))),
            'sessions': self.get_training_sessions_snapshot(),
        }

    def get_run_report(self):
        """Return pipeline-level status, timing, resume, and environment metadata."""
        return {
            'status': self.training_status,
            'termination_reason': self.termination_reason,
            'failure': self.failure,
            'resume_training': self.resume_training,
            'resume_checkpoint_path': str(self.get_checkpoint_path('last_epoch')) if self.resume_training else None,
            'resume_state_validated': self.resume_state_validated,
            'runtime_environment': self.runtime_metadata,
            'timing': self.get_timing_summary(),
        }

    def synchronise_device(self):
        """Synchronise CUDA before wall-clock phase timing."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def label_validation_metrics(validation_metrics):
        """Return externally stored validation metrics with unambiguous names."""
        if validation_metrics is None:
            return None

        if 'validation_loss' in validation_metrics:
            return {'validation_loss': float(validation_metrics['validation_loss']),
                    'validation_error_px': float(validation_metrics['validation_error_px'])}

        return {'validation_loss': float(validation_metrics['loss']), 'validation_error_px': float(validation_metrics['error_px'])}

    def build_augmentation_metadata(self):
        """Return the oversampling and augmentation policy stored in checkpoints."""
        oversampling_factor = int(self.data_config.oversampling_factor)
        return {'enabled': oversampling_factor > 1, 'oversampling_factor': oversampling_factor, 'applies_to': 'training split only',
                'policy': get_augmentation_policy()}

    @staticmethod
    def serialise(value):
        """Convert paths and nested values to serialisable objects."""
        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {key: TrainModel.serialise(item) for key, item in value.items()}

        if isinstance(value, tuple):
            return [TrainModel.serialise(item) for item in value]

        if isinstance(value, list):
            return [TrainModel.serialise(item) for item in value]

        return value

    def validate_configs(self):
        """Validate core configuration values."""
        if int(self.data_config.repetition) < 1:
            raise ValueError(f'repetition must be at least 1. Got: {self.data_config.repetition}')

        self.data_config.fold = normalise_fold(self.data_config.fold)

        if int(self.data_config.num_of_points) < MIN_POINTS_PER_IMAGE or int(self.data_config.num_of_points) > MAX_POINTS_PER_IMAGE:
            raise ValueError(f'num_of_points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}. Got: {self.data_config.num_of_points}')

        if len(tuple(self.data_config.image_size)) != 2:
            raise ValueError('image_size must be a two-item tuple: height, width.')

        image_height, image_width = (int(value) for value in self.data_config.image_size)

        if image_height < 1 or image_width < 1:
            raise ValueError(f'image_size values must be positive. Got: {self.data_config.image_size}')

        self.validate_model_config(image_height=image_height, image_width=image_width)

        if self.data_config.heatmap_sigma <= 0:
            raise ValueError(f'heatmap_sigma must be greater than 0. Got: {self.data_config.heatmap_sigma}')

        if self.train_config.random_seed < 0:
            raise ValueError(f'random_seed must be at least 0. Got: {self.train_config.random_seed}')

        if self.train_config.batch_size < 1:
            raise ValueError(f'batch_size must be at least 1. Got: {self.train_config.batch_size}')

        if self.train_config.learning_rate <= 0:
            raise ValueError(f'learning_rate must be greater than 0. Got: {self.train_config.learning_rate}')

        if self.train_config.max_training_epochs < 1:
            raise ValueError(f'max_training_epochs must be at least 1. Got: {self.train_config.max_training_epochs}')

        if self.train_config.num_workers < 0:
            raise ValueError(f'num_workers must be at least 0. Got: {self.train_config.num_workers}')

        if int(self.data_config.oversampling_factor) < 1:
            raise ValueError(f'oversampling_factor must be at least 1. Got: {self.data_config.oversampling_factor}')

        if self.train_config.positive_weight < 0:
            raise ValueError(f'positive_weight must be at least 0. Got: {self.train_config.positive_weight}')

        if self.train_config.weight_decay < 0:
            raise ValueError(f'weight_decay must be at least 0. Got: {self.train_config.weight_decay}')

        if self.train_config.momentum < 0:
            raise ValueError(f'momentum must be at least 0. Got: {self.train_config.momentum}')

        if self.train_config.lr_step_size < 1:
            raise ValueError(f'lr_step_size must be at least 1. Got: {self.train_config.lr_step_size}')

        if self.train_config.lr_gamma <= 0:
            raise ValueError(f'lr_gamma must be greater than 0. Got: {self.train_config.lr_gamma}')

        if self.train_config.early_stop_patience < 1:
            raise ValueError(f'early_stop_patience must be at least 1. Got: {self.train_config.early_stop_patience}')

        if self.train_config.early_stop_min_delta < 0:
            raise ValueError(f'early_stop_min_delta must be at least 0. Got: {self.train_config.early_stop_min_delta}')

        if self.train_config.early_stop_warmup_epochs < 0:
            raise ValueError(f'early_stop_warmup_epochs must be at least 0. Got: {self.train_config.early_stop_warmup_epochs}')

        if self.train_config.loss_name == 'bce_logits' and self.model_config.output_activation != 'none':
            raise ValueError('loss_name=bce_logits requires output_activation=none because BCEWithLogitsLoss expects raw logits.')

    def validate_model_config(self, image_height, image_width):
        """Validate the selected architecture and its image-size requirements."""
        network_name = str(self.model_config.network_name).lower()
        get_model_config_fields(network_name)

        if self.model_config.dropout < 0 or self.model_config.dropout >= 1:
            raise ValueError(f'dropout must be in the range [0, 1). Got: {self.model_config.dropout}')

        if self.model_config.vit_dropout < 0 or self.model_config.vit_dropout >= 1:
            raise ValueError(f'vit_dropout must be in the range [0, 1). Got: {self.model_config.vit_dropout}')

        if self.model_config.auxiliary_loss_weight < 0:
            raise ValueError(f'auxiliary_loss_weight must be at least 0. Got: {self.model_config.auxiliary_loss_weight}')

        if network_name == 'unet_basic':
            if self.model_config.base_channels < 1 or self.model_config.depth < 1 or self.model_config.channel_multiplier < 1:
                raise ValueError('U-Net base_channels, depth, and channel_multiplier must be at least 1.')

            if self.model_config.max_channels < self.model_config.base_channels:
                raise ValueError('max_channels must be greater than or equal to base_channels.')

            minimum_image_size = 2 ** int(self.model_config.depth)
            deepest_height = image_height // minimum_image_size
            deepest_width = image_width // minimum_image_size

            if self.model_config.normalisation in ('batch', 'instance') and deepest_height * deepest_width < 2:
                raise ValueError(f'image_size produces a {deepest_height} x {deepest_width} deepest U-Net feature map. Use a larger image, a shallower network, or normalisation=None.')

            if self.model_config.normalisation == 'group':
                deepest_channels = min(int(self.model_config.base_channels) * (int(self.model_config.channel_multiplier) ** int(self.model_config.depth)), int(self.model_config.max_channels))
                groups = min(8, deepest_channels)

                while deepest_channels % groups != 0:
                    groups -= 1

                if (deepest_channels // groups) * deepest_height * deepest_width < 2:
                    raise ValueError('The deepest U-Net feature map does not contain enough values per group for group normalisation.')

            if self.model_config.padding_mode == 'reflect' and (deepest_height < 2 or deepest_width < 2):
                raise ValueError('padding_mode=reflect requires both deepest U-Net feature-map dimensions to be at least 2.')
        elif network_name == 'hrnet':
            if self.model_config.hrnet_width < 4 or self.model_config.hrnet_modules < 1 or self.model_config.hrnet_blocks < 1:
                raise ValueError('HRNet requires hrnet_width >= 4, hrnet_modules >= 1, and hrnet_blocks >= 1.')

            minimum_image_size = 64
        elif network_name == 'stacked_hourglass':
            if self.model_config.hourglass_features < 16 or self.model_config.hourglass_stacks < 1 or self.model_config.hourglass_depth < 1 or self.model_config.hourglass_blocks < 1:
                raise ValueError('Stacked hourglass requires hourglass_features >= 16 and positive stack, depth, and block counts.')

            minimum_image_size = 8 * (2 ** int(self.model_config.hourglass_depth))
        elif network_name == 'vitpose':
            patch_size = int(self.model_config.vit_patch_size)

            if patch_size < 2 or patch_size & (patch_size - 1):
                raise ValueError('vit_patch_size must be a power of two greater than or equal to 2.')

            if self.model_config.vit_embed_dim < 8 or self.model_config.vit_depth < 1 or self.model_config.vit_heads < 1:
                raise ValueError('ViTPose requires vit_embed_dim >= 8 and positive transformer depth and head counts.')

            if self.model_config.vit_embed_dim % self.model_config.vit_heads != 0:
                raise ValueError('vit_heads must divide vit_embed_dim exactly.')

            if self.model_config.vit_mlp_ratio <= 0 or self.model_config.vit_decoder_channels < 16:
                raise ValueError('ViTPose requires vit_mlp_ratio > 0 and vit_decoder_channels >= 16.')

            minimum_image_size = patch_size
        else:
            raise ValueError(f'Unknown heatmap model: {network_name}')

        if image_height < minimum_image_size or image_width < minimum_image_size:
            raise ValueError(f'image_size must be at least {minimum_image_size} x {minimum_image_size} for {network_name}. Got: {image_height} x {image_width}')

    @staticmethod
    def set_random_seed(seed):
        """Seed Python, NumPy, PyTorch, and CUDA RNGs for repeatable runs."""
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

    def get_checkpoint_path(self, checkpoint_type):
        """Return a checkpoint path."""
        return self.output_path / f'model_{checkpoint_type}.pth'

    def get_checkpoint_summary_path(self):
        """Return the validation checkpoint summary path."""
        return self.output_path / 'validation_checkpoint_summary.json'

    def get_log_path(self):
        """Return the combined training and validation log path."""
        return self.output_path / 'training_validation_log.csv'

    def get_plot_path(self):
        """Return the combined training and validation plot path."""
        return self.output_path / 'training_validation_plot.png'


    @staticmethod
    def get_current_lr(optimiser):
        """Return the current optimiser learning rate."""
        return optimiser.param_groups[0]['lr']

    @staticmethod
    def empty_history():
        """Create the training history store."""
        return {field_name: [] for field_name in HISTORY_FIELDS}

    @staticmethod
    def update_history(history, epoch, epoch_started_at, epoch_completed_at, epoch_lr, training_metrics, validation_metrics,
                       training_duration_seconds, validation_duration_seconds, epoch_duration_seconds):
        """Append one epoch to the training history."""
        values = {
            'epoch': int(epoch),
            'epoch_started_at': epoch_started_at,
            'epoch_completed_at': epoch_completed_at,
            'lr': float(epoch_lr),
            'training_loss': float(training_metrics['loss']),
            'training_error_px': float(training_metrics['error_px']),
            'validation_loss': float(validation_metrics['loss']),
            'validation_error_px': float(validation_metrics['error_px']),
            'training_duration_seconds': float(training_duration_seconds),
            'validation_duration_seconds': float(validation_duration_seconds),
            'epoch_duration_seconds': float(epoch_duration_seconds),
        }

        for field_name in HISTORY_FIELDS:
            history[field_name].append(values[field_name])

    def write_history_log(self, history):
        """Atomically rebuild the CSV from checkpoint-backed history."""
        log_path = self.get_log_path()
        temporary_path = log_path.with_name(f'.{log_path.name}.tmp')

        try:
            with open(temporary_path, 'w', newline='', encoding='utf-8') as log_file:
                writer = csv.DictWriter(log_file, fieldnames=list(HISTORY_FIELDS))
                writer.writeheader()

                for row_index in range(len(history['epoch'])):
                    writer.writerow({field_name: history[field_name][row_index] for field_name in HISTORY_FIELDS})

            os.replace(temporary_path, log_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def validate_history(history, completed_epoch):
        """Validate that saved history represents every committed epoch exactly once."""
        if not isinstance(history, dict):
            raise ValueError('Resume checkpoint history must be a dictionary of columns.')

        missing_fields = sorted(set(HISTORY_FIELDS) - set(history))

        if missing_fields:
            raise ValueError(f'Resume checkpoint history is missing fields: {missing_fields}.')

        lengths = {field_name: len(history[field_name]) for field_name in HISTORY_FIELDS}

        if len(set(lengths.values())) != 1:
            raise ValueError(f'Resume checkpoint history columns have inconsistent lengths: {lengths}.')

        expected_epochs = list(range(1, int(completed_epoch) + 1))

        if list(history['epoch']) != expected_epochs:
            raise ValueError(f'Resume checkpoint history epochs must be {expected_epochs}; got {history["epoch"]}.')

        numeric_fields = set(HISTORY_FIELDS) - {'epoch_started_at', 'epoch_completed_at'}

        for field_name in numeric_fields:
            if not all(np.isfinite(value) for value in history[field_name]):
                raise ValueError(f'Resume checkpoint history contains a non-finite value in {field_name}.')

        for field_name in ('training_duration_seconds', 'validation_duration_seconds', 'epoch_duration_seconds'):
            if any(float(value) < 0 for value in history[field_name]):
                raise ValueError(f'Resume checkpoint history contains a negative duration in {field_name}.')

    def save_history_plot(self, history):
        """Save loss and endpoint-error traces in the training plot."""
        if not history['epoch']:
            return

        plt.clf()
        figure, loss_axis = plt.subplots(figsize=(9, 5))
        error_axis = loss_axis.twinx()
        loss_axis.plot(history['epoch'], history['training_loss'], label='training_loss')
        loss_axis.plot(history['epoch'], history['validation_loss'], label='validation_loss')
        error_axis.plot(history['epoch'], history['training_error_px'], linestyle='--', label='training_error_px')
        error_axis.plot(history['epoch'], history['validation_error_px'], linestyle='--', label='validation_error_px')
        loss_axis.set_xlabel('Epoch')
        loss_axis.set_ylabel('Loss')
        error_axis.set_ylabel('Mean endpoint error (px)')
        loss_lines, loss_labels = loss_axis.get_legend_handles_labels()
        error_lines, error_labels = error_axis.get_legend_handles_labels()
        loss_axis.legend(loss_lines + error_lines, loss_labels + error_labels, loc='best')
        figure.tight_layout()
        figure.savefig(self.get_plot_path())
        plt.close(figure)

    def get_validation_output_path(self):
        """Return the validation output directory."""
        return self.output_path / 'validation_results'

    def get_validation_staging_path(self):
        """Return the same-volume staging directory for a complete validation export."""
        return self.output_path / '.validation_results.tmp'

    def get_validation_backup_path(self):
        """Return the temporary backup used while committing a validation export."""
        return self.output_path / '.validation_results.backup'

    def prepare_validation_staging_path(self):
        """Recover a previously committed export and clear only uncommitted staging data."""
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
        """Swap a complete staged validation export into place while preserving the prior export on failure."""
        final_path = self.get_validation_output_path()
        staging_path = self.get_validation_staging_path()
        backup_path = self.get_validation_backup_path()

        if not staging_path.is_dir():
            raise ValueError(f'Validation staging directory is missing: {staging_path}')

        if backup_path.exists():
            shutil.rmtree(backup_path)

        moved_existing_output = False

        try:
            if final_path.exists():
                os.replace(final_path, backup_path)
                moved_existing_output = True

            os.replace(staging_path, final_path)
        except BaseException:
            if moved_existing_output and not final_path.exists() and backup_path.exists():
                os.replace(backup_path, final_path)
            raise

        if backup_path.exists():
            shutil.rmtree(backup_path)

    @staticmethod
    def get_checkpoint_type_from_path(checkpoint_path):
        """Infer checkpoint type from the checkpoint filename."""
        if checkpoint_path is None:
            return None

        name = Path(checkpoint_path).stem.lower()

        if name == 'model_best_validation_loss':
            return 'best_validation_loss'

        if name == 'model_last_epoch':
            return 'last_epoch'

        return None

    def create_image_summary_row(self, sample_name, image_path, image_height, image_width, point_errors, checkpoint_type=None):
        """Create one image-level validation summary row."""
        point_errors = np.asarray(point_errors, dtype=np.float32)
        return {'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
                'sample_name': sample_name, 'image_path': image_path, 'image_height': int(image_height), 'image_width': int(image_width),
                'num_points': int(point_errors.size), 'mean_error_px': float(np.mean(point_errors)), 'median_error_px': float(np.median(point_errors)),
                'max_error_px': float(np.max(point_errors)), 'checkpoint_type': checkpoint_type}

    def create_endpoint_rows(self, sample_name, image_path, target_points, predicted_points, point_errors, checkpoint_type=None):
        """Create one endpoint-level validation row per landmark."""
        rows = []

        for point_index, (target, predicted, error) in enumerate(zip(target_points, predicted_points, point_errors), start=1):
            rows.append({'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
                         'sample_name': sample_name, 'image_path': image_path, 'point_index': point_index, 'target_x': float(target[0]),
                         'target_y': float(target[1]), 'pred_x': float(predicted[0]), 'pred_y': float(predicted[1]),
                         'error_px': float(error), 'checkpoint_type': checkpoint_type})

        return rows

    def write_validation_outputs(self, validation_output_path, image_summary_rows, endpoint_rows, prediction_rows):
        """Write validation CSV and Excel outputs in an IPV-like format."""
        output_paths = {'validation_summary_xlsx': validation_output_path / 'validation_summary.xlsx',
                        'validation_image_summary_csv': validation_output_path / 'validation_image_summary.csv',
                        'validation_endpoints_csv': validation_output_path / 'validation_endpoints.csv',
                        'validation_predictions_csv': validation_output_path / 'validation_predictions.csv'}
        self.write_rows_csv(output_paths['validation_image_summary_csv'], image_summary_rows)
        self.write_rows_csv(output_paths['validation_endpoints_csv'], endpoint_rows)
        self.write_rows_csv(output_paths['validation_predictions_csv'], prediction_rows)
        self.write_validation_workbook(output_paths['validation_summary_xlsx'], image_summary_rows=image_summary_rows, endpoint_rows=endpoint_rows)
        return output_paths

    @staticmethod
    def write_validation_workbook(output_xlsx, image_summary_rows, endpoint_rows):
        """Write validation summaries to an Excel workbook matching the IPV sheet layout."""
        workbook = Workbook()
        image_sheet = workbook.active
        image_sheet.title = 'validation_image_summary'
        TrainModel.write_rows_to_sheet(image_sheet, image_summary_rows)
        endpoint_sheet = workbook.create_sheet('validation_endpoints')
        TrainModel.write_rows_to_sheet(endpoint_sheet, endpoint_rows)
        workbook.save(output_xlsx)

    @staticmethod
    def write_rows_to_sheet(sheet, rows):
        """Write dictionaries to one worksheet."""
        if not rows:
            return

        headers = list(rows[0].keys())
        sheet.append(headers)

        for row in rows:
            sheet.append([row.get(header) for header in headers])

    @staticmethod
    def write_rows_csv(output_csv, rows):
        """Write dictionaries to CSV."""
        if not rows:
            return

        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_validation_run_metadata(self, validation_output_path, checkpoint_path, checkpoint_type, checkpoint_payload,
                                      image_summary_rows, endpoint_rows, output_paths):
        """Write validation run metadata."""
        if checkpoint_payload is None:
            raise ValueError('Validation metadata requires the concrete checkpoint used for prediction.')

        logs_path = validation_output_path / 'validation_logs'
        logs_path.mkdir(exist_ok=True, parents=True)
        checkpoint_epoch = None if checkpoint_payload is None else checkpoint_payload.get('epoch')
        checkpoint_metrics = None if checkpoint_payload is None else checkpoint_payload.get('validation_metrics')
        raw_checkpoint_metrics = None if checkpoint_metrics is None else self.unlabel_validation_metrics(checkpoint_metrics)
        metadata = {'schema': 'heatmap_validation_run_metadata', 'schema_version': CHECKPOINT_SCHEMA_VERSION, 'created_at': utc_now_iso(),
                    'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
                    'task_name': self.data_config.task_name, 'num_points': int(self.data_config.num_of_points),
                    'image_count': len(image_summary_rows), 'endpoint_count': len(endpoint_rows),
                    'training_status': self.training_status, 'termination_reason': self.termination_reason,
                    'checkpoint': self.build_checkpoint_descriptor(path=checkpoint_path, checkpoint_type=checkpoint_type,
                                                                   epoch=checkpoint_epoch, validation_metrics=raw_checkpoint_metrics),
                    'output_paths': {name: str(path) for name, path in output_paths.items()},
                    'runtime_environment': self.runtime_metadata, 'timing': self.get_timing_summary(),
                    'metadata': (self.build_checkpoint_metadata(checkpoint_type=checkpoint_type, epoch=checkpoint_epoch,
                                                                validation_metrics=raw_checkpoint_metrics)
                                 if checkpoint_payload is not None else self.build_run_metadata())}

        with open(logs_path / 'validation_run_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4, default=str)

    def create_prediction_row(self, sample_name, target_points, predicted_points, point_errors):
        """Create one prediction CSV row."""
        row = {'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': normalise_fold(self.data_config.fold),
               'sample_name': sample_name, 'mean_error_px': float(np.mean(point_errors))}

        for point_index, (target, predicted, error) in enumerate(zip(target_points, predicted_points, point_errors), start=1):
            row[f'target_x{point_index}'] = float(target[0])
            row[f'target_y{point_index}'] = float(target[1])
            row[f'pred_x{point_index}'] = float(predicted[0])
            row[f'pred_y{point_index}'] = float(predicted[1])
            row[f'error_px{point_index}'] = float(error)

        return row

