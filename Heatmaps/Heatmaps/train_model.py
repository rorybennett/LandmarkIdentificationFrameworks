"""
Training and validation routines for heatmap landmark models.
"""

import csv
import datetime as dt
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
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
from .utils.io_utils import heatmaps_to_points, infer_image_channel_count, safe_file_stem, scale_points_to_original
from .utils.progress_bar import ProgressBar
from .utils.visualisation_utils import save_validation_overlays

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def seed_worker(_worker_id):
    """Seed NumPy and Python RNGs inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_NAME = 'heatmap_checkpoint_metadata'
MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30


@dataclass
class HeatmapDataConfig:
    repetition: int
    fold: int
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

    def __init__(self, data_config, train_config, model_config, output_save_path, device=None):
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.output_path = Path(output_save_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validate_configs()

    def train(self, on_dataset_validated=None):
        """Run the fold training workflow."""
        self.set_random_seed(self.train_config.random_seed)
        self.output_path.mkdir(exist_ok=True, parents=True)
        training_loader, validation_loader = self.build_data_loaders()

        if on_dataset_validated is not None:
            on_dataset_validated()

        model = self.build_model()
        criterion = self.build_criterion()
        optimiser = self.build_optimiser(model)
        scheduler = self.build_scheduler(optimiser)
        scaler = torch.amp.GradScaler('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda')
        history = self.empty_history()
        log_path = self.get_log_path()
        best_epoch = None
        best_validation_loss = float('inf')
        early_stop_best_validation_loss = float('inf')
        last_epoch = 0
        last_validation_loss = None
        best_checkpoint_path = None
        last_checkpoint_path = None
        bad_epochs = 0

        print(f'\tNetwork loaded on {self.device}. Trainable parameters: {count_trainable_parameters(model):,}', flush=True)

        with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['epoch', 'lr', 'training_loss', 'training_error_px', 'validation_loss', 'validation_error_px'])

            for epoch in range(1, self.train_config.max_training_epochs + 1):
                print(f"\t{dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Epoch {epoch}/{self.train_config.max_training_epochs}", flush=True)
                epoch_lr = self.get_current_lr(optimiser)
                training_metrics = self.train_epoch(model=model, loader=training_loader, criterion=criterion, optimiser=optimiser, scaler=scaler)
                validation_metrics = self.validate(model=model, loader=validation_loader, criterion=criterion)
                self.validate_finite_metrics(phase='training', metrics=training_metrics)
                self.validate_finite_metrics(phase='validation', metrics=validation_metrics)

                if scheduler is not None:
                    scheduler.step(validation_metrics['loss']) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

                log_writer.writerow([epoch, epoch_lr, training_metrics['loss'], training_metrics['error_px'], validation_metrics['loss'], validation_metrics['error_px']])
                log_file.flush()
                self.update_history(history=history, epoch=epoch, training_metrics=training_metrics, validation_metrics=validation_metrics)
                self.save_history_plot(history)
                last_epoch = epoch
                last_validation_loss = validation_metrics['loss']
                last_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, checkpoint_type='last_epoch', epoch=epoch,
                                                            validation_metrics=validation_metrics)
                is_new_best = validation_metrics['loss'] < best_validation_loss
                is_early_stop_improvement = validation_metrics['loss'] < early_stop_best_validation_loss - self.train_config.early_stop_min_delta

                if is_new_best:
                    best_epoch = epoch
                    best_validation_loss = validation_metrics['loss']
                    best_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, checkpoint_type='best_validation_loss', epoch=epoch,
                                                                validation_metrics=validation_metrics)
                    print(f"\tNew best model saved from epoch {epoch} with validation_loss={best_validation_loss:.6f} and "
                          f"validation_error={validation_metrics['error_px']:.2f}px", flush=True)

                if is_early_stop_improvement:
                    early_stop_best_validation_loss = validation_metrics['loss']

                if epoch >= self.train_config.early_stop_warmup_epochs:
                    bad_epochs = 0 if is_early_stop_improvement else bad_epochs + 1

                    if bad_epochs >= self.train_config.early_stop_patience:
                        print(f'\tEarly stop: validation loss stopped improving by at least {self.train_config.early_stop_min_delta:g}. '
                              f'Best checkpoint epoch: {best_epoch}; early-stop reference loss: {early_stop_best_validation_loss:.6f}', flush=True)
                        break

        validation_output_paths = None

        if self.train_config.save_validation_predictions:
            validation_output_paths = self.save_validation_predictions(model=model, validation_loader=validation_loader,
                                                                       checkpoint_path=best_checkpoint_path or last_checkpoint_path)

        self.write_checkpoint_summary(best_epoch=best_epoch, last_epoch=last_epoch, best_validation_loss=best_validation_loss,
                                      last_validation_loss=last_validation_loss,
                                      best_checkpoint_path=best_checkpoint_path, last_checkpoint_path=last_checkpoint_path,
                                      validation_output_paths=validation_output_paths)
        plt.clf()
        return best_checkpoint_path or last_checkpoint_path

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
        if checkpoint_path is not None:
            self.load_checkpoint_state(model=model, checkpoint_path=checkpoint_path)

        model.eval()
        validation_output_path = self.get_validation_output_path()

        if validation_output_path.exists():
            print(f'\tExisting validation output directory found at {validation_output_path}. Clearing it before export.', flush=True)
            shutil.rmtree(validation_output_path)

        logs_path = validation_output_path / 'validation_logs'
        validation_output_path.mkdir(exist_ok=True, parents=True)
        logs_path.mkdir(exist_ok=True, parents=True)

        image_summary_rows = []
        endpoint_rows = []
        prediction_rows = []
        checkpoint_type = self.get_checkpoint_type_from_path(checkpoint_path)

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

        output_paths = self.write_validation_outputs(validation_output_path=validation_output_path, image_summary_rows=image_summary_rows,
                                                     endpoint_rows=endpoint_rows, prediction_rows=prediction_rows)
        self.write_validation_run_metadata(validation_output_path=validation_output_path, checkpoint_path=checkpoint_path, checkpoint_type=checkpoint_type,
                                           image_summary_rows=image_summary_rows, endpoint_rows=endpoint_rows, output_paths=output_paths)
        print(f"\tValidation summary saved to {output_paths['validation_summary_xlsx']}", flush=True)
        return output_paths

    def build_data_loaders(self):
        """Build train and validation data loaders after resolving input channels."""
        training_dataset = HeatmapDataset(self.build_dataset_config(split_name='training'))
        validation_dataset = HeatmapDataset(self.build_dataset_config(split_name='validation'))
        self.validate_dataset_membership(training_dataset=training_dataset, validation_dataset=validation_dataset)
        self.resolve_input_channels(training_dataset=training_dataset, validation_dataset=validation_dataset)
        validated_training_records = training_dataset.validate_all_records()
        validated_validation_records = validation_dataset.validate_all_records()
        generator = torch.Generator()
        generator.manual_seed(int(self.train_config.random_seed))
        print(f'\tDataset validation complete: {validated_training_records} training and {validated_validation_records} validation records.', flush=True)
        print(f'\tTraining samples: {len(training_dataset)} ({len(training_dataset.records)} original, '
              f'oversampling_factor={training_dataset.oversampling_factor}).', flush=True)
        print(f'\tValidation samples: {len(validation_dataset)}.', flush=True)
        training_loader = DataLoader(training_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers,
                                     pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=generator)
        validation_loader = DataLoader(validation_dataset, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.num_workers,
                                       pin_memory=self.device.type == 'cuda', worker_init_fn=seed_worker, generator=generator)
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
                                    oversampling_factor=self.data_config.oversampling_factor)

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

    def save_checkpoint(self, model, optimiser, checkpoint_type, epoch, validation_metrics):
        """Save one model checkpoint with reconstruction-focused metadata."""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type)
        labelled_validation_metrics = self.label_validation_metrics(validation_metrics)
        metadata = self.build_metadata(checkpoint_type=checkpoint_type, epoch=epoch, validation_metrics=validation_metrics)
        torch.save({'format_version': CHECKPOINT_FORMAT_VERSION, 'schema': 'heatmap_training_checkpoint', 'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'created_at': dt.datetime.now().isoformat(), 'epoch': int(epoch), 'checkpoint_type': checkpoint_type, 'state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(), 'validation_metrics': labelled_validation_metrics, 'metadata': metadata}, checkpoint_path)
        return checkpoint_path

    def load_checkpoint_state(self, model, checkpoint_path):
        """Load checkpoint weights into a model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None

        if state_dict is None:
            raise ValueError(f'Checkpoint does not contain a state_dict: {checkpoint_path}')

        model.load_state_dict(state_dict)
        model.eval()

    def write_checkpoint_summary(self, best_epoch, last_epoch, best_validation_loss, last_validation_loss, best_checkpoint_path, last_checkpoint_path,
                                 validation_output_paths):
        """Write validation-based checkpoint-selection and run metadata."""
        validation_summary_path = None if validation_output_paths is None else validation_output_paths.get('validation_summary_xlsx')
        validation_predictions_path = None if validation_output_paths is None else validation_output_paths.get('validation_predictions_csv')
        summary = {'format_version': CHECKPOINT_FORMAT_VERSION, 'schema': 'heatmap_validation_checkpoint_summary',
                   'schema_version': CHECKPOINT_SCHEMA_VERSION, 'created_at': dt.datetime.now().isoformat(),
                   'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold), 'task_name': self.data_config.task_name,
                   'num_of_points': int(self.data_config.num_of_points), 'checkpoints': {
                'best_validation_loss': {'epoch': best_epoch, 'validation_loss': best_validation_loss,
                                         'path': str(best_checkpoint_path) if best_checkpoint_path is not None else None},
                'last_epoch': {'epoch': last_epoch, 'validation_loss': last_validation_loss,
                               'path': str(last_checkpoint_path) if last_checkpoint_path is not None else None}},
                   'validation_summary_path': str(validation_summary_path) if validation_summary_path is not None else None,
                   'validation_predictions_path': str(validation_predictions_path) if validation_predictions_path is not None else None,
                   'metadata': self.build_metadata()}

        with open(self.get_checkpoint_summary_path(), 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=4, default=str)

    def build_metadata(self, checkpoint_type=None, epoch=None, validation_metrics=None):
        """Build serialisable checkpoint metadata using IPV-style sections."""
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
            'created_at': dt.datetime.now().isoformat(),
            'checkpoint': {'format_version': CHECKPOINT_FORMAT_VERSION, 'type': checkpoint_type, 'epoch': None if epoch is None else int(epoch),
                           'validation_metrics': self.label_validation_metrics(validation_metrics)},
            'task': {'name': self.data_config.task_name, 'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
                     'num_points': int(self.data_config.num_of_points),
                     'output_heads': int(self.data_config.num_of_points), 'prediction_type': 'landmark_heatmap_regression'},
            'model': {'registry_name': self.model_config.network_name, 'module': registry_entry['module'], 'class_name': registry_entry['class_name'],
                      'init_args': model_init_args},
            'data': {'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
                     'fold_lists_path': str(self.data_config.fold_lists_path),
                     'mark_list_file': str(self.data_config.mark_list_file), 'image_data_dir': str(self.data_config.image_data_dir),
                     'recursive_image_search': bool(self.data_config.recursive_image_search), 'input_channels': input_channels},
            'preprocessing': {'image_size': {'height': image_height, 'width': image_width}, 'heatmap_sigma': float(self.data_config.heatmap_sigma),
                              'input_channels': input_channels, 'tensor_shape': ['batch', input_channels, image_height, image_width],
                              'channel_order': 'channels_first', 'image_value_range': 'float32_0_to_1',
                              'resize': {'library': 'cv2.resize', 'interpolation': 'INTER_AREA'},
                              'target_heatmaps': {'channels': int(self.data_config.num_of_points), 'generation': 'normalised_gaussian_per_landmark'}},
            'inference': {'heatmap_to_point': 'argmax', 'output_coordinate_space': 'original_image_pixels', 'scale_back_to_original': True,
                          'resized_coordinate_space': {'height': image_height, 'width': image_width}},
            'augmentation': self.build_augmentation_metadata(),
            'training': {'train_config': train_config, 'optimiser': self.train_config.optimiser_name, 'loss': self.train_config.loss_name,
                         'random_seed': int(self.train_config.random_seed), 'auxiliary_loss_weight': float(self.model_config.auxiliary_loss_weight) if self.model_config.network_name == 'stacked_hourglass' else None},
            'raw_configs': {'data_config': data_config, 'train_config': train_config, 'model_config': model_config}
        }

    @staticmethod
    def label_validation_metrics(validation_metrics):
        """Return externally stored validation metrics with unambiguous names."""
        if validation_metrics is None:
            return None

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

        if int(self.data_config.fold) < 1:
            raise ValueError(f'fold must be at least 1. Got: {self.data_config.fold}')

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
        return {'epoch': [], 'training_loss': [], 'training_error_px': [], 'validation_loss': [], 'validation_error_px': []}

    @staticmethod
    def update_history(history, epoch, training_metrics, validation_metrics):
        """Append one epoch to the training history."""
        history['epoch'].append(epoch)
        history['training_loss'].append(training_metrics['loss'])
        history['training_error_px'].append(training_metrics['error_px'])
        history['validation_loss'].append(validation_metrics['loss'])
        history['validation_error_px'].append(validation_metrics['error_px'])

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
        return {'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
                'sample_name': sample_name, 'image_path': image_path, 'image_height': int(image_height), 'image_width': int(image_width),
                'num_points': int(point_errors.size), 'mean_error_px': float(np.mean(point_errors)), 'median_error_px': float(np.median(point_errors)),
                'max_error_px': float(np.max(point_errors)), 'checkpoint_type': checkpoint_type}

    def create_endpoint_rows(self, sample_name, image_path, target_points, predicted_points, point_errors, checkpoint_type=None):
        """Create one endpoint-level validation row per landmark."""
        rows = []

        for point_index, (target, predicted, error) in enumerate(zip(target_points, predicted_points, point_errors), start=1):
            rows.append({'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
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

    def write_validation_run_metadata(self, validation_output_path, checkpoint_path, checkpoint_type, image_summary_rows, endpoint_rows, output_paths):
        """Write validation run metadata."""
        logs_path = validation_output_path / 'validation_logs'
        logs_path.mkdir(exist_ok=True, parents=True)
        metadata = {'schema': 'heatmap_validation_run_metadata', 'schema_version': CHECKPOINT_SCHEMA_VERSION, 'created_at': dt.datetime.now().isoformat(),
                    'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
                    'task_name': self.data_config.task_name, 'num_points': int(self.data_config.num_of_points),
                    'image_count': len(image_summary_rows), 'endpoint_count': len(endpoint_rows), 'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None,
                    'checkpoint_type': checkpoint_type, 'output_paths': {name: str(path) for name, path in output_paths.items()}, 'metadata': self.build_metadata()}

        with open(logs_path / 'validation_run_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4, default=str)

    def create_prediction_row(self, sample_name, target_points, predicted_points, point_errors):
        """Create one prediction CSV row."""
        row = {'dataset_split': 'validation', 'repetition': int(self.data_config.repetition), 'fold': int(self.data_config.fold),
               'sample_name': sample_name, 'mean_error_px': float(np.mean(point_errors))}

        for point_index, (target, predicted, error) in enumerate(zip(target_points, predicted_points, point_errors), start=1):
            row[f'target_x{point_index}'] = float(target[0])
            row[f'target_y{point_index}'] = float(target[1])
            row[f'pred_x{point_index}'] = float(predicted[0])
            row[f'pred_y{point_index}'] = float(predicted[1])
            row[f'error_px{point_index}'] = float(error)

        return row

