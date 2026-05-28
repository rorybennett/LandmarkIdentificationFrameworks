"""
Training and validation routines for heatmap landmark models.
"""

import csv
import datetime as dt
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from .model_registry import build_heatmap_model
from .models import count_trainable_parameters
from .utils.io_utils import heatmaps_to_points, safe_file_stem, scale_points_to_original
from .utils.visualisation_utils import save_validation_overlays

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class HeatmapDataConfig:
    fold: int
    task_name: str
    num_of_points: int
    fold_lists_path: Path
    mark_list_file: Path
    image_data_dir: Path
    image_size: tuple[int, int]
    heatmap_sigma: float = 8.0
    pixels_per_cm: float = 40.0
    input_channels: int = 1
    recursive_image_search: bool = False


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    max_training_epochs: int
    num_workers: int = 8
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
    save_validation_overlays: bool = False


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

    def train(self):
        """Run the fold training workflow."""
        self.output_path.mkdir(exist_ok=True, parents=True)
        train_loader, val_loader = self.build_data_loaders()
        model = self.build_model()
        criterion = self.build_criterion()
        optimiser = self.build_optimiser(model)
        scheduler = self.build_scheduler(optimiser)
        scaler = torch.amp.GradScaler('cuda', enabled=self.train_config.use_amp and self.device.type == 'cuda')
        history = self.empty_history()
        log_path = self.get_log_path()
        best_epoch = None
        best_val_loss = float('inf')
        last_epoch = 0
        last_val_loss = None
        best_checkpoint_path = None
        last_checkpoint_path = None
        bad_epochs = 0

        print(f'\tNetwork loaded on {self.device}. Trainable parameters: {count_trainable_parameters(model):,}', flush=True)

        with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['epoch', 'lr', 'train_loss', 'train_error_px', 'train_error_mm', 'val_loss', 'val_error_px', 'val_error_mm'])

            for epoch in range(1, self.train_config.max_training_epochs + 1):
                print(f"\t{dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Epoch {epoch}/{self.train_config.max_training_epochs}", flush=True)
                train_metrics = self.train_epoch(model=model, loader=train_loader, criterion=criterion, optimiser=optimiser, scaler=scaler)
                val_metrics = self.validate(model=model, loader=val_loader, criterion=criterion)

                if scheduler is not None:
                    scheduler.step(val_metrics['loss']) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

                current_lr = self.get_current_lr(optimiser)
                log_writer.writerow([epoch, current_lr, train_metrics['loss'], train_metrics['error_px'], train_metrics['error_mm'], val_metrics['loss'], val_metrics['error_px'], val_metrics['error_mm']])
                log_file.flush()
                self.update_history(history=history, epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)
                self.save_history_plot(history)
                print(f"\tlr={current_lr:.6g} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_error={val_metrics['error_px']:.2f}px", flush=True)
                last_epoch = epoch
                last_val_loss = val_metrics['loss']
                last_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, checkpoint_type='last', epoch=epoch, metrics=val_metrics)
                is_new_best = val_metrics['loss'] < best_val_loss
                is_early_stop_improvement = val_metrics['loss'] < best_val_loss - self.train_config.early_stop_min_delta

                if is_new_best:
                    best_epoch = epoch
                    best_val_loss = val_metrics['loss']
                    best_checkpoint_path = self.save_checkpoint(model=model, optimiser=optimiser, checkpoint_type='best', epoch=epoch, metrics=val_metrics)
                    print(f"\tNew best model saved from epoch {epoch} with val_loss={best_val_loss:.6f}", flush=True)

                if epoch >= self.train_config.early_stop_warmup_epochs:
                    bad_epochs = 0 if is_early_stop_improvement else bad_epochs + 1

                    if bad_epochs >= self.train_config.early_stop_patience:
                        print(f'\tEarly stop: validation loss stopped improving. Best epoch: {best_epoch}', flush=True)
                        break

        validation_predictions_path = None

        if self.train_config.save_validation_predictions:
            validation_predictions_path = self.save_validation_predictions(model=model, val_loader=val_loader, checkpoint_path=best_checkpoint_path or last_checkpoint_path)

        self.write_checkpoint_summary(best_epoch=best_epoch, last_epoch=last_epoch, best_val_loss=best_val_loss, last_val_loss=last_val_loss, best_checkpoint_path=best_checkpoint_path, last_checkpoint_path=last_checkpoint_path, validation_predictions_path=validation_predictions_path)
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
                outputs = model(images)
                loss = criterion(outputs, targets)

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
                outputs = model(images)
                loss = criterion(outputs, targets)
                batch_error = self.calculate_batch_error(outputs=outputs, points_original=points_original, original_size=original_size)
                total_loss += loss.item() * images.size(0)
                total_error_px += batch_error.sum().item()
                total_points += batch_error.numel()

        return self.format_metrics(loss=total_loss / max(len(loader.dataset), 1), error_px=total_error_px / max(total_points, 1))

    def save_validation_predictions(self, model, val_loader, checkpoint_path):
        """Save validation endpoint predictions from the selected checkpoint."""
        if checkpoint_path is not None:
            self.load_checkpoint_state(model=model, checkpoint_path=checkpoint_path)

        model.eval()
        output_csv = self.output_path / f'validation_predictions_f{self.data_config.fold}.csv'
        rows = []

        with torch.inference_mode():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                points_original = batch['points_original'].to(self.device, non_blocking=True)
                original_size = batch['original_size'].to(self.device, non_blocking=True)
                outputs = model(images)
                predicted_resized = heatmaps_to_points(outputs)
                predicted_original = scale_points_to_original(points=predicted_resized, original_sizes=original_size, image_size=self.data_config.image_size)
                errors = torch.linalg.norm(predicted_original - points_original, dim=2)

                for index, sample_name in enumerate(batch['sample_name']):
                    target_points = points_original[index].detach().cpu().numpy()
                    predicted_points = predicted_original[index].detach().cpu().numpy()
                    point_errors = errors[index].detach().cpu().numpy()
                    row = self.create_prediction_row(sample_name=sample_name, target_points=target_points, predicted_points=predicted_points, point_errors=point_errors)
                    rows.append(row)

                    if self.train_config.save_validation_overlays:
                        output_stem = safe_file_stem(sample_name)
                        predicted_heatmaps = outputs[index].detach().cpu().numpy()
                        save_validation_overlays(image_path=Path(batch['image_path'][index]), output_dir=self.get_validation_overlay_path(), output_stem=output_stem, target_points=target_points, predicted_points=predicted_points, predicted_heatmaps=predicted_heatmaps)

        self.write_prediction_csv(output_csv=output_csv, rows=rows)
        print(f'\tValidation predictions saved to {output_csv}', flush=True)
        return output_csv

    def build_data_loaders(self):
        """Build train and validation data loaders."""
        train_dataset = HeatmapDataset(self.build_dataset_config(split_name='train'))
        val_dataset = HeatmapDataset(self.build_dataset_config(split_name='val'))
        train_loader = DataLoader(train_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers, pin_memory=self.device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.num_workers, pin_memory=self.device.type == 'cuda')
        return train_loader, val_loader

    def build_dataset_config(self, split_name):
        """Build one dataset configuration."""
        return HeatmapDatasetConfig(fold=self.data_config.fold, split_name=split_name, num_of_points=self.data_config.num_of_points, fold_lists_path=self.data_config.fold_lists_path, mark_list_file=self.data_config.mark_list_file, image_data_dir=self.data_config.image_data_dir, image_size=self.data_config.image_size, heatmap_sigma=self.data_config.heatmap_sigma, input_channels=self.data_config.input_channels, recursive_image_search=self.data_config.recursive_image_search)

    def build_model(self):
        """Build the configured heatmap model."""
        model_kwargs = {
            'base_channels': self.model_config.base_channels,
            'depth': self.model_config.depth,
            'channel_multiplier': self.model_config.channel_multiplier,
            'max_channels': self.model_config.max_channels,
            'normalisation': self.model_config.normalisation,
            'activation': self.model_config.activation,
            'dropout': self.model_config.dropout,
            'upsampling': self.model_config.upsampling,
            'output_activation': self.model_config.output_activation,
            'padding_mode': self.model_config.padding_mode,
            'final_kernel_size': self.model_config.final_kernel_size,
        }
        model = build_heatmap_model(network_name=self.model_config.network_name, num_of_points=self.data_config.num_of_points, input_channels=self.data_config.input_channels, **model_kwargs)
        return model.to(self.device)

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

        if schedule in ('none', 'false', '0'):
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
        """Return loss, pixel error, and millimetre error."""
        pixels_per_mm = float(self.data_config.pixels_per_cm) / 10.0
        error_mm = float(error_px) / pixels_per_mm if pixels_per_mm > 0 else float('nan')
        return {'loss': float(loss), 'error_px': float(error_px), 'error_mm': float(error_mm)}

    def save_checkpoint(self, model, optimiser, checkpoint_type, epoch, metrics):
        """Save one model checkpoint."""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type)
        torch.save({'format_version': CHECKPOINT_FORMAT_VERSION, 'created_at': dt.datetime.now().isoformat(), 'epoch': int(epoch), 'checkpoint_type': checkpoint_type, 'state_dict': model.state_dict(), 'optimiser_state_dict': optimiser.state_dict(), 'metrics': metrics, 'metadata': self.build_metadata()}, checkpoint_path)
        return checkpoint_path

    def load_checkpoint_state(self, model, checkpoint_path):
        """Load checkpoint weights into a model."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None

        if state_dict is None:
            raise ValueError(f'Checkpoint does not contain a state_dict: {checkpoint_path}')

        model.load_state_dict(state_dict)
        model.eval()

    def write_checkpoint_summary(self, best_epoch, last_epoch, best_val_loss, last_val_loss, best_checkpoint_path, last_checkpoint_path, validation_predictions_path):
        """Write checkpoint and run metadata."""
        summary = {'format_version': CHECKPOINT_FORMAT_VERSION, 'created_at': dt.datetime.now().isoformat(), 'fold': int(self.data_config.fold), 'task_name': self.data_config.task_name, 'num_of_points': int(self.data_config.num_of_points), 'checkpoints': {'best': {'epoch': best_epoch, 'val_loss': best_val_loss, 'path': str(best_checkpoint_path) if best_checkpoint_path is not None else None}, 'last': {'epoch': last_epoch, 'val_loss': last_val_loss, 'path': str(last_checkpoint_path) if last_checkpoint_path is not None else None}}, 'validation_predictions_path': str(validation_predictions_path) if validation_predictions_path is not None else None, 'metadata': self.build_metadata()}
        with open(self.get_checkpoint_summary_path(), 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=4, default=str)

    def build_metadata(self):
        """Build serialisable metadata."""
        return {'data_config': self.serialise(asdict(self.data_config)), 'train_config': self.serialise(asdict(self.train_config)), 'model_config': self.serialise(asdict(self.model_config))}

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
        if int(self.data_config.num_of_points) < 1:
            raise ValueError('num_of_points must be at least 1.')

        if int(self.data_config.input_channels) not in (1, 3):
            raise ValueError('input_channels must be 1 or 3.')

        if len(tuple(self.data_config.image_size)) != 2:
            raise ValueError('image_size must be a two-item tuple: height, width.')

    def copy_outputs_to(self, save_path):
        """Copy output files to an external save path."""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

        for entry in self.output_path.iterdir():
            destination = save_path / entry.name
            if entry.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(entry, destination)
            else:
                shutil.copy2(entry, destination)

    def get_checkpoint_path(self, checkpoint_type):
        """Return a checkpoint path."""
        return self.output_path / f'model_f{self.data_config.fold}_{checkpoint_type}.pth'

    def get_checkpoint_summary_path(self):
        """Return the checkpoint summary path."""
        return self.output_path / f'checkpoint_summary_f{self.data_config.fold}.json'

    def get_log_path(self):
        """Return the training log path."""
        return self.output_path / f'train_log_f{self.data_config.fold}.csv'

    def get_plot_path(self):
        """Return the training plot path."""
        return self.output_path / f'train_plot_f{self.data_config.fold}.png'

    def get_validation_overlay_path(self):
        """Return the validation overlay path."""
        return self.output_path / f'validation_overlays_F{self.data_config.fold}'

    @staticmethod
    def get_current_lr(optimiser):
        """Return the current optimiser learning rate."""
        return optimiser.param_groups[0]['lr']

    @staticmethod
    def empty_history():
        """Create the training history store."""
        return {'epoch': [], 'train_loss': [], 'train_error_px': [], 'val_loss': [], 'val_error_px': []}

    @staticmethod
    def update_history(history, epoch, train_metrics, val_metrics):
        """Append one epoch to the training history."""
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_error_px'].append(train_metrics['error_px'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_error_px'].append(val_metrics['error_px'])

    def save_history_plot(self, history):
        """Save loss and endpoint-error plots."""
        if not history['epoch']:
            return

        plt.clf()
        plt.figure(figsize=(8, 5))
        plt.plot(history['epoch'], history['train_loss'], label='train_loss')
        plt.plot(history['epoch'], history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.get_plot_path())
        plt.close()

    @staticmethod
    def create_prediction_row(sample_name, target_points, predicted_points, point_errors):
        """Create one prediction CSV row."""
        row = {'sample_name': sample_name, 'mean_error_px': float(np.mean(point_errors))}

        for point_index, (target, predicted, error) in enumerate(zip(target_points, predicted_points, point_errors), start=1):
            row[f'target_x{point_index}'] = float(target[0])
            row[f'target_y{point_index}'] = float(target[1])
            row[f'pred_x{point_index}'] = float(predicted[0])
            row[f'pred_y{point_index}'] = float(predicted[1])
            row[f'error_px{point_index}'] = float(error)

        return row

    @staticmethod
    def write_prediction_csv(output_csv, rows):
        """Write validation predictions to CSV."""
        if not rows:
            return

        with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
