import ast
import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .custom_dataset import CustomDataset, ToTensor
from .utils.landmark_inference_utils import LandmarkInferenceConfig, run_validation_inference_for_trained_model

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
CSV_METADATA_COLUMNS = 5
CHECKPOINT_FORMAT_VERSION = 1


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
    loss_print_interval: int
    num_workers: int = 8
    momentum: float = 0.9
    lr_schedule: bool = False
    lr_step_size: int = 1
    lr_gamma: float = 0.1
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    early_stop_warmup_epochs: int = 3
    save_validation_results: bool = True
    validation_inference_batch_size: int = 2048
    validation_vote_smoothing_sigma: float = 7.0
    validation_save_raw_vote_maps: bool = False


class TrainModel:
    def __init__(self, current_fold, num_of_points, data_save_path, tasks_classes, train_config, quadruplet_config, output_save_path=None, device=None):
        self.fold = current_fold
        self.num_of_points = num_of_points
        self.train_path = Path(data_save_path)
        self.output_path = Path(output_save_path) if output_save_path is not None else self.train_path
        self.tasks_classes = tasks_classes
        self.train_config = train_config
        self.quadruplet_config = quadruplet_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validate_num_of_points(self.num_of_points)
        self.validate_tasks_classes_structure(self.tasks_classes)
        self.tasks_per_point = len(self.tasks_classes)
        self.expected_label_count = self.num_of_points * self.tasks_per_point
        self.input_channels = None
        self.num_of_classes = [len(task_classes) for _ in range(self.num_of_points) for task_classes in self.tasks_classes]

    def train(self):
        """Run training for one fold."""
        self.validate_training_inputs()
        self.output_path.mkdir(exist_ok=True, parents=True)
        train_loader, val_loader = self.build_data_loaders()
        self.input_channels = self.resolve_input_channels(train_loader.dataset, val_loader.dataset)
        model = self.build_model(input_channels=self.input_channels)
        criterion = nn.CrossEntropyLoss()
        optimiser = SGD(model.parameters(), lr=self.train_config.learning_rate, momentum=self.train_config.momentum)
        scheduler = StepLR(optimiser, step_size=self.train_config.lr_step_size, gamma=self.train_config.lr_gamma) if self.train_config.lr_schedule else None

        history = self.empty_history()
        previous_val_accuracy = None
        log_path = self.get_log_path()

        best_epoch = None
        last_epoch = 0
        best_val_loss = float('inf')
        last_val_loss = None
        best_checkpoint_path = None
        last_checkpoint_path = None
        bad_epochs = 0

        print('\tData loaded...', flush=True)
        print(f'\tNetwork loaded on {self.device}. Training network...', flush=True)

        with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['lr', 'epoch', 'step', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

            for epoch in range(1, self.train_config.max_training_epochs + 1):
                print(f"\t{dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Epoch {epoch}/{self.train_config.max_training_epochs}", flush=True)

                epoch_result = self.train_epoch(model, train_loader, val_loader, criterion, optimiser, log_writer, history, epoch)

                if scheduler is not None and previous_val_accuracy is not None and epoch_result['val_accuracy'] < previous_val_accuracy:
                    scheduler.step()

                previous_val_accuracy = epoch_result['val_accuracy']
                last_epoch = epoch
                last_val_loss = epoch_result['val_loss']
                last_checkpoint_path = self.save_checkpoint(model=model, checkpoint_type='last', epoch=epoch, metrics=epoch_result)

                previous_best_val_loss = best_val_loss
                is_new_best = epoch_result['val_loss'] < best_val_loss
                is_early_stop_improvement = epoch_result['val_loss'] < best_val_loss - self.train_config.early_stop_min_delta

                if is_new_best:
                    best_epoch = epoch
                    best_val_loss = epoch_result['val_loss']
                    best_checkpoint_path = self.save_checkpoint(model=model, checkpoint_type='best', epoch=epoch, metrics=epoch_result)
                    print(f"\tNew best model saved from epoch {epoch} with val_loss={best_val_loss:.6f}", flush=True)

                self.save_history_plot(history)

                if epoch >= self.train_config.early_stop_warmup_epochs:
                    if is_early_stop_improvement:
                        bad_epochs = 0
                    else:
                        bad_epochs += 1

                    if bad_epochs >= self.train_config.early_stop_patience:
                        print(f"\tEarly stop: validation loss stopped improving. Best val loss: {previous_best_val_loss:.6f}", flush=True)
                        break

        validation_results_path = None

        if self.train_config.save_validation_results:
            validation_results_path = self.run_validation_inference(model=model, best_checkpoint_path=best_checkpoint_path, last_checkpoint_path=last_checkpoint_path)

        self.write_checkpoint_summary(best_epoch=best_epoch, last_epoch=last_epoch, best_val_loss=best_val_loss, last_val_loss=last_val_loss,
                                      best_checkpoint_path=best_checkpoint_path, last_checkpoint_path=last_checkpoint_path,
                                      validation_results_path=validation_results_path)
        plt.clf()

    def run_validation_inference(self, model, best_checkpoint_path=None, last_checkpoint_path=None):
        """Run full-image inference on validation images and save overlays and Excel metrics."""
        checkpoint_path = best_checkpoint_path or last_checkpoint_path
        checkpoint_type = 'best' if best_checkpoint_path is not None else 'last'

        if checkpoint_path is not None:
            self.load_checkpoint_state(model=model, checkpoint_path=checkpoint_path)

        data_metadata = self.read_data_creation_metadata()
        validation_output_path = self.get_validation_output_path()
        config = LandmarkInferenceConfig(
            fold=int(self.fold),
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
            save_raw_vote_maps=bool(self.train_config.validation_save_raw_vote_maps),
            checkpoint_path=checkpoint_path,
            checkpoint_type=checkpoint_type
        )

        print(f'	Running validation image inference with {checkpoint_type} checkpoint...', flush=True)
        run_validation_inference_for_trained_model(model=model, config=config, device=self.device)
        print(f'	Validation image inference outputs saved to {validation_output_path}', flush=True)
        return validation_output_path

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
        metadata_paths = sorted(self.train_path.glob(f'run_info_*_f{self.fold}.json'))

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

        train_dataset = CustomDataset(train_csv_path, num_sub_patches=self.quadruplet_config.num_sub_patches, transform=ToTensor())
        val_dataset = CustomDataset(val_csv_path, num_sub_patches=self.quadruplet_config.num_sub_patches, transform=ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers,
                                  pin_memory=self.device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=self.train_config.batch_size, shuffle=False, num_workers=self.train_config.num_workers,
                                pin_memory=self.device.type == 'cuda')

        return train_loader, val_loader

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

    def train_epoch(self, model, train_loader, val_loader, criterion, optimiser, log_writer, history, epoch):
        """Train one epoch and run periodic validation."""
        model.train()

        total_batches = len(train_loader)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_predictions = 0

        window_loss = 0.0
        window_correct = 0
        window_predictions = 0
        window_samples = 0

        latest_val_loss = 0.0
        latest_val_accuracy = 0.0
        last_batch_index = 0
        last_validation_batch = 0

        for batch_index, data in enumerate(train_loader, start=1):
            last_batch_index = batch_index
            images = data['image'].to(self.device, non_blocking=True)
            labels = data['labels'].to(self.device, non_blocking=True).long()
            batch_size = labels.shape[0]

            optimiser.zero_grad(set_to_none=True)
            outputs = model(images)

            train_loss = self.calculate_loss(outputs, labels, criterion)
            train_loss.backward()
            optimiser.step()

            batch_correct = self.count_correct(outputs, labels)
            batch_predictions = batch_size * len(outputs)

            epoch_loss += train_loss.item() * batch_size
            epoch_correct += batch_correct
            epoch_predictions += batch_predictions

            window_loss += train_loss.item() * batch_size
            window_correct += batch_correct
            window_predictions += batch_predictions
            window_samples += batch_size

            should_validate = batch_index % self.train_config.loss_print_interval == 0 or batch_index == 1

            if should_validate:
                latest_val_loss, latest_val_accuracy = self.validate(model, val_loader, criterion)

                average_window_loss = window_loss / max(window_samples, 1)
                average_window_accuracy = window_correct / max(window_predictions, 1)

                self.update_history(history, epoch, batch_index, total_batches, average_window_loss, average_window_accuracy, latest_val_loss, latest_val_accuracy)
                self.write_log(log_writer, optimiser, epoch, batch_index, average_window_loss, average_window_accuracy, latest_val_loss, latest_val_accuracy)

                last_validation_batch = batch_index
                window_loss = 0.0
                window_correct = 0
                window_predictions = 0
                window_samples = 0

            model.train()

        average_epoch_loss = epoch_loss / max(len(train_loader.dataset), 1)
        average_epoch_accuracy = epoch_correct / max(epoch_predictions, 1)

        if last_validation_batch != last_batch_index:
            latest_val_loss, latest_val_accuracy = self.validate(model, val_loader, criterion)
            self.update_history(history, epoch, last_batch_index, total_batches, average_epoch_loss, average_epoch_accuracy, latest_val_loss, latest_val_accuracy)
            self.write_log(log_writer, optimiser, epoch, last_batch_index, average_epoch_loss, average_epoch_accuracy, latest_val_loss, latest_val_accuracy)

        return {
            'train_loss': average_epoch_loss,
            'train_accuracy': average_epoch_accuracy,
            'val_loss': latest_val_loss,
            'val_accuracy': latest_val_accuracy
        }

    def validate(self, model, val_loader, criterion):
        """Evaluate the model on the validation set."""
        model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_predictions = 0

        with torch.no_grad():
            for data in val_loader:
                images = data['image'].to(self.device, non_blocking=True)
                labels = data['labels'].to(self.device, non_blocking=True).long()
                batch_size = labels.shape[0]

                outputs = model(images)
                loss = self.calculate_loss(outputs, labels, criterion)

                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_correct += self.count_correct(outputs, labels)
                total_predictions += batch_size * len(outputs)

        val_loss = total_loss / max(total_samples, 1)
        val_accuracy = total_correct / max(total_predictions, 1)

        return val_loss, val_accuracy

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

    def write_log(self, log_writer, optimiser, epoch, batch_index, train_loss, train_accuracy, val_loss, val_accuracy):
        """Write one training status row to CSV."""
        step = batch_index * self.train_config.batch_size
        lr = self.get_current_lr(optimiser)

        log_writer.writerow([lr, epoch, step, train_loss, train_accuracy, val_loss, val_accuracy])

    def update_history(self, history, epoch, batch_index, total_batches, train_loss, train_accuracy, val_loss, val_accuracy):
        """Store losses and accuracies for plotting."""
        step = (epoch - 1) + batch_index / max(total_batches, 1)

        history['step'].append(step)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

    def save_checkpoint(self, model, checkpoint_type, epoch=None, metrics=None):
        """Save model weights with one consolidated metadata block."""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type)
        created_at = dt.datetime.now().isoformat()
        checkpoint = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'created_at': created_at,
            'state_dict': model.state_dict(),
            'metadata': self.build_checkpoint_metadata(checkpoint_type=checkpoint_type, epoch=epoch, metrics=metrics or {}, created_at=created_at)
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def save_history_plot(self, history):
        """Save a loss and accuracy plot."""
        plot_path = self.get_plot_path()

        plt.clf()
        plt.plot(history['step'], history['train_loss'], '--')
        plt.plot(history['step'], history['train_accuracy'], '-')
        plt.plot(history['step'], history['val_loss'], '--')
        plt.plot(history['step'], history['val_accuracy'], '-')
        plt.legend(['Training Loss', 'Train accuracy', 'Val Loss', 'Val accuracy'])
        plt.xlabel('Epoch progress')
        plt.ylabel('Loss / Accuracy')
        plt.savefig(plot_path)

    def build_checkpoint_metadata(self, checkpoint_type=None, epoch=None, metrics=None, created_at=None):
        """Build the single metadata structure saved in every checkpoint."""
        data_metadata = self.read_data_creation_metadata()

        return {
            'schema': 'ipv_checkpoint_metadata',
            'schema_version': CHECKPOINT_FORMAT_VERSION,
            'created_at': created_at or dt.datetime.now().isoformat(),
            'checkpoint': {
                'type': checkpoint_type,
                'epoch': epoch,
                'metrics': metrics or {}
            },
            'task': self.build_task_metadata(data_metadata),
            'model': self.build_model_metadata(),
            'data': self.build_data_metadata(data_metadata),
            'preprocessing': self.build_preprocessing_metadata(data_metadata),
            'inference': self.build_inference_metadata(data_metadata),
            'training': self.build_training_metadata()
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
            'fold': int(self.fold),
            'data_save_path': str(self.train_path),
            'output_save_path': str(self.output_path),
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
            'image_value_range': 'float32_0_to_1',
            'patch_resize': {
                'library': 'skimage.transform.resize',
                'preserve_range': True,
                'anti_aliasing': True
            }
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
                'use_probability_weights': True,
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
        data_info_path = self.train_path / f'data_info_f{self.fold}.csv'

        if data_info_path.is_file():
            return self.read_data_info_csv(data_info_path)

        run_info_metadata = self.read_run_info_metadata()

        if run_info_metadata:
            return run_info_metadata

        raise ValueError(
            f'Cannot save inference metadata because no data metadata was found for fold {self.fold}. '
            f'Expected {data_info_path} or run_info_*_f{self.fold}.json in {self.train_path}.'
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
            'image_data_dir': raw_metadata.get('IMAGE_DATA_DIR')
        }

    def read_run_info_metadata(self):
        """Read full run_info JSON metadata when compact data_info CSV is unavailable."""
        metadata_paths = sorted(self.train_path.glob(f'run_info_*_f{self.fold}.json'))

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
            'image_data_dir': data_config.get('image_data_dir')
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

    def write_checkpoint_summary(self, best_epoch, last_epoch, best_val_loss, last_val_loss, best_checkpoint_path, last_checkpoint_path, validation_results_path=None):
        """Write a compact run-level checkpoint summary."""
        summary_path = self.get_checkpoint_summary_path()
        metadata = self.build_checkpoint_metadata(
            checkpoint_type='summary',
            epoch=last_epoch,
            metrics={'best_val_loss': best_val_loss, 'last_val_loss': last_val_loss}
        )
        summary = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'created_at': dt.datetime.now().isoformat(),
            'run_name': self.get_run_name(),
            'fold': int(self.fold),
            'task': metadata['task'],
            'model': metadata['model'],
            'data': metadata['data'],
            'preprocessing': metadata['preprocessing'],
            'inference': metadata['inference'],
            'training': metadata['training'],
            'checkpoints': {
                'best': {
                    'epoch': best_epoch,
                    'val_loss': best_val_loss,
                    'path': str(best_checkpoint_path) if best_checkpoint_path is not None else None
                },
                'last': {
                    'epoch': last_epoch,
                    'val_loss': last_val_loss,
                    'path': str(last_checkpoint_path) if last_checkpoint_path is not None else None
                }
            },
            'validation_inference': {
                'enabled': bool(self.train_config.save_validation_results),
                'path': str(validation_results_path) if validation_results_path is not None else None
            }
        }

        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            json.dump(summary, summary_file, indent=4, default=str)

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
        """Build a consistent name for logs, checkpoints, and plots."""
        lr_label = self.format_number(self.train_config.learning_rate)
        stem_label = int(self.quadruplet_config.small_input_stem)

        channel_label = self.input_channels if self.input_channels is not None else self.quadruplet_config.input_channels
        channel_part = f'_ch{channel_label}' if channel_label is not None else ''

        schedule_label = int(self.train_config.lr_schedule)
        lr_gamma_label = self.format_number(self.train_config.lr_gamma)
        early_delta_label = self.format_number(self.train_config.early_stop_min_delta)

        return (f'points{self.num_of_points}_'
                f'{self.quadruplet_config.network_name}_'
                f'bf{self.quadruplet_config.branch_features}_'
                f'fs{self.quadruplet_config.frozen_stages}_'
                f'stem{stem_label}{channel_part}_'
                f'bs{self.train_config.batch_size}_'
                f'lr{lr_label}_sched{schedule_label}_'
                f'lrs{self.train_config.lr_step_size}_'
                f'lrg{lr_gamma_label}_'
                f'ep{self.train_config.max_training_epochs}_'
                f'esp{self.train_config.early_stop_patience}_'
                f'esd{early_delta_label}_'
                f'esw{self.train_config.early_stop_warmup_epochs}')

    def get_train_csv_path(self):
        """Return the generated training CSV path."""
        return self.train_path / f'Train_f{self.fold}.csv'

    def get_val_csv_path(self):
        """Return the generated validation CSV path."""
        return self.train_path / f'Val_f{self.fold}.csv'

    def get_log_path(self):
        """Return the log CSV path."""
        return self.output_path / f'train_log_f{self.fold}.csv'

    def get_checkpoint_path(self, checkpoint_type):
        """Return the latest or best checkpoint path."""
        return self.output_path / f'model_f{self.fold}_{checkpoint_type}.pth'

    def get_checkpoint_summary_path(self):
        """Return the checkpoint summary JSON path."""
        return self.output_path / f'checkpoint_summary_f{self.fold}.json'

    def get_plot_path(self):
        """Return the plot path."""
        return self.output_path / f'train_plot_f{self.fold}.png'

    def get_validation_output_path(self):
        """Return the validation-image inference output directory."""
        return self.output_path / f'validation_inference_f{self.fold}'

    @staticmethod
    def format_number(value):
        """Format numeric values safely for file names."""
        return f'{value:g}'.replace('-', 'm').replace('.', 'p')

    @staticmethod
    def empty_history():
        """Create the training history store."""
        return {
            'step': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }


def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load a Quadruplet model from a self-describing checkpoint."""
    from .quadruplet import Quadruplet

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
