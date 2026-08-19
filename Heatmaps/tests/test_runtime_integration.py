"""Integration checks that require the Heatmaps runtime dependencies."""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from Heatmaps.heatmap_training_pipeline import HeatmapTrainingPipeline, RunConfig, calculate_fold_collection_sha256
from Heatmaps.train_model import HeatmapDataConfig, HeatmapModelConfig, TrainConfig, TrainModel
from Heatmaps.utils.generate_folds import create_repeated_kfold_lists
from Heatmaps.utils.io_utils import validate_repeated_kfold_lists
from Heatmaps.utils.verify_transforms import read_mark_rows


class RuntimeIntegrationTests(unittest.TestCase):
    def test_runtime_validator_accepts_generated_repeated_kfold_lists(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            mark_list = temporary_path / 'marks.txt'
            mark_list.write_text(''.join(f'patient_{index}.png (1, 2)\n' for index in range(1, 13)), encoding='utf-8')
            fold_root = temporary_path / 'folds'
            create_repeated_kfold_lists(mark_list_path=mark_list, output_dir=fold_root, num_repetitions=2, num_folds=3, base_seed=9,
                                        test_sample_ids=['patient_2', 'patient_11'])

            info = validate_repeated_kfold_lists(fold_root)
            self.assertEqual(info, {'repetitions': [1, 2], 'folds': [1, 2, 3], 'sample_count': 10})
            self.assertTrue((fold_root / 'test_cases.xlsx').is_file())

    def test_runtime_validator_rejects_an_orphan_validation_fold(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            repetition_dir = Path(temporary_dir) / 'folds' / 'repetition_1'
            repetition_dir.mkdir(parents=True)
            (repetition_dir / 'training_f1.txt').write_text('patient_1\n', encoding='utf-8')
            (repetition_dir / 'val_f1.txt').write_text('patient_2\n', encoding='utf-8')
            (repetition_dir / 'val_f2.txt').write_text('patient_1\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'must match exactly'):
                validate_repeated_kfold_lists(repetition_dir.parent)

    def test_transform_verifier_rejects_missing_and_extra_landmarks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            for row, expected_difference in (
                ('patient.png (1, 2)', '1 missing'),
                ('patient.png (1, 2) (3, 4) (5, 6)', '1 extra'),
            ):
                mark_list = root / f'{expected_difference.replace(" ", "_")}.txt'
                mark_list.write_text(f'{row}\n', encoding='utf-8')

                with self.subTest(expected_difference=expected_difference):
                    with self.assertRaisesRegex(ValueError, expected_difference):
                        read_mark_rows(mark_list, expected_points=2)

    def test_repetition_and_fold_output_leaves_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_config = HeatmapDataConfig(repetition=1, fold=1, task_name='task', num_of_points=2, fold_lists_path=root,
                                            mark_list_file=root / 'marks.txt', image_data_dir=root, image_size=(32, 32))
            train_config = TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0)
            model_config = HeatmapModelConfig(depth=1)

            def make_pipeline(repetition, fold):
                run_config = RunConfig(repetition=repetition, fold=fold, task_name='task', num_of_points=2, train_model=True,
                                       copy_files=False, run_dir=root, save_dir=None, run_name='run', fold_collection_sha256='manual-test')
                per_run_data = HeatmapDataConfig(**{**data_config.__dict__, 'repetition': repetition, 'fold': fold})
                return HeatmapTrainingPipeline(run_config=run_config, data_config=per_run_data, train_config=train_config,
                                               model_config=model_config)

            first_path = make_pipeline(1, 1).run_results_path
            second_path = make_pipeline(2, 1).run_results_path
            third_path = make_pipeline(1, 2).run_results_path
            self.assertEqual(first_path, root / 'TRAINING_RESULTS' / 'task' / 'run' / 'repetition_1' / 'fold_1')
            self.assertEqual(len({first_path, second_path, third_path}), 3)

    def test_fold_collection_digest_changes_with_membership(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            fold_root = Path(temporary_dir) / 'folds'
            repetition_dir = fold_root / 'repetition_1'
            repetition_dir.mkdir(parents=True)
            (repetition_dir / 'training_f1.txt').write_text('patient_1\npatient_2\n', encoding='utf-8')
            (repetition_dir / 'val_f1.txt').write_text('patient_3\n', encoding='utf-8')
            first_digest = calculate_fold_collection_sha256(fold_root, repetition_numbers=[1], fold_numbers=[1])

            (repetition_dir / 'val_f1.txt').write_text('patient_4\n', encoding='utf-8')
            second_digest = calculate_fold_collection_sha256(fold_root, repetition_numbers=[1], fold_numbers=[1])

            self.assertEqual(len(first_digest), 64)
            self.assertNotEqual(first_digest, second_digest)

    def test_validation_landmark_mismatch_prevents_cleanup_callback(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fold_dir = root / 'folds' / 'repetition_1'
            image_dir = root / 'images'
            output_dir = root / 'outputs'
            fold_dir.mkdir(parents=True)
            image_dir.mkdir()
            output_dir.mkdir()
            (fold_dir / 'training_f1.txt').write_text('patient_training\n', encoding='utf-8')
            (fold_dir / 'val_f1.txt').write_text('patient_validation\n', encoding='utf-8')
            mark_list = root / 'marks.txt'
            mark_list.write_text('patient_training.png (2, 2) (4, 4)\npatient_validation.png (3, 3)\n', encoding='utf-8')
            self.assertTrue(cv2.imwrite(str(image_dir / 'patient_training.png'), np.zeros((8, 8), dtype=np.uint8)))
            sentinel = output_dir / 'existing_result.txt'
            sentinel.write_text('preserve me', encoding='utf-8')
            callback_called = False

            def cleanup_callback():
                nonlocal callback_called
                callback_called = True
                sentinel.unlink()

            trainer = TrainModel(
                data_config=HeatmapDataConfig(repetition=1, fold=1, task_name='task', num_of_points=2,
                                              fold_lists_path=root / 'folds', mark_list_file=mark_list, image_data_dir=image_dir,
                                              image_size=(8, 8), heatmap_sigma=1.0),
                train_config=TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0),
                model_config=HeatmapModelConfig(base_channels=4, depth=1, max_channels=8),
                output_save_path=output_dir,
            )

            with self.assertRaisesRegex(ValueError, r'repetition 1, fold 1, validation split.*1 missing'):
                trainer.train(on_dataset_validated=cleanup_callback)

            self.assertFalse(callback_called)
            self.assertTrue(sentinel.is_file())

    def test_validation_paths_and_rows_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            trainer = TrainModel(
                data_config=HeatmapDataConfig(repetition=2, fold=3, task_name='task', num_of_points=1, fold_lists_path=root,
                                              mark_list_file=root / 'marks.txt', image_data_dir=root, image_size=(8, 8)),
                train_config=TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0),
                model_config=HeatmapModelConfig(base_channels=4, depth=1, max_channels=8),
                output_save_path=root,
            )
            row = trainer.create_prediction_row(sample_name='patient', target_points=np.asarray([[1, 2]]),
                                                predicted_points=np.asarray([[2, 3]]), point_errors=np.asarray([1.4]))
            self.assertEqual(row['dataset_split'], 'validation')
            self.assertEqual((row['repetition'], row['fold']), (2, 3))
            self.assertEqual(trainer.get_validation_output_path(), root / 'validation_results')
            self.assertEqual(trainer.get_checkpoint_path('best_validation_loss').name, 'model_best_validation_loss.pth')

    def test_one_epoch_training_writes_explicit_validation_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fold_dir = root / 'folds' / 'repetition_1'
            image_dir = root / 'images'
            output_dir = root / 'outputs'
            fold_dir.mkdir(parents=True)
            image_dir.mkdir()
            (fold_dir / 'training_f1.txt').write_text('patient_training\n', encoding='utf-8')
            (fold_dir / 'val_f1.txt').write_text('patient_validation\n', encoding='utf-8')
            mark_list = root / 'marks.txt'
            mark_list.write_text(
                'patient_training.png (2, 2) (5, 5)\n'
                'patient_validation.png (3, 3) (6, 6)\n'
                'patient_held_out.png (1, 1)\n',
                encoding='utf-8',
            )

            for patient_name in ('patient_training', 'patient_validation'):
                self.assertTrue(cv2.imwrite(str(image_dir / f'{patient_name}.png'), np.zeros((8, 8), dtype=np.uint8)))

            callback_called = False

            def validated_callback():
                nonlocal callback_called
                callback_called = True

            trainer = TrainModel(
                data_config=HeatmapDataConfig(repetition=1, fold=1, task_name='task', num_of_points=2,
                                              fold_lists_path=root / 'folds', mark_list_file=mark_list, image_data_dir=image_dir,
                                              image_size=(8, 8), heatmap_sigma=1.0),
                train_config=TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0,
                                         save_validation_predictions=True),
                model_config=HeatmapModelConfig(base_channels=4, depth=1, max_channels=8),
                output_save_path=output_dir,
                device=torch.device('cpu'),
            )
            trainer.train(on_dataset_validated=validated_callback)

            self.assertTrue(callback_called)
            expected_outputs = {
                'model_best_validation_loss.pth', 'model_last_epoch.pth', 'validation_checkpoint_summary.json',
                'training_validation_log.csv', 'training_validation_plot.png', 'validation_results',
            }
            self.assertTrue(expected_outputs.issubset({path.name for path in output_dir.iterdir()}))
            validation_dir = output_dir / 'validation_results'
            expected_validation_outputs = {
                'validation_summary.xlsx', 'validation_image_summary.csv', 'validation_endpoints.csv', 'validation_predictions.csv',
                'validation_heatmap_overlays', 'validation_point_overlays', 'validation_logs',
            }
            self.assertTrue(expected_validation_outputs.issubset({path.name for path in validation_dir.iterdir()}))

            with open(validation_dir / 'validation_predictions.csv', 'r', encoding='utf-8') as predictions_file:
                row = next(csv.DictReader(predictions_file))
            self.assertEqual((row['dataset_split'], row['repetition'], row['fold']), ('validation', '1', '1'))

            checkpoint = torch.load(output_dir / 'model_best_validation_loss.pth', map_location='cpu', weights_only=False)
            self.assertIn('validation_metrics', checkpoint)
            self.assertEqual(checkpoint['checkpoint_type'], 'best_validation_loss')
            self.assertNotIn(None, checkpoint['metadata']['checkpoint'].values())
            self.assertEqual(checkpoint['metadata']['runtime_environment']['framework']['version'], '0.1.0')
            self.assertEqual(checkpoint['metadata']['runtime_environment']['compute']['device_type'], 'cpu')

            last_checkpoint = torch.load(output_dir / 'model_last_epoch.pth', map_location='cpu', weights_only=False)
            self.assertTrue(last_checkpoint['resume_capable'])
            self.assertIn('scheduler_state_dict', last_checkpoint)
            self.assertIn('grad_scaler_state_dict', last_checkpoint)
            self.assertIn('training_state', last_checkpoint)
            self.assertIn('rng_state', last_checkpoint)
            self.assertIn('data_loader_generator_states', last_checkpoint)
            self.assertIn('best_model_state_dict', last_checkpoint)
            self.assertIn('resume_signature', last_checkpoint)

            summary = json.loads((output_dir / 'validation_checkpoint_summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['repetition'], summary['fold']), (1, 1))
            self.assertIn('validation_loss', summary['checkpoints']['best_validation_loss'])
            self.assertIn('validation_error_px', summary['checkpoints']['best_validation_loss'])
            self.assertNotIn('checkpoint', summary['metadata'])
            self.assertEqual(summary['runtime_environment']['framework']['version'], '0.1.0')
            self.assertEqual(summary['termination_reason'], 'max_epochs_reached')
            self.assertGreaterEqual(summary['timing']['cumulative_epoch_duration_seconds'], 0)

            validation_metadata = json.loads(
                (validation_dir / 'validation_logs' / 'validation_run_metadata.json').read_text(encoding='utf-8')
            )
            self.assertEqual(validation_metadata['checkpoint']['type'], 'best_validation_loss')
            self.assertEqual(validation_metadata['checkpoint']['epoch'], checkpoint['epoch'])
            self.assertEqual(validation_metadata['checkpoint']['validation_metrics'] if 'validation_metrics' in validation_metadata['checkpoint'] else {
                'validation_loss': validation_metadata['checkpoint']['validation_loss'],
                'validation_error_px': validation_metadata['checkpoint']['validation_error_px'],
            }, checkpoint['validation_metrics'])
            self.assertEqual(validation_metadata['metadata']['checkpoint']['type'], 'best_validation_loss')
            self.assertIsNotNone(validation_metadata['metadata']['checkpoint']['epoch'])
            self.assertIsNotNone(validation_metadata['metadata']['checkpoint']['validation_metrics'])

            with open(output_dir / 'training_validation_log.csv', 'r', encoding='utf-8') as log_file:
                log_row = next(csv.DictReader(log_file))
            for timing_field in ('epoch_started_at', 'epoch_completed_at', 'training_duration_seconds',
                                 'validation_duration_seconds', 'epoch_duration_seconds'):
                self.assertIn(timing_field, log_row)
            self.assertGreaterEqual(float(log_row['training_duration_seconds']), 0)
            self.assertGreaterEqual(float(log_row['validation_duration_seconds']), 0)
            self.assertGreaterEqual(float(log_row['epoch_duration_seconds']), 0)

    def test_interrupted_training_resumes_from_the_last_completed_epoch_exactly(self):
        class InterruptBeforeSecondEpoch(TrainModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.training_epoch_calls = 0

            def train_epoch(self, *args, **kwargs):
                self.training_epoch_calls += 1

                if self.training_epoch_calls == 2:
                    raise KeyboardInterrupt('simulated interruption before epoch 2')

                return super().train_epoch(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fold_dir = root / 'folds' / 'repetition_1'
            image_dir = root / 'images'
            uninterrupted_output = root / 'uninterrupted'
            resumed_output = root / 'resumed'
            fold_dir.mkdir(parents=True)
            image_dir.mkdir()
            training_names = ['patient_1', 'patient_2', 'patient_3']
            validation_names = ['patient_4']
            (fold_dir / 'training_f1.txt').write_text('\n'.join(training_names) + '\n', encoding='utf-8')
            (fold_dir / 'val_f1.txt').write_text('\n'.join(validation_names) + '\n', encoding='utf-8')
            mark_list = root / 'marks.txt'
            mark_list.write_text(''.join(f'{name}.png ({5 + index}, {7 + index})\n' for index, name in enumerate(training_names + validation_names)),
                                 encoding='utf-8')

            for index, patient_name in enumerate(training_names + validation_names):
                image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
                image = np.roll(image, shift=index, axis=1)
                self.assertTrue(cv2.imwrite(str(image_dir / f'{patient_name}.png'), image))

            data_config = HeatmapDataConfig(
                repetition=1, fold=1, task_name='resume_test', num_of_points=1, fold_lists_path=root / 'folds',
                mark_list_file=mark_list, image_data_dir=image_dir, image_size=(32, 32), heatmap_sigma=1.0,
                oversampling_factor=2, fold_collection_sha256='resume-test-fold-digest',
            )
            train_config = TrainConfig(
                batch_size=2, learning_rate=1e-3, max_training_epochs=3, num_workers=0, random_seed=17,
                lr_schedule='step', lr_step_size=1, lr_gamma=0.5, early_stop_patience=10,
                save_validation_predictions=False,
            )
            model_config = HeatmapModelConfig(base_channels=2, depth=1, max_channels=4, normalisation=None, dropout=0.2)

            TrainModel(data_config=replace(data_config), train_config=replace(train_config),
                       model_config=replace(model_config), output_save_path=uninterrupted_output,
                       device=torch.device('cpu')).train()

            interrupted = InterruptBeforeSecondEpoch(
                data_config=replace(data_config), train_config=replace(train_config),
                model_config=replace(model_config), output_save_path=resumed_output, device=torch.device('cpu')
            )
            with self.assertRaises(KeyboardInterrupt):
                interrupted.train()

            committed_before_resume = torch.load(resumed_output / 'model_last_epoch.pth', map_location='cpu', weights_only=False)
            self.assertEqual(committed_before_resume['epoch'], 1)
            self.assertEqual(committed_before_resume['training_state']['termination_reason'], 'in_progress')

            TrainModel(data_config=replace(data_config), train_config=replace(train_config),
                       model_config=replace(model_config), output_save_path=resumed_output,
                       device=torch.device('cpu'), resume_training=True).train()

            uninterrupted = torch.load(uninterrupted_output / 'model_last_epoch.pth', map_location='cpu', weights_only=False)
            resumed = torch.load(resumed_output / 'model_last_epoch.pth', map_location='cpu', weights_only=False)
            self.assertEqual((uninterrupted['epoch'], resumed['epoch']), (3, 3))
            self.assertEqual(uninterrupted['scheduler_state_dict'], resumed['scheduler_state_dict'])

            for parameter_name in uninterrupted['state_dict']:
                self.assertTrue(torch.equal(uninterrupted['state_dict'][parameter_name], resumed['state_dict'][parameter_name]), parameter_name)

            for field_name in ('epoch', 'lr', 'training_loss', 'training_error_px', 'validation_loss', 'validation_error_px'):
                self.assertEqual(uninterrupted['training_state']['history'][field_name], resumed['training_state']['history'][field_name])

            self.assertTrue(torch.equal(uninterrupted['rng_state']['torch_cpu'], resumed['rng_state']['torch_cpu']))
            self.assertTrue(torch.equal(uninterrupted['data_loader_generator_states']['training'],
                                        resumed['data_loader_generator_states']['training']))
            self.assertTrue(torch.equal(uninterrupted['data_loader_generator_states']['validation'],
                                        resumed['data_loader_generator_states']['validation']))
            self.assertEqual(len(resumed['training_state']['training_sessions']), 2)

            with open(resumed_output / 'training_validation_log.csv', 'r', encoding='utf-8') as log_file:
                resumed_rows = list(csv.DictReader(log_file))
            self.assertEqual([int(row['epoch']) for row in resumed_rows], [1, 2, 3])

    def test_atomic_checkpoint_failure_preserves_the_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'model_last_epoch.pth'
            checkpoint_path.write_bytes(b'previous committed checkpoint')

            with patch('Heatmaps.train_model.torch.save', side_effect=OSError('simulated write failure')):
                with self.assertRaisesRegex(OSError, 'simulated write failure'):
                    TrainModel.atomic_torch_save({'new': 'payload'}, checkpoint_path)

            self.assertEqual(checkpoint_path.read_bytes(), b'previous committed checkpoint')
            self.assertFalse((checkpoint_path.parent / '.model_last_epoch.pth.tmp').exists())


if __name__ == '__main__':
    unittest.main()
