"""Runtime integration checks for comparison-ready IPV outputs."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from IPV.ipv_training_pipeline import DataCreationConfig, IPVTrainingPipeline, RunConfig
from IPV.train_model import HISTORY_FIELDS, QuadrupletConfig, TrainConfig, TrainModel
from IPV.utils.landmark_inference_utils import (LandmarkImageRecord, build_prediction_rows, build_result,
                                                load_model_from_checkpoint)


class RuntimeIntegrationTests(unittest.TestCase):
    @staticmethod
    def make_configs(root, repetition=1, fold=1):
        run_config = RunConfig(repetition=repetition, fold=fold, task_name='task', num_of_points=1,
                               create_data=False, train_model=True, copy_files=False, delete_files=False,
                               run_dir=root, save_dir=None, run_name='run', fold_collection_sha256='digest')
        data_config = DataCreationConfig(distance_intervals=[(0, 10)], angle_intervals=[(0, 360)],
                                         num_of_repetitions=2, num_of_folds=3, sub_patch_scales=[8, 12, 16, 20],
                                         patches_per_training_sample=2, val_grid_spacing=4, fold_lists_path=root / 'folds',
                                         mark_list_file=root / 'marks.txt', image_data_dir=root / 'images',
                                         sampling_variances=(1,), num_workers=1, random_seed=42,
                                         keep_part_csvs=False, fold_collection_sha256='digest')
        train_config = TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0)
        model_config = QuadrupletConfig(network_name='small_cnn', branch_features=8, small_input_stem=False)
        return run_config, data_config, train_config, model_config

    def test_repetition_and_fold_output_leaves_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = []
            for repetition, fold in ((1, 1), (2, 1), (1, 2)):
                configs = self.make_configs(root, repetition=repetition, fold=fold)
                paths.append(IPVTrainingPipeline(*configs).run_results_path)

            self.assertEqual(paths[0], root / 'TRAINING_RESULTS' / 'task' / 'run' / 'repetition_1' / 'fold_1')
            self.assertEqual(len(set(paths)), 3)

    def test_validation_rows_expose_common_and_ipv_specific_metrics(self):
        record = LandmarkImageRecord('patient', Path('patient.png'), [(2.0, 3.0)])
        result = build_result(record=record, detected_points=[(5, 7)], ground_truth_points=record.ground_truth_points,
                              peak_values=[0.8], num_centres=10, grid_spacing=4,
                              checkpoint_type='best_validation_loss', image_shape=(20, 30),
                              dataset_split='validation', repetition=2, fold=3)
        summary = result['summary']
        endpoint = result['endpoint_rows'][0]
        self.assertEqual((summary['dataset_split'], summary['repetition'], summary['fold']), ('validation', 2, 3))
        self.assertAlmostEqual(summary['mean_error_px'], 5.0)
        self.assertEqual((endpoint['target_x'], endpoint['target_y'], endpoint['pred_x'], endpoint['pred_y']), (2.0, 3.0, 5, 7))
        self.assertEqual(endpoint['error_px'], endpoint['point_error_px'])
        prediction = build_prediction_rows([summary], [endpoint])[0]
        self.assertEqual((prediction['target_x1'], prediction['pred_x1']), (2.0, 5))

    def test_epoch_history_uses_comparison_ready_schema(self):
        history = TrainModel.empty_history()
        TrainModel.update_history(history=history, epoch=1, epoch_started_at='start', epoch_completed_at='end', epoch_lr=0.01,
                                  training_metrics={'loss': 1.0, 'accuracy': 0.5},
                                  validation_metrics={'loss': 0.8, 'accuracy': 0.6, 'error_px': 4.2},
                                  training_duration_seconds=1.0, validation_duration_seconds=2.0,
                                  epoch_duration_seconds=3.0)
        self.assertEqual(tuple(history), HISTORY_FIELDS)
        self.assertEqual(history['validation_error_px'], [4.2])

    def test_atomic_checkpoint_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / 'model_last_epoch.pth'
            path.write_bytes(b'previous')

            with patch('IPV.train_model.torch.save', side_effect=OSError('simulated failure')):
                with self.assertRaisesRegex(OSError, 'simulated failure'):
                    TrainModel.atomic_torch_save({'new': True}, path)

            self.assertEqual(path.read_bytes(), b'previous')
            self.assertFalse((path.parent / '.model_last_epoch.pth.tmp').exists())

    def test_one_epoch_training_writes_comparison_ready_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / 'data'
            output_dir = root / 'results' / 'task' / 'run' / 'repetition_1' / 'fold_1'
            patch_dir = data_dir / 'patches'
            image_dir = root / 'images'
            patch_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            source_image = np.full((8, 8), 127, dtype=np.uint8)
            source_path = image_dir / 'patient.png'
            self.assertTrue(cv2.imwrite(str(source_path), source_image))
            mark_list = root / 'marks.txt'
            mark_list.write_text('patient.png (4, 4)\n', encoding='utf-8')

            def write_split(split_name, centres):
                rows = []
                for group_index, (x, y) in enumerate(centres, start=1):
                    patch_id = f'{split_name}_{group_index}'
                    for scale_index in range(1, 5):
                        patch_path = patch_dir / f'{patch_id}_{scale_index}.png'
                        patch_image = np.full((8, 8), 30 * (group_index + scale_index), dtype=np.uint8)
                        self.assertTrue(cv2.imwrite(str(patch_path), patch_image))
                        rows.append([patch_id, patch_path.as_posix(), 'patient', x, y, 0, 0])
                with open(data_dir / f'{split_name}_f1.csv', 'w', newline='', encoding='utf-8') as split_file:
                    csv.writer(split_file).writerows(rows)

            write_split('Train', [(2, 2), (5, 5)])
            write_split('Val', [(3, 3)])
            metadata_headers = ['TASK_NAME', 'NUM_OF_POINTS', 'SUB_PATCH_SCALES', 'PATCH_SIZE',
                                'PATCHES_PER_TRAINING_SAMPLE', 'GRID_DATA_STEP', 'SAMPLING_VARIANCES',
                                'RANDOM_SEED', 'MARK_LIST_FILE', 'IMAGE_DATA_DIR']
            metadata_values = ['task', 1, '[8, 10, 12, 14]', 8, 2, 4, '(1,)', 42, mark_list, image_dir]
            with open(data_dir / 'data_info.csv', 'w', newline='', encoding='utf-8') as metadata_file:
                writer = csv.writer(metadata_file)
                writer.writerow(metadata_headers)
                writer.writerow(metadata_values)

            trainer = TrainModel(
                repetition=1,
                current_fold=1,
                num_of_points=1,
                data_save_path=data_dir,
                output_save_path=output_dir,
                tasks_classes=[[(0, 2)], [(0, 360)]],
                train_config=TrainConfig(batch_size=2, learning_rate=1e-3, max_training_epochs=1,
                                         num_workers=0, lr_schedule='none', save_validation_results=False),
                quadruplet_config=QuadrupletConfig(network_name='small_cnn', branch_features=2,
                                                   frozen_stages=0, small_input_stem=False),
                device='cpu',
                fold_collection_sha256='test-digest',
            )
            returned_checkpoint = trainer.train()

            self.assertEqual(returned_checkpoint, output_dir / 'model_best_validation_loss.pth')
            for output_name in ('model_best_validation_loss.pth', 'model_last_epoch.pth',
                                'validation_checkpoint_summary.json', 'training_validation_log.csv',
                                'training_validation_plot.png'):
                self.assertTrue((output_dir / output_name).is_file(), output_name)

            with open(output_dir / 'training_validation_log.csv', newline='', encoding='utf-8') as log_file:
                log_rows = list(csv.DictReader(log_file))
            self.assertEqual(len(log_rows), 1)
            self.assertEqual(tuple(log_rows[0]), HISTORY_FIELDS)
            self.assertTrue(np.isfinite(float(log_rows[0]['validation_error_px'])))

            summary = json.loads((output_dir / 'validation_checkpoint_summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['repetition'], summary['fold']), (1, 1))
            self.assertEqual(summary['training_status'], 'completed')
            self.assertEqual(summary['termination_reason'], 'max_epochs_reached')

            loaded_checkpoint = load_model_from_checkpoint(returned_checkpoint, device='cpu')
            self.assertEqual(loaded_checkpoint.metadata['num_points'], 1)
            self.assertEqual(loaded_checkpoint.metadata['checkpoint_type'], 'best_validation_loss')


if __name__ == '__main__':
    unittest.main()
