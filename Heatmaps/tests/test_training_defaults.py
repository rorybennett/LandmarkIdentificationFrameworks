"""Regression coverage for recommended CLI recipes and explicit overrides."""

import json
from dataclasses import asdict
from importlib.resources import files

import cv2
import numpy as np
import pytest

from Heatmaps.heatmap_training_pipeline import build_configs, parse_args
from Heatmaps.model_registry import get_available_model_names
from Heatmaps.train_model import TrainModel


@pytest.fixture
def dataset_args(tmp_path):
    fold = tmp_path / 'folds' / 'repetition_1'
    fold.mkdir(parents=True)
    for number, train, val in ((1, 'one', 'two'), (2, 'two', 'one')):
        (fold / f'training_f{number}.txt').write_text(train + '\n')
        (fold / f'val_f{number}.txt').write_text(val + '\n')
    marks = tmp_path / 'marks.txt'
    marks.write_text('one.png (10, 8)\ntwo.png (12, 7)\n')
    for name in ('one', 'two'):
        cv2.imwrite(str(tmp_path / f'{name}.png'), np.arange(16 * 32, dtype=np.uint8).reshape(16, 32))
    # Configuration validation checks that a local path exists, without loading
    # weights; the existing model integration tests exercise checkpoint loading.
    pretrained = tmp_path / 'pretrained.pth'
    pretrained.touch()
    return ['1', '1', 'test', '--run-dir', str(tmp_path / 'outputs'), '--num-points', '1',
            '--fold-lists-path', str(tmp_path / 'folds'), '--mark-list-file', str(marks),
            '--image-data-dir', str(tmp_path), '--pretrained-checkpoint', str(pretrained)]


@pytest.mark.parametrize('name,batch_size,learning_rate,normalise,blocks', [
    ('unet_basic', 32, 0.001, True, None),
    ('hrnet', 32, 0.001, True, None),
    ('stacked_hourglass', 32, 0.001, True, None),
    ('vitpose', 8, 0.0001, False, 4),
    ('vit-medsam', 8, 0.0001, False, 8),
])
def test_minimal_cli_builds_recommended_configs(dataset_args, name, batch_size, learning_rate, normalise, blocks):
    args = parse_args(dataset_args + ['--network-name', name])
    run, data, train, model = build_configs(args)
    assert data.image_size == 512
    assert data.heatmap_sigma == 8.0
    assert data.oversampling_factor == 8
    assert data.enforce_greyscale is True
    assert data.normalise_inputs is normalise
    assert train.batch_size == batch_size
    assert train.learning_rate == learning_rate
    assert train.max_training_epochs == 300
    assert train.early_stop_patience == 30
    assert train.num_workers == 8
    assert train.landmark_constraint_loss is None
    assert train.constraint_angle_weight == train.constraint_side_weight == 0.0
    assert train.use_amp is False
    assert model.network_name == name
    if blocks is not None:
        assert train.finetune_last_blocks == blocks
    if name == 'stacked_hourglass':
        assert model.hourglass_stacks == 6
    assert f'of8_bs{batch_size}_' in run.run_name

    # Every applicable reference value must reach the actual configuration,
    # including model constructor fields and less visible scheduler options.
    reference = json.loads(files('Heatmaps').joinpath('recommended_normal_loss_defaults.json').read_text(encoding='utf-8'))
    for section, config in (('data_config', data), ('train_config', train), ('model_config', model)):
        expected = {**reference['common'][section], **reference['models'][name][section]}
        actual = asdict(config)
        for key, value in expected.items():
            if key != 'input_channels':  # Resolved from the dataset below.
                assert actual[key] == value, (name, section, key)


def test_default_preprocessing_resolves_three_channels_and_training_statistics(dataset_args, tmp_path):
    args = parse_args(dataset_args + ['--train-workers', '0', '--device', 'cpu'])
    _, data, train, model = build_configs(args)
    trainer = TrainModel(data, train, model, tmp_path / 'results')
    training, validation = trainer.build_data_loaders()
    assert data.input_channels == 3
    assert training.dataset.config.input_channels == validation.dataset.config.input_channels == 3
    assert len(training.dataset) == 8
    assert len(validation.dataset) == 1
    assert data.normalisation_mean is not None
    assert len(set(data.normalisation_mean)) == len(set(data.normalisation_std)) == 1


@pytest.mark.parametrize('name', get_available_model_names())
def test_explicit_options_override_recipe_in_either_argument_order(dataset_args, name):
    before = ['--batch-size=3', '--learning-rate', '0.002', '--max-training-epochs', '2',
              '--train-workers', '0', '--positive-weight', '0', '--save-validation-predictions', 'false']
    after = ['--freeze-encoder-epochs', '0', '--finetune-last-blocks=2', '--hourglass-stacks', '1',
             '--gradient-checkpointing', 'false', '--use-amp', 'true', '--early-stop-patience', '7']
    args = parse_args(before + dataset_args + ['--network-name', name] + after)
    _, _, train, model = build_configs(args)
    assert train.batch_size == 3 and train.learning_rate == 0.002
    assert train.max_training_epochs == 2 and train.early_stop_patience == 7
    assert train.num_workers == train.positive_weight == train.freeze_encoder_epochs == 0
    assert train.save_validation_predictions is False
    assert train.use_amp is True
    assert train.finetune_last_blocks == 2
    assert model.hourglass_stacks == 1
    assert model.gradient_checkpointing is False


def test_cnn_preprocessing_and_geometry_can_be_overridden(dataset_args):
    args = parse_args(dataset_args + ['--normalise-inputs', 'false', '--enforce-greyscale', 'false',
        '--landmark-constraint-loss', 'prostate_saus', '--constraint-angle-weight', '0', '--constraint-side-weight', '0.02'])
    _, data, train, _ = build_configs(args)
    assert data.normalise_inputs is data.enforce_greyscale is False
    assert train.landmark_constraint_loss == 'prostate-saus'
    assert train.constraint_angle_weight == 0 and train.constraint_side_weight == 0.02


@pytest.mark.parametrize('name,batch_size', [('unet_basic', 32), ('vitpose', 8), ('vit-medsam', 8)])
def test_help_uses_selected_model_defaults(name, batch_size, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(['--help', '--network-name', name])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert f'Training batch size. (default: {batch_size})' in help_text
    assert 'Maximum training epochs. (default: 300)' in help_text
    blocks = 8 if name == 'vit-medsam' else 4
    assert f'Number of final encoder blocks to fine-tune. (default: {blocks})' in ' '.join(help_text.split())


def test_every_registered_model_has_a_bundled_recipe():
    reference = json.loads(files('Heatmaps').joinpath('recommended_normal_loss_defaults.json').read_text(encoding='utf-8'))
    assert set(reference['models']) == set(get_available_model_names())
