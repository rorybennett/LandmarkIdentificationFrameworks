"""Behavioural coverage for features transferred from Heatmaps."""
import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from IPV.landmark_losses import expected_landmark_offsets, constraint_terms, prostate_taus_terms, prostate_saus_terms
from IPV.train_model import TrainModel, TrainConfig, QuadrupletConfig
from IPV.training_options import add_training_arguments, validate_training_options
from IPV.validation_progress import ValidationProgress
from IPV.utils.landmark_inference_utils import draw_points, draw_points_with_point_colours


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer_for(root, **overrides):
    config = TrainConfig(batch_size=2, learning_rate=0.001, max_training_epochs=2,
                         num_workers=0, save_validation_results=False)
    config = replace(config, **overrides)
    return TrainModel(1, 'all', 2, root/'data',
        [[(0, 4), (4, 24)], [(0, 90), (90, 180), (180, 270), (270, 360)]],
        config, QuadrupletConfig(network_name='small_cnn', branch_features=2, small_input_stem=False),
        output_save_path=root/'results', device='cpu')


def create_data(root):
    data, images = root/'data', root/'images'
    data.mkdir(parents=True)
    images.mkdir()
    patches = data/'patches'
    patches.mkdir()
    marks = root/'marks.txt'
    names = ['patient_a', 'patient_b']
    marks.write_text(''.join(f'{name}.png (8, 4) (8, 12)\n' for name in names))
    folds = root/'folds'/'repetition_1'
    folds.mkdir(parents=True)
    for filename in ('training_fall.txt','val_fall.txt'):
        (folds/filename).write_text('\n'.join(names)+'\n')
    for name in names:
        assert cv2.imwrite(str(images/f'{name}.png'), np.full((16,16),100,dtype=np.uint8))
    for phase in ('Train','Val'):
        rows=[]
        for i,name in enumerate(names):
            for scale in range(4):
                path=patches/f'{phase}_{name}_{scale}.png'
                assert cv2.imwrite(str(path),np.full((8,8),30+20*scale+i,dtype=np.uint8))
                rows.append([f'{phase}_{name}',str(path),name,8,8,1,0,1,2])
        with (data/f'{phase}_fall.csv').open('w',newline='') as f:
            csv.writer(f).writerows(rows)
    with (data/'data_info.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['TASK_NAME','NUM_OF_POINTS','SUB_PATCH_SCALES','PATCH_SIZE','PATCHES_PER_TRAINING_SAMPLE',
                         'GRID_DATA_STEP','SAMPLING_VARIANCES','RANDOM_SEED','MARK_LIST_FILE','IMAGE_DATA_DIR','FOLD_LISTS_PATH','ENFORCE_GREYSCALE'])
        writer.writerow(['sagittal',2,'[8, 10, 12, 14]',8,2,8,'(1,)',42,marks,images,folds.parent,False])


def test_expected_offsets_follow_ipv_angle_direction_and_wrap():
    sizes=torch.tensor([[3.,4.]])
    # Equal probabilities either side of zero point left from the patch centre.
    offsets=expected_landmark_offsets([torch.zeros(1,1),torch.zeros(1,2)],
        [[(4,6)],[(350,360),(0,10)]],sizes)
    assert offsets[0,0,0] < -0.99
    assert abs(offsets[0,0,1].item()) < 1e-6
    half=expected_landmark_offsets([torch.zeros(1,1),torch.zeros(1,2)],
        [[(4,6)],[(350,360),(0,10)]],sizes*2)
    torch.testing.assert_close(half,offsets/2)


def test_geometry_penalises_wrong_order_and_collapsed_axes():
    good=torch.tensor([[[0.,-0.2],[0.2,0.], [0.,0.2],[-0.2,0.]]])
    angle,side=prostate_taus_terms(good)
    assert angle == 0 and side == 0
    assert prostate_taus_terms(good[:,[2,1,0,3]])[1] > 0
    assert prostate_taus_terms(torch.zeros_like(good))[1] > 0
    sagittal=torch.tensor([[[0.,-0.2],[0.,0.2]]])
    assert prostate_saus_terms(sagittal)[1] == 0
    assert prostate_saus_terms(sagittal.flip(1))[1] > 0


@pytest.mark.parametrize('kind,points',[('prostate_taus',4),('prostate-saus',2)])
def test_geometry_reaches_both_classification_heads(kind,points):
    torch.manual_seed(7)
    outputs=[torch.randn(3,n,requires_grad=True) for _ in range(points) for n in (2,4)]
    angle,side=constraint_terms(outputs,[[(0,4),(4,24)],[(0,90),(90,180),(180,270),(270,360)]],
        torch.tensor([[16.,16.]]*3), margin=1.,constraint_type=kind)
    (angle+side).backward()
    for output in outputs:
        assert torch.isfinite(output.grad).all()
        assert output.grad.abs().sum() > 0


def test_disabled_geometry_is_exact_original_classification_loss(tmp_path):
    trainer=trainer_for(tmp_path)
    outputs=[torch.randn(2,n,requires_grad=True) for n in (2,4,2,4)]
    labels=torch.zeros((2,4),dtype=torch.long)
    criterion=torch.nn.CrossEntropyLoss()
    expected=sum(criterion(output,labels[:,i]) for i,output in enumerate(outputs))/4
    torch.testing.assert_close(trainer.calculate_loss(outputs,labels,criterion),expected,rtol=0,atol=0)


def test_cli_aliases_and_invalid_geometry(tmp_path):
    parser=argparse.ArgumentParser()
    add_training_arguments(parser)
    assert parser.parse_args(['--landmark-constraint-loss','prostate_saus']).landmark_constraint_loss=='prostate-saus'
    assert parser.parse_args(['--landmark-constraint-loss','none']).landmark_constraint_loss is None
    with pytest.raises(ValueError,match='requires 4'):
        trainer_for(tmp_path,landmark_constraint_loss='prostate_taus')
    for key,value in [('constraint_temperature',0),('constraint_side_weight',-1),('constraint_margin',float('nan')),
                      ('visualise_validation_progress_images',-1),('lr_decay_epochs',0),('lr_min_factor',2)]:
        with pytest.raises(ValueError):
            trainer_for(tmp_path,**{key:value})


@pytest.mark.parametrize('schedule',['linear','cosine','exponential','plateau'])
def test_schedulers_reach_floor_and_restore(schedule,tmp_path):
    trainer=trainer_for(tmp_path,lr_schedule=schedule,lr_decay_epochs=4,lr_min_factor=0.1,lr_plateau_patience=0)
    model=torch.nn.Linear(1,1)
    optimiser=trainer.build_optimiser(model)
    scheduler=trainer.build_scheduler(optimiser)
    for i in range(12):
        optimiser.step()
        scheduler.step(1.) if schedule=='plateau' else scheduler.step()
    assert optimiser.param_groups[0]['lr']==pytest.approx(trainer.train_config.learning_rate*0.1)
    other_optimiser=trainer.build_optimiser(model)
    other=trainer.build_scheduler(other_optimiser)
    other_optimiser.load_state_dict(optimiser.state_dict())
    other.load_state_dict(scheduler.state_dict())
    optimiser.step(); other_optimiser.step()
    scheduler.step(1.) if schedule=='plateau' else scheduler.step()
    other.step(1.) if schedule=='plateau' else other.step()
    assert optimiser.param_groups[0]['lr']==other_optimiser.param_groups[0]['lr']


@pytest.mark.parametrize('coloured',[False,True])
def test_point_markers_have_visible_white_backing(coloured):
    image=np.zeros((64,64,3),dtype=np.uint8)
    if coloured:
        draw_points_with_point_colours(image,[(32,32)],'P')
    else:
        draw_points(image,[(32,32)],(0,0,255),'P')
    np.testing.assert_array_equal(image[32,32],[0,0,255])
    assert np.any(np.all(image[20:44,20:44]==255,axis=-1))


def test_progress_restores_weights_rng_and_generators_after_failure(tmp_path):
    trainer=trainer_for(tmp_path)
    trainer.training_generator=torch.Generator().manual_seed(4)
    trainer.validation_generator=torch.Generator().manual_seed(5)
    model=torch.nn.Linear(1,1).train()
    original=trainer.clone_state_dict_to_cpu(model.state_dict())
    state=trainer.capture_rng_state()
    generators=trainer.capture_data_loader_generator_states()
    progress=object.__new__(ValidationProgress)
    progress.interval=1
    progress.records=[object()]
    progress.sample_names=['patient']
    def fail(*args,**kwargs):
        model.eval()
        with torch.no_grad(): model.weight.add_(99)
        torch.rand(1); np.random.rand(); random.random()
        torch.rand(1,generator=trainer.training_generator)
        raise RuntimeError('simulated export failure')
    with patch.object(trainer,'load_checkpoint_state',side_effect=fail):
        with pytest.raises(RuntimeError,match='simulated'):
            progress.render(trainer,model,1)
    assert model.training
    assert trainer.state_dicts_equal(original,model.state_dict())
    actual=trainer.capture_rng_state()
    for key,value in state.items():
        if torch.is_tensor(value): torch.testing.assert_close(actual[key],value)
        elif key=='numpy':
            assert actual[key][0]==value[0]
            np.testing.assert_array_equal(actual[key][1],value[1])
            assert actual[key][2:]==value[2:]
        else: assert actual[key]==value
    for key,value in generators.items():
        torch.testing.assert_close(trainer.capture_data_loader_generator_states()[key],value)


def test_training_progress_exports_constraints_and_resume(tmp_path):
    create_data(tmp_path)
    options=dict(landmark_constraint_loss='prostate-saus',visualise_validation_progress_images=1,
                 visualise_validation_progress_epochs=1,lr_schedule='cosine',save_validation_results=True)
    trainer=trainer_for(tmp_path,**options)
    original_train_epoch=trainer.train_epoch
    calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2: raise KeyboardInterrupt()
        return original_train_epoch(*args,**kwargs)
    with patch.object(trainer,'train_epoch',side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt): trainer.train()
    committed=torch.load(trainer.get_checkpoint_path('last_epoch'),weights_only=False)
    assert committed['epoch']==1
    assert committed['best_pixel_checkpoint']['epoch']==1
    # A newer stray sibling must be replaced by the committed last-epoch snapshot.
    torch.save({'stale':True},trainer.get_checkpoint_path('best_validation_pixel_error'))
    resumed=trainer_for(tmp_path,**options)
    resumed.resume_training=True
    resumed.train()
    assert resumed.resume_state_validated
    assert len(resumed.history['epoch'])==2
    assert resumed.history['training_loss_component_constraint_side'][0] >= 0
    for phase in ('training','validation'):
        total=np.array(resumed.history[f'{phase}_loss'])
        components=sum(np.array(resumed.history[f'{phase}_loss_component_{name}']) for name in ('classification','constraint_angle','constraint_side'))
        np.testing.assert_allclose(total,components,rtol=1e-6,atol=1e-7)
    selections=[]
    for epoch in (1,2):
        for kind in ('best_validation_loss','best_validation_pixel_error'):
            directory=tmp_path/'results'/'validation_progress'/f'epoch_{epoch:04d}'/kind
            selection=json.loads((directory/'selection.json').read_text())
            assert selection['observed_epoch']==epoch
            assert selection['checkpoint_epoch']<=epoch
            selections.append(selection['sample_names'])
            assert list(directory.rglob('*.png'))
    assert all(names==selections[0] for names in selections)
    assert list((tmp_path/'results'/'validation_best_loss').rglob('*.png'))
    assert list((tmp_path/'results'/'validation_best_pixel_error').rglob('*.png'))
    summary = json.loads(resumed.get_checkpoint_summary_path().read_text())
    assert set(summary['validation_exports']) == {'validation_best_loss', 'validation_best_pixel_error'}
    for label, paths in summary['validation_exports'].items():
        directory = tmp_path/'results'/label
        assert set(paths) == {'validation_summary_xlsx', 'validation_image_summary_csv',
                              'validation_endpoints_csv', 'validation_predictions_csv'}
        assert all(Path(path).is_file() and Path(path).parent == directory for path in paths.values())
        metadata = json.loads((directory/'validation_logs'/'validation_run_metadata.json').read_text())
        assert Path(metadata['config']['output_dir']) == directory
        assert all(Path(path).is_file() and Path(path).parent == directory
                   for path in metadata['output_paths'].values())
        kind = label.replace('validation_best_', 'best_validation_')
        checkpoint = torch.load(resumed.get_checkpoint_path(kind), weights_only=False)
        assert metadata['checkpoint_metadata']['checkpoint']['type'] == kind
        assert metadata['checkpoint_metadata']['checkpoint']['epoch'] == summary['checkpoints'][kind]['epoch']
        with Path(paths['validation_endpoints_csv']).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert {row['checkpoint_type'] for row in rows} == {kind}
        for row in rows:
            error = np.hypot(float(row['pred_x'])-float(row['target_x']),
                             float(row['pred_y'])-float(row['target_y']))
            assert float(row['error_px']) == pytest.approx(error)
    assert not (tmp_path/'results'/'validation_results').exists()
    # A clean uninterrupted run must have exactly the same numerical trajectory.
    baseline=trainer_for(tmp_path,**options)
    baseline.output_path=tmp_path/'baseline'
    baseline.train()
    resumed_last=torch.load(resumed.get_checkpoint_path('last_epoch'),weights_only=False)
    baseline_last=torch.load(baseline.get_checkpoint_path('last_epoch'),weights_only=False)
    assert resumed.state_dicts_equal(resumed_last['state_dict'],baseline_last['state_dict'])
    for field in ('training_loss','validation_loss','validation_error_px','lr'):
        assert resumed.history[field]==baseline.history[field]


def test_training_defaults_match_heatmap_cli_and_allow_overrides(tmp_path):
    import sys
    from IPV.ipv_training_pipeline import parse_args
    heatmap_defaults = json.loads((Path(__file__).resolve().parents[2] / 'Heatmaps' / 'Heatmaps' /
                                  'recommended_normal_loss_defaults.json').read_text())['common']['train_config']
    argv = ['ipv-train', '1', '1', 'prostate_sagittal', 'false', 'true', 'false', 'false',
            '--run-dir', str(tmp_path), '--num-points', '2', '--fold-lists-path', str(tmp_path),
            '--mark-list-file', str(tmp_path/'marks.txt'), '--image-data-dir', str(tmp_path),
            '--data-creation-workers', '1', '--train-workers', '0', '--random-seed', '42',
            '--keep-part-csvs', 'false', '--batch-size', '2', '--learning-rate', '0.001',
            '--patches-per-training-sample', '2', '--val-grid-spacing', '8',
            '--network-name', 'small_cnn', '--branch-features', '2', '--frozen-stages', '0',
            '--small-input-stem', 'false']
    with patch.object(sys, 'argv', argv):
        args = parse_args()
    config = TrainConfig(batch_size=2, learning_rate=0.001)
    for name in ('max_training_epochs', 'early_stop_patience', 'early_stop_min_delta',
                 'early_stop_warmup_epochs', 'lr_schedule', 'lr_gamma', 'lr_plateau_patience', 'lr_min_factor'):
        assert getattr(args, name) == getattr(config, name) == heatmap_defaults[name]
    with patch.object(sys, 'argv', argv + ['--max-training-epochs', '17', '--early-stop-patience', '4',
                                         '--lr-plateau-patience', '2', '--save-validation-predictions', 'false']):
        args = parse_args()
    assert (args.max_training_epochs, args.early_stop_patience, args.lr_plateau_patience) == (17, 4, 2)
    assert args.save_validation_results is False


@pytest.mark.parametrize('interrupt_after', [None, 3])
def test_loss_stopping_and_pixel_scheduler_remain_independent_on_resume(tmp_path, interrupt_after):
    create_data(tmp_path)
    options = dict(max_training_epochs=10, early_stop_patience=2, early_stop_min_delta=0.1,
                   early_stop_warmup_epochs=3, lr_schedule='plateau', lr_plateau_patience=0)
    trainer = trainer_for(tmp_path, **options)
    # Loss always improves, but only epoch 4 improves enough to reset stopping patience.
    # Pixel error plateaus at epochs 3, 5 and 6 and must reduce LR independently of loss.
    losses = [1.0, 0.96, 0.92, 0.85, 0.84, 0.83]
    errors = [10.0, 9.0, 9.0, 8.0, 8.0, 8.0]
    metrics = [{'loss': loss, 'accuracy': 0.5, 'error_px': error} for loss, error in zip(losses, errors)]
    if interrupt_after is not None:
        with patch.object(trainer, 'validate', side_effect=metrics[:interrupt_after] + [KeyboardInterrupt()]):
            with pytest.raises(KeyboardInterrupt):
                trainer.train()
        saved = torch.load(trainer.get_checkpoint_path('last_epoch'), weights_only=False)
        assert saved['epoch'] == interrupt_after
        assert saved['training_state']['bad_epochs'] == 1
        trainer = trainer_for(tmp_path, **options)
        trainer.resume_training = True
    with patch.object(trainer, 'validate', side_effect=metrics[interrupt_after or 0:]):
        trainer.train()
    assert trainer.history['epoch'] == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(trainer.history['lr'], [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.00025])
    assert trainer.termination_reason == 'early_stopping'
    summary = json.loads(trainer.get_checkpoint_summary_path().read_text())
    for kind, epoch in (('best_validation_loss', 6), ('best_validation_pixel_error', 4), ('last_epoch', 6)):
        descriptor = summary['checkpoints'][kind]
        checkpoint = torch.load(trainer.get_checkpoint_path(kind), weights_only=False)
        assert descriptor['type'] == checkpoint['checkpoint_type'] == kind
        assert descriptor['epoch'] == checkpoint['epoch'] == epoch
        assert descriptor['validation_loss'] == losses[epoch - 1]
        assert descriptor['validation_error_px'] == errors[epoch - 1]
    assert summary['training']['early_stopping_monitor'] == 'validation_loss'
    assert summary['training']['lr_plateau_monitor'] == 'validation_error_px'
    assert summary['validation_exports'] is None
    assert summary['validation_summary_path'] is None
    assert summary['validation_best_pixel_error_path'] is None
    assert not (trainer.output_path/'validation_best_loss').exists()
    assert not (trainer.output_path/'validation_best_pixel_error').exists()
    for filename in ('training_validation_plot.png', 'individual_loss_plot.png', 'learning_rate_plot.png'):
        assert (trainer.output_path/filename).is_file()


def test_plots_separate_components_and_record_epoch_start_lr(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    trainer = trainer_for(tmp_path)
    trainer.output_path.mkdir(parents=True)
    history = {'epoch': [1, 2], 'training_loss': [2.1, 1.2], 'validation_loss': [3.2, 2.1],
               'validation_error_px': [5, 4], 'lr': [0.001, 0.0005],
               'training_loss_component_classification': [2, 1],
               'validation_loss_component_classification': [3, 2],
               'training_loss_component_constraint_side': [0.1, 0.2],
               'validation_loss_component_constraint_side': [0.2, 0.1],
               'training_loss_component_constraint_angle': [0, 0],
               'validation_loss_component_constraint_angle': [0, 0]}
    figures = {}
    def capture(figure, path, *args, **kwargs):
        figures[Path(path).name] = figure
    with patch.object(Figure, 'savefig', capture):
        trainer.save_history_plot(history)
    main = figures['training_validation_plot.png']
    assert len(main.axes) == 2
    assert [line.get_label() for line in main.axes[1].lines] == ['validation_error_px']
    components = figures['individual_loss_plot.png'].axes[0].lines
    assert len(components) == 4
    for train_line, val_line in zip(components[::2], components[1::2]):
        assert train_line.get_color() == val_line.get_color()
        assert train_line.get_linestyle() == '-' and val_line.get_linestyle() == '--'
    np.testing.assert_array_equal(figures['learning_rate_plot.png'].axes[0].lines[0].get_ydata(), history['lr'])
    assert all(not plt.fignum_exists(figure.number) for figure in figures.values())


def test_fresh_run_clears_new_and_legacy_outputs_only(tmp_path):
    from IPV.ipv_training_pipeline import IPVTrainingPipeline
    pipeline = object.__new__(IPVTrainingPipeline)
    pipeline.run_results_path = tmp_path/'fold_1'
    pipeline.run_results_path.mkdir()
    for name in ('validation_best_loss', '.validation_best_loss.tmp', '.validation_best_loss.backup',
                 'validation_best_pixel_error', 'validation_results', 'individual_loss_plot.png',
                 'learning_rate_plot.png'):
        path = pipeline.run_results_path/name
        if path.suffix == '.png':
            path.write_bytes(b'old plot')
        else:
            path.mkdir()
            (path/'old.txt').write_text('stale')
    notes = pipeline.run_results_path/'notes.txt'
    notes.write_text('retain')
    sibling = tmp_path/'fold_2'
    sibling.mkdir()
    (sibling/'validation_best_loss').mkdir()
    pipeline.clear_existing_fold_outputs()
    assert list(pipeline.run_results_path.iterdir()) == [notes]
    assert (sibling/'validation_best_loss').is_dir()
