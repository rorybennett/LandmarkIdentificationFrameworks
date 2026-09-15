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
    assert list((tmp_path/'results'/'validation_results').rglob('*.png'))
    assert list((tmp_path/'results'/'validation_best_pixel_error').rglob('*.png'))
    # A clean uninterrupted run must have exactly the same numerical trajectory.
    baseline=trainer_for(tmp_path,**options)
    baseline.output_path=tmp_path/'baseline'
    baseline.train()
    resumed_last=torch.load(resumed.get_checkpoint_path('last_epoch'),weights_only=False)
    baseline_last=torch.load(baseline.get_checkpoint_path('last_epoch'),weights_only=False)
    assert resumed.state_dicts_equal(resumed_last['state_dict'],baseline_last['state_dict'])
    for field in ('training_loss','validation_loss','validation_error_px','lr'):
        assert resumed.history[field]==baseline.history[field]
