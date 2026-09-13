import math
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from Heatmaps.heatmap_losses import prostate_taus_terms, constraint_terms
from Heatmaps.train_model import TrainModel
from Heatmaps.custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from Heatmaps.utils.io_utils import scale_points, valid_content_mask
from medsam_helpers import configs, pretrained, encoder_double, limited_threads


def shape():
    return torch.tensor([[[.5,.2],[.8,.5],[.5,.8],[.2,.5]]])


def test_geometry_and_swapped_sides():
    angle, sides = prostate_taus_terms(shape())
    assert angle == 0 and sides == 0
    for swap in ([2,1,0,3], [0,3,2,1], [2,3,0,1]):
        assert prostate_taus_terms(shape()[:,swap])[1] > 0
    collapsed = torch.ones(1,4,2)*.5
    assert prostate_taus_terms(collapsed)[1] > 0
    for degrees, penalised in [(19,False),(21,True)]:
        points=shape()
        dx=.3*math.sin(math.radians(degrees));dy=.3*math.cos(math.radians(degrees))
        points[0,0]=torch.tensor([.5-dx,.5-dy]);points[0,2]=torch.tensor([.5+dx,.5+dy])
        assert bool(prostate_taus_terms(points)[0]>0) == penalised


@pytest.mark.parametrize('linear', [[[1.,0.],[0.,1.]], [[0.,-1.],[1.,0.]], [[-1.,0.],[0.,1.]], [[.8,.4],[.2,1.1]]])
def test_inverse_affine_padding_and_gradients(linear):
    size=(80,120); canvas=128
    points=shape()[0].numpy()*30 + np.array([40,20])
    matrix=np.eye(3);matrix[:2,:2]=linear
    centre=np.array([60,40]);matrix[:2,2]=centre-matrix[:2,:2]@centre
    transformed=np.c_[points,np.ones(4)]@matrix.T
    mapped=scale_points(transformed[:,:2],size,canvas)
    yy,xx=torch.meshgrid(torch.arange(canvas),torch.arange(canvas),indexing='ij')
    # Gaussian logits whose spatial expectation closely matches each coordinate.
    logits=torch.stack([-((xx-float(x))**2+(yy-float(y))**2)/2 for x,y in mapped])[None].requires_grad_()
    angle,sides=constraint_terms(logits,valid_content_mask([size],canvas),torch.tensor([size]),torch.tensor(np.linalg.inv(matrix)[None],dtype=torch.float32),temperature=1)
    assert angle.item()<1e-6 and sides.item()<1e-6
    # Swap top/bottom, producing a nonzero loss and useful finite gradient.
    a,b=constraint_terms(logits[:,[2,1,0,3]],valid_content_mask([size],canvas),torch.tensor([size]),torch.tensor(np.linalg.inv(matrix)[None],dtype=torch.float32),temperature=1)
    (a+b).backward()
    assert b>0 and torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_disabled_loss_unchanged_and_enabled_gradients(tmp_path,pretrained):
    data,config,model=configs(tmp_path,pretrained)
    trainer=TrainModel(data,config,model,tmp_path/'run',device='cpu')
    outputs=torch.randn(1,4,16,16,requires_grad=True)
    targets=torch.zeros_like(outputs);mask=torch.ones(1,1,16,16,dtype=torch.bool)
    criterion=trainer.build_criterion()
    base=trainer.calculate_model_loss(outputs,[],targets,criterion,mask)
    torch.testing.assert_close(base,criterion(outputs,targets).mean(),rtol=0,atol=0)
    trainer.train_config=replace(config,landmark_constraint_loss='prostate_taus')
    loss=trainer.calculate_model_loss(outputs,[],targets,criterion,mask,{'original_size':torch.tensor([[16,16]]),'inverse_augmentation':torch.eye(3)[None]})
    assert loss>=base
    loss.backward();assert torch.isfinite(outputs.grad).all()
    with pytest.raises(ValueError,match='4 landmarks'):
        TrainModel(data,trainer.train_config,model,tmp_path/'bad',device='cpu')


def test_dataset_records_exact_inverse_augmentation(tmp_path, pretrained):
    data, config, model = configs(tmp_path, pretrained)
    dataset=HeatmapDataset(HeatmapDatasetConfig(1,1,'training',1,data.fold_lists_path,data.mark_list_file,data.image_data_dir,512,8,input_channels=3,enforce_greyscale=True,oversampling_factor=2))
    original=dataset[0]
    augmented=dataset[len(dataset.records)]
    coordinates=torch.cat((augmented['points_original'],torch.ones(1,1)),dim=-1)
    restored=coordinates@augmented['inverse_augmentation'].T
    torch.testing.assert_close(restored[:,:2],original['points_original'],rtol=1e-5,atol=1e-4)
    torch.testing.assert_close(original['inverse_augmentation'],torch.eye(3))
