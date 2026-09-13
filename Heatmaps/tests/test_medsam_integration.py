import json
import sys
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from medsam_helpers import configs, pretrained, encoder_double, limited_threads
from Heatmaps.models import vit_medsam as models
from Heatmaps.train_model import TrainModel
from Heatmaps.heatmap_training_pipeline import parse_args, build_configs, HeatmapTrainingPipeline
from Heatmaps.utils.heatmap_inference_utils import load_model_from_checkpoint, build_config_from_checkpoint_metadata, run_heatmap_inference_for_records, HeatmapImageRecord


@pytest.mark.parametrize('size',[384,500,512,640])
def test_variable_size(size,pretrained):
    model=models.MedSAMHeatmap(2,image_size=size,decoder_channels=16,pretrained_checkpoint=pretrained).eval()
    with torch.no_grad():
        assert model(torch.zeros(1,3,size,size)).shape==(1,2,size,size)
    assert model.image_encoder.pos_embed.shape[1:3]==((size+15)//16,)*2


def test_public_pipeline_and_dual_exports(tmp_path,pretrained):
    data,train,model=configs(tmp_path,pretrained)
    argv=['run','1','1','test','true','false','--network-name','vit-medsam','--device','cpu','--image-size','64','--num-points','1',
          '--fold-lists-path',str(data.fold_lists_path),'--mark-list-file',str(data.mark_list_file),'--image-data-dir',str(data.image_data_dir),
          '--run-dir',str(tmp_path/'outputs'),'--pretrained-checkpoint',str(pretrained),'--max-training-epochs','3','--freeze-encoder-epochs','1',
          '--early-stop-warmup-epochs','2','--train-workers','0','--decoder-channels','16']
    with patch.object(sys,'argv',argv):
        args=parse_args();run,data,train,model=build_configs(args)
    assert train.learning_rate==1e-4 and train.num_workers==0 and data.enforce_greyscale
    pipeline=HeatmapTrainingPipeline(run,data,train,model)
    with patch.object(TrainModel,'validate',side_effect=[{'loss':1.,'error_px':10.},{'loss':2.,'error_px':8.},{'loss':3.,'error_px':9.}]):
        pipeline.run()
    folder=pipeline.run_results_path
    for label,epoch in [('validation_best_loss',1),('validation_best_pixel_error',2)]:
        export=folder/label
        assert (export/'validation_predictions.csv').exists()
        metadata=json.loads((export/'validation_logs/validation_run_metadata.json').read_text())
        assert metadata['checkpoint']['epoch']==epoch
    loaded=load_model_from_checkpoint(folder/'model_best_validation_pixel_error.pth',device='cpu')
    config=build_config_from_checkpoint_metadata(loaded.metadata,tmp_path/'infer')
    run_heatmap_inference_for_records(loaded.model,config,[HeatmapImageRecord('two',tmp_path/'two.png',[(12,7)])],device='cpu')


def test_saus_inverse_rotation():
    from Heatmaps.heatmap_losses import constraint_terms
    y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    # A 180 degree rotated top/bottom pair; inverse returns original order.
    logits=torch.stack([-((x-16)**2+(y-24)**2),-((x-16)**2+(y-8)**2)])[None].float().requires_grad_()
    inverse=torch.tensor([[[-1.,0.,32.],[0.,-1.,32.],[0.,0.,1.]]])
    mask=torch.ones(1,1,32,32,dtype=torch.bool);sizes=torch.tensor([[32,32]])
    a,b=constraint_terms(logits,mask,sizes,inverse,temperature=1,constraint_type='prostate-saus')
    assert a==0 and b==0
    a,b=constraint_terms(logits,mask,sizes,torch.eye(3)[None],temperature=1,constraint_type='prostate-saus')
    b.backward();assert b>0 and torch.isfinite(logits.grad).all()
