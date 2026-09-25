import copy
import torch
import torch.nn.functional as F
from recursive_geodesic_composition.architecture import GeodesicMerge
from recursive_geodesic_composition.train import step_loss_batch,_is_decoder_parameter


def test_detaching_parent_removes_only_decoder_facing_encoder_gradient():
    torch.manual_seed(52)
    source=GeodesicMerge(d=8,hidden=16,layers=1,heads=2,arity=4,decoder=True,
        decoder_layers=1,decoder_mode='strict',decoder_length_mode='predicted',
        importance_mode='semantic',geodesic_parameterization='tangent_expmap').eval()
    rows=[{'leaves':F.normalize(torch.randn(4,8),dim=-1),'lengths':torch.tensor([1.,2.,3.,4.]),
           'targets':[F.normalize(torch.randn(8),dim=-1)]} for _ in range(3)]
    values={}
    for detach in (False,True):
        model=copy.deepcopy(source)
        loss,_=step_loss_batch(model,rows,True,torch.device('cpu'),1.,teacher_weight=0.,
            auxiliary_weight_multiplier=0.,importance_weight_multiplier=0.,detach_decoder_parent=detach)
        loss.backward()
        encoder={n:p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n,p in model.named_parameters() if not _is_decoder_parameter(n)}
        decoder={n:p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n,p in model.named_parameters() if _is_decoder_parameter(n)}
        values[detach]=(loss.item(),encoder,decoder)
    assert values[False][0]==values[True][0]
    assert sum(v.square().sum().item() for v in values[False][1].values())>0
    assert sum(v.square().sum().item() for v in values[True][1].values())==0
    for key in values[False][2]:
        torch.testing.assert_close(values[False][2][key],values[True][2][key])
