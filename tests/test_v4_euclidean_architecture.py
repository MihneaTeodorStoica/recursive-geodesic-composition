import torch
import torch.nn.functional as F
from recursive_geodesic_composition.architecture import GeodesicMerge

def build(geometry):
    torch.manual_seed(7)
    return GeodesicMerge(d=16,hidden=32,layers=1,heads=4,decoder=True,decoder_mode='strict',
        decoder_layers=1,decoder_length_mode='predicted',importance_mode='semantic',
        geodesic_parameterization='tangent_expmap',arity=4,composition_geometry=geometry,max_angle=.5)

def test_variants_parameter_matched_finite_normalized_and_variable_arity():
    models=[build(x) for x in ('geodesic','euclidean_residual','free_euclidean')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    children=F.normalize(torch.randn(3,4,16),dim=-1)
    lengths=torch.tensor([[3.,2.,0.,0.],[3.,2.,1.,0.],[3.,2.,1.,4.]])
    for m in models:
        roots,aux=m.merge(children,lengths,return_aux=True)
        assert torch.isfinite(roots).all()
        assert torch.allclose(roots.norm(dim=-1),torch.ones(3),atol=1e-6)
        assert aux['composition_geometry']==m.composition_geometry

def test_residual_zero_initialization_is_exact_euclidean_anchor():
    m=build('euclidean_residual'); children=F.normalize(torch.randn(2,4,16),dim=-1); lengths=torch.ones(2,4)
    roots,aux=m.merge(children,lengths,return_aux=True)
    expected=F.normalize(children.mean(1),dim=-1)
    assert torch.allclose(roots,expected,atol=1e-7,rtol=0)
    assert torch.equal(aux['residual_norm'],torch.zeros(2))

def test_free_euclidean_is_not_anchor_constrained():
    m=build('free_euclidean'); children=F.normalize(torch.randn(2,4,16),dim=-1); lengths=torch.ones(2,4)
    roots,aux=m.merge(children,lengths,return_aux=True)
    assert torch.isfinite(roots).all()
    assert (aux['angle']>.1).all()
    assert not torch.allclose(roots,aux['anchor'])
