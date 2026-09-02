import torch
from recursive_geodesic_composition.v2_2_covariance import covariance_frobenius,whitened_covariance,balanced

def test_zero_and_gradients():
 torch.manual_seed(3);x=torch.randn(8,5,requires_grad=True);assert covariance_frobenius(x,x).item()<1e-12
 y=torch.randn(8,5);loss=covariance_frobenius(x,y);loss.backward();assert torch.isfinite(x.grad).all()
def test_whiten_and_degenerate_rejection():
 x=torch.randn(8,5);v,c=whitened_covariance(x,x);assert v.item()<1e-10 and torch.isfinite(c)
 try: covariance_frobenius(torch.randn(3,5),torch.randn(3,5));assert False
 except ValueError: pass
def test_balanced():
 assert balanced([torch.tensor(1.),torch.tensor(2.),torch.tensor(3.),torch.tensor(4.)]).item()==2.5
