import torch
from recursive_geodesic_composition.v2_2_covariance import whitened_covariance,balanced

def test_two_step_detached_equal_corpus_block():
 torch.manual_seed(9); params=[torch.randn(2,6,requires_grad=True) for _ in range(4)]; teachers=[torch.randn(2,6) for _ in range(4)]
 old=[(x.detach().clone(),y.detach().clone()) for x,y in zip(params,teachers)]
 cur=[torch.randn(2,6,requires_grad=True) for _ in range(4)]; tar=[torch.randn(2,6) for _ in range(4)]
 terms=[whitened_covariance(torch.cat((o[0],x)),torch.cat((o[1],y)))[0] for o,x,y in zip(old,cur,tar)]
 loss=balanced(terms);loss.backward()
 assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in cur)
 assert all(x.grad is None for x in params)
 assert torch.equal(balanced([torch.tensor(float(i)) for i in (1,2,3,4)]),torch.tensor(2.5))
