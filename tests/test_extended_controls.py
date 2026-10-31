import itertools
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from recursive_geodesic_composition.extended_controls import *


class LinearMass:
    def merge(self, x, n):
        return (x*n[..., None]).sum(1)/n.sum(1)[:, None]


def test_all_five_binary_trees_preserve_order_and_mass():
    trees = binary_trees()
    assert len(trees) == len(set(map(str, trees))) == 5
    x, n = torch.randn(3,4,7), torch.tensor([[1.,2.,3.,4.]]*3)
    expected = (x*n[...,None]).sum(1)/n.sum(1)[:,None]
    for tree in trees:
        def flatten(t):
            return [t] if isinstance(t,int) else flatten(t[0])+flatten(t[1])
        assert flatten(tree) == [0,1,2,3]
        value, count = binary_tree(LinearMass(), x, n, tree)
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(count, n.sum(1))


@pytest.mark.parametrize('arity',[2,4])
def test_ragged_reduction_singletons_and_masses(arity):
    xs = [torch.randn(k,7) for k in (1,2,3,4,7,8,13,16,58)]
    ns = [torch.arange(1,len(x)+1,dtype=torch.float32) for x in xs]
    roots, audit = ragged_reduce(LinearMass(),xs,ns,arity,3)
    for i,(x,n) in enumerate(zip(xs,ns)):
        torch.testing.assert_close(roots[i], (x*n[:,None]).sum(0)/n.sum())
        assert audit['total_masses'][i] == n.sum().item()
    assert audit['depths'][0] == audit['merges'][0] == 0
    assert audit['depths'][7] == (4 if arity == 2 else 2)


def test_slot_permutation_following_cancels_fixed_predictions():
    x,p = torch.randn(5,4,11),torch.randn(5,4,11)
    gaps=[]
    for perm in PERMUTATIONS:
        ordered, shuffled = slot_scores(p,x[:,perm])
        gaps.append(ordered-shuffled)
    torch.testing.assert_close(torch.stack(gaps).mean(0),torch.zeros(5,dtype=torch.float64),atol=1e-14,rtol=0)
    ordered, shuffled = slot_scores(x,x)
    assert (ordered-shuffled > 0).all()


@pytest.mark.parametrize('family',['bounded','unbounded','direct'])
def test_probe_initialization_and_parent_only_shapes(family):
    centroid=torch.randn(4,9)
    p=F.normalize(torch.randn(6,9),dim=-1)
    probe=FlexibleProbe(9,7,family,centroid)
    out=probe(p)
    assert out.shape == (6,4,9)
    torch.testing.assert_close(out.norm(dim=-1),torch.ones(6,4))
    expected=F.normalize(centroid,dim=-1)[None].expand(6,-1,-1) if family=='direct' else p[:,None].expand(-1,4,-1)
    torch.testing.assert_close(out,expected)


def test_unbounded_and_direct_can_leave_residual_cap():
    for family in ('unbounded','direct'):
        probe=FlexibleProbe(8,5,family,torch.ones(4,8))
        with torch.no_grad():
            probe.network[-1].bias.fill_(-100.)
        p=F.normalize(torch.ones(2,8),dim=-1)
        assert (F.cosine_similarity(probe(p),p[:,None],dim=-1) < 0).all()


def test_retrieval_matches_sklearn_without_ties_and_known_ranks():
    scores=np.array([[.9,.6,.3,.1],[.1,.5,.9,.3]])
    rel=np.array([[1,0,1,0],[1,0,0,0]])
    m=retrieval_metrics(scores,rel)
    for i in range(2):
        assert m['ndcg_at_10'][i] == pytest.approx(ndcg_score(rel[i:i+1],scores[i:i+1],k=10))
    np.testing.assert_allclose(m['mrr_at_10'],[1,.25])
    np.testing.assert_allclose(m['recall_at_1'],[.5,0])


def test_stable_retrieval_ties_and_positive_components():
    scores=np.ones((3,3));rel=np.array([[1,1,0],[0,1,0],[0,0,1]])
    m=retrieval_metrics(scores,rel)
    np.testing.assert_allclose(m['mrr_at_10'],[1,.5,1/3])
    groups=positive_components(rel)
    assert groups[0]==groups[1] and groups[2]!=groups[0]


def test_cluster_bootstrap_is_query_weighted_not_cluster_mean():
    values=np.array([0.,0.,1.]);groups=np.array([0,0,1])
    result=cluster_bootstrap(values,groups,100,52)[:,0]
    assert set(result)=={0.,1/3,1.}
    rng=np.random.default_rng(52)
    reference=[]
    for _ in range(100):
        sampled=rng.integers(0,2,2)
        reference.append(np.concatenate([values[groups==g] for g in sampled]).mean())
    np.testing.assert_array_equal(result,reference)
