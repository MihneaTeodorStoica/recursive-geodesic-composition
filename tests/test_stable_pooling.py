import pytest
import torch
import torch.nn.functional as F
from recursive_geodesic_composition.stable_pooling import stable_masked_mean


def test_fp32_accumulation_prevents_long_context_half_overflow():
    hidden=torch.full((1,8190,3),16.,dtype=torch.float16)
    mask=torch.ones(1,8190,dtype=torch.long)
    assert not torch.isfinite(hidden.sum(1)).all()
    value=stable_masked_mean(hidden,mask)
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value,F.normalize(torch.ones(1,3),dim=-1))


def test_stable_mean_matches_double_reference_and_ignores_padding():
    torch.manual_seed(51)
    hidden=torch.randn(3,15,8,dtype=torch.float16)
    mask=torch.ones(3,15,dtype=torch.long);mask[1,7:]=0;mask[2,11:]=0
    expected=F.normalize((hidden.double()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True),dim=-1)
    torch.testing.assert_close(stable_masked_mean(hidden,mask).double(),expected,atol=1e-7,rtol=0)


def test_invalid_attention_mask_rejected():
    with pytest.raises(ValueError):
        stable_masked_mean(torch.ones(1,4,8),torch.zeros(1,4))
