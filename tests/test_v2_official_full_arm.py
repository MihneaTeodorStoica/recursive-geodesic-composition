import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_v2',ROOT/'scripts/v2_cls/official_v2_1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Model:
 def merge(self,c,l): return F.normalize((c*l[...,None]).sum(1),dim=-1)
 def decode(self,p,return_aux=False): return p[:,None,:].expand(-1,4,-1)
class Probe(torch.nn.Module):
 def forward(self,p): return p[:,None,:].expand(-1,4,-1)

def test_full_arm_tracks_leaf_and_recursive_children_exactly():
 torch.manual_seed(1);leaves=F.normalize(torch.randn(5,8),dim=-1);lengths=torch.tensor([2.,3.,5.,7.,11.])
 root,entries=m.full_merge_diagnostics(Model(),Probe(),leaves,lengths,'cpu')
 assert len(entries)==2
 assert entries[0]['arity']==4 and entries[0]['child_types']==['leaf']*4
 assert entries[1]['arity']==2 and entries[1]['child_types']==['recursive','leaf']
 s=m.summarize_full_merges(entries)
 assert s['genuine_merges']==2 and s['leaf_child_slots']==5 and s['recursive_child_slots']==1
 assert s['probe_exact4_parent_merges']==1 and s['probe_exact4_parent_retention'] is not None
 assert root.shape==(1,8) and torch.isfinite(root).all()

def test_chunk_remainders_and_singleton_depth():
 assert [len(x) for x in m.chunks(list(range(2051)))]==[1024,1024,3]
 assert m.depth(2)==1 and m.depth(3)==1 and m.depth(5)==2
