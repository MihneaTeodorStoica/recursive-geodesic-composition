#!/usr/bin/env python3
import copy,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from recursive_geodesic_composition.train import run
from scripts.v2_2_cls.pilot import _manifest
ARMS={'BASELINE_R':0.,'F8192':8192.,'F32768':32768.,'F131072':131072.,'F524288':524288.}
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--arm',action='append',choices=ARMS);a=p.parse_args()
 base=yaml.safe_load((ROOT/'configs/v2_2_cls/pilot.yaml').read_text());spec=yaml.safe_load((ROOT/'configs/v2_2_cls/covariance_refinement_v1.yaml').read_text());_manifest(base['cache_dir']);out=ROOT/'results/v2_2_cls/covariance_refinement_v1';out.mkdir(parents=True,exist_ok=True)
 for name,w in ARMS.items():
  if a.arm and name not in a.arm:continue
  c=copy.deepcopy(base);c['output_dir']=str(out/name.lower());c['covariance_mode']='none' if w==0 else 'frobenius';c['covariance_weight']=w;c['residual_mean_weight']=0.;c['study']='v2_2_covariance_refinement_v1_'+name
  if Path(c['output_dir']).exists():raise FileExistsError(c['output_dir'])
  run(c,'encoder',424242,torch.device('cuda'))
if __name__=='__main__':main()
