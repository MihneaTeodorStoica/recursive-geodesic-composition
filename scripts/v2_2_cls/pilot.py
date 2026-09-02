#!/usr/bin/env python3
"""V2.2 pilot CLI. Synthetic mode is the only enabled mode in this milestone."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch.nn.functional as F
from recursive_geodesic_composition.v2_2_covariance import covariance_frobenius,whitened_covariance,balanced
ARMS=('BASELINE','A','B','C')
def provenance(m):
 req={'teacher_pooling':'cls','teacher_id':'Alibaba-NLP/gte-modernbert-base','teacher_revision':'e7f32e3c00f91d699e8c43b53106206bcc72bb22','namespace':'v2_2_cls'}
 if any(m.get(k)!=v for k,v in req.items()):raise ValueError('V2.2 CLS provenance rejection')
def objective(g,t,corpus,arm):
 """Distribution additions only; base legacy terms are supplied by training path."""
 if g.shape[0]<4:raise ValueError('fewer than four matched rows')
 cov=torch.zeros((),device=g.device);res=torch.zeros((),device=g.device);cond=torch.tensor(0.,device=g.device)
 if arm=='A':cov=covariance_frobenius(g,t)
 if arm in ('B','C'):
  if arm=='B':cov,cond=whitened_covariance(g,t)
  else:
   terms=[];conds=[]
   for c in range(4):
    x=(corpus==c)
    if x.sum()<4:raise ValueError('C requires four valid rows per corpus')
    z,q=whitened_covariance(g[x],t[x]);terms.append(z);conds.append(q)
   cov=balanced(terms);cond=torch.stack(conds).max();res=(g-t).mean(0).square().mean()
 if not torch.isfinite(cov+res+cond) or cond>1e8:raise FloatingPointError('covariance health')
 return {'covariance':cov,'residual_mean':res,'condition':cond,'addition':64*cov+(.01*res if arm=='C' else 0*res)}
def synthetic(arm,steps=3):
 torch.manual_seed(424242);t=F.normalize(torch.arange(128,dtype=torch.float32).reshape(16,8)+1,dim=-1);p=torch.nn.Parameter(t+.01*torch.randn_like(t));initial=hashlib.sha256(p.detach().numpy().tobytes()).hexdigest();opt=torch.optim.AdamW([p],lr=1e-3);corp=torch.tensor([0]*4+[1]*4+[2]*4+[3]*4);log=[]
 for _ in range(steps):
  g=F.normalize(p,dim=-1);base=(1-(g*t).sum(1)).mean()+.25*(g.mean(0)-t.mean(0)).square().mean()+256*(g.var(0)-t.var(0)).square().mean();q=objective(g,t,corp,arm);loss=base+q['addition'];opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_([p],1000);assert torch.isfinite(gn) and torch.isfinite(p.grad).all();opt.step();log.append({'total':loss.detach().item(),'covariance':q['covariance'].detach().item(),'gradient_norm':gn.detach().item()})
 return {'arm':arm,'initial_hash':initial,'log':log}
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('synthetic','prepare','run-pilot'));p.add_argument('--arm',choices=ARMS,default='BASELINE');a=p.parse_args()
 if a.command!='synthetic':raise RuntimeError('TRAIN access intentionally disabled until separate authorization')
 print(json.dumps(synthetic(a.arm),indent=2))
if __name__=='__main__':main()
