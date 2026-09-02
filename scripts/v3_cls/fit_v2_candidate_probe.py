#!/usr/bin/env python3
"""Fit the predeclared V2 candidate's parent-only probe on V3 TRAIN only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from recursive_geodesic_composition.architecture import GeodesicMerge
from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe
CORPORA=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 if (ROOT/'artifacts/v3_cls/OFFICIAL_ACCESS_SPENT.json').exists(): raise PermissionError('probe fitting forbidden after official access')
 device=torch.device('cuda'); cp=ROOT/a.checkpoint; ck=torch.load(cp,map_location='cpu',weights_only=True); c=ck['config'];
 if int(ck.get('seed',-1))!=a.seed: raise ValueError('seed/checkpoint mismatch')
 m=GeodesicMerge(d=768,hidden=c['encoder_hidden'],layers=c['encoder_layers'],heads=c['heads'],max_angle=c['max_geodesic_angle'],decoder=True,decoder_layers=c['decoder_layers'],decoder_length_mode=c['decoder_length_mode'],importance_mode=c['importance_mode'],importance_beta=c['importance_beta'],decoder_mode=c['decoder_mode'],geodesic_parameterization=c['geodesic_parameterization'],arity=4).to(device); m.load_state_dict(ck['state_dict']); m.eval()
 fit=[]; floor=[]; audit={}
 for corpus in CORPORA:
  rows=torch.load(ROOT/f'artifacts/v3_cls/cache_train_cls/{corpus}/train.pt',weights_only=False); by={str(r['id']):r for r in rows if len(r['leaves'])==4 and r.get('segmentation')=='train'}; ranked=sorted(by,key=lambda x:hashlib.sha256(f'v3-probe|730241|{corpus}|{x}'.encode()).digest()); fg,vg=ranked[:80],ranked[80:160]; fit += [(corpus,by[x]) for x in fg]; floor += [(corpus,by[x]) for x in vg]; audit[corpus]={'fit_ids_sha256':hashlib.sha256(json.dumps(fg,sort_keys=True).encode()).hexdigest(),'floor_ids_sha256':hashlib.sha256(json.dumps(vg,sort_keys=True).encode()).hexdigest()}
 @torch.inference_mode()
 def roots(rows):
  out=[]
  for i in range(0,len(rows),128):
   p=rows[i:i+128]; out.append(m.merge(torch.stack([r['leaves'].float() for _,r in p]).to(device),torch.stack([r['lengths'].float() for _,r in p]).to(device)).cpu())
  return torch.cat(out)
 xr,xf=roots(fit),roots(floor); yf=torch.stack([r['leaves'].float() for _,r in fit]); yfloor=torch.stack([r['leaves'].float() for _,r in floor]); labels=np.array([x for x,_ in floor])
 torch.manual_seed(730241); probe=ParentOnlyRetentionProbe(768,4,512).to(device); opt=torch.optim.AdamW(probe.parameters(),lr=3e-4,weight_decay=.001); schedule=torch.randint(len(fit),(400,128),generator=torch.Generator().manual_seed(730242))
 for ind in schedule:
  pred=probe(xr[ind].to(device)); target=yf[ind].to(device); loss=1-F.cosine_similarity(pred,target,dim=-1).mean()+.25*F.mse_loss(pred,target); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(),1,error_if_nonfinite=True); opt.step()
 probe.eval()
 with torch.inference_mode(): pred=torch.cat([probe(xf[i:i+512].to(device)).cpu() for i in range(0,len(xf),512)])
 val=F.cosine_similarity(pred,yfloor,dim=-1).mean(-1).numpy(); scores={'pooled':float(val.mean()),**{x:float(val[labels==x].mean()) for x in CORPORA}}; floors={k:v-.001 for k,v in scores.items()}; out=ROOT/a.output
 if out.exists(): raise FileExistsError(out)
 out.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':probe.cpu().state_dict(),'seed':a.seed,'fit_role':'v3_cls_train','arity':4,'hidden':512,'updates':400,'selection_audit':audit,'schedule_sha256':hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),'train_floor_scores':scores,'frozen_floors':floors,'candidate_checkpoint_sha256':sha(cp)},out); print(out,sha(out),json.dumps(floors))
if __name__=='__main__':main()
