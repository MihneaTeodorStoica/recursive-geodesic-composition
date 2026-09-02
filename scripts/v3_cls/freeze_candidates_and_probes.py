#!/usr/bin/env python3
"""Fit V3 TRAIN-only parent probes and freeze candidates before development."""
from __future__ import annotations
import hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from recursive_geodesic_composition.architecture import GeodesicMerge  # noqa:E402
from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe  # noqa:E402
SEEDS=(424242,271828,314159); CORPORA=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def stable(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def model_from(cp,device):
 ck=torch.load(cp,map_location='cpu',weights_only=True); c=ck['config']; m=GeodesicMerge(d=768,hidden=c['encoder_hidden'],layers=c['encoder_layers'],heads=c['heads'],max_angle=c['max_geodesic_angle'],decoder=True,decoder_layers=c['decoder_layers'],decoder_length_mode=c['decoder_length_mode'],importance_mode=c['importance_mode'],importance_beta=c['importance_beta'],decoder_mode=c['decoder_mode'],geodesic_parameterization=c['geodesic_parameterization'],arity=4).to(device); m.load_state_dict(ck['state_dict'],strict=True); return m.eval(),ck
@torch.inference_mode()
def roots(model,rows,device):
 out=[]
 for i in range(0,len(rows),128):
  p=rows[i:i+128]; out.append(model.merge(torch.stack([r['leaves'].float() for _,r in p]).to(device),torch.stack([r['lengths'].float() for _,r in p]).to(device)).cpu())
 return torch.cat(out)
def scores(probe,x,y,labels,device):
 with torch.inference_mode(): pred=torch.cat([probe(x[i:i+512].to(device)).cpu() for i in range(0,len(x),512)])
 v=F.cosine_similarity(pred,y,dim=-1).mean(-1).numpy(); labels=np.asarray(labels); return {'pooled':float(v.mean()),**{c:float(v[labels==c].mean()) for c in CORPORA}}
def main():
 if (ROOT/'artifacts/v3_cls/DEVELOPMENT_ACCESS_SPENT.json').exists(): raise PermissionError('development already opened')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows={c:torch.load(ROOT/f'artifacts/v3_cls/cache_train_cls/{c}/train.pt',weights_only=False) for c in CORPORA}
 fit=[]; floor=[]; audit={}
 for c in CORPORA:
  by={str(r['id']):r for r in rows[c] if len(r['leaves'])==4 and r.get('segmentation')=='train'}; ranked=sorted(by,key=lambda x:hashlib.sha256(f'v3-probe|730241|{c}|{x}'.encode()).digest());
  if len(ranked)<160: raise RuntimeError(f'insufficient exact4 {c}')
  fg,vg=ranked[:80],ranked[80:160]; fit += [(c,by[x]) for x in fg]; floor += [(c,by[x]) for x in vg]; audit[c]={'fit':80,'floor':80,'fit_ids_sha256':stable(fg),'floor_ids_sha256':stable(vg)}
 fit_y=torch.stack([r['leaves'].float() for _,r in fit]); floor_y=torch.stack([r['leaves'].float() for _,r in floor]); labels=[c for c,_ in floor]
 out=ROOT/'artifacts/v3_cls/predevelopment_freeze';
 if out.exists(): raise FileExistsError(out)
 out.mkdir(); seed_results={}
 for seed in SEEDS:
  cp=ROOT/f'results/v3_cls/final_seed{seed}/autoencoder_seed{seed}.pt'; model,ck=model_from(cp,device)
  history=[json.loads(x) for x in (cp.parent/f'autoencoder_seed{seed}.jsonl').read_text().splitlines()];
  if len(history)!=2400 or [x['step'] for x in history]!=list(range(1,2401)) or ck.get('selected_step')!=2400: raise ValueError('nonconsecutive/nonfinal checkpoint')
  if any(not math.isfinite(float(v)) for r in history for v in r.values() if isinstance(v,(int,float))): raise FloatingPointError('nonfinite history')
  xr=roots(model,fit,device); xf=roots(model,floor,device)
  torch.manual_seed(730241); probe=ParentOnlyRetentionProbe(768,4,512).to(device); opt=torch.optim.AdamW(probe.parameters(),lr=3e-4,weight_decay=.001); gen=torch.Generator().manual_seed(730242); schedule=torch.randint(len(fit),(400,128),generator=gen)
  probe.train(); final={}
  for ind in schedule:
   pred=probe(xr[ind].to(device)); target=fit_y[ind].to(device); cosine=1-F.cosine_similarity(pred,target,dim=-1).mean(); mse=F.mse_loss(pred,target); loss=cosine+.25*mse; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(),1,error_if_nonfinite=True); opt.step(); final={'loss':float(loss),'cosine_loss':float(cosine),'mse':float(mse)}
  probe.eval(); floor_scores=scores(probe,xf,floor_y,labels,device); floors={k:v-.001 for k,v in floor_scores.items()}; pp=out/f'parent_only_seed{seed}.pt'; torch.save({'state_dict':probe.cpu().state_dict(),'seed':seed,'fit_role':'v3_cls_train','arity':4,'hidden':512,'updates':400,'selection_audit':audit,'schedule_sha256':hashlib.sha256(schedule.numpy().tobytes()).hexdigest()},pp)
  seed_results[str(seed)]={'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'config':f'configs/v3_cls/seed{seed}.yaml','config_sha256':sha(ROOT/f'configs/v3_cls/seed{seed}.yaml'),'history_sha256':sha(cp.parent/f'autoencoder_seed{seed}.jsonl'),'probe':str(pp.relative_to(ROOT)),'probe_sha256':sha(pp),'probe_fit_final':final,'train_floor_scores':floor_scores,'frozen_floors':floors,'training_health':{'max_grad_norm':max(x['grad_norm'] for x in history),'max_geodesic_saturation_fraction':max(x['geodesic_saturation_fraction'] for x in history),'max_geodesic_angle':max(x['geodesic_angle_max'] for x in history),'all_finite':True}}
 manifest={'study':'v3_cls','freeze':'predevelopment_candidates_and_train_parent_probes','development_accessed':False,'source_range':[296000,308000],'teacher_pooling':'cls','selection':'fixed final step 2400','identical_protocol_except_seed_output':True,'probe_selection_audit':audit,'seeds':seed_results,'hashes':{'training_cache_manifest_sha256':sha(ROOT/'artifacts/v3_cls/cache_train_cls/CACHE_MANIFEST.json'),'training_code_sha256':sha(ROOT/'src/recursive_geodesic_composition/train.py'),'evaluator_code_sha256':sha(ROOT/'src/recursive_geodesic_composition/evaluate_v3_cls.py'),'evaluator_protocol_sha256':sha(ROOT/'configs/v3_cls/evaluator_two_arm.yaml'),'attack_config_sha256':sha(ROOT/'configs/evaluator_attacks_v1.yaml'),'cache_preparer_sha256':sha(ROOT/'scripts/v3_cls/prepare_two_arm_cache.py')}}
 mp=out/'PREDEVELOPMENT_FREEZE.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); (out/'COMPLETE').write_text(f'{sha(mp)}  PREDEVELOPMENT_FREEZE.json\n'); print(mp)
if __name__=='__main__': main()
