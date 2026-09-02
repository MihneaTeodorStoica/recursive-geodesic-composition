#!/usr/bin/env python3
"""Fit V2 CLS parent-only probes/floors from TRAIN cache only."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch, torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src')]
from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe,build_model
CORPORA=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101');SEEDS=(424242,271828,314159)
CPS={424242:'results/v2_cls/amended_v2_1_seed424242/stage_c/autoencoder_seed424242.pt',271828:'results/v2_cls/amended_v2_1_seed271828/stage_c/autoencoder_seed271828.pt',314159:'results/v2_cls/amended_v2_1_seed314159/stage_c_retry_after_enospc/autoencoder_seed314159.pt'}
def sh(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 out=ROOT/'artifacts/v2_cls/train_probes_v2_1';out.mkdir(exist_ok=False)
 cache=ROOT/'artifacts/v2_cls/cache_train_cls';rows={c:torch.load(cache/c/'train.pt',map_location='cpu',weights_only=False) for c in CORPORA}; manifest=json.loads((cache/'CACHE_MANIFEST.json').read_text())
 if manifest.get('teacher_pooling')!='cls':raise ValueError('non-CLS cache')
 chosen={};fit=[];floor=[]
 for c in CORPORA:
  by={str(r['id']):r for r in rows[c] if len(r['leaves'])==4 and r.get('segmentation')=='train'}
  keys=sorted(by,key=lambda x:hashlib.sha256(f'v2probe|730241|{c}|{x}'.encode()).digest())
  if len(keys)<160:raise ValueError(c)
  chosen[c]={'fit':keys[:80],'floor':keys[80:160]};fit += [(c,by[x]) for x in keys[:80]];floor += [(c,by[x]) for x in keys[80:160]]
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); result={'role':'v2_train_only_parent_probe','cache_manifest_sha256':sh(cache/'CACHE_MANIFEST.json'),'selection':{c:{k:hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in d.items()} for c,d in chosen.items()},'seeds':{}}
 def roots(model,items):
  x=torch.stack([r['leaves'] for _,r in items]).to(device);m=torch.stack([r['lengths'] for _,r in items]).to(device)
  with torch.inference_mode():return model.merge(x,m).cpu()
 yfit=torch.stack([r['leaves'] for _,r in fit]);yfloor=torch.stack([r['leaves'] for _,r in floor]); labels=[c for c,_ in floor]
 for seed in SEEDS:
  model,_=build_model(ROOT/CPS[seed],device); rf,rr=roots(model,fit),roots(model,floor)
  torch.manual_seed(730241);probe=ParentOnlyRetentionProbe(768,4,512).to(device); init={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()};opt=torch.optim.AdamW(probe.parameters(),lr=3e-4,weight_decay=.001);g=torch.Generator().manual_seed(730242);sched=torch.randint(len(fit),(400,128),generator=g)
  for idx in sched:
   pred=probe(rf[idx].to(device));loss=1-F.cosine_similarity(pred,yfit[idx].to(device),dim=-1).mean()+.25*F.mse_loss(pred,yfit[idx].to(device));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True);opt.step()
  with torch.inference_mode(): vals=F.cosine_similarity(probe(rr.to(device)).cpu(),yfloor,dim=-1).mean(-1)
  scores={'pooled':float(vals.mean())};scores.update({c:float(vals[[i for i,x in enumerate(labels) if x==c]].mean()) for c in CORPORA}); floors={k:v-.001 for k,v in scores.items()}
  p=out/f'parent_only_seed{seed}.pt';torch.save({'state_dict':probe.cpu().state_dict(),'seed':seed,'teacher_pooling':'cls','fit_role':'v2_train_only','schedule_sha256':hashlib.sha256(sched.numpy().tobytes()).hexdigest(),'initialization_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in init.values())).hexdigest()},p);result['seeds'][str(seed)]={'checkpoint':CPS[seed],'checkpoint_sha256':sh(ROOT/CPS[seed]),'probe':str(p.relative_to(ROOT)),'probe_sha256':sh(p),'train_scores':scores,'floors':floors}
 (out/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
