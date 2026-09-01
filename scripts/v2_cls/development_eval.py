#!/usr/bin/env python3
"""Create-only V2 CLS development safe-span cache and frozen evaluation."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v2_cls')]
from protocol import DEVELOPMENT_SCAN, TEACHER_ID, TEACHER_REVISION  # noqa
from recursive_geodesic_composition.data import FrozenTeacher, stream_document_ranges, tokenize_text, chunk_token_ids  # noqa
from recursive_geodesic_composition.architecture import recursive_reduce  # noqa
from recursive_geodesic_composition.evaluate_kary_validation import build_model  # noqa
from recursive_geodesic_composition.evaluate import evaluate_attacks  # noqa

CORPORA=(('fineweb_edu','HuggingFaceFW/fineweb-edu','sample-10BT','87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'),('fineweb','HuggingFaceFW/fineweb',None,'9bb295ddab0e05d785b879661af7260fed5140fc'),('openwebtext','Skylion007/openwebtext',None,'79d93d786212f7344586290adb811d4ae6a1762c'),('wikipedia_en_20231101','wikimedia/wikipedia','20231101.en','b04c8d1ceb2f5cd4588862100d08de323dccfbaa'))
def sh(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def prepare(device, n=100):
 out=ROOT/'artifacts/v2_cls/development_safe_cls'
 if out.exists(): raise FileExistsError(out)
 teacher=FrozenTeacher(TEACHER_ID,TEACHER_REVISION,device,pooling='cls',trust_remote_code=True)
 out.mkdir(parents=True); manifest={'role':'development_safe','source_scan':list(DEVELOPMENT_SCAN),'teacher':teacher.provenance(),'per_corpus':{}}
 for cid,name,conf,rev in CORPORA:
  docs=stream_document_ranges(name,conf,rev,{'development':(DEVELOPMENT_SCAN[0],DEVELOPMENT_SCAN[1]-DEVELOPMENT_SCAN[0])})['development']; rows=[]
  for offset,doc in enumerate(docs):
   ids=tokenize_text(teacher.tokenizer,doc.text)
   if not (1025<=len(ids)<=4096): continue
   chunks=chunk_token_ids(ids,1024)
   if not 2<=len(chunks)<=4 or sum(map(len,chunks))!=len(ids): continue
   leaves=teacher.embed(chunks,32); target=teacher.embed([ids],1)[0]
   rows.append({'id':f'{cid}:{doc.id}','source_index':DEVELOPMENT_SCAN[0]+offset,'leaves':leaves,'lengths':torch.tensor(list(map(len,chunks)),dtype=torch.float32),'target':target,'teacher_pooling':'cls'})
   if len(rows)>=n: break
  if len(rows)<n: raise RuntimeError(f'{cid} insufficient safe rows')
  torch.save(rows,out/f'{cid}.pt');manifest['per_corpus'][cid]={'rows':len(rows),'sha256':sh(out/f'{cid}.pt')}
 (out/'CACHE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
def evaluate(device):
 cache=ROOT/'artifacts/v2_cls/development_safe_cls'; m=json.loads((cache/'CACHE_MANIFEST.json').read_text())
 if m['source_scan']!=list(DEVELOPMENT_SCAN) or m['teacher']['teacher_pooling']!='cls':raise ValueError('bad dev cache')
 allrows={cid:torch.load(cache/f'{cid}.pt',map_location='cpu',weights_only=False) for cid,*_ in CORPORA}
 cps={424242:ROOT/'results/v2_cls/amended_v2_1_seed424242/stage_c/autoencoder_seed424242.pt',271828:ROOT/'results/v2_cls/amended_v2_1_seed271828/stage_c/autoencoder_seed271828.pt',314159:ROOT/'results/v2_cls/amended_v2_1_seed314159/stage_c_retry_after_enospc/autoencoder_seed314159.pt'}
 result={'role':'development_safe','cache_manifest_sha256':sh(cache/'CACHE_MANIFEST.json'),'seeds':{}}
 for seed,cp in cps.items():
  model,_=build_model(cp,torch.device(device)); fake=[]; real=[]; groups=[]; rec=[]; corp={}
  for cid,rows in allrows.items():
   f=[];r=[];rr=[]
   with torch.inference_mode():
    for row in rows:
     root,_=recursive_reduce(model,row['leaves'].to(device)[None],row['lengths'].to(device)[None]); root=root[0]
     f.append(root.cpu());r.append(row['target']); decoded=model.decode(root[None])[0,:len(row['lengths'])];rr.append(float(F.cosine_similarity(decoded,row['leaves'],dim=-1).mean()))
   f=np.stack([x.numpy() for x in f]);r=np.stack([x.numpy() for x in r]);corp[cid]={'rows':len(rows),'teacher_cosine':float(np.mean(np.sum(f*r,axis=1))),'ordered_child_cosine':float(np.mean(rr))};fake.append(f);real.append(r);groups += [cid+':'+str(i) for i in range(len(rows))]
  fake=np.concatenate(fake);real=np.concatenate(real); attacks=evaluate_attacks(real,fake,groups,bootstrap=400)
  result['seeds'][str(seed)]={'checkpoint_sha256':sh(cp),'pooled_teacher_cosine':float(np.mean(np.sum(fake*real,axis=1))),'per_corpus':corp,'attacks':attacks}
 result['three_seed_mean_teacher_cosine']=float(np.mean([x['pooled_teacher_cosine'] for x in result['seeds'].values()]))
 out=ROOT/'results/v2_cls/DEVELOPMENT_SAFE_RESULTS.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'mean_cosine':result['three_seed_mean_teacher_cosine']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate'));p.add_argument('--device',default='cuda');p.add_argument('--rows',type=int,default=100);a=p.parse_args();prepare(a.device,a.rows) if a.mode=='prepare' else evaluate(a.device)
