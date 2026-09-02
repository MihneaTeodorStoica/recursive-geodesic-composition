#!/usr/bin/env python3
"""Create immutable V3 development/official two-arm CLS caches.

This script is the only authorized source-data entrypoint for the evaluator.  It
marks a role spent before opening the streaming dataset and never overwrites.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, uuid
from pathlib import Path
import torch, yaml
from datasets import load_dataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text  # noqa:E402

MODEL='Alibaba-NLP/gte-modernbert-base'; REV='e7f32e3c00f91d699e8c43b53106206bcc72bb22'
RANGES={'development':(308000,317000),'official':(318000,348000)}
CORPORA=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def tensor_sha(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def fsync_file(p):
 with Path(p).open('rb') as f: os.fsync(f.fileno())
def exact_chunks(ids,n=1024): return [ids[i:i+n] for i in range(0,len(ids),n)]
def source_rows(corpus,start,end):
 ds=load_dataset(corpus['dataset_name'],corpus.get('dataset_config'),split='train',streaming=True,revision=corpus['dataset_revision'])
 for i,row in enumerate(ds):
  if i>=end: break
  if i<start: continue
  text=row.get('text') or row.get('content') or ''
  if isinstance(text,str) and text.strip(): yield i,str(row.get('id',i)),text

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--role',choices=RANGES,required=True); ap.add_argument('--config',default='configs/v3_cls/seed424242.yaml'); ap.add_argument('--protocol',default='configs/v3_cls/evaluator_two_arm.yaml'); ap.add_argument('--device',default='cuda'); ap.add_argument('--documents-per-corpus',type=int,default=1000); a=ap.parse_args()
 cfg=yaml.safe_load((ROOT/a.config).read_text()); protocol=yaml.safe_load((ROOT/a.protocol).read_text()); role=a.role; start,end=RANGES[role]
 if cfg.get('teacher_pooling')!='cls' or cfg.get('model_name')!=MODEL or cfg.get('model_revision')!=REV or not cfg.get('v3_cls_experiment'): raise ValueError('V3 pinned CLS config required')
 if protocol['source_ranges'][role]!=[start,end] or protocol['safe_arm']['minimum_tokens']!=1025 or protocol['safe_arm']['maximum_tokens']!=4096: raise ValueError('protocol/range mismatch')
 if a.documents_per_corpus<1: raise ValueError('positive count required')
 dest=ROOT/f'artifacts/v3_cls/{role}_two_arm_cls'; marker=ROOT/f'artifacts/v3_cls/{role.upper()}_ACCESS_SPENT.json'
 if dest.exists() or marker.exists(): raise FileExistsError(f'{role} is create-only/already spent')
 if role=='official':
  freeze=ROOT/'reproducibility/v3_cls/OFFICIAL_PREACCESS_FREEZE.json'
  if not freeze.is_file(): raise PermissionError('official cache requires completed preaccess freeze')
 dest.parent.mkdir(parents=True,exist_ok=True)
 # SPENT is durable before constructing a dataset object or iterating one row.
 fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
 with os.fdopen(fd,'w') as f:
  json.dump({'role':role,'source_range':[start,end],'selection':'first eligible source-order documents with >=1025 tokens','documents_per_corpus':a.documents_per_corpus,'config':a.config,'config_sha256':sha(ROOT/a.config),'protocol':a.protocol,'protocol_sha256':sha(ROOT/a.protocol)},f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 stage=dest.with_name('.'+dest.name+'.staging-'+uuid.uuid4().hex); stage.mkdir()
 try:
  teacher=FrozenTeacher(MODEL,REV,a.device,pooling='cls',trust_remote_code=True); artifacts={}; audit={}
  for corpus in cfg['corpora']:
   cid=corpus['id']; out=stage/cid; (out/'full').mkdir(parents=True); safe=[]; index=[]; scanned=0
   for source_index,doc_id,text in source_rows(corpus,start,end):
    scanned+=1; ids=tokenize_text(teacher.tokenizer,text)
    if len(ids)<1025: continue
    chunks=exact_chunks(ids); leaves=teacher.embed(chunks,32).float(); lengths=torch.tensor([len(x) for x in chunks],dtype=torch.float32)
    safe_ids=ids[:min(len(ids),4096)]; safe_count=(len(safe_ids)+1023)//1024
    target=teacher.embed([safe_ids],1)[0].float(); leaf_hash=tensor_sha(leaves)
    row={'id':doc_id,'source_index':source_index,'leaves':leaves[:safe_count].clone(),'lengths':lengths[:safe_count].clone(),'teacher_target':target,'teacher_pooling':'cls','safe_token_count':len(safe_ids),'full_leaf_tensor_sha256':leaf_hash}
    shard_rel=f'{cid}/full/{len(index):06d}.pt'; shard={'id':doc_id,'source_index':source_index,'leaves':leaves,'lengths':lengths,'token_count':len(ids),'truncated':False,'token_cap':None,'leaf_cap':None,'leaf_tensor_sha256':leaf_hash}
    shard_path=stage/shard_rel; torch.save(shard,shard_path); artifacts[shard_rel]={'sha256':sha(shard_path),'bytes':shard_path.stat().st_size}
    safe.append(row); index.append({'id':doc_id,'source_index':source_index,'token_count':len(ids),'leaf_count':len(chunks),'shard':shard_rel})
    if len(safe)>=a.documents_per_corpus: break
   if len(safe)!=a.documents_per_corpus: raise RuntimeError(f'{cid}: only {len(safe)} eligible docs in frozen range')
   safe_path=out/'safe.pt'; torch.save(safe,safe_path); idx_path=out/'full_index.json'; idx_path.write_text(json.dumps({'documents':index},indent=2)+'\n')
   for p in (safe_path,idx_path): artifacts[str(p.relative_to(stage))]={'sha256':sha(p),'bytes':p.stat().st_size}
   audit[cid]={'accepted':len(safe),'source_rows_examined_through_acceptance':scanned,'first_source_index':safe[0]['source_index'],'last_source_index':safe[-1]['source_index'],'max_tokens':max(x['token_count'] for x in index),'max_leaves':max(x['leaf_count'] for x in index)}
  manifest={'schema':'v3_cls_two_arm_cache_v1','study':'v3_cls','role':role,'source_range':[start,end],'selection':'first eligible source-order documents with >=1025 tokenizer tokens','documents_per_corpus':a.documents_per_corpus,'teacher_provenance':{**teacher.provenance(),'implementation_sha256':sha(ROOT/'src/recursive_geodesic_composition/data.py'),'preparer_sha256':sha(__file__)},'safe_arm':protocol['safe_arm'],'full_arm':protocol['full_arm'],'tensor_identical_leaf_reuse':'safe leaves are clone of exact prefix of full leaves; per-row full_leaf_tensor_sha256 records full tensor','audit':audit,'artifacts':artifacts}
  mp=stage/'CACHE_MANIFEST.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); fsync_file(mp); (stage/'COMPLETE').write_text(f'{sha(mp)}  CACHE_MANIFEST.json\n'); fsync_file(stage/'COMPLETE'); os.replace(stage,dest)
 except BaseException:
  shutil.rmtree(stage,ignore_errors=True); raise
 print(dest)
if __name__=='__main__': main()
