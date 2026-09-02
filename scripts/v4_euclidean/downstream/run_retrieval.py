#!/usr/bin/env python3
"""V3 CLS downstream retrieval with one exact shared leaf cache per benchmark."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import pytrec_eval
import torch
import torch.nn.functional as F
from datasets import load_dataset

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'src'))
from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text  # noqa:E402
from recursive_geodesic_composition.evaluate_v4_euclidean import (load_checkpoint, reduce_full, _new_telemetry, _finish_telemetry, _depth)  # noqa:E402

OUT=ROOT/'results/v4_euclidean/downstream'; PRECOMMIT=OUT/'PRECOMMIT.json'
TRAIN_MANIFEST='artifacts/v4_euclidean/cache_train_cls/CACHE_MANIFEST.json'
TEACHER_ID='Alibaba-NLP/gte-modernbert-base'; TEACHER_REV='e7f32e3c00f91d699e8c43b53106206bcc72bb22'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def tensor_sha(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write_json(path,obj):
 p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,p)
def norm(x):
 n=np.linalg.norm(x,axis=-1,keepdims=True)
 if not np.isfinite(x).all() or np.any(n<=0): raise FloatingPointError('nonfinite/zero embedding')
 return (x/n).astype(np.float32)
def rows_unique(rows,label):
 ids=[x[0] for x in rows]
 if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)): raise ValueError(f'{label}: empty or duplicate IDs')
 return rows

def load_nfc():
 base=OUT/'source/nfcorpus'; corpus=[]
 with (base/'corpus.jsonl').open() as f:
  for line in f:
   x=json.loads(line); corpus.append((str(x['_id']), (str(x.get('title',''))+'\n'+str(x.get('text',''))).strip()))
 queries=[]
 with (base/'queries.jsonl').open() as f:
  for line in f:
   x=json.loads(line); queries.append((str(x['_id']),str(x['text']).strip()))
 qrels={}; seen=set(); path=base/'qrels/test.tsv'
 with path.open() as f:
  header=f.readline().rstrip('\n').split('\t')
  if header!=['query-id','corpus-id','score']: raise ValueError(f'nfcorpus malformed qrels header: {header}')
  for number,line in enumerate(f,2):
   fields=line.rstrip('\n').split('\t')
   if len(fields)!=3: raise ValueError(f'nfcorpus malformed qrel line {number}')
   q,d,v=fields
   try:
    numeric=float(v); rel=int(numeric)
   except ValueError as e: raise ValueError(f'nfcorpus nonnumeric qrel line {number}') from e
   if not math.isfinite(numeric) or numeric!=rel or (q,d) in seen: raise ValueError(f'nfcorpus duplicate/nonfinite/noninteger qrel line {number}')
   seen.add((q,d)); qrels.setdefault(q,{})[d]=rel
 all_queries=rows_unique(queries,'nfcorpus all-split queries'); by_query=dict(all_queries)
 missing=set(qrels)-set(by_query)
 if missing: raise ValueError(f'nfcorpus test qrels reference {len(missing)} missing queries')
 queries=[(q,by_query[q]) for q in sorted(qrels)]
 return rows_unique(corpus,'nfcorpus corpus'),queries,qrels,{"qrels_sha256":sha(path),"query_rows_total_all_splits":len(all_queries),"query_rows_selected_test":len(queries),"query_rows_structurally_excluded_non_test":len(all_queries)-len(queries),"precommit_amendment":"results/v4_euclidean/downstream/PRECOMMIT_AMENDMENT_1.json","precommit_amendment_sha256":sha(OUT/'PRECOMMIT_AMENDMENT_1.json')}

def load_longembed(revision):
 kw=dict(path='dwzhu/LongEmbed',name='narrativeqa',revision=revision)
 c=load_dataset(**kw,split='corpus'); q=load_dataset(**kw,split='queries'); r=load_dataset(**kw,split='qrels')
 corpus=rows_unique([(str(x['doc_id']),str(x['text'])) for x in c],'narrativeqa corpus')
 queries=rows_unique([(str(x['qid']),str(x['text']).strip()) for x in q],'narrativeqa queries')
 qrels={}; seen=set()
 for number,x in enumerate(r,1):
  qid,did=str(x['qid']),str(x['doc_id'])
  if not qid or not did or (qid,did) in seen: raise ValueError(f'narrativeqa malformed/duplicate qrel row {number}')
  seen.add((qid,did)); qrels.setdefault(qid,{})[did]=1
 return corpus,queries,qrels,{"dataset_revision":revision,"rows":{"corpus":len(c),"queries":len(q),"qrels":len(r)}}

def validate(corpus,queries,qrels):
 dids={x[0] for x in corpus}; qids={x[0] for x in queries}
 if set(qrels)!=qids: raise ValueError(f'qrels/query mismatch: qrels={len(qrels)} queries={len(qids)} missing_qrels={len(qids-set(qrels))} extra_qrels={len(set(qrels)-qids)}')
 for q,judgments in qrels.items():
  if not judgments or not any(v>0 for v in judgments.values()): raise ValueError(f'{q}: no positive qrel')
  unknown=set(judgments)-dids
  if unknown: raise ValueError(f'{q}: qrels contain {len(unknown)} unknown documents')

def prepare_cache(dataset,corpus,queries,teacher,batch_size):
 cache=OUT/dataset/'leaf_cache'; cache.mkdir(parents=True,exist_ok=True); manifest_path=cache/'MANIFEST.json'
 if manifest_path.exists(): return json.loads(manifest_path.read_text())
 docs_dir=cache/'documents'; docs_dir.mkdir(exist_ok=True); records=[]; total_leaves=0; started=time.perf_counter()
 for index,(doc_id,text) in enumerate(corpus):
  ids=tokenize_text(teacher.tokenizer,text)
  if not ids: raise ValueError(f'{dataset}/{doc_id}: empty tokenization')
  chunks=[ids[i:i+1024] for i in range(0,len(ids),1024)]
  leaves=teacher.embed(chunks,batch_size).float()
  lengths=torch.tensor([len(x) for x in chunks],dtype=torch.float32)
  if leaves.ndim!=2 or len(leaves)!=len(lengths) or not torch.isfinite(leaves).all(): raise FloatingPointError('invalid leaf tensor')
  shard={'id':doc_id,'leaves':leaves,'lengths':lengths,'token_count':len(ids),'leaf_count':len(chunks),'depth':_depth(len(chunks))}
  path=docs_dir/f'{index:06d}.pt'; torch.save(shard,path)
  records.append({'id':doc_id,'shard':str(path.relative_to(cache)),'sha256':sha(path),'leaf_tensor_sha256':tensor_sha(leaves),'tokens':len(ids),'leaves':len(chunks),'depth':shard['depth']}); total_leaves+=len(chunks)
  if (index+1)%100==0: print(f'{dataset}: cached {index+1}/{len(corpus)} documents',flush=True)
 qids=[]; token_counts=[]
 for qid,text in queries:
  ids=tokenize_text(teacher.tokenizer,text)
  if not ids or len(ids)>8190: raise ValueError(f'{dataset}/{qid}: query is empty or exceeds direct-teacher limit ({len(ids)})')
  qids.append(ids); token_counts.append(len(ids))
 query_emb=teacher.embed(qids,64).float(); query_path=cache/'queries.pt'; torch.save({'ids':[x[0] for x in queries],'embeddings':query_emb,'token_counts':token_counts},query_path)
 max_tokens=max(x['tokens'] for x in records); direct_eligible=max_tokens<=8190
 direct_path=None
 if direct_eligible:
  values=[]
  for _,text in corpus: values.append(teacher.embed([tokenize_text(teacher.tokenizer,text)],1)[0])
  direct_path=cache/'direct_documents.pt'; torch.save({'ids':[x[0] for x in corpus],'embeddings':torch.stack(values)},direct_path)
 manifest={"schema":"v4_euclidean_downstream_leaf_cache_v1","dataset":dataset,"teacher":{"id":TEACHER_ID,"revision":TEACHER_REV,"pooling":"normalized_cls"},"chunking":{"raw_tokens":1024,"contiguous":True,"final_short_leaf_retained":True,"dtype":"float32"},"documents":records,"query_cache":{"path":"queries.pt","sha256":sha(query_path)},"counts":{"documents":len(corpus),"queries":len(queries),"tokens":sum(x['tokens'] for x in records),"leaves":total_leaves,"max_tokens":max_tokens,"max_leaves":max(x['leaves'] for x in records),"max_depth":max(x['depth'] for x in records)},"direct_teacher":{"eligible_for_entire_benchmark":direct_eligible,"rule":"all documents and queries <=8190 raw tokens","path":None if direct_path is None else direct_path.name,"sha256":None if direct_path is None else sha(direct_path)},"seconds":time.perf_counter()-started}
 write_json(manifest_path,manifest); (cache/'COMPLETE').write_text(f'{sha(manifest_path)}  MANIFEST.json\n'); return manifest

def load_shared(cache,manifest):
 q=torch.load(cache/'queries.pt',map_location='cpu',weights_only=False)
 if sha(cache/'queries.pt')!=manifest['query_cache']['sha256']: raise ValueError('query cache hash mismatch')
 return q

def shared_roots(cache,manifest,weighted):
 values=[]
 for x in manifest['documents']:
  p=cache/x['shard']
  if sha(p)!=x['sha256']: raise ValueError('leaf shard hash mismatch')
  s=torch.load(p,map_location='cpu',weights_only=False); leaves=s['leaves'].float()
  if tensor_sha(leaves)!=x['leaf_tensor_sha256']: raise ValueError('leaf tensor identity hash mismatch')
  root=(leaves*s['lengths'].view(-1,1)).sum(0)/s['lengths'].sum() if weighted else leaves.mean(0)
  values.append(root.numpy())
 return norm(np.stack(values))

def recursive_roots(cache,manifest,seed,entry,device):
 spec={'architecture':'euclidean_residual','checkpoint':entry['path'],'checkpoint_sha256':entry['sha256'],'training_cache_manifest':TRAIN_MANIFEST,'training_cache_manifest_sha256':sha(ROOT/TRAIN_MANIFEST),'seed':int(seed)}
 model,_,prov=load_checkpoint(ROOT,spec,torch.device(device)); telemetry=_new_telemetry(); roots=[]; depths=[]; started=time.perf_counter()
 for i,x in enumerate(manifest['documents']):
  p=cache/x['shard']
  if sha(p)!=x['sha256']: raise ValueError('leaf shard hash mismatch before recursion')
  s=torch.load(p,map_location='cpu',weights_only=False)
  roots.append(reduce_full(model,s,torch.device(device),telemetry,64).numpy()); depths.append(x['depth'])
  if (i+1)%100==0: print(f'recursive seed {seed}: {i+1}/{len(manifest["documents"])}',flush=True)
 seconds=time.perf_counter()-started
 report=_finish_telemetry(telemetry,documents=len(roots),leaves=manifest['counts']['leaves'],depths=depths,seconds=seconds) if manifest['counts']['leaves']>len(roots) else {"documents":len(roots),"leaves":manifest['counts']['leaves'],"depth":{"minimum":0,"mean":0.0,"maximum":0},"seconds":seconds}
 del model; torch.cuda.empty_cache(); return norm(np.stack(roots)),{**prov,"telemetry":report}

def per_query_and_metrics(query_ids,doc_ids,qrels,scores):
 if scores.shape!=(len(query_ids),len(doc_ids)) or not np.isfinite(scores).all(): raise ValueError('missing/nonfinite score matrix')
 run={}; top={}
 docs=np.asarray(doc_ids)
 for i,q in enumerate(query_ids):
  order=np.lexsort((docs,-scores[i])); run[q]={doc_ids[j]:float(scores[i,j]) for j in order}; top[q]=order[:100]
 evaluator=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10','recall.10','recall.100'})
 evaluated=evaluator.evaluate(run)
 if set(evaluated)!=set(query_ids): raise ValueError('pytrec_eval silently dropped queries')
 per=[]
 for q in query_ids:
  e=evaluated[q]; rr=0.0
  for rank,j in enumerate(top[q][:10],1):
   if qrels[q].get(doc_ids[j],0)>0: rr=1.0/rank; break
  per.append({'query_id':q,'nDCG@10':float(e['ndcg_cut_10']),'Recall@10':float(e['recall_10']),'Recall@100':float(e['recall_100']),'MRR@10':rr})
 metrics={k:float(np.mean([x[k] for x in per])) for k in ('nDCG@10','Recall@10','Recall@100','MRR@10')}
 return per,top,metrics

def stratify(per,qrels,metadata):
 byid={x['id']:x for x in metadata}; groups={}
 for row in per:
  judged=qrels[row['query_id']]; best=max(judged.values()); primary=sorted(d for d,v in judged.items() if v==best)[0]; m=byid[primary]
  assignments={'tokens':next(f'{a}-{b or "inf"}' for a,b in [(1,1024),(1025,4096),(4097,8190),(8191,32768),(32769,None)] if m['tokens']>=a and (b is None or m['tokens']<=b)), 'leaves':next(f'{a}-{b or "inf"}' for a,b in [(1,1),(2,4),(5,8),(9,32),(33,None)] if m['leaves']>=a and (b is None or m['leaves']<=b)), 'depth':str(m['depth'])}
  for axis,label in assignments.items(): groups.setdefault(axis,{}).setdefault(label,[]).append(row)
 out={}
 for axis,bins in groups.items(): out[axis]={label:{'queries':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('nDCG@10','Recall@10','Recall@100','MRR@10')}} for label,rows in sorted(bins.items())}
 return out

def evaluate_dataset(dataset,corpus,queries,qrels,source,manifest,precommit,device):
 result_dir=OUT/dataset; final=result_dir/'RESULTS.json'
 if final.exists(): raise FileExistsError(f'create-only result exists: {final}')
 validate(corpus,queries,qrels); cache=result_dir/'leaf_cache'; qcache=load_shared(cache,manifest); qids=[x[0] for x in queries]; docids=[x[0] for x in corpus]
 if qcache['ids']!=qids: raise ValueError('query embedding ID/order mismatch')
 query_vectors=norm(qcache['embeddings'].numpy()); methods={}; provenance={}; started=time.perf_counter()
 methods['token_weighted_leaf_mean']=shared_roots(cache,manifest,True); methods['unweighted_leaf_mean']=shared_roots(cache,manifest,False)
 if manifest['direct_teacher']['eligible_for_entire_benchmark']:
  p=cache/manifest['direct_teacher']['path']
  if sha(p)!=manifest['direct_teacher']['sha256']: raise ValueError('direct teacher cache hash mismatch')
  d=torch.load(p,map_location='cpu',weights_only=False)
  if d['ids']!=docids: raise ValueError('direct teacher document order mismatch')
  methods['direct_teacher']=norm(d['embeddings'].numpy())
 for seed,entry in precommit['checkpoints'].items(): methods[f'recursive_seed{seed}'],provenance[f'recursive_seed{seed}']=recursive_roots(cache,manifest,seed,entry,device)
 method_results={}; rankings=result_dir/'rankings'; rankings.mkdir(exist_ok=True)
 for name,docs in methods.items():
  per,top,metrics=per_query_and_metrics(qids,docids,qrels,query_vectors@docs.T)
  per_path=result_dir/f'{name}_per_query.jsonl'; per_path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in per))
  run_path=rankings/f'{name}.trec'
  with run_path.open('w') as f:
   scores=query_vectors@docs.T
   for i,q in enumerate(qids):
    for rank,j in enumerate(top[q],1): f.write(f'{q} Q0 {docids[j]} {rank} {float(scores[i,j]):.9g} {name}\n')
  emb_path=result_dir/f'{name}_document_embeddings.npz'; np.savez_compressed(emb_path,ids=np.asarray(docids),embeddings=docs)
  method_results[name]={"metrics":metrics,"stratified":stratify(per,qrels,manifest['documents']),"artifacts":{"per_query":{"path":per_path.name,"sha256":sha(per_path)},"top100_run":{"path":str(run_path.relative_to(result_dir)),"sha256":sha(run_path)},"document_embeddings":{"path":emb_path.name,"sha256":sha(emb_path)}}}
  print(dataset,name,metrics,flush=True)
 result={"schema":"v4_euclidean_downstream_retrieval_results_v1","dataset":dataset,"metrics_scale":"0..1","counts":{"corpus":len(corpus),"queries":len(queries),"qrel_queries":len(qrels),"qrel_pairs":sum(map(len,qrels.values()))},"methods":method_results,"direct_teacher_included":'direct_teacher' in methods,"source":source,"runtime_seconds":time.perf_counter()-started,"provenance":{"precommit_sha256":sha(PRECOMMIT),"leaf_cache_manifest_sha256":sha(cache/'MANIFEST.json'),"training_cache_manifest_sha256":sha(ROOT/TRAIN_MANIFEST),"checkpoints":provenance,"software":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"pytrec_eval":getattr(pytrec_eval,'__version__','unknown')},"script_sha256":sha(__file__)}}
 write_json(final,result); (result_dir/'COMPLETE').write_text(f'{sha(final)}  RESULTS.json\n'); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--leaf-batch-size',type=int,default=8); a=ap.parse_args()
 pre=json.loads(PRECOMMIT.read_text()); expected=pre['blind_inspection_sha256']
 if sha(ROOT/pre['blind_inspection'])!=expected: raise ValueError('precommit blind inspection hash mismatch')
 datasets={}; datasets['nfcorpus']=(*load_nfc(),)
 le=next(x for x in pre['benchmarks'] if x['id']=='longembed_narrativeqa'); datasets['longembed_narrativeqa']=(*load_longembed(le['revision']),)
 for values in datasets.values(): validate(values[0],values[1],values[2])
 teacher=FrozenTeacher(TEACHER_ID,TEACHER_REV,a.device,pooling='cls',trust_remote_code=True)
 manifests={name:prepare_cache(name,v[0],v[1],teacher,a.leaf_batch_size) for name,v in datasets.items()}
 del teacher; torch.cuda.empty_cache()
 summaries={name:evaluate_dataset(name,v[0],v[1],v[2],v[3],manifests[name],pre,a.device) for name,v in datasets.items()}
 aggregate={"schema":"v4_euclidean_downstream_summary_v1","precommit_sha256":sha(PRECOMMIT),"datasets":{name:{"results":f"results/v4_euclidean/downstream/{name}/RESULTS.json","sha256":sha(OUT/name/'RESULTS.json'),"metrics":{m:x['metrics'] for m,x in result['methods'].items()},"runtime_seconds":result['runtime_seconds']} for name,result in summaries.items()}}
 write_json(OUT/'SUMMARY.json',aggregate); print(OUT/'SUMMARY.json')
if __name__=='__main__': main()
