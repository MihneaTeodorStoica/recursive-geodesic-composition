#!/usr/bin/env python3
"""Frozen V4 LongEmbed suite: shared CLS leaves, BEIR metrics, scale strata."""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,sys,time
from pathlib import Path
import numpy as np, pytrec_eval, torch
from datasets import load_dataset
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from recursive_geodesic_composition.data import FrozenTeacher,tokenize_text
from recursive_geodesic_composition.evaluate_v4_euclidean import load_checkpoint,reduce_full,_new_telemetry,_finish_telemetry,_depth
OUT=ROOT/'results/v4_euclidean/downstream';PRE=OUT/'SUITE_PRECOMMIT.json';REV='10039a580487dacecf79db69166e17ace3ede392';TRAIN='artifacts/v4_euclidean/cache_train_cls/CACHE_MANIFEST.json';TID='Alibaba-NLP/gte-modernbert-base';TREV='e7f32e3c00f91d699e8c43b53106206bcc72bb22'
REAL=['qmsum','2wikimqa','summ_screen_fd'];SYNTH=['needle','passkey']; METRICS=('nDCG@1','nDCG@10','Recall@10','Recall@100','MRR@10')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def norm(x):
 n=np.linalg.norm(x,axis=-1,keepdims=True)
 if not np.isfinite(x).all() or np.any(n<=0):raise FloatingPointError('invalid embeddings')
 return (x/n).astype(np.float32)
def merge_count(n):
 z=0
 while n>1:q,r=divmod(n,4);z+=q+(r>1);n=q+(r>0)
 return int(z)
def load_task(cfg):
 ds={s:load_dataset('dwzhu/LongEmbed',cfg,revision=REV,split=s) for s in ('corpus','queries','qrels')}
 corpus=[{'id':str(x['doc_id']),'text':str(x['text']),'context_length':x.get('context_length')} for x in ds['corpus']];queries=[{'id':str(x['qid']),'text':str(x['text']),'context_length':x.get('context_length')} for x in ds['queries']];qrels={};seen=set()
 for i,x in enumerate(ds['qrels']):
  q,d=str(x['qid']),str(x['doc_id']);ctx=x.get('context_length')
  if not q or not d or (q,d) in seen:raise ValueError(f'{cfg}: malformed qrel {i}')
  seen.add((q,d));qrels.setdefault(q,{})[d]=1
 did=[x['id'] for x in corpus];qid=[x['id'] for x in queries]
 if len(did)!=len(set(did)) or len(qid)!=len(set(qid)) or set(qrels)!=set(qid) or any(d not in set(did) for x in qrels.values() for d in x):raise ValueError(f'{cfg}: ID/qrel corruption')
 return corpus,queries,qrels,{'dataset':'dwzhu/LongEmbed','revision':REV,'config':cfg,'splits':{s:len(ds[s]) for s in ds}}
def prepare(cfg,corpus,queries,teacher,batch):
 cache=OUT/cfg/'leaf_cache';mp=cache/'MANIFEST.json'
 if mp.exists():
  if not (cache/'COMPLETE').exists() or (cache/'COMPLETE').read_text().strip()!=f'{sha(mp)}  MANIFEST.json':raise RuntimeError('incomplete cache')
  return json.loads(mp.read_text())
 cache.mkdir(parents=True);(cache/'documents').mkdir();records=[]
 for i,x in enumerate(corpus):
  ids=tokenize_text(teacher.tokenizer,x['text']);chunks=[ids[j:j+1024] for j in range(0,len(ids),1024)]
  if not chunks:raise ValueError(f'{cfg}/{x["id"]}: empty')
  leaves=teacher.embed(chunks,batch).float();lens=torch.tensor(list(map(len,chunks)),dtype=torch.float32)
  shard={'id':x['id'],'leaves':leaves,'lengths':lens,'token_count':len(ids),'leaf_count':len(chunks),'depth':_depth(len(chunks)),'merge_count':merge_count(len(chunks)),'context_length':x['context_length']};p=cache/'documents'/f'{i:06d}.pt';torch.save(shard,p)
  records.append({'id':x['id'],'shard':str(p.relative_to(cache)),'sha256':sha(p),'leaf_tensor_sha256':tsha(leaves),'tokens':len(ids),'leaves':len(chunks),'depth':shard['depth'],'merge_count':shard['merge_count'],'context_length':x['context_length']})
  if (i+1)%100==0:print(cfg,'cache',i+1,'/',len(corpus),flush=True)
 qtok=[tokenize_text(teacher.tokenizer,x['text']) for x in queries]
 if any(not x or len(x)>8190 for x in qtok):raise ValueError(f'{cfg}: malformed/overlength query')
 qe=teacher.embed(qtok,64).float();qp=cache/'queries.pt';torch.save({'ids':[x['id'] for x in queries],'embeddings':qe,'token_counts':list(map(len,qtok)),'context_lengths':[x['context_length'] for x in queries]},qp)
 # Direct teacher is condition-specific for synthetic tasks and whole-task only for real tasks.
 groups={}
 if cfg in SYNTH:
  for ctx in sorted(set(int(x['context_length']) for x in corpus)):
   idx=[i for i,x in enumerate(records) if int(x['context_length'])==ctx]
   if max(records[i]['tokens'] for i in idx)<=8190:groups[str(ctx)]=idx
 elif max(x['tokens'] for x in records)<=8190:groups['all']=list(range(len(records)))
 direct={}
 for label,idx in groups.items():
  vals=[teacher.embed([tokenize_text(teacher.tokenizer,corpus[i]['text'])],1)[0] for i in idx];p=cache/f'direct_{label}.pt';torch.save({'ids':[corpus[i]['id'] for i in idx],'embeddings':torch.stack(vals)},p);direct[label]={'path':p.name,'sha256':sha(p),'document_indexes':idx}
 m={'schema':'v4_longembed_shared_cls_cache_v2','dataset':{'id':'dwzhu/LongEmbed','revision':REV,'config':cfg},'teacher':{'id':TID,'revision':TREV,'pooling':'normalized_cls'},'chunking':{'raw_tokens':1024,'contiguous':True,'retain_final_short':True},'documents':records,'query_cache':{'path':qp.name,'sha256':sha(qp)},'counts':{'documents':len(records),'queries':len(queries),'tokens':sum(x['tokens'] for x in records),'leaves':sum(x['leaves'] for x in records),'merges':sum(x['merge_count'] for x in records),'max_depth':max(x['depth'] for x in records)},'direct_teacher_groups':direct,'suite_precommit_sha256':sha(PRE)};write(mp,m);(cache/'COMPLETE').write_text(f'{sha(mp)}  MANIFEST.json\n');return m
def pooled(cache,m,weighted):
 out=[]
 for x in m['documents']:
  p=cache/x['shard'];s=torch.load(p,map_location='cpu',weights_only=False)
  if sha(p)!=x['sha256'] or tsha(s['leaves'])!=x['leaf_tensor_sha256']:raise ValueError('leaf identity failure')
  v=(s['leaves']*s['lengths'][:,None]).sum(0)/s['lengths'].sum() if weighted else s['leaves'].mean(0);out.append(v.numpy())
 return norm(np.stack(out))
def recursive(cache,m,seed,entry,device):
 spec={'architecture':'euclidean_residual','checkpoint':entry['path'],'checkpoint_sha256':entry['sha256'],'training_cache_manifest':TRAIN,'training_cache_manifest_sha256':sha(ROOT/TRAIN),'seed':int(seed)};model,_,prov=load_checkpoint(ROOT,spec,torch.device(device));tel=_new_telemetry();out=[];deps=[];t=time.time()
 for i,x in enumerate(m['documents']):
  p=cache/x['shard'];s=torch.load(p,map_location='cpu',weights_only=False)
  if sha(p)!=x['sha256'] or tsha(s['leaves'])!=x['leaf_tensor_sha256']:raise ValueError('recursive leaf identity failure')
  out.append(reduce_full(model,s,torch.device(device),tel,64).numpy());deps.append(x['depth'])
  if (i+1)%100==0:print(m['dataset']['config'],'seed',seed,i+1,'/',len(m['documents']),flush=True)
 sec=time.time()-t;report=_finish_telemetry(tel,documents=len(out),leaves=m['counts']['leaves'],depths=deps,seconds=sec) if m['counts']['leaves']>len(out) else {'documents':len(out),'leaves':len(out),'seconds':sec,'depth':{'minimum':0,'mean':0,'maximum':0}};del model;torch.cuda.empty_cache();return norm(np.stack(out)),{**prov,'telemetry':report}
def metrics(qids,dids,qrels,scores):
 if scores.shape!=(len(qids),len(dids)) or not np.isfinite(scores).all():raise ValueError('score alignment')
 docs=np.asarray(dids);run={};tops={}
 for i,q in enumerate(qids):
  order=np.lexsort((docs,-scores[i]));run[q]={dids[j]:float(scores[i,j]) for j in order};tops[q]=order[:100]
 ev=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.1','ndcg_cut.10','recall.10','recall.100'}).evaluate(run)
 if set(ev)!=set(qids):raise ValueError('metric dropped query')
 per=[]
 for q in qids:
  rr=0.
  for rank,j in enumerate(tops[q][:10],1):
   if qrels[q].get(dids[j],0)>0:rr=1/rank;break
  per.append({'query_id':q,'nDCG@1':float(ev[q]['ndcg_cut_1']),'nDCG@10':float(ev[q]['ndcg_cut_10']),'Recall@10':float(ev[q]['recall_10']),'Recall@100':float(ev[q]['recall_100']),'MRR@10':rr})
 return per,{k:float(np.mean([x[k] for x in per])) for k in METRICS}
def ci(per,seed):
 rng=np.random.default_rng(seed);n=len(per);a={k:np.asarray([x[k] for x in per]) for k in METRICS};out={}
 for k,v in a.items():z=v[rng.integers(0,n,size=(1000,n))].mean(1);out[k]=[float(np.quantile(z,.025)),float(np.quantile(z,.975))]
 return out
def binlabel(v,bins):
 for a,b in bins:
  if v>=a and (b is None or v<=b):return f'{a}-{b if b is not None else "inf"}'
 raise ValueError(v)
def strata(per,qrels,meta):
 by={x['id']:x for x in meta};bins={'tokens':[[1,1024],[1025,4096],[4097,8190],[8191,16384],[16385,32768],[32769,65536],[65537,None]],'leaves':[[1,1],[2,4],[5,8],[9,16],[17,32],[33,64],[65,None]]};out={}
 for axis in ('tokens','leaves','depth'):
  groups={}
  for r in per:
   d=sorted(qrels[r['query_id']])[0];v=by[d][axis];lab=str(v) if axis=='depth' else binlabel(v,bins[axis]);groups.setdefault(lab,[]).append(r)
  out[axis]={lab:{'queries':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in METRICS}} for lab,rs in groups.items()}
 return out
def evaluate(cfg,corpus,queries,qrels,source,m,pre,device):
 rd=OUT/cfg;final=rd/'SUITE_RESULTS.json'
 if final.exists():raise FileExistsError(final)
 cache=rd/'leaf_cache';q=torch.load(cache/'queries.pt',map_location='cpu',weights_only=False)
 if sha(cache/'queries.pt')!=m['query_cache']['sha256'] or q['ids']!=[x['id'] for x in queries]:raise ValueError('query identity failure')
 qv=norm(q['embeddings'].numpy());allids=[x['id'] for x in corpus];methods={'token_weighted_leaf_mean':pooled(cache,m,True),'unweighted_leaf_mean':pooled(cache,m,False)};prov={}
 for seed,e in pre['checkpoints'].items():methods[f'recursive_seed{seed}'],prov[f'recursive_seed{seed}']=recursive(cache,m,seed,e,device)
 results={};per_dir=rd/'suite_per_query';per_dir.mkdir()
 def one(name,qidx,didx,tag):
  qids=[queries[i]['id'] for i in qidx];dids=[allids[i] for i in didx];qr={qid:{d:v for d,v in qrels[qid].items() if d in set(dids)} for qid in qids}
  if any(not x for x in qr.values()):raise ValueError(f'{cfg}/{tag}: missing condition qrel')
  pv=methods[name][didx];per,met=metrics(qids,dids,qr,qv[qidx]@pv.T);pp=per_dir/f'{name}__{tag}.jsonl';pp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in per));return {'metrics':met,'ci95_query_bootstrap':ci(per,int(hashlib.sha256(f'{cfg}|{name}|{tag}'.encode()).hexdigest()[:8],16)),'stratified':strata(per,qr,[m['documents'][i] for i in didx]),'per_query':{'path':str(pp.relative_to(rd)),'sha256':sha(pp)}}
 if cfg in SYNTH:
  contexts={}
  for ctx in sorted(set(int(x['context_length']) for x in corpus)):
   di=[i for i,x in enumerate(corpus) if int(x['context_length'])==ctx];qi=[i for i,x in enumerate(queries) if int(x['context_length'])==ctx];tag=f'context_{ctx}';cr={name:one(name,qi,di,tag) for name in methods}
   if str(ctx) in m['direct_teacher_groups']:
    z=m['direct_teacher_groups'][str(ctx)];p=cache/z['path']
    if sha(p)!=z['sha256'] or z['document_indexes']!=di:raise ValueError('direct group identity')
    d=torch.load(p,map_location='cpu',weights_only=False);methods['direct_teacher_tmp']=norm(d['embeddings'].numpy());saved=methods['direct_teacher_tmp'];methods['direct_teacher_tmp_full']=methods.get('direct_teacher_tmp_full');methods['direct_teacher_tmp']=saved
    # local direct computation because indexes are condition-local
    qids=[queries[i]['id'] for i in qi];dids=[allids[i] for i in di];qr={q:{d:v for d,v in qrels[q].items() if d in set(dids)} for q in qids};per,met=metrics(qids,dids,qr,qv[qi]@saved.T);pp=per_dir/f'direct_teacher__{tag}.jsonl';pp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in per));cr['direct_teacher']={'metrics':met,'ci95_query_bootstrap':ci(per,ctx+9182),'per_query':{'path':str(pp.relative_to(rd)),'sha256':sha(pp)}};methods.pop('direct_teacher_tmp');methods.pop('direct_teacher_tmp_full',None)
   contexts[str(ctx)]=cr
  # Unweighted context-condition mean, matching LongEmbed's synthetic evaluation convention.
  agg={}
  for name in list(methods):agg[name]={k:float(np.mean([contexts[c][name]['metrics'][k] for c in contexts])) for k in METRICS}
  results={'evaluation':'separate retrieval corpus/query/qrels per official context_length condition','contexts':contexts,'context_unweighted_mean':agg,'main_synthetic_score':'nDCG@1'}
 else:
  qi=list(range(len(queries)));di=list(range(len(corpus)));results={'methods':{name:one(name,qi,di,'all') for name in methods}}
  if 'all' in m['direct_teacher_groups']:
   z=m['direct_teacher_groups']['all'];p=cache/z['path'];d=torch.load(p,map_location='cpu',weights_only=False);dv=norm(d['embeddings'].numpy());per,met=metrics([x['id'] for x in queries],allids,qrels,qv@dv.T);results['methods']['direct_teacher']={'metrics':met,'ci95_query_bootstrap':ci(per,9917)}
 out={'schema':'v4_longembed_suite_results_v2','task':cfg,'source':source,'counts':{'documents':len(corpus),'queries':len(queries),'qrels':sum(map(len,qrels.values()))},'metrics_implementation':'pytrec_eval ndcg_cut.1/10 recall.10/100 plus explicit positive-qrel MRR@10','results':results,'provenance':{'suite_precommit_sha256':sha(PRE),'cache_manifest_sha256':sha(cache/'MANIFEST.json'),'checkpoints':prov,'software':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}};write(final,out);(rd/'SUITE_COMPLETE').write_text(f'{sha(final)}  SUITE_RESULTS.json\n');return out
def smoke():
 q=['q'];d=['a','b'];r={'q':{'a':1}};p,m=metrics(q,d,r,np.array([[1.,0.]]));assert m['nDCG@1']==1 and m['MRR@10']==1 and len(p)==1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--leaf-batch-size',type=int,default=8);ap.add_argument('--tasks',nargs='+',default=REAL+SYNTH);a=ap.parse_args();pre=json.loads(PRE.read_text());smoke()
 amend_path=OUT/'SUITE_ENGINEERING_AMENDMENT_1.json';amend=json.loads(amend_path.read_text()) if amend_path.exists() else None;accepted_evaluator=pre['evaluator_sha256'] if amend is None else amend['corrected_evaluator_sha256']
 if sha(ROOT/pre['inspection'])!=pre['inspection_sha256'] or sha(__file__)!=accepted_evaluator:raise ValueError('precommit/evaluator/inspection hash mismatch')
 tasks={c:load_task(c) for c in a.tasks};teacher=FrozenTeacher(TID,TREV,a.device,pooling='cls',trust_remote_code=True);man={c:prepare(c,x[0],x[1],teacher,a.leaf_batch_size) for c,x in tasks.items()};del teacher;torch.cuda.empty_cache()
 summary={c:evaluate(c,*tasks[c],man[c],pre,a.device) for c in a.tasks};write(OUT/'LONGEMBED_SUITE_NEW_TASKS_SUMMARY.json',{'schema':'v4_longembed_new_tasks_summary_v1','suite_precommit_sha256':sha(PRE),'tasks':{c:{'result':f'results/v4_euclidean/downstream/{c}/SUITE_RESULTS.json','sha256':sha(OUT/c/'SUITE_RESULTS.json')} for c in summary}})
if __name__=='__main__':main()
