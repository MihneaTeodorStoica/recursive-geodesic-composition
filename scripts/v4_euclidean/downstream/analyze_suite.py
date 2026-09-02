#!/usr/bin/env python3
"""Consolidate frozen retrieval outputs and predeclared scale/gap analyses."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from datasets import load_dataset
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v4_euclidean/downstream';REV='10039a580487dacecf79db69166e17ace3ede392';SEEDS=('424242','271828','314159');REAL=('narrativeqa','qmsum','2wikimqa','summ_screen_fd');ALL=('nfcorpus',)+REAL
TB=[[1,1024],[1025,4096],[4097,8190],[8191,16384],[16385,32768],[32769,65536],[65537,None]];LB=[[1,1],[2,4],[5,8],[9,16],[17,32],[33,64],[65,None]]
def load(p):return json.loads((ROOT/p).read_text())
def binlab(v,b):
 for a,z in b:
  if v>=a and (z is None or v<=z):return f'{a}-{z if z else "inf"}'
 raise ValueError(v)
def merges(n):
 x=0
 while n>1:q,r=divmod(n,4);x+=q+(r>1);n=q+(r>0)
 return int(x)
def qrels(task):
 if task=='nfcorpus':
  x={};p=OUT/'source/nfcorpus/qrels/test.tsv'
  with p.open() as f:
   next(f)
   for line in f:q,d,v=line.rstrip().split('\t');x.setdefault(q,{})[d]=int(v)
  return x
 d=load_dataset('dwzhu/LongEmbed',task,revision=REV,split='qrels');x={}
 for r in d:x.setdefault(str(r['qid']),{})[str(r['doc_id'])]=1
 return x
def manifest(task):
 name='MANIFEST.json';m=load(f'results/v4_euclidean/downstream/{"longembed_"+task if task=="narrativeqa" else task}/leaf_cache/{name}')
 for x in m['documents']:x.setdefault('merge_count',merges(x['leaves']))
 return m
def result_dir(task):return OUT/('longembed_narrativeqa' if task=='narrativeqa' else task)
def method_files(task):
 rd=result_dir(task)
 if task in ('nfcorpus','narrativeqa'):
  r=load(str((rd/'RESULTS.json').relative_to(ROOT)));return r['methods'],{k:rd/v['artifacts']['per_query']['path'] for k,v in r['methods'].items()},r
 r=load(str((rd/'SUITE_RESULTS.json').relative_to(ROOT)));methods=r['results']['methods'];return methods,{k:rd/v['per_query']['path'] for k,v in methods.items()},r
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def stats(rs):
 keys=[x for x in ('nDCG@1','nDCG@10','Recall@10','Recall@100','MRR@10') if x in rs[0]];return {k:float(np.mean([x[k] for x in rs])) for k in keys}
def paired_ci(a,b,key,tag):
 va=np.array([x[key] for x in a]);vb=np.array([x[key] for x in b]);assert len(va)==len(vb);rng=np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8],16));z=(va-vb)[rng.integers(0,len(va),(1000,len(va)))].mean(1);return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def single_ci(a,key,tag):
 v=np.array([x[key] for x in a]);rng=np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8],16));z=v[rng.integers(0,len(v),(1000,len(v)))].mean(1);return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def main():
 suite={'schema':'v4_complete_downstream_suite_v1','dataset_revision':REV,'methods':['recursive seeds and arithmetic metric mean','token-weighted leaf mean','unweighted leaf mean','direct teacher only where complete candidate set fits'],'real_world':{},'synthetic':{},'scale_gap_rows':[]};metric_rows=[];strata_rows=[]
 for task in ALL:
  m=manifest(task);qr=qrels(task);methods,files,raw=method_files(task);by={x['id']:x for x in m['documents']};per={k:rows(p) for k,p in files.items()};order={k:[x['query_id'] for x in v] for k,v in per.items()}
  if len({tuple(v) for v in order.values()})!=1:raise ValueError(f'{task}: method query alignment mismatch')
  groups={'tokens':{},'leaves':{},'depth':{}}
  for i,q in enumerate(next(iter(order.values()))):
   rel=sorted(d for d,v in qr[q].items() if v==max(qr[q].values()))[0];x=by[rel]
   labs={'tokens':binlab(x['tokens'],TB),'leaves':binlab(x['leaves'],LB),'depth':str(x['depth'])}
   for axis,lab in labs.items():groups[axis].setdefault(lab,[]).append(i)
  taskout={'counts':{'documents':len(m['documents']),'queries':len(next(iter(per.values()))),'tokens':sum(x['tokens'] for x in m['documents']),'leaves':sum(x['leaves'] for x in m['documents']),'merges':sum(x['merge_count'] for x in m['documents']),'max_depth':max(x['depth'] for x in m['documents'])},'methods':{},'method_ci95':{},'stratified':{},'gaps':{}}
  for name,rs in per.items():
   overall=stats(rs);taskout['methods'][name]=overall;taskout['method_ci95'][name]={k:single_ci(rs,k,f'{task}|{name}|{k}') for k in overall}
   for k,v in overall.items():metric_rows.append({'task':task,'method':name,'metric':k,'value':v})
  for axis,bins in groups.items():
   taskout['stratified'][axis]={}
   for lab,idx in bins.items():
    taskout['stratified'][axis][lab]={'queries':len(idx),'methods':{name:stats([rs[i] for i in idx]) for name,rs in per.items()}}
    for name,rs in per.items():
     for k,v in stats([rs[i] for i in idx]).items():strata_rows.append({'task':task,'axis':axis,'bin':lab,'queries':len(idx),'method':name,'metric':k,'value':v})
  # metric-space three-seed mean and paired per-query V4 ensemble mean for gaps.
  rec=[per[f'recursive_seed{s}'] for s in SEEDS];meanrows=[]
  for i in range(len(rec[0])):
   z={'query_id':rec[0][i]['query_id']}
   for k in rec[0][i]:
    if k!='query_id':z[k]=float(np.mean([r[i][k] for r in rec]))
   meanrows.append(z)
  taskout['methods']['recursive_three_seed_mean']=stats(meanrows);taskout['method_ci95']['recursive_three_seed_mean']={k:single_ci(meanrows,k,f'{task}|recursive_mean|{k}') for k in taskout['methods']['recursive_three_seed_mean']}
  for baseline in ('token_weighted_leaf_mean','unweighted_leaf_mean'):
   taskout['gaps'][baseline]={'overall':{k:taskout['methods']['recursive_three_seed_mean'][k]-taskout['methods'][baseline][k] for k in taskout['methods']['recursive_three_seed_mean']},'overall_paired_ci95':{k:paired_ci(meanrows,per[baseline],k,f'{task}|{baseline}|overall|{k}') for k in taskout['methods']['recursive_three_seed_mean']},'stratified':{}}
   for axis,bins in groups.items():
    taskout['gaps'][baseline]['stratified'][axis]={}
    for lab,idx in bins.items():
     a=[meanrows[i] for i in idx];b=[per[baseline][i] for i in idx];sa,sb=stats(a),stats(b);z={k:{'delta':sa[k]-sb[k],'paired_bootstrap_ci95':paired_ci(a,b,k,f'{task}|{baseline}|{axis}|{lab}|{k}')} for k in sa};taskout['gaps'][baseline]['stratified'][axis][lab]={'queries':len(idx),'metrics':z}
     for k,v in z.items():suite['scale_gap_rows'].append({'task':task,'baseline':baseline,'axis':axis,'bin':lab,'queries':len(idx),'metric':k,**v})
  suite['real_world'][task]=taskout
 for task in ('needle','passkey'):
  raw=load(f'results/v4_euclidean/downstream/{task}/SUITE_RESULTS.json');contexts=raw['results']['contexts'];z={'counts':raw['counts'],'main_score':'nDCG@1','contexts':{},'context_unweighted_mean':raw['results']['context_unweighted_mean']}
  for ctx,v in contexts.items():
   z['contexts'][ctx]={m:{'metrics':x['metrics'],'ci95_query_bootstrap':x['ci95_query_bootstrap']} for m,x in v.items()};
   for m,x in v.items():
    for k,val in x['metrics'].items():metric_rows.append({'task':task,'context_length':ctx,'method':m,'metric':k,'value':val})
  # seed mean and pooling gaps by context
  for ctx,v in z['contexts'].items():
   mean={k:float(np.mean([v[f'recursive_seed{s}']['metrics'][k] for s in SEEDS])) for k in v[f'recursive_seed{SEEDS[0]}']['metrics']};v['recursive_three_seed_mean']={'metrics':mean}
   v['gaps']={b:{k:mean[k]-v[b]['metrics'][k] for k in mean} for b in ('token_weighted_leaf_mean','unweighted_leaf_mean')}
  suite['synthetic'][task]=z
 suite['nonretrieval']=load('results/v4_euclidean/downstream/DOWNSTREAM_RESULTS.json');write=OUT/'CONSOLIDATED_SUITE_RESULTS.json';write.write_text(json.dumps(suite,indent=2)+'\n')
 pub=ROOT/'results/v4_euclidean/publication_sources';pub.mkdir(exist_ok=True)
 docrows=[]
 for task in ALL+('needle','passkey'):
  mm=manifest(task)
  for x in mm['documents']:docrows.append({'task':task,'document_id':x['id'],'token_count':x['tokens'],'leaf_count':x['leaves'],'recursion_depth':x['depth'],'merge_count':x.get('merge_count',merges(x['leaves'])),'context_length':x.get('context_length')})
 for p,rs in [(pub/'complete_retrieval_metrics.csv',metric_rows),(pub/'retrieval_scale_strata.csv',strata_rows),(pub/'retrieval_scale_gaps.csv',suite['scale_gap_rows']),(pub/'document_scale_metadata.csv',docrows)]:
  keys=sorted(set().union(*(r.keys() for r in rs)));f=p.open('w',newline='');w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs);f.close()
 print(write)
if __name__=='__main__':main()
