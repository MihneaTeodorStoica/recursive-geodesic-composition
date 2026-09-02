#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v4_euclidean/downstream';checks=[]
def h(p):
 z=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):z.update(b)
 return z.hexdigest()
def add(n,v,d=''):checks.append({'name':n,'pass':bool(v),'detail':d})
def load(p):return json.loads(Path(p).read_text())
def main():
 add('branch',subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='v4-paper-euclidean')
 pre=load(OUT/'SUITE_PRECOMMIT.json');add('immutable dataset revision',pre['dataset']['suite_revision']=='10039a580487dacecf79db69166e17ace3ede392');rev=load(ROOT/pre['dataset']['revision_audit']);add('MTEB revision content identity',rev['all_benchmark_files_content_identical_between_revisions'])
 add('inspection frozen',h(ROOT/pre['inspection'])==pre['inspection_sha256']);ins=load(ROOT/pre['inspection']);add('all qrels prevalidated',all(x['id_audit']['query_set_equals_qrel_set'] and x['id_audit']['all_qrel_documents_exist'] and x['id_audit']['duplicates']==0 and x['id_audit']['malformed']==0 for x in ins['tasks'].values()))
 for s,e in pre['checkpoints'].items():add(f'frozen checkpoint {s}',h(ROOT/e['path'])==e['sha256'])
 for task in ('qmsum','2wikimqa','summ_screen_fd','needle','passkey'):
  cache=OUT/task/'leaf_cache';mp=cache/'MANIFEST.json';m=load(mp);add(f'{task} cache complete',(cache/'COMPLETE').read_text().strip()==f'{h(mp)}  MANIFEST.json');add(f'{task} document fields',all(all(k in x for k in ('tokens','leaves','depth','merge_count','sha256','leaf_tensor_sha256')) for x in m['documents']))
  ok=True
  for x in m['documents']:
   p=cache/x['shard'];ok=ok and p.is_file() and h(p)==x['sha256']
  add(f'{task} all shard hashes',ok);add(f'{task} query hash',h(cache/m['query_cache']['path'])==m['query_cache']['sha256'])
  rp=OUT/task/'SUITE_RESULTS.json';r=load(rp);add(f'{task} result complete',(OUT/task/'SUITE_COMPLETE').read_text().strip()==f'{h(rp)}  SUITE_RESULTS.json');add(f'{task} no count exclusion',r['counts']['documents']==ins['tasks'][task]['schema']['corpus']['rows'] and r['counts']['queries']==ins['tasks'][task]['schema']['queries']['rows'] and r['counts']['qrels']==ins['tasks'][task]['qrel_pairs'])
  if task in ('needle','passkey'):
   contexts=r['results']['contexts'];add(f'{task} standard contexts',set(contexts)==set(map(str,(256,512,1024,2048,4096,8192,16384,32768))) and all(len(x)==6 if int(c)<=8192 else len(x)==5 for c,x in contexts.items()))
  else:add(f'{task} all methods',set(r['results']['methods'])=={f'recursive_seed{s}' for s in ('424242','271828','314159')}|{'token_weighted_leaf_mean','unweighted_leaf_mean'})
 con=load(OUT/'CONSOLIDATED_SUITE_RESULTS.json');add('all requested retrieval tasks',set(con['real_world'])=={'nfcorpus','narrativeqa','qmsum','2wikimqa','summ_screen_fd'} and set(con['synthetic'])=={'needle','passkey'});add('all scale analyses',all(set(z['stratified'])=={'tokens','leaves','depth'} and set(z['gaps'])=={'token_weighted_leaf_mean','unweighted_leaf_mean'} for z in con['real_world'].values()))
 da=load(OUT/'DOWNSTREAM_AUDIT.json');add('prior classification/cache audit',all(da[k] for k in ('tensor_identical_leaf_cache','no_silent_exclusions','standard_metrics','all_hashes_match')))
 figs=list((ROOT/'figures/v4_euclidean/downstream_suite').glob('*'));add('publication figures',len(figs)==20 and all(p.stat().st_size>1000 for p in figs));sources=list((ROOT/'results/v4_euclidean/publication_sources').glob('figure_*.csv'));add('figure sources',len(sources)>=9 and all(p.stat().st_size>20 for p in sources));add('per-document metadata',sum(1 for _ in (ROOT/'results/v4_euclidean/publication_sources/document_scale_metadata.csv').open())==6422)
 test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env={**__import__('os').environ,'PYTHONPATH':'src'},stdout=subprocess.PIPE);add('tests',test.returncode==0,test.stdout.decode()[-200:])
 out={'schema':'v4_downstream_suite_final_audit_v1','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks};p=OUT/'SUITE_FINAL_AUDIT.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__':main()
