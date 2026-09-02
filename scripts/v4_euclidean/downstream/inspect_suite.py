#!/usr/bin/env python3
"""Pre-performance schema/ID/qrel/token-length inspection for the frozen V4 suite."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'results/v4_euclidean/downstream'
REV='10039a580487dacecf79db69166e17ace3ede392'; TEACHER='Alibaba-NLP/gte-modernbert-base'; TREV='e7f32e3c00f91d699e8c43b53106206bcc72bb22'
CONFIGS=['narrativeqa','qmsum','2wikimqa','summ_screen_fd','needle','passkey']
TOKEN_BINS=[[1,1024],[1025,4096],[4097,8190],[8191,16384],[16385,32768],[32769,65536],[65537,None]]
LEAF_BINS=[[1,1],[2,4],[5,8],[9,16],[17,32],[33,64],[65,None]]
def pct(v):
 a=np.asarray(v);return {'min':int(a.min()),'p10':float(np.quantile(a,.1)),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'p99':float(np.quantile(a,.99)),'max':int(a.max()),'mean':float(a.mean())}
def merges(n):
 total=0
 while n>1:
  q,r=divmod(n,4);total+=q+(1 if r>1 else 0);n=q+(1 if r else 0)
 return total
def depth(n):
 d=0
 while n>1:n=(n+3)//4;d+=1
 return d
def main():
 tok=AutoTokenizer.from_pretrained(TEACHER,revision=TREV,trust_remote_code=True); result={'schema':'v4_downstream_suite_preperformance_inspection_v1','created_unix':time.time(),'restriction':'No representation model, retrieval score, ranking, or task performance was computed.','dataset':{'id':'dwzhu/LongEmbed','revision':REV},'teacher':{'id':TEACHER,'revision':TREV,'pooling':'normalized last_hidden_state[:,0]'},'predeclared_bins':{'tokens':TOKEN_BINS,'leaves':LEAF_BINS,'depth':'exact integer','synthetic':'official context_length field'},'tasks':{}}
 for cfg in CONFIGS:
  splits={s:load_dataset('dwzhu/LongEmbed',cfg,revision=REV,split=s) for s in ('corpus','queries','qrels')}; rows={s:[dict(x) for x in d] for s,d in splits.items()}
  schema={s:{'rows':len(splits[s]),'columns':splits[s].column_names,'features':str(splits[s].features)} for s in splits}
  ids={s:[str(x['doc_id'] if s=='corpus' else x['qid']) for x in rows[s]] for s in rows};
  if any(not x for x in ids['corpus']+ids['queries']) or len(set(ids['corpus']))!=len(ids['corpus']) or len(set(ids['queries']))!=len(ids['queries']):raise ValueError(f'{cfg}: empty/duplicate IDs')
  did=set(ids['corpus']);qid=set(ids['queries']);seen=set();qrels={}
  for i,x in enumerate(rows['qrels']):
   q,d=str(x['qid']),str(x['doc_id'])
   if not q or not d or (q,d) in seen:raise ValueError(f'{cfg}: malformed/duplicate qrel {i}')
   seen.add((q,d));qrels.setdefault(q,set()).add(d)
  if set(qrels)!=qid or any(not ds for ds in qrels.values()) or any(d not in did for ds in qrels.values() for d in ds):raise ValueError(f'{cfg}: qrel/query/document corruption')
  if any(not isinstance(x['text'],str) or not x['text'].strip() for x in rows['corpus']+rows['queries']):raise ValueError(f'{cfg}: malformed empty text')
  token_counts=[len(tok(x['text'],add_special_tokens=False,truncation=False)['input_ids']) for x in rows['corpus']]; query_tokens=[len(tok(x['text'],add_special_tokens=False,truncation=False)['input_ids']) for x in rows['queries']]
  leaves=[math.ceil(n/1024) for n in token_counts];deps=[depth(n) for n in leaves];mrs=[merges(n) for n in leaves]
  task={'schema':schema,'qrel_pairs':len(seen),'id_audit':{'query_set_equals_qrel_set':set(qrels)==qid,'all_qrel_documents_exist':all(d in did for ds in qrels.values() for d in ds),'duplicates':0,'malformed':0},'documents':{'tokens':pct(token_counts),'leaves':pct(leaves),'depth':pct(deps),'merges':pct(mrs),'total_tokens':sum(token_counts),'total_leaves':sum(leaves)},'queries':{'tokens':pct(query_tokens),'all_directly_encodable':max(query_tokens)<=8190},'direct_teacher_entire_task_eligible':max(token_counts)<=8190 and max(query_tokens)<=8190}
  if cfg in ('needle','passkey'):
   contexts=sorted(set(int(x['context_length']) for x in rows['corpus']));task['context_length_conditions']={str(c):{'documents':sum(int(x['context_length'])==c for x in rows['corpus']),'queries':sum(int(x['context_length'])==c for x in rows['queries']),'qrels':sum(int(x['context_length'])==c for x in rows['qrels']),'token_counts':pct([n for n,x in zip(token_counts,rows['corpus']) if int(x['context_length'])==c])} for c in contexts}
   if any(int(x['context_length']) not in contexts for x in rows['queries']+rows['qrels']):raise ValueError(f'{cfg}: context condition corruption')
  result['tasks'][cfg]=task;print(cfg,task['documents'],flush=True)
 p=OUT/'SUITE_PREPERFORMANCE_INSPECTION.json';p.write_text(json.dumps(result,indent=2)+'\n');print(p)
if __name__=='__main__':main()
