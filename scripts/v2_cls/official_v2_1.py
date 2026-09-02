#!/usr/bin/env python3
"""V2.1 create-only official CLS cache and two-arm evaluator.

Imports are corpus-free. Only ``prepare`` may open the frozen official scan.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F, yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src')]
from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text
from recursive_geodesic_composition.architecture import recursive_reduce, contiguous_reduction_groups
from recursive_geodesic_composition.evaluate_kary_validation import build_model, ParentOnlyRetentionProbe
from recursive_geodesic_composition.evaluate import evaluate_attacks
CORPORA=(('fineweb_edu','HuggingFaceFW/fineweb-edu','sample-10BT','87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'),('fineweb','HuggingFaceFW/fineweb',None,'9bb295ddab0e05d785b879661af7260fed5140fc'),('openwebtext','Skylion007/openwebtext',None,'79d93d786212f7344586290adb811d4ae6a1762c'),('wikipedia_en_20231101','wikimedia/wikipedia','20231101.en','b04c8d1ceb2f5cd4588862100d08de323dccfbaa'))
def sh(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def chunks(ids,w=1024):
 x=[ids[i:i+w] for i in range(0,len(ids),w)]
 if not x or sum(map(len,x))!=len(ids):raise ValueError('exact chunk failure')
 return x
def depth(n):
 d=0
 while n>1:n=(n+3)//4;d+=1
 return d
def cfg():return yaml.safe_load((ROOT/'configs/v2_cls/official_v2_1.yaml').read_text())
def preflight(c):
 if c['scan']!=[214000,244000] or c['teacher']['pooling']!='cls':raise ValueError('official config')
 if (ROOT/c['cache_dir']).exists() or (ROOT/c['output_dir']).exists():raise FileExistsError('one-shot namespace exists')
 probes=json.loads((ROOT/c['probe_manifest']).read_text())
 for s,p in c['checkpoints'].items():
  x=torch.load(ROOT/p,map_location='cpu',weights_only=True);want={'teacher_id':c['teacher']['id'],'teacher_revision':c['teacher']['revision'],'teacher_pooling':'cls'}
  if x.get('teacher_provenance')!=want or probes['seeds'][s]['checkpoint_sha256']!=sh(ROOT/p):raise ValueError('checkpoint/probe mismatch')
 return probes
def prepare(device):
 c=cfg();probes=preflight(c);out=ROOT/c['cache_dir']; marker=out.with_name(out.name+'.OFFICIAL_SPENT')
 fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o444);os.write(fd,b'official access claimed\n');os.close(fd)
 stage=out.with_name('.'+out.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True);teacher=FrozenTeacher(c['teacher']['id'],c['teacher']['revision'],device,pooling='cls',trust_remote_code=True);mani={'stage':c['stage'],'scan':c['scan'],'teacher':teacher.provenance(),'safe':{},'full':{},'config_sha256':sh(ROOT/'configs/v2_cls/official_v2_1.yaml'),'probes_sha256':sh(ROOT/c['probe_manifest'])}
 try:
  from datasets import load_dataset
  for cid,name,conf,rev in CORPORA:
   ds=load_dataset(name,conf,split='train',streaming=True,revision=rev);safe=[];idx=[]
   for off,raw in enumerate(ds.skip(c['scan'][0]).take(c['scan'][1]-c['scan'][0])):
    text=raw.get('text') or raw.get('content') or ''
    if not isinstance(text,str) or not text.strip():continue
    ids=tokenize_text(teacher.tokenizer,text)
    if not (1025<=len(ids)) :continue
    plen=min(len(ids),4096); pref=ids[:plen]; pc=chunks(pref)
    if not 2<=len(pc)<=4:continue
    did=str(raw.get('id',c['scan'][0]+off)); leaves=teacher.embed(pc,32);target=teacher.embed([pref],1)[0]
    fullc=chunks(ids);fullleaves=[]
    for j in range(0,len(fullc),32):fullleaves.append(teacher.embed(fullc[j:j+32],32))
    ident={'id':f'{cid}:{did}','source_index':c['scan'][0]+off,'source_document_id':did,'corpus':cid,'full_token_count':len(ids),'full_leaf_count':len(fullc),'tree_depth':depth(len(fullc))}
    safe.append({**ident,'arm':'safe','leaves':leaves,'lengths':torch.tensor(list(map(len,pc)),dtype=torch.float32),'teacher_target':target,'teacher_pooling':'cls','token_hash':hashlib.sha256(json.dumps(pref).encode()).hexdigest()})
    torch.save({**ident,'arm':'full','leaves':torch.cat(fullleaves).half(),'lengths':torch.tensor(list(map(len,fullc)),dtype=torch.float32),'teacher_pooling':'cls','token_hash':hashlib.sha256(json.dumps(ids).encode()).hexdigest()},stage/cid/f'{len(safe)-1:06d}.pt') if (stage/cid).mkdir(exist_ok=True) is None else None
    idx.append(ident)
    if len(safe)>=c['selected_documents_per_corpus']:break
   if len(safe)!=c['selected_documents_per_corpus']:raise RuntimeError('insufficient selected safe rows')
   torch.save(safe,stage/f'{cid}_safe.pt');(stage/f'{cid}_index.json').write_text(json.dumps(idx,indent=2)+'\n');mani['safe'][cid]={'sha256':sh(stage/f'{cid}_safe.pt'),'rows':len(safe)};mani['full'][cid]={'index_sha256':sh(stage/f'{cid}_index.json'),'rows':len(idx)}
  (stage/'CACHE_MANIFEST.json').write_text(json.dumps(mani,indent=2)+'\n');os.replace(stage,out)
 except BaseException:
  raise
@torch.inference_mode()
def full_merge_diagnostics(model, probe, leaves, lengths, device):
 """Decode every genuine merge; retain exact child-type/order strata.

 ``leaf`` means an original input leaf. ``recursive`` means a node produced at
 an earlier genuine merge. The TRAIN probe was fit only on exact four-child
 parents, so probe values are defined only for arity-four merges; decoder values
 are defined for all genuine arities with their active ordered prefix.
 """
 nodes=leaves.float().to(device)[None]; masses=lengths.float().to(device)[None]; kinds=[[True]*nodes.shape[1]]; out=[]
 while nodes.shape[1]>1:
  groups=contiguous_reduction_groups(nodes.shape[1],4); nextn=[]; nextm=[]; nextk=[]
  for start,width in groups:
   if width==1:
    nextn.append(nodes[:,start:start+1]);nextm.append(masses[:,start:start+1]);nextk.append(kinds[0][start]);continue
   child=nodes[:,start:start+width]; cm=masses[:,start:start+width]; pad=4-width
   pchild=F.pad(child,(0,0,0,pad));pmass=F.pad(cm,(0,pad));parent=model.merge(pchild,pmass)
   decoded=model.decode(parent,return_aux=False)[:,:width]
   cos=F.cosine_similarity(decoded,child,dim=-1)[0]
   isleaf=kinds[0][start:start+width]
   entry={'arity':width,'ordered_child_cosines':[float(x) for x in cos],'child_types':['leaf' if x else 'recursive' for x in isleaf], 'probe_retention':None}
   if width==4:
    pred=probe(parent);entry['probe_retention']=float(F.cosine_similarity(pred,child,dim=-1).mean())
   out.append(entry);nextn.append(parent[:,None]);nextm.append(pmass.sum(1,keepdim=True));nextk.append(False)
  nodes=torch.cat(nextn,1);masses=torch.cat(nextm,1);kinds=[nextk]
 return nodes[:,0],out

def summarize_full_merges(entries):
 vals=[];leaf=[];rec=[];probe=[]
 for e in entries:
  vals+=e['ordered_child_cosines']
  for v,t in zip(e['ordered_child_cosines'],e['child_types']):(leaf if t=='leaf' else rec).append(v)
  if e['probe_retention'] is not None:probe.append(e['probe_retention'])
 return {'ordered_child_cosine_overall':float(np.mean(vals)) if vals else None,'ordered_child_cosine_leaf_children':float(np.mean(leaf)) if leaf else None,'ordered_child_cosine_recursive_children':float(np.mean(rec)) if rec else None,'genuine_merges':len(entries),'leaf_child_slots':len(leaf),'recursive_child_slots':len(rec),'probe_exact4_parent_retention':float(np.mean(probe)) if probe else None,'probe_exact4_parent_merges':len(probe)}

def evaluate(device):
 c=cfg();cache=ROOT/c['cache_dir'];m=json.loads((cache/'CACHE_MANIFEST.json').read_text());probes=json.loads((ROOT/c['probe_manifest']).read_text());out=ROOT/c['output_dir'];out.mkdir(parents=True);result={'safe':{},'full':{},'cache_manifest_sha256':sh(cache/'CACHE_MANIFEST.json')}
 for seed,cp in c['checkpoints'].items():
  model,_=build_model(ROOT/cp,torch.device(device));probe=ParentOnlyRetentionProbe(768,4,512).to(device);probe.load_state_dict(torch.load(ROOT/probes['seeds'][seed]['probe'],map_location='cpu',weights_only=True)['state_dict']);fake=[];real=[];groups=[];per={};fullstats=[]
  for cid,*_ in CORPORA:
   rows=torch.load(cache/f'{cid}_safe.pt',map_location='cpu',weights_only=False);fs=[];rs=[];dec=[];ret=[]
   with torch.inference_mode():
    for r in rows:
     root,_=recursive_reduce(model,r['leaves'].to(device)[None],r['lengths'].to(device)[None]);root=root[0];fs.append(root.cpu());rs.append(r['teacher_target']);dec.append(float(F.cosine_similarity(model.decode(root[None])[0,:len(r['lengths'])],r['leaves'].to(device),dim=-1).mean()));ret.append(float(F.cosine_similarity(probe(root[None]),r['leaves'].to(device)[None],dim=-1).mean()))
   f=np.stack([x.numpy() for x in fs]);r=np.stack([x.numpy() for x in rs]);per[cid]={'teacher_cosine':float(np.mean(np.sum(f*r,1))),'decoder_child_cosine':float(np.mean(dec)),'probe_retention':float(np.mean(ret))};fake.append(f);real.append(r);groups += [x['id'] for x in rows]
   for sp in sorted((cache/cid).glob('*.pt')):
    x=torch.load(sp,map_location='cpu',weights_only=False)
    root,entries=full_merge_diagnostics(model,probe,x['leaves'],x['lengths'],device)
    fullstats.append({'id':x['id'],'corpus':cid,'tokens':x['full_token_count'],'leaves':x['full_leaf_count'],'depth':x['tree_depth'],'finite':bool(torch.isfinite(root).all()),'norm':float(root.norm()),**summarize_full_merges(entries)})
  f=np.concatenate(fake);r=np.concatenate(real);result['safe'][seed]={'pooled_teacher_cosine':float(np.mean(np.sum(f*r,1))),'per_corpus':per,'attacks':evaluate_attacks(r,f,groups,bootstrap=c['bootstrap']['replicates'])};result['full'][seed]=fullstats
 (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate'));p.add_argument('--device',default='cuda');a=p.parse_args();prepare(a.device) if a.mode=='prepare' else evaluate(a.device)
