import json,sys,os
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src')]
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.evaluate import evaluate_attacks
C=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101');CP={424242:'results/v2_cls/amended_v2_1_seed424242/stage_c/autoencoder_seed424242.pt',271828:'results/v2_cls/amended_v2_1_seed271828/stage_c/autoencoder_seed271828.pt',314159:'results/v2_cls/amended_v2_1_seed314159/stage_c_retry_after_enospc/autoencoder_seed314159.pt'}
def main():
 out=ROOT/'reproducibility/v2_cls/auc_diagnostic';out.mkdir(parents=True,exist_ok=True);cache=ROOT/'artifacts/v2_cls/official_v2_1_cache';dev=ROOT/'artifacts/v2_cls/development_safe_cls';res={'seeds':{}}
 for s,p in CP.items():
  m,_=build_model(ROOT/p,torch.device('cuda')); fs=[];ts=[];cs=[];meta=[];ints=[]
  with torch.inference_mode():
   for c in C:
    for r in torch.load(cache/f'{c}_safe.pt',map_location='cpu',weights_only=False):
     ll=r['leaves'].cuda();mm=r['lengths'].cuda();root,aux=m.merge(F.pad(ll,(0,0,0,4-len(ll)))[None],F.pad(mm,(0,4-len(mm)))[None],return_aux=True);root=root[0];fs.append(root.cpu().numpy());ts.append(r['teacher_target'].numpy());cs.append(c);meta.append((int(r['lengths'].sum()),len(r['lengths'])));ints.append([float(aux['angle']),float(aux['raw_tangent_norm']),float(aux['correction_norm'])])
  f=np.asarray(fs);t=np.asarray(ts);resid=f-t; entry={'pooled':{},'per_corpus':{}}
  for name,ix in [('pooled',np.arange(len(f)))]+[(c,np.array([i for i,x in enumerate(cs) if x==c])) for c in C]:
   a,b=f[ix],t[ix]; cov=np.cov(a,rowvar=False)-np.cov(b,rowvar=False); sv=np.linalg.svd(a-a.mean(0),compute_uv=False);q=sv/sv.sum(); stats={'n':len(ix),'cosine':float(np.mean(np.sum(a*b,1))),'mean_diff_norm':float(np.linalg.norm(a.mean(0)-b.mean(0))),'var_diff_norm':float(np.linalg.norm(a.var(0)-b.var(0))),'cov_fro':float(np.linalg.norm(cov,'fro')),'effective_rank_generated':float(np.exp(-np.sum(q*np.log(np.maximum(q,1e-30)))))}
   if name=='pooled':entry['pooled']['stats']=stats
   else:entry['per_corpus'][name]=stats
   if name!='pooled': entry['per_corpus'][name]['attack']=evaluate_attacks(b,a,[str(i) for i in ix],bootstrap=0)['max_attack_auc']
  z=np.r_[t,f];y=np.r_[np.ones(len(t)),np.zeros(len(f))];lr=LogisticRegression(max_iter=1000).fit(z,y);score=lr.decision_function(z);entry['logistic']={'auc':float(roc_auc_score(y,score)),'coefficient_norm':float(np.linalg.norm(lr.coef_)),'top_coordinates':np.argsort(np.abs(lr.coef_[0]))[-10:].tolist()};entry['residual']={'mean_norm':float(np.linalg.norm(resid.mean(0))),'mean_sample_norm':float(np.mean(np.linalg.norm(resid,axis=1))),'mean_alignment':float(np.mean(F.cosine_similarity(torch.tensor(resid),torch.tensor(resid.mean(0))[None]).numpy()))};entry['internal']={'angle_mean':float(np.mean(ints,0)[0]),'angle_max':float(np.max(ints,0)[0]),'raw_tangent_mean':float(np.mean(ints,0)[1])};np.savez_compressed(out/f'seed{s}_safe_arrays.npz',generated=f,teacher=t,residual=resid,corpus=np.asarray(cs),meta=np.asarray(meta),internal=np.asarray(ints),score=score);res['seeds'][str(s)]=entry
 (out/'SUMMARY.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
