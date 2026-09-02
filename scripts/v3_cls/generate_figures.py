#!/usr/bin/env python3
"""Generate V3 figures exclusively from preserved machine-readable results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'figures/v3_cls'; SRC=ROOT/'results/v3_cls/publication_sources'; OUT.mkdir(parents=True,exist_ok=True); SRC.mkdir(parents=True,exist_ok=True)
SEEDS=(424242,271828,314159); CORPORA=('pooled','fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')
def load(seed):return json.loads((ROOT/f'results/v3_cls/official_seed{seed}/TWO_ARM_EVALUATION.json').read_text())
def write(name,rows):
 if not rows:return
 with (SRC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
 ds={s:load(s) for s in SEEDS}; rows=[]
 for s,d in ds.items():
  for c in CORPORA:
   p=d['safe_direct']['scopes'][c]; rows.append({'seed':s,'corpus':c,'rows':p['rows'],'teacher_cosine':p['teacher_cosine'],'cosine_ci_low':p['teacher_cosine_ci95_bootstrap'][0],'cosine_ci_high':p['teacher_cosine_ci95_bootstrap'][1],'logistic_auc':p['attacks']['attacks']['standard_scaled_logistic']['orientation_safe_auc'],'rbf_auc':p['attacks']['attacks']['standard_scaled_rbf_svm']['orientation_safe_auc'],'strongest_auc':p['attacks']['strongest_orientation_safe_auc'],'effective_rank':p['health']['effective_rank']})
 write('official_safe_metrics.csv',rows)
 x=np.arange(len(CORPORA)); labels=['All','FineWeb-Edu','FineWeb','OpenWebText','Wikipedia']; fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 for metric,title,ax in [('teacher_cosine','Paired normalized-CLS cosine',axes[0]),('strongest_auc','Strongest orientation-safe ROC-AUC',axes[1])]:
  vals=np.array([[r[metric] for r in rows if r['seed']==s] for s in SEEDS]); mean=vals.mean(0); ax.errorbar(x,mean,yerr=vals.std(0),fmt='o-',capsize=3,color='#315a9e'); ax.set_xticks(x,labels,rotation=25,ha='right'); ax.set_title(title); ax.grid(alpha=.25); ax.set_ylim((.98,.992) if metric=='teacher_cosine' else (.5,.82));
 fig.tight_layout(); fig.savefig(OUT/'official_cosine_auc.pdf'); fig.savefig(OUT/'official_cosine_auc.png',dpi=220); plt.close(fig)
 ret=[]
 for s,d in ds.items():
  r=d['safe_direct']['retention']; t=d['full_document']['telemetry']; ret.append({'seed':s,'decoder_child_cosine':r['built_in_parent_only_decoder']['pooled_ordered_child_cosine'],'probe_child_cosine':r['train_fitted_parent_only_probe']['pooled_ordered_child_cosine'],'probe_all_floors_pass':r['train_fitted_parent_only_probe']['all_floors_pass'],'documents':t['documents'],'leaves':t['leaves'],'internal_merges':t['internal_merges'],'max_depth':t['depth']['maximum'],'max_angle':t['geodesic_angle_radians']['maximum'],'geodesic_saturation_count':t['geodesic_saturation_count'],'importance_saturation_count':t['importance_saturation_count']})
 write('official_retention_recursion.csv',ret)
 fig,ax=plt.subplots(figsize=(6,4)); width=.35; vals=np.array([[r['decoder_child_cosine'],r['probe_child_cosine']] for r in ret]); ax.bar(np.arange(3)-width/2,vals[:,0],width,label='Built-in decoder'); ax.bar(np.arange(3)+width/2,vals[:,1],width,label='TRAIN-fitted probe'); ax.set_xticks(range(3),[str(s) for s in SEEDS]); ax.set_ylabel('Ordered child cosine'); ax.set_ylim(.88,.93); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'official_retention.pdf'); fig.savefig(OUT/'official_retention.png',dpi=220); plt.close(fig)
 summary={'checkpoint_seeds':list(SEEDS),'safe_metrics_csv':'results/v3_cls/publication_sources/official_safe_metrics.csv','retention_recursion_csv':'results/v3_cls/publication_sources/official_retention_recursion.csv','source_results':[f'results/v3_cls/official_seed{s}/TWO_ARM_EVALUATION.json' for s in SEEDS]}; (SRC/'FIGURE_PROVENANCE.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
