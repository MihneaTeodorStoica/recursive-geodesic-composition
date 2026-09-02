#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'figures/v4_euclidean'; PUB=ROOT/'results/v4_euclidean/publication_sources';OUT.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True)
d=json.loads((ROOT/'results/v4_euclidean/FINAL_METRICS_SUMMARY.json').read_text());dev=d['development']; order=['geodesic','euclidean_residual','free_euclidean'];labels=['Geodesic control','Euclidean residual','Free Euclidean'];colors=['#4c78a8','#f58518','#54a24b']
rows=[]
for g,l in zip(order,labels):
 for seed,x in dev[g]['seeds'].items(): rows.append({'architecture':g,'label':l,'seed':seed,'cosine':x['cosine'],'auc':x['strongest_auc'],'rank':x['effective_rank'],'saturation':x['full_saturation_count']})
with open(PUB/'development_architecture_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(7,4.5))
for g,l,c in zip(order,labels,colors):
 r=[x for x in rows if x['architecture']==g];ax.scatter([x['cosine'] for x in r],[x['auc'] for x in r],s=70,label=l,color=c)
 for x in r:ax.annotate(x['seed'],(x['cosine'],x['auc']),fontsize=7,xytext=(3,3),textcoords='offset points')
ax.axhline(.5,color='gray',ls='--',lw=1);ax.set_xlabel('Direct safe teacher cosine');ax.set_ylabel('Strongest orientation-safe AUC');ax.set_title('V4 development: fidelity versus detectability');ax.legend();fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'detectability_vs_fidelity.{ext}',dpi=220)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,4))
for i,(metric,title) in enumerate([('rank','Effective rank'),('saturation','Full-recursion cap saturations')]):
 vals=[[x[metric] for x in rows if x['architecture']==g] for g in order];axs[i].bar(range(3),[sum(v)/len(v) for v in vals],color=colors);axs[i].set_xticks(range(3),['Geo','Euc-res','Free']);axs[i].set_title(title)
fig.suptitle('V4 architecture stability');fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'architecture_stability.{ext}',dpi=220)
plt.close(fig)
# Downstream concise table/figure
r=d['downstream']['retrieval']['datasets']; c=d['downstream']['classification_clustering']; seeds=['424242','271828','314159'];
def avg(vals):return sum(vals)/len(vals)
metrics={'NFCorpus nDCG@10':avg([r['nfcorpus']['metrics'][f'recursive_seed{s}']['nDCG@10'] for s in seeds]),'NarrativeQA nDCG@10':avg([r['longembed_narrativeqa']['metrics'][f'recursive_seed{s}']['nDCG@10'] for s in seeds]),'ECtHR-A macro F1':avg([c['ecthr_a'][f'recursive_seed{s}']['test']['macro_f1'] for s in seeds]),'20NG ARI':avg([c['20_newsgroups'][f'recursive_seed{s}']['ari'] for s in seeds])}
with open(PUB/'downstream_metrics.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['metric','v4_recursive_three_seed_mean']);w.writerows(metrics.items())
fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(4),metrics.values(),color='#f58518');ax.set_xticks(range(4),['NFCorpus\nnDCG@10','NarrativeQA\nnDCG@10','ECtHR-A\nmacro F1','20NG\nARI']);ax.set_ylim(0,.65);ax.set_title('Selected V4 Euclidean-residual downstream results');fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'downstream_summary.{ext}',dpi=220)
