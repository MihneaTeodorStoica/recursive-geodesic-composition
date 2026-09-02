#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'figures/v3_cls'; SRC=ROOT/'results/v3_cls/publication_sources'; OUT.mkdir(exist_ok=True);SRC.mkdir(exist_ok=True)
def main():
 rows=[]
 for ds in ('nfcorpus','longembed_narrativeqa'):
  d=json.loads((ROOT/f'results/v3_cls/downstream/{ds}/RESULTS.json').read_text())
  for method,v in d['methods'].items(): rows.append({'dataset':ds,'method':method,**v['metrics']})
 with (SRC/'downstream_retrieval_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 long=[r for r in rows if r['dataset']=='longembed_narrativeqa']; labels=[r['method'].replace('recursive_seed','R').replace('token_weighted_leaf_mean','Weighted').replace('unweighted_leaf_mean','Unweighted') for r in long]; x=np.arange(len(long)); fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(x,[r['nDCG@10'] for r in long],color=['#315a9e']*3+['#d97904','#77964a']);ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_ylabel('nDCG@10');ax.set_ylim(.52,.57);ax.set_title('LongEmbed NarrativeQA (355 long documents)');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'narrativeqa_retrieval.pdf');fig.savefig(OUT/'narrativeqa_retrieval.png',dpi=220);plt.close(fig)
 d=json.loads((ROOT/'results/v3_cls/downstream/longembed_narrativeqa/RESULTS.json').read_text()); methods=['recursive_seed424242','recursive_seed271828','recursive_seed314159']; depths=sorted(d['methods'][methods[0]]['stratified']['depth'],key=int); gap=[]
 for dep in depths:
  rec=np.mean([d['methods'][m]['stratified']['depth'][dep]['nDCG@10'] for m in methods]); base=d['methods']['token_weighted_leaf_mean']['stratified']['depth'][dep]['nDCG@10']; gap.append({'depth':int(dep),'recursive_mean_ndcg10':rec,'weighted_ndcg10':base,'recursive_minus_weighted':rec-base,'queries':d['methods'][methods[0]]['stratified']['depth'][dep]['queries']})
 with (SRC/'narrativeqa_recursive_minus_pooling_by_depth.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=gap[0]);w.writeheader();w.writerows(gap)
 fig,ax=plt.subplots(figsize=(6,4));ax.axhline(0,color='black',lw=.8);ax.plot([x['depth'] for x in gap],[x['recursive_minus_weighted'] for x in gap],'o-',color='#9d3a3a');ax.set_xlabel('K=4 recursion depth');ax.set_ylabel('Mean recursive − weighted nDCG@10');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'narrativeqa_recursive_gap_depth.pdf');fig.savefig(OUT/'narrativeqa_recursive_gap_depth.png',dpi=220);plt.close(fig)
 cls=json.loads((ROOT/'results/v3_cls/downstream/DOWNSTREAM_RESULTS.json').read_text()); methods=list(cls['ecthr_a']); clsrows=[]
 for m in methods: clsrows.append({'method':m,'ecthr_micro_f1':cls['ecthr_a'][m]['test']['micro_f1'],'ecthr_macro_f1':cls['ecthr_a'][m]['test']['macro_f1'],'newsgroups_nmi':cls['20_newsgroups'][m]['nmi'],'newsgroups_ari':cls['20_newsgroups'][m]['ari']})
 with (SRC/'downstream_classification_clustering.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=clsrows[0]);w.writeheader();w.writerows(clsrows)
 fig,axes=plt.subplots(1,2,figsize=(10,4)); x=np.arange(len(methods)); labels=[m.replace('recursive_seed','R').replace('token_weighted_mean','Weighted').replace('unweighted_mean','Unweighted') for m in methods]; axes[0].bar(x,[r['ecthr_micro_f1'] for r in clsrows],color='#315a9e');axes[0].set_title('ECtHR-A multilabel');axes[0].set_ylabel('Micro-F1');axes[0].set_ylim(.60,.65);axes[1].bar(x,[r['newsgroups_nmi'] for r in clsrows],color='#77964a');axes[1].set_title('20 Newsgroups clustering');axes[1].set_ylabel('NMI');axes[1].set_ylim(.57,.63)
 for ax in axes: ax.set_xticks(x,labels,rotation=20,ha='right');ax.grid(axis='y',alpha=.25)
 fig.tight_layout();fig.savefig(OUT/'classification_clustering.pdf');fig.savefig(OUT/'classification_clustering.png',dpi=220);plt.close(fig)
if __name__=='__main__':main()
