#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'figures/v4_euclidean/downstream_suite';PUB=ROOT/'results/v4_euclidean/publication_sources';OUT.mkdir(parents=True,exist_ok=True);PUB.mkdir(exist_ok=True)
x=json.loads((ROOT/'results/v4_euclidean/downstream/CONSOLIDATED_SUITE_RESULTS.json').read_text());tasks=['nfcorpus','narrativeqa','qmsum','2wikimqa','summ_screen_fd'];names={'nfcorpus':'NFCorpus','narrativeqa':'NarrativeQA','qmsum':'QMSum','2wikimqa':'2WikiMQA','summ_screen_fd':'SummScreenFD'};methods=['recursive_three_seed_mean','token_weighted_leaf_mean','unweighted_leaf_mean','direct_teacher'];mlabel=['V4 recursive mean','Token-weighted mean','Unweighted mean','Direct CLS'];colors=['#f58518','#4c78a8','#54a24b','#b279a2']
def save(fig,name):
 fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
def writecsv(name,rows):
 keys=sorted(set().union(*(r.keys() for r in rows)));f=(PUB/name).open('w',newline='');w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows);f.close()
# overall real retrieval
rows=[]
for t in tasks:
 for m in methods:
  if m in x['real_world'][t]['methods']:rows.append({'task':t,'method':m,'nDCG@10':x['real_world'][t]['methods'][m]['nDCG@10']})
writecsv('figure_retrieval_by_method.csv',rows);fig,ax=plt.subplots(figsize=(10,4.8));width=.18;xx=np.arange(len(tasks))
for j,(m,l,c) in enumerate(zip(methods,mlabel,colors)):ax.bar(xx+(j-1.5)*width,[next((r['nDCG@10'] for r in rows if r['task']==t and r['method']==m),np.nan) for t in tasks],width,label=l,color=c)
ax.set_xticks(xx,[names[t] for t in tasks],rotation=15);ax.set_ylabel('nDCG@10');ax.set_title('Frozen V4 downstream retrieval');ax.legend(ncol=2);save(fig,'retrieval_by_representation')
# scales
for axis in ('tokens','leaves','depth'):
 rows=[];fig,axs=plt.subplots(2,2,figsize=(10,7),sharey=False); 
 for ax,t in zip(axs.flat,tasks[1:]):
  bins=x['real_world'][t]['stratified'][axis];ordered=sorted(bins,key=lambda z:int(z.split('-')[0]))
  for m,l,c in zip(methods[:3],mlabel[:3],colors[:3]):
   vals=[]
   for b in ordered:
    if m=='recursive_three_seed_mean':v=np.mean([bins[b]['methods'][f'recursive_seed{s}']['nDCG@10'] for s in ('424242','271828','314159')])
    else:v=bins[b]['methods'][m]['nDCG@10']
    vals.append(v);rows.append({'task':t,'axis':axis,'bin':b,'queries':bins[b]['queries'],'method':m,'nDCG@10':v})
   ax.plot(range(len(ordered)),vals,marker='o',label=l,color=c)
  ax.set_xticks(range(len(ordered)),ordered,rotation=35,ha='right',fontsize=8);ax.set_title(names[t]);ax.set_ylabel('nDCG@10')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3);fig.suptitle(f'Retrieval versus {axis} (predeclared strata)',y=1.02);writecsv(f'figure_retrieval_vs_{axis}.csv',rows);save(fig,f'retrieval_vs_{axis}')
# scale gaps by axis
for axis in ('tokens','leaves','depth'):
 rows=[];fig,axs=plt.subplots(2,2,figsize=(10,7),sharey=False)
 for ax,t in zip(axs.flat,tasks[1:]):
  for baseline,label,c in [('token_weighted_leaf_mean','V4 − token-weighted','#e45756'),('unweighted_leaf_mean','V4 − unweighted','#72b7b2')]:
   bins=x['real_world'][t]['gaps'][baseline]['stratified'][axis];ordered=sorted(bins,key=lambda z:int(z.split('-')[0]));vals=[bins[b]['metrics']['nDCG@10']['delta'] for b in ordered];ax.plot(range(len(ordered)),vals,marker='o',label=label,color=c)
   for b,v in zip(ordered,vals):rows.append({'task':t,'axis':axis,'bin':b,'queries':bins[b]['queries'],'baseline':baseline,'nDCG@10_delta':v})
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(ordered)),ordered,rotation=35,ha='right',fontsize=8);ax.set_title(names[t]);ax.set_ylabel('nDCG@10 gap')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2);fig.suptitle(f'V4-minus-pooling gap versus {axis}',y=1.02);writecsv(f'figure_gap_vs_{axis}.csv',rows);save(fig,f'gap_vs_{axis}')
# synthetic conditions
rows=[];fig,axs=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
for ax,t in zip(axs,['needle','passkey']):
 ctx=x['synthetic'][t]['contexts'];cs=sorted(map(int,ctx))
 for m,l,c in zip(methods,mlabel,colors):
  vals=[];xs=[]
  for z in cs:
   if m in ctx[str(z)]:v=ctx[str(z)][m]['metrics']['nDCG@1'];vals.append(v);xs.append(z);rows.append({'task':t,'context_length':z,'method':m,'nDCG@1':v})
  if xs:ax.plot(xs,vals,marker='o',label=l,color=c)
 ax.set_xscale('log',base=2);ax.set_xticks(cs,[str(z) for z in cs],rotation=35);ax.set_title(t.title());ax.set_xlabel('Official context length');ax.set_ylabel('nDCG@1')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2);fig.suptitle('LongEmbed synthetic retrieval versus context length',y=1.08);writecsv('figure_synthetic_context.csv',rows);save(fig,'synthetic_vs_context_length')
# nonretrieval
n=x['nonretrieval'];rows=[]
for m,z in n['ecthr_a'].items():
 for metric in ('macro_f1','micro_f1'):rows.append({'method':m,'metric':metric,'value':z['test'][metric]})
writecsv('figure_ecthr_a.csv',rows);fig,ax=plt.subplots(figsize=(9,4.5));ms=list(n['ecthr_a']);xx=np.arange(len(ms));ax.bar(xx-.18,[n['ecthr_a'][m]['test']['macro_f1'] for m in ms],.36,label='Macro-F1');ax.bar(xx+.18,[n['ecthr_a'][m]['test']['micro_f1'] for m in ms],.36,label='Micro-F1');ax.set_xticks(xx,[m.replace('recursive_','rec\n') for m in ms],rotation=20);ax.set_ylim(.45,.68);ax.set_title('ECtHR-A frozen-representation OVR logistic');ax.legend();save(fig,'ecthr_a_comparison')
rows=[]
for m,z in n['20_newsgroups'].items():
 for metric in ('nmi','ari'):rows.append({'method':m,'metric':metric.upper(),'value':z[metric]})
writecsv('figure_20newsgroups.csv',rows);fig,ax=plt.subplots(figsize=(9,4.5));ms=list(n['20_newsgroups']);xx=np.arange(len(ms));ax.bar(xx-.18,[n['20_newsgroups'][m]['nmi'] for m in ms],.36,label='NMI');ax.bar(xx+.18,[n['20_newsgroups'][m]['ari'] for m in ms],.36,label='ARI');ax.set_xticks(xx,[m.replace('recursive_','rec\n') for m in ms],rotation=20);ax.set_ylim(.4,.65);ax.set_title('20 Newsgroups fixed K-means');ax.legend();save(fig,'20newsgroups_clustering')
