#!/usr/bin/env python3
import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(p): return json.loads((ROOT/p).read_text())
def main():
 dev=load('results/v4_euclidean/DEVELOPMENT_ARCHITECTURE_COMPARISON.json')
 official={}; rows=[]
 for seed in (424242,271828,314159):
  d=load(f'results/v4_euclidean/official_euclidean_residual_seed{seed}/TWO_ARM_EVALUATION.json'); p=d['safe_direct']['scopes']['pooled']; r=d['safe_direct']['retention']; f=d['full_document']
  x={'cosine':p['teacher_cosine'],'cosine_ci95':p['teacher_cosine_ci95_bootstrap'],'logistic_auc':p['attacks']['attacks']['standard_scaled_logistic']['orientation_safe_auc'],'logistic_auc_ci95':p['attacks']['attacks']['standard_scaled_logistic']['orientation_safe_auc_ci95_group_bootstrap'],'rbf_auc':p['attacks']['attacks']['standard_scaled_rbf_svm']['orientation_safe_auc'],'rbf_auc_ci95':p['attacks']['attacks']['standard_scaled_rbf_svm']['orientation_safe_auc_ci95_group_bootstrap'],'strongest_auc':p['attacks']['strongest_orientation_safe_auc'],'effective_rank':p['health']['effective_rank'],'decoder_child_cosine':r['built_in_parent_only_decoder']['pooled_ordered_child_cosine'],'probe_child_cosine':r['train_fitted_parent_only_probe']['pooled_ordered_child_cosine'],'probe_floors_pass':r['train_fitted_parent_only_probe']['all_floors_pass'],'geometry':p['architecture_geometry'],'per_corpus':{c:{'cosine':d['safe_direct']['scopes'][c]['teacher_cosine'],'strongest_auc':d['safe_direct']['scopes'][c]['attacks']['strongest_orientation_safe_auc']} for c in ('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')},'full_document':{'rows':f['scopes']['pooled']['rows'],'health':f['scopes']['pooled']['health'],'telemetry':f['telemetry']}}
  official[str(seed)]=x;rows.append(x)
 keys=('cosine','logistic_auc','rbf_auc','strongest_auc','effective_rank','decoder_child_cosine','probe_child_cosine')
 mean={k:statistics.mean(x[k] for x in rows) for k in keys}
 geo=dev['geodesic']['seeds']; eu=dev['euclidean_residual']['seeds']; matched={s:{'cosine_euclidean_minus_geodesic':eu[s]['cosine']-geo[s]['cosine'],'auc_geodesic_minus_euclidean':geo[s]['strongest_auc']-eu[s]['strongest_auc']} for s in ('424242','271828')}
 matched['mean_healthy_geodesic_pairs']={k:statistics.mean(x[k] for x in matched.values()) for k in ('cosine_euclidean_minus_geodesic','auc_geodesic_minus_euclidean')}
 retrieval=load('results/v4_euclidean/downstream/SUMMARY.json'); tasks=load('results/v4_euclidean/downstream/DOWNSTREAM_RESULTS.json')
 out={'study':'v4_euclidean','selected_architecture':'euclidean_residual','development':dev,'matched_development_comparison':matched,'official':{'seeds':official,'mean':mean},'downstream':{'retrieval':retrieval,'classification_clustering':tasks},'v3_reference':load('results/v3_cls/FINAL_METRICS_SUMMARY.json'),'conclusion':'Removing spherical/geodesic operations did not reduce the held-out detectability signature. Euclidean residual preserved fidelity and rank and avoided a geodesic saturation failure, but official AUC remained about 0.695. Free Euclidean failed. Geometry is therefore not supported as the primary bottleneck; stable whole-span CLS prediction from child CLS remains the likely limitation.'}
 (ROOT/'results/v4_euclidean/FINAL_METRICS_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
