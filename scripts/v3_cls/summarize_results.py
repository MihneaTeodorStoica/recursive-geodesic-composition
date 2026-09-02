#!/usr/bin/env python3
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; seeds=(424242,271828,314159); corpora=('fineweb_edu','fineweb','openwebtext','wikipedia_en_20231101')
def extract(prefix):
 out={}
 for s in seeds:
  d=json.loads((ROOT/f'results/v3_cls/{prefix}{s}/TWO_ARM_EVALUATION.json').read_text()); p=d['safe_direct']['scopes']['pooled']; out[str(s)]={'teacher_cosine':p['teacher_cosine'],'teacher_cosine_ci95':p['teacher_cosine_ci95_bootstrap'],'logistic_auc':p['attacks']['attacks']['standard_scaled_logistic']['orientation_safe_auc'],'logistic_auc_ci95':p['attacks']['attacks']['standard_scaled_logistic']['orientation_safe_auc_ci95_group_bootstrap'],'rbf_auc':p['attacks']['attacks']['standard_scaled_rbf_svm']['orientation_safe_auc'],'rbf_auc_ci95':p['attacks']['attacks']['standard_scaled_rbf_svm']['orientation_safe_auc_ci95_group_bootstrap'],'per_corpus':{c:{'teacher_cosine':d['safe_direct']['scopes'][c]['teacher_cosine'],'strongest_auc':d['safe_direct']['scopes'][c]['attacks']['strongest_orientation_safe_auc']} for c in corpora},'decoder_child_cosine':d['safe_direct']['retention']['built_in_parent_only_decoder']['pooled_ordered_child_cosine'],'probe_child_cosine':d['safe_direct']['retention']['train_fitted_parent_only_probe']['pooled_ordered_child_cosine'],'probe_floors_pass':d['safe_direct']['retention']['train_fitted_parent_only_probe']['all_floors_pass'],'safe_effective_rank':p['health']['effective_rank'],'full_telemetry':d['full_document']['telemetry']}
 summary={}
 for metric in ('teacher_cosine','logistic_auc','rbf_auc','decoder_child_cosine','probe_child_cosine','safe_effective_rank'):
  vals=[out[str(s)][metric] for s in seeds]; summary[metric]=statistics.mean(vals); summary[metric+'_sd']=statistics.pstdev(vals)
 out['three_seed_mean']=summary
 return out
def main():
 result={'development_round2_final':extract('final_development_round2_seed'),'official':extract('official_seed'),'official_cache_audit':json.loads((ROOT/'artifacts/v3_cls/official_two_arm_cls/CACHE_MANIFEST.json').read_text())['audit']}
 p=ROOT/'results/v3_cls/FINAL_METRICS_SUMMARY.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(p)
if __name__=='__main__':main()
