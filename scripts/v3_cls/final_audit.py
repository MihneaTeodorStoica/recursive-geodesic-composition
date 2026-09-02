#!/usr/bin/env python3
"""Independent fail-closed final V3 provenance/scientific audit."""
import hashlib,json,subprocess,yaml
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def main():
 checks={}; checks['branch']=git('branch','--show-current')=='v3-paper-cls'; checks['base']=git('merge-base','HEAD','origin/main')=='c6b3f28830e7965533e1d036c2a749836b7848b6'; checks['cls_reference']=json.loads((ROOT/'reproducibility/v3_cls/CLS_REFERENCE_VALIDATION.json').read_text())['passed']; alloc=json.loads((ROOT/'reproducibility/v3_cls/SOURCE_ALLOCATION.json').read_text())['v3_cls']; ranges=list(alloc.values()); checks['source_nonoverlap']=all(ranges[i][1]<=ranges[i+1][0] for i in range(len(ranges)-1)) and ranges[0][0]==296000
 freeze=json.loads((ROOT/'reproducibility/v3_cls/OFFICIAL_PREACCESS_FREEZE.json').read_text()); checks['three_seeds']=set(freeze['seeds'])=={'424242','271828','314159'}; normalized=[]; checkpoint_hashes={}; probe_hashes={}
 for seed,s in freeze['seeds'].items():
  checks[f'checkpoint_hash_{seed}']=sha(ROOT/s['checkpoint'])==s['checkpoint_sha256']; checks[f'probe_hash_{seed}']=sha(ROOT/s['probe'])==s['probe_sha256']; checkpoint_hashes[seed]=s['checkpoint_sha256'];probe_hashes[seed]=s['probe_sha256']; ck=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=True); c=ck['config']; checks[f'fresh_cls_{seed}']=c.get('teacher_pooling')=='cls' and c.get('model_revision')=='e7f32e3c00f91d699e8c43b53106206bcc72bb22' and not any(c.get(k) for k in ('initialize_from_encoder_checkpoint','continue_encoder_from_checkpoint','continue_autoencoder_from_checkpoint','encoder_initialization_checkpoint')); normalized.append({k:v for k,v in c.items() if k not in ('seed','study','output_dir')})
 checks['identical_protocol']=all(x==normalized[0] for x in normalized[1:]); checks['freeze_before_official_git_history']=int(git('rev-list','--count','1bf1cd7..744f4f7'))>=1; checks['official_spent_marker']=(ROOT/'artifacts/v3_cls/OFFICIAL_ACCESS_SPENT.json').is_file(); checks['official_results']=all((ROOT/f'results/v3_cls/official_seed{s}/TWO_ARM_EVALUATION_COMPLETE').is_file() for s in freeze['seeds']); checks['teacher_full_arm_not_fabricated']=all(not json.loads((ROOT/f'results/v3_cls/official_seed{s}/TWO_ARM_EVALUATION.json').read_text())['full_document']['teacher_comparison_performed'] for s in freeze['seeds'])
 modified=git('diff','--name-only','origin/main...HEAD').splitlines(); checks['historical_results_unchanged']=not any(p.startswith(('results/clean_v6','results/kary_scaling','reproducibility/v2_','artifacts/v2_')) for p in modified)
 # Downstream audit sidecar is optional until downstream jobs finish; final run requires it.
 downstream=ROOT/'results/v3_cls/downstream/DOWNSTREAM_AUDIT.json'; checks['downstream_audit_present']=downstream.is_file()
 if downstream.is_file():
  d=json.loads(downstream.read_text()); checks['leaf_cache_identity']=bool(d.get('tensor_identical_leaf_cache')); checks['no_silent_exclusions']=bool(d.get('no_silent_exclusions')); checks['standard_metrics']=bool(d.get('standard_metrics'))
 result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'checkpoint_hashes':checkpoint_hashes,'probe_hashes':probe_hashes,'head':git('rev-parse','HEAD'),'modified_paths':modified}; out=ROOT/'reproducibility/v3_cls/FINAL_AUDIT.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__':main()
