#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; checks=[]
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def add(name,ok,detail=''):checks.append({'name':name,'pass':bool(ok),'detail':detail})
def load(p):return json.loads((ROOT/p).read_text())
def main():
 add('branch',subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='v4-paper-euclidean')
 plan=(ROOT/'reproducibility/v4_euclidean/PLAN.md').read_text(); add('ranges',all(x in plan for x in ('[348000,360000)','[360000,369000)','[369000,370000)','[370000,400000)')))
 pre=load('reproducibility/v4_euclidean/PREDEVELOPMENT_FREEZE.json'); add('nine frozen',sum(len(x) for x in pre['models'].values())==9 and pre['parameter_count_each']==123616258)
 for g,seeds in pre['models'].items():
  for s,x in seeds.items():
   cp=ROOT/x['checkpoint']; alt=Path('/root/v4_unselected_archive')/x['checkpoint']; actual=cp if cp.exists() else alt
   add(f'checkpoint {g}/{s}',actual.exists() and sha(actual)==x['checkpoint_sha256'],str(actual))
   add(f'probe {g}/{s}',sha(ROOT/x['probe'])==x['probe_sha256'])
 off=load('reproducibility/v4_euclidean/OFFICIAL_PREACCESS_FREEZE.json');add('selected frozen',off['selected_architecture']=='euclidean_residual' and not off['official_accessed'])
 spent=load('artifacts/v4_euclidean/OFFICIAL_ACCESS_SPENT.json');add('official spent',spent['role']=='official')
 man=load('artifacts/v4_euclidean/official_two_arm_cls/CACHE_MANIFEST.json');add('official source/provenance',man['source_range']==[370000,400000] and man['teacher_provenance']['teacher_revision']=='e7f32e3c00f91d699e8c43b53106206bcc72bb22' and man['teacher_provenance']['teacher_pooling']=='cls')
 for s,x in off['seeds'].items():
  p=ROOT/f'results/v4_euclidean/official_euclidean_residual_seed{s}/TWO_ARM_EVALUATION.json';d=json.loads(p.read_text());po=d['safe_direct']['scopes']['pooled'];add(f'official {s}',d['training_performed'] is False and po['health']['finite_fraction']==1 and d['full_document']['telemetry']['geodesic_saturation_count']==0)
 da=load('results/v4_euclidean/downstream/DOWNSTREAM_AUDIT.json');add('downstream audit',all(da[k] for k in ('tensor_identical_leaf_cache','no_silent_exclusions','standard_metrics','all_hashes_match')))
 summary=load('results/v4_euclidean/FINAL_METRICS_SUMMARY.json');add('conclusion bounded',summary['selected_architecture']=='euclidean_residual' and summary['official']['mean']['strongest_auc']>0.69)
 add('tests',subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env={**__import__('os').environ,'PYTHONPATH':'src'},stdout=subprocess.PIPE).returncode==0)
 out={'study':'v4_euclidean','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks};(ROOT/'results/v4_euclidean/FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__':main()
