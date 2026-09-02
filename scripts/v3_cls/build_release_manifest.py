#!/usr/bin/env python3
"""Prepare (never publish) the proposed future v3-paper release manifest."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def entry(p):
 p=Path(p); return {'path':str(p),'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size}
def main():
 freeze=json.loads((ROOT/'reproducibility/v3_cls/OFFICIAL_PREACCESS_FREEZE.json').read_text()); groups={}
 groups['checkpoints']=[entry(x['checkpoint']) for x in freeze['seeds'].values()]; groups['probes']=[entry(x['probe']) for x in freeze['seeds'].values()]
 groups['configs']=[entry(p) for p in sorted(Path('configs/v3_cls').glob('adversarial_v2_*seed*.yaml'))]+[entry('configs/v3_cls/evaluator_two_arm.yaml')]
 groups['protocols']=[entry(p) for p in ['reproducibility/v3_cls/TRAINING_PLAN.md','reproducibility/v3_cls/TRAIN_PROTOCOL_AMENDMENT_V1.md','reproducibility/v3_cls/TRAIN_PROTOCOL_AMENDMENT_V2.md','reproducibility/v3_cls/AMENDMENT_V2_SELECTION_DECISION.md','reproducibility/v3_cls/EVALUATION_PROTOCOL.md','reproducibility/v3_cls/OFFICIAL_PREACCESS_FREEZE.json']]
 result_paths=list(Path('results/v3_cls').glob('official_seed*/TWO_ARM_EVALUATION.json'))+[Path('results/v3_cls/FINAL_METRICS_SUMMARY.json')]
 downstream=Path('results/v3_cls/downstream');
 if downstream.exists(): result_paths += [p for p in downstream.rglob('*') if p.is_file() and p.suffix in ('.json','.csv') and 'source/' not in str(p)]
 groups['machine_results']=[entry(p) for p in sorted(set(result_paths)) if (ROOT/p).is_file()]
 groups['figures']=[entry(p) for p in sorted(Path('figures/v3_cls').glob('*')) if p.is_file()]
 manifest={'proposed_release':'v3-paper','published':False,'branch':'v3-paper-cls','base_commit':'c6b3f28830e7965533e1d036c2a749836b7848b6','teacher':'pinned GTE ModernBERT normalized CLS','groups':groups}; out=ROOT/'reproducibility/v3_cls/PROPOSED_V3_PAPER_RELEASE_MANIFEST.json'; out.write_text(json.dumps(manifest,indent=2)+'\n'); print(out)
if __name__=='__main__':main()
