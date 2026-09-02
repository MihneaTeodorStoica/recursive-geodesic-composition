#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 files=[]
 for base in ('reproducibility/v4_euclidean','configs/v4_euclidean','scripts/v4_euclidean','figures/v4_euclidean','results/v4_euclidean/publication_sources'):
  for p in sorted((ROOT/base).rglob('*')):
   if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 for rel in ('results/v4_euclidean/FINAL_METRICS_SUMMARY.json','results/v4_euclidean/FINAL_AUDIT.json','results/v4_euclidean/DEVELOPMENT_ARCHITECTURE_COMPARISON.json','results/v4_euclidean/downstream/DOWNSTREAM_AUDIT.json'):
  p=ROOT/rel;files.append({'path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
 official=[]
 for s in (424242,271828,314159):
  for name in ('TWO_ARM_EVALUATION.json','TWO_ARM_EVALUATION_COMPLETE','TWO_ARM_EVALUATION_STARTED'):
   p=ROOT/f'results/v4_euclidean/official_euclidean_residual_seed{s}/{name}';official.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 manifest={'schema':'proposed_v4_paper_release_manifest_v1','release_published':False,'branch':'v4-paper-euclidean','source_commit_before_manifest':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'selected_architecture':'euclidean_residual','files':files+official,'large_checkpoint_provenance':json.loads((ROOT/'reproducibility/v4_euclidean/PREDEVELOPMENT_FREEZE.json').read_text())['models'],'note':'Proposed manifest only; no tag, merge, or release was created.'}
 (ROOT/'PROPOSED_V4_PAPER_RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
