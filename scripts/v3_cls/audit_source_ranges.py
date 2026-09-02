#!/usr/bin/env python
"""Programmatically audit protected source-index ranges across V1/V2/V2.2."""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ['origin/main', 'origin/v2-paper-cls']
RANGE_RE = re.compile(r'(?P<key>(?:train|validation|development|scan|official|source|buffer)[_a-z]*):\s*(?P<value>\d+)')

MANUAL_PROTECTED = {
  'official_v5': [110000, 140000],
  'v6_train': [140000, 152000],
  'v6_development': [152000, 161000],
  'v6_buffer': [161000, 162000],
  'v6_official': [162000, 192000],
  'v2_1_train': [192000, 204000],
  'v2_1_development': [204000, 213000],
  'v2_1_buffer': [213000, 214000],
  'v2_1_official': [214000, 244000],
  'v2_2_train': [244000, 256000],
  'v2_2_development': [256000, 265000],
  'v2_2_buffer': [265000, 266000],
  'v2_2_official': [266000, 296000]
}
V3 = {
  'train': [296000, 308000],
  'development': [308000, 317000],
  'buffer_unused': [317000, 318000],
  'future_official_one_shot': [318000, 348000]
}

def git_files(branch):
    out=subprocess.check_output(['git','ls-tree','-r','--name-only',branch], cwd=ROOT, text=True)
    return [x for x in out.splitlines() if x.endswith(('.yaml','.yml','.json','.md','.py'))]

def git_show(branch, path):
    return subprocess.check_output(['git','show',f'{branch}:{path}'], cwd=ROOT, text=True, errors='replace')

def overlaps(a,b): return max(a[0],b[0]) < min(a[1],b[1])

def main():
    observations=[]
    for br in BRANCHES:
        try: files=git_files(br)
        except Exception: continue
        for p in files:
            try: txt=git_show(br,p)
            except Exception: continue
            for i,line in enumerate(txt.splitlines(),1):
                if any(tok in line for tok in ('start','scan','range','documents')) and re.search(r'\d{4,}', line):
                    observations.append({'branch':br,'path':p,'line':i,'text':line[:500]})
    highest=max(v[1] for v in MANUAL_PROTECTED.values())
    collisions=[]
    for name,r in MANUAL_PROTECTED.items():
        for vname,vr in V3.items():
            if overlaps(r,vr): collisions.append([name,r,vname,vr])
    result={'observations':observations,'protected_ranges':MANUAL_PROTECTED,'protected_highest_exclusive':highest,'v3_allocation':V3,'v3_after_highest': all(r[0] >= highest for r in V3.values()),'collisions':collisions,'non_overlap': not collisions}
    out=ROOT/'reproducibility/v3_cls/RANGE_AUDIT.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('protected_highest_exclusive','v3_allocation','non_overlap')}, indent=2))
    if collisions: raise SystemExit(1)
if __name__=='__main__': main()
