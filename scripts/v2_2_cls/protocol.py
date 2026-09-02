"""V2.2 source allocation: validated before any source loader is permitted."""
from pathlib import Path
HISTORICAL={
 'official_v5':(110000,140000),'v6_train':(140000,152000),'v6_development':(152000,161000),'v6_buffer':(161000,162000),'v6_official':(162000,192000),
 'v2_1_train':(192000,204000),'v2_1_development':(204000,213000),'v2_1_buffer':(213000,214000),'v2_1_official':(214000,244000)}
V2_2={'train':(244000,256000),'development':(256000,265000),'buffer':(265000,266000),'official':(266000,296000)}
def overlap(a,b):return max(a[0],b[0])<min(a[1],b[1])
def audit():
 for n,r in V2_2.items():
  for hn,hr in HISTORICAL.items():
   if overlap(r,hr):raise ValueError(f'overlap {n} {r} with {hn} {hr}')
 for n,r in V2_2.items():
  for nn,rr in V2_2.items():
   if n<nn and overlap(r,rr):raise ValueError(f'internal overlap {n}/{nn}')
 return {'historical':HISTORICAL,'v2_2':V2_2,'non_overlap':True}
if __name__=='__main__':
 import json;print(json.dumps(audit(),indent=2))
