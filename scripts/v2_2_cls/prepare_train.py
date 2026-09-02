"""Create-only V2.2 TRAIN cache; real source loading is deliberately gated."""
import hashlib,json
from pathlib import Path
from scripts.v2_2_cls.protocol import V2_2,audit
REQ={'namespace':'v2_2_cls','teacher_id':'Alibaba-NLP/gte-modernbert-base','teacher_revision':'e7f32e3c00f91d699e8c43b53106206bcc72bb22','teacher_pooling':'cls','chunk_tokens':1024}
def validate_manifest(m):
 audit()
 if any(m.get(k)!=v for k,v in REQ.items()):raise ValueError('reject non-V2.2/mean-pooled/mismatched cache')
 if tuple(m.get('range',()))!=V2_2['train']:raise ValueError('wrong range')
 return True
def synthetic_manifest():return {**REQ,'range':list(V2_2['train']),'tokenizer_class':'mock','model_class':'mock','rows':[{'id':'synthetic:0','corpus':'fineweb','tokens':1024,'leaf_masses':[1024]}]}
def main():
 import argparse;p=argparse.ArgumentParser();p.add_argument('command',choices=('synthetic','prepare'));a=p.parse_args()
 if a.command=='prepare':raise RuntimeError('real TRAIN access requires explicit later authorization')
 print(json.dumps(synthetic_manifest(),indent=2))
if __name__=='__main__':main()
