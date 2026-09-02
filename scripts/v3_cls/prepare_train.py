#!/usr/bin/env python
"""Create-only V3 CLS TRAIN cache from frozen source ranges."""
from __future__ import annotations
import copy, json, os, shutil, sys, uuid, hashlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recursive_geodesic_composition.data import FrozenTeacher, stream_document_ranges  # noqa:E402
from recursive_geodesic_composition.prepare_kary import prepare_kary_split, require_free_storage  # noqa:E402

TRAIN_RANGE = (296000, 308000)
DEV_RANGE = (308000, 317000)
BUFFER_RANGE = (317000, 318000)
OFFICIAL_RANGE = (318000, 348000)
TEACHER_ID = "Alibaba-NLP/gte-modernbert-base"
TEACHER_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"

def sha256_file(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def reject(path: Path):
    s=str(path)
    if 'v3_cls' not in s:
        raise ValueError(f"V3 path must be namespaced under v3_cls: {path}")
    for token in ('v1', 'v2_cls', 'v2_2_cls', 'kary4_strict_autoencoder_v6'):
        if token in s:
            raise ValueError(f"V3 refuses historical namespace token {token!r}: {path}")

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if not cfg.get('v3_cls_experiment'):
        raise ValueError('config must declare v3_cls_experiment: true')
    if cfg.get('teacher_pooling') != 'cls':
        raise ValueError('V3 requires teacher_pooling: cls')
    if cfg.get('model_name') != TEACHER_ID or cfg.get('model_revision') != TEACHER_REVISION:
        raise ValueError('V3 requires pinned GTE ModernBERT teacher identity')
    dest=ROOT / cfg['cache_dir']; reject(dest)
    marker=dest.with_name(dest.name+'.PREPARATION_STARTED')
    if dest.exists() or marker.exists():
        raise FileExistsError(f'V3 cache is create-only and already started/exists: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'range': TRAIN_RANGE, 'config': args.config, 'config_sha256': sha256_file(args.config)})+'\n')
    require_free_storage(dest.parent, int(cfg.get('minimum_free_bytes_before_prepare', 0) or 0), 'V3 TRAIN CLS prepare')
    stage=dest.with_name('.'+dest.name+'.staging-'+uuid.uuid4().hex); stage.mkdir()
    try:
        teacher=FrozenTeacher(cfg['model_name'], cfg['model_revision'], args.device, pooling='cls', trust_remote_code=cfg.get('teacher_trust_remote_code', True))
        stage_cfg=copy.deepcopy(cfg); stage_cfg['cache_dir']=str(stage)
        if args.smoke:
            count=min(4, TRAIN_RANGE[1]-TRAIN_RANGE[0])
        else:
            count=TRAIN_RANGE[1]-TRAIN_RANGE[0]
        audits={}
        for corpus in cfg['corpora']:
            if int(corpus['train_start']) != TRAIN_RANGE[0] or int(corpus['train_documents']) > count:
                raise ValueError('corpus train range must match frozen V3 allocation')
            docs=stream_document_ranges(corpus['dataset_name'], corpus.get('dataset_config'), corpus['dataset_revision'], {'train': (TRAIN_RANGE[0], count)})['train']
            local=dict(corpus); local['train_start']=TRAIN_RANGE[0]; local['train_documents']=count
            prepare_kary_split(stage_cfg, teacher, local, 'train', docs)
            audits[corpus['id']]={'source_range':[TRAIN_RANGE[0], TRAIN_RANGE[0]+count], 'usable_sidecar': json.loads((stage/corpus['id']/ 'train.pt.json').read_text()).get('usable')}
        artifacts={}
        for p in sorted(stage.glob('*/*')):
            if p.is_file(): artifacts[str(p.relative_to(stage))]={'bytes':p.stat().st_size,'sha256':sha256_file(p)}
        manifest={'study':'v3_cls','role':'train','source_range':list(TRAIN_RANGE),'development_range':list(DEV_RANGE),'buffer_range':list(BUFFER_RANGE),'future_official_range':list(OFFICIAL_RANGE),'teacher_provenance':teacher.provenance(),'teacher_pooling':'cls','config':args.config,'config_sha256':sha256_file(args.config),'preparer_sha256':sha256_file(__file__),'corpus_audit':audits,'artifacts':artifacts}
        (stage/'CACHE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.replace(stage, dest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

if __name__ == '__main__':
    main()
