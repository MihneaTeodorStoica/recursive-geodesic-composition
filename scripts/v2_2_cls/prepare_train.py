#!/usr/bin/env python3
"""Atomically prepare the create-only V2.2 CLS TRAIN cache."""
from __future__ import annotations
import copy, json, os, shutil, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from recursive_geodesic_composition.data import FrozenTeacher, stream_document_ranges
from recursive_geodesic_composition.prepare_kary import prepare_kary_split, require_free_storage
from scripts.v2_2_cls.protocol import V2_2, audit

REQ = {'namespace': 'v2_2_cls', 'teacher_id': 'Alibaba-NLP/gte-modernbert-base',
       'teacher_revision': 'e7f32e3c00f91d699e8c43b53106206bcc72bb22',
       'teacher_pooling': 'cls', 'chunk_tokens': 1024}

def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

def validate_manifest(manifest):
    audit()
    if manifest.get('namespace', manifest.get('cache_namespace')) != REQ['namespace']:
        raise ValueError('reject non-V2.2 cache namespace')
    teacher = manifest.get('teacher', manifest)
    for key in ('teacher_id', 'teacher_revision', 'teacher_pooling'):
        if teacher.get(key) != REQ[key]: raise ValueError('reject non-V2.2 CLS provenance')
    if tuple(manifest.get('source_scan', manifest.get('range', ()))) != V2_2['train']:
        raise ValueError('wrong V2.2 TRAIN range')
    if manifest.get('role', 'train') != 'train': raise ValueError('not a TRAIN manifest')
    return True

def _fsync_tree(root):
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle: os.fsync(handle.fileno())

def prepare(config_path: Path, device='cuda') -> Path:
    import yaml
    audit()
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg.get('cache_dir') != 'artifacts/v2_2_cls/cache_train_cls': raise ValueError('unexpected V2.2 cache path')
    if tuple(cfg.get('source_scan', ())) != V2_2['train']: raise ValueError('config does not bind V2.2 TRAIN')
    if cfg.get('teacher_pooling') != 'cls': raise ValueError('V2.2 requires CLS pooling')
    destination = ROOT / cfg['cache_dir']; marker = destination.with_name(destination.name + '.PREPARATION_STARTED')
    if destination.exists() or marker.exists(): raise FileExistsError('V2.2 TRAIN namespace is create-only')
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as out:
        out.write(f'config_sha256={sha256_file(config_path)}\nauthorized_scan={V2_2["train"]}\n')
        out.flush(); os.fsync(out.fileno())
    require_free_storage(ROOT, int(cfg.get('minimum_free_bytes_before_prepare', 1800000000)), 'V2.2 TRAIN preparation')
    stage = destination.with_name(f'.{destination.name}.staging-{uuid.uuid4().hex}'); stage.mkdir(parents=True)
    try:
        teacher = FrozenTeacher(cfg['model_name'], cfg['model_revision'], device, pooling='cls', trust_remote_code=True)
        audit_rows = {}
        for corpus in cfg['corpora']:
            documents = stream_document_ranges(corpus['dataset_name'], corpus.get('dataset_config'), corpus['dataset_revision'],
                                               {'train': (V2_2['train'][0], V2_2['train'][1] - V2_2['train'][0])})['train']
            stage_cfg = copy.deepcopy(cfg); stage_cfg['cache_dir'] = str(stage)
            prepare_kary_split(stage_cfg, teacher, corpus, 'train', documents)
            sidecar = stage / corpus['id'] / 'train.pt.json'
            data = json.loads(sidecar.read_text())
            if data.get('teacher_pooling') != 'cls' or (data.get('source_start'), data.get('requested')) != (V2_2['train'][0], V2_2['train'][1] - V2_2['train'][0]):
                raise ValueError('TRAIN sidecar escaped V2.2 authorization')
            audit_rows[corpus['id']] = {'source_rows_scanned': V2_2['train'][1] - V2_2['train'][0], 'usable_augmented_rows': data['usable'], 'sidecar_sha256': sha256_file(sidecar)}
        artifacts = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in sorted(stage.glob('*/*')) if p.is_file()}
        manifest = {'namespace': REQ['namespace'], 'cache_namespace': REQ['namespace'], 'role': 'train', 'teacher': teacher.provenance(), 'source_scan': list(V2_2['train']), 'config': str(Path(config_path).relative_to(ROOT)), 'config_sha256': sha256_file(config_path), 'preparer_sha256': sha256_file(Path(__file__)), 'corpus_audit': audit_rows, 'artifacts': artifacts}
        validate_manifest(manifest)
        (stage / 'CACHE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (stage / 'COMPLETE').write_text(f'{sha256_file(stage / "CACHE_MANIFEST.json")}  CACHE_MANIFEST.json\n')
        _fsync_tree(stage); os.replace(stage, destination); return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise

def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default=str(ROOT / 'configs/v2_2_cls/pilot.yaml')); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); print(prepare(Path(args.config).resolve(), args.device))
if __name__ == '__main__': main()
