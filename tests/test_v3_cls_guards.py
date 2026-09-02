import json
from pathlib import Path

import pytest

from recursive_geodesic_composition.train import _enforce_v3_cls_guards, _assert_v3_cache_sidecar


def base_cfg(tmp_path):
    return {
        'v3_cls_experiment': True,
        'teacher_pooling': 'cls',
        'model_name': 'Alibaba-NLP/gte-modernbert-base',
        'model_revision': 'e7f32e3c00f91d699e8c43b53106206bcc72bb22',
        'cache_dir': str(tmp_path / 'artifacts/v3_cls/cache'),
        'output_dir': str(tmp_path / 'results/v3_cls/out'),
    }


def test_v3_requires_cls(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg['teacher_pooling'] = 'masked_mean'
    with pytest.raises(ValueError, match='teacher_pooling'):
        _enforce_v3_cls_guards(cfg)


def test_v3_forbids_historical_initialization(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg['initialize_from_encoder_checkpoint'] = 'results/v2_cls/final_seed424242/checkpoint.pt'
    with pytest.raises(ValueError, match='forbids'):
        _enforce_v3_cls_guards(cfg)


def test_v3_cache_sidecar_must_be_cls(tmp_path):
    cfg = base_cfg(tmp_path)
    tensor = tmp_path / 'artifacts/v3_cls/cache/fineweb/train.pt'
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b'placeholder')
    Path(str(tensor) + '.json').write_text(json.dumps({
        'teacher_pooling': 'masked_mean',
        'model': cfg['model_name'],
        'model_revision': cfg['model_revision'],
    }))
    with pytest.raises(ValueError, match='non-CLS'):
        _assert_v3_cache_sidecar(cfg, tensor)


def test_v3_cache_sidecar_accepts_cls(tmp_path):
    cfg = base_cfg(tmp_path)
    tensor = tmp_path / 'artifacts/v3_cls/cache/fineweb/train.pt'
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b'placeholder')
    Path(str(tensor) + '.json').write_text(json.dumps({
        'teacher_pooling': 'cls',
        'model': cfg['model_name'],
        'model_revision': cfg['model_revision'],
    }))
    _assert_v3_cache_sidecar(cfg, tensor)
