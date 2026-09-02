import json
from pathlib import Path
import yaml

from scripts.v2_2_cls import pilot
from scripts.v2_2_cls.prepare_train import validate_manifest


def _manifest():
    return {'namespace': 'v2_2_cls', 'role': 'train', 'source_scan': [244000, 256000],
            'teacher': {'teacher_id': 'Alibaba-NLP/gte-modernbert-base',
                        'teacher_revision': 'e7f32e3c00f91d699e8c43b53106206bcc72bb22',
                        'teacher_pooling': 'cls'}}


def test_manifest_rejects_non_train_or_non_cls():
    assert validate_manifest(_manifest())
    bad = _manifest(); bad['teacher']['teacher_pooling'] = 'masked_mean'
    try:
        validate_manifest(bad)
        assert False, 'masked mean cache must not be accepted'
    except ValueError:
        pass


def test_four_arm_runner_uses_only_frozen_objective_deltas(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'; cache.mkdir()
    (cache / 'CACHE_MANIFEST.json').write_text(json.dumps(_manifest()))
    (cache / 'COMPLETE').write_text('complete\n')
    cfg = {'source_scan': [244000, 256000], 'cache_dir': str(cache), 'output_dir': str(tmp_path / 'out'),
           'seed': 424242, 'steps': 250, 'batch_size': 8}
    config = tmp_path / 'pilot.yaml'; config.write_text(yaml.safe_dump(cfg))
    calls = []
    def fake_run(arm_cfg, variant, seed, device):
        calls.append((arm_cfg, variant, seed, device))
        output = Path(arm_cfg['output_dir']); output.mkdir(parents=True)
        (output / f'encoder_seed{seed}_run_status.json').write_text(json.dumps({'status': 'completed'}))
        (output / f'encoder_seed{seed}.jsonl').write_text(json.dumps({'teacher': .99, 'covariance': .1}) + '\n')
    monkeypatch.setattr(pilot, 'run', fake_run)
    result = pilot.run_pilot(config, 'cpu')
    assert tuple(result) == pilot.ARMS
    assert [entry[0]['covariance_mode'] for entry in calls] == ['none', 'frobenius', 'whitened', 'balanced_whitened']
    assert [entry[0]['seed'] for entry in calls] == [424242] * 4
    assert all(entry[1] == 'encoder' for entry in calls)
