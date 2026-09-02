#!/usr/bin/env python3
"""Run the frozen four-arm V2.2 TRAIN-only covariance pilot."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import yaml, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from recursive_geodesic_composition.train import run
from scripts.v2_2_cls.prepare_train import prepare, validate_manifest

ARMS = ('BASELINE', 'A', 'B', 'C')
ARM_OBJECTIVES = {
    'BASELINE': {'covariance_mode': 'none', 'covariance_weight': 0., 'residual_mean_weight': 0.},
    'A': {'covariance_mode': 'frobenius', 'covariance_weight': 64., 'residual_mean_weight': 0.},
    'B': {'covariance_mode': 'whitened', 'covariance_weight': 64., 'residual_mean_weight': 0.},
    'C': {'covariance_mode': 'balanced_whitened', 'covariance_weight': 64., 'residual_mean_weight': .01},
}

def provenance(manifest):
    return validate_manifest(manifest)

def _manifest(cache_dir):
    path = Path(cache_dir) / 'CACHE_MANIFEST.json'
    if not path.is_file() or not (path.parent / 'COMPLETE').is_file():
        raise FileNotFoundError('V2.2 TRAIN cache is incomplete')
    data = json.loads(path.read_text()); provenance(data); return data

def _last_record(output_dir):
    logs = sorted(Path(output_dir).glob('encoder_seed*.jsonl'))
    if not logs: return {}
    lines = [line for line in logs[-1].read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}

def run_pilot(config_path, device='cuda', arms=ARMS):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if tuple(cfg.get('source_scan', ())) != (244000, 256000): raise ValueError('pilot config is not V2.2 TRAIN-only')
    _manifest(cfg['cache_dir'])
    requested = tuple(arms)
    if not requested or any(arm not in ARMS for arm in requested): raise ValueError('unknown pilot arm')
    results = {}
    for arm in requested:
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.update(ARM_OBJECTIVES[arm])
        arm_cfg['output_dir'] = str(Path(cfg['output_dir']) / arm.lower())
        arm_cfg['v2_2_pilot_arm'] = arm
        # run() resets both Python and Torch RNG. Thus every arm has identical
        # fresh initialization and sampling; only the frozen objective differs.
        if Path(arm_cfg['output_dir']).exists(): raise FileExistsError(f'create-only pilot output exists: {arm_cfg["output_dir"]}')
        run(arm_cfg, 'encoder', int(cfg['seed']), torch.device(device))
        status = json.loads((Path(arm_cfg['output_dir']) / f'encoder_seed{cfg["seed"]}_run_status.json').read_text())
        record = _last_record(arm_cfg['output_dir'])
        results[arm] = {'status': status, 'final': {key: record.get(key) for key in ('teacher', 'covariance', 'residual_mean', 'distribution_variance', 'covariance_condition', 'grad_norm')}}
    out = Path(cfg['output_dir']) / 'PILOT_RESULTS.json'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({'arms': results, 'seed': cfg['seed'], 'train_only': True}, indent=2) + '\n')
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run-pilot'))
    parser.add_argument('--config', default=str(ROOT / 'configs/v2_2_cls/pilot.yaml'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arm', choices=ARMS, action='append')
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(Path(args.config).resolve(), args.device))
    else: print(json.dumps(run_pilot(Path(args.config).resolve(), args.device, tuple(args.arm or ARMS)), indent=2))
if __name__ == '__main__': main()
