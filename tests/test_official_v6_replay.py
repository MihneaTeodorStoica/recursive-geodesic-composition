from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts", ROOT / "scripts/official_v6"):
    sys.path.insert(0, str(directory))

import evaluate_official_test_v6_recovery as official


def test_replay_namespace_is_create_only_without_historical_markers(tmp_path, monkeypatch):
    historical = tmp_path / "results_kary4_strict_autoencoder_official_test_v6"
    historical.mkdir()
    retained = historical / "results.json"
    retained.write_text('{"historical": true}\n')
    original = retained.read_bytes()
    monkeypatch.setattr(official, "ROOT", tmp_path)

    output = official._claim_replay_output()
    assert output == tmp_path / "replays/official_v6"
    assert output.is_dir()
    (output / "FAILED.json").write_text('{"status": "failed_not_complete"}\n')
    with pytest.raises(FileExistsError):
        official._claim_replay_output()
    assert retained.read_bytes() == original
    assert not list(tmp_path.glob("*.EVALUATION*"))


def test_replay_refuses_output_namespace_symlinks(tmp_path, monkeypatch):
    outside = tmp_path / "historical"
    outside.mkdir()
    (outside / "results.json").write_text("retained historical evidence\n")
    (tmp_path / "replays").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(official, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="symbolic link"):
        official._claim_replay_output()
    assert (outside / "results.json").read_text() == "retained historical evidence\n"
    assert not (outside / "official_v6").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_revision", "different-teacher-pin"),
        ("bootstrap", {"reps": 1}),
        ("collapse", {"minimum_effective_rank": 0}),
    ],
)
def test_replay_rejects_changed_scientific_config_before_cache_contract(key, value):
    cfg = yaml.safe_load(official.CONFIG.read_text())
    changed = copy.deepcopy(cfg)
    changed[key] = value
    with pytest.raises(ValueError, match="pinned public config"):
        official._validate_public_cache_contract(changed, {}, {}, {})
