from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/official_v6"
RECOVERY = STAGE / "evaluate_official_test_v6_recovery.py"


def test_recovery_is_minimal_and_uses_v6_development_provenance():
    original = (STAGE / "evaluate_official_test_v6_initial.py").read_text()
    recovery = RECOVERY.read_text()
    assert 'cfg["confirmation"]' in original
    assert 'cfg["confirmation"]' not in recovery
    assert 'cfg["development_provenance"]' in recovery
    assert 'immutable.get("development_final_results_sha256")' in recovery
    assert 'immutable.get("post_final_verification_sha256")' in recovery
    assert 'immutable.get("implementation_parent_commit")' in recovery
    assert "protected_data_accessed_before_freeze" in recovery


def test_recovery_has_distinct_one_shot_claim_and_discloses_failure():
    recovery = RECOVERY.read_text()
    launcher = (
        ROOT / "reproducibility/protocols/official_v6/run_evaluation_recovery.sh"
    ).read_text()
    assert ".EVALUATION_RECOVERY_STARTED" in recovery
    assert "recovery requires the spent initial evaluation marker" in recovery
    assert '"initial_failure": "KeyError: \'confirmation\'"' in recovery
    assert "prepare_official_test_v6" not in launcher
    assert "RECOVERY_SHA256SUMS" in launcher


def test_recovery_does_not_change_the_sole_target_or_scan():
    recovery = RECOVERY.read_text()
    assert 'reporting_gate(seed_results, primary["mean_auc"], cfg)' in recovery
    assert "load_cache(cfg)" in recovery
    config = (ROOT / "configs/official_v6.yaml").read_text()
    assert "sole_auc_target: 0.60" in config
    assert "scan_start: 162000" in config
    assert "scan_end_exclusive: 192000" in config
