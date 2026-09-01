from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/v2_cls"))

from protocol import (  # noqa: E402
    TEACHER_POOLING,
    TRAIN_SCAN,
    DEVELOPMENT_SCAN,
    RESERVE_BUFFER,
    FUTURE_OFFICIAL_SCAN,
    assert_source_ranges_nonoverlap,
    audit_configs,
)


def test_v2_ranges_are_after_v1_and_disjoint():
    ranges = assert_source_ranges_nonoverlap()
    assert ranges["official_v6"] == [162000, 192000]
    assert ranges["v2_cls_train"] == list(TRAIN_SCAN)
    assert TRAIN_SCAN[1] == DEVELOPMENT_SCAN[0]
    assert DEVELOPMENT_SCAN[1] == RESERVE_BUFFER[0]
    assert RESERVE_BUFFER[1] == FUTURE_OFFICIAL_SCAN[0]


def test_v2_configs_explicitly_require_cls_and_no_v1_initialization():
    audit = audit_configs()
    assert len(audit["configs"]) == 3
    for name, item in audit["configs"].items():
        text = (ROOT / "configs/v2_cls" / name).read_text()
        assert "teacher_pooling: cls" in text
        assert "initialize_from_encoder_checkpoint" not in text
        assert "initialize_encoder_from_autoencoder_checkpoint: false" in text
        assert TEACHER_POOLING == "cls"
