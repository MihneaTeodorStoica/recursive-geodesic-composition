"""Corpus/cache-free immutable verification for the strict-K4 AE v6 launch."""

from __future__ import annotations

import json
import sys

from official_test_v6_protocol import (
    CONFIG,
    ROOT,
    load_config,
    validate_immutable_provenance,
)


def main() -> None:
    cfg = load_config(CONFIG)
    cache = ROOT / cfg["cache_dir"]
    output = ROOT / cfg["output_dir"]
    forbidden = (
        cache,
        cache.with_name(cache.name + ".PREPARATION_STARTED"),
        output,
        output.with_name(output.name + ".EVALUATION_STARTED"),
    )
    if any(path.exists() for path in forbidden):
        raise FileExistsError("official-v6 namespace was already claimed or populated")
    if "datasets" in sys.modules:
        raise RuntimeError("corpus loader was imported before preaccess verification")
    proof = validate_immutable_provenance(cfg, check_checkpoint_tensors=True)
    if "datasets" in sys.modules:
        raise RuntimeError("immutable verification imported a corpus loader")
    print(
        json.dumps(
            {
                "preaccess_verified": True,
                "official_data_or_cache_opened": False,
                "implementation_parent_commit": proof["implementation_parent_commit"],
                "checkpoint_seeds": list(proof["checkpoints"]),
                "checkpoint_sha256": {
                    str(seed): value["checkpoint_sha256"]
                    for seed, value in proof["checkpoints"].items()
                },
                "strict_decoder_state_tensors": {
                    str(seed): value["model"]["decoder_state_tensors"]
                    for seed, value in proof["checkpoints"].items()
                },
                "genuine_encoder_decoder_change": {
                    str(seed): {
                        "encoder": value["training"]["encoder_change"]["nonzero"],
                        "decoder": value["training"]["decoder_change"]["nonzero"],
                    }
                    for seed, value in proof["checkpoints"].items()
                },
                "seed_specific_train_probes": {
                    str(seed): value["train_probes"]["candidate"]
                    for seed, value in proof["checkpoints"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
