# Preaccess audit: official strict-K4 AE v6

- Status: **FROZEN, NOT LAUNCHED**.
- Development base: committed clean-v6 PASS `f38f75d2d70a9af92d3bdeae959281173c7998ee`.
- Authorized future scan: `[162000,192000)` only.
- Refused partition: every source index in `[0,162000)`, including official v5, clean-v6 fresh TRAIN, clean-v6 fresh development, and the entire `[161000,162000)` buffer.
- Population: first 1000 safe-prefix-length-eligible documents per pinned corpus, same documents in both arms.
- Inputs checked without opening an official cache or importing a corpus loader: three exact candidate checkpoints, three source checkpoints, six seed-specific TRAIN probe artifacts, three pre-development freezes, clean-development final verification, fixed attack config, and actual strict-decoder/finite tensor state.
- Unique cache, result, staging, and exclusive-start marker namespaces do not exist at freeze time.
- Protocol imports its own v6 preparation/evaluation modules directly and does not import another official protocol.
- Paper paths are excluded from the freeze changes.
- The sole AUC target in this v6 protocol is `<0.60`; no competing numeric AUC target is declared.
- Launch command intentionally not run: `experiments/kary4_strict_autoencoder_official_test_v6/run.sh`.
