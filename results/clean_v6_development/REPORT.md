# CLEAN DEVELOPMENT ONLY v6 retraining and validation

**Promotion: PASS**

No reserve-buffer or protected-data access occurred. Checkpoints were fixed, probes were fitted, and retention floors were recorded before the development cache existed.

## Frozen gate

- Three-seed mean strongest AUC: `0.589482667` (target: `<0.60`)
- Every-seed AUC guard passed: `True`
- Pooled cosine guard passed: `True`
- Built-in decoder guard passed: `True`
- TRAIN-fitted matched-probe retention passed: `True`
- Every encoder and decoder parameter tensor changed: `True`
- Finite/noncollapsed/no-saturation: `True`

| seed | strongest AUC | teacher cosine | built-in child cosine | retention |
|---:|---:|---:|---:|:---:|
| 424242 | 0.587464250 | 0.984856367 | 0.912145495 | True |
| 271828 | 0.589812750 | 0.984854102 | 0.912112474 | True |
| 314159 | 0.591171000 | 0.984867930 | 0.912180841 | True |

The unrestricted full-document diagnostic used every available token from the same selected documents, with no teacher comparison or AUC.
