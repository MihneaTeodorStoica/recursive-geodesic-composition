# V2 CLS validation log

Commands run before any V2 training/development/official access:

```bash
/venv/main/bin/python scripts/v2_cls/validate_cls_reference.py --device cuda
PYTHONPATH=src /venv/main/bin/python scripts/v2_cls/protocol.py > reproducibility/v2_cls/PROTOCOL_AUDIT.json
PYTHONPATH=src /venv/main/bin/python scripts/v2_cls/smoke_invariants.py > /tmp/v2_smoke_stdout
PYTHONPATH=src /venv/main/bin/python -m compileall -q src scripts/v2_cls
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/test_v2_cls_protocol.py tests/test_geodesic_parameterization.py tests/test_kary_scaling.py
```

Results:

- CLS reference validation: PASS (`reproducibility/v2_cls/CLS_REFERENCE_VALIDATION.json`), max absolute difference `0.0`, cosines all >= `1.0` within float display.
- Protocol/range audit: PASS (`reproducibility/v2_cls/PROTOCOL_AUDIT.json`).
- Smoke invariants: PASS (`reproducibility/v2_cls/SMOKE_INVARIANTS.json`, stdout copy `SMOKE_INVARIANTS.stdout`).
- Compileall: PASS.
- Pytest subset: `20 passed in 5.92s`.

No V2 development or official data was accessed.
