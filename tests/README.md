# cifsQuant Smoke Tests

Pytest-based tests that verify the pipeline's core logic without requiring real data files.

## Setup

```bash
conda activate cifsquant
cd /path/to/cifsQuant
```

## Run all tests

```bash
pytest tests/ -v
```

## Run specific test files

```bash
pytest tests/smoke_test_config.py -v       # Config validation
pytest tests/smoke_test_gating.py -v       # Stage 2 gating logic
pytest tests/smoke_test_spatial.py -v      # Stage 3 spatial analysis
```

## What each file tests

| File | Tests |
|---|---|
| `smoke_test_config.py` | project.yaml validation: YAML parsing, required keys, gate/panel consistency, phenotype marker references, `load_config()` dual-format detection, `validate_project()` |
| `smoke_test_gating.py` | `normalize_column()`, `apply_gate()`, `load_project_config()` dict overrides |
| `smoke_test_spatial.py` | `PhenotypeBuilder`, `MetadataManager`, `PopulationDynamics`/`DistanceAnalysis` init, `load_config()` both formats |

## Notes

- Tests create synthetic data — no real .h5ad or image files required
- `manual_gating.py` tests assume `normalize_column()` and `apply_gate()` are importable functions; if they are private helpers inside `main()`, tests for those will be skipped
- Analysis class init tests only check instantiation; `.run()` is not called (avoids heavy I/O and full dependency chain)
- SpatialCells is NOT required for these smoke tests — tests only cover code paths that don't invoke SpatialCells directly
