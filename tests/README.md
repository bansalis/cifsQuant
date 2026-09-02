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
pytest tests/smoke_test_rawdata.py -v      # Per-channel rawdata mode (orchestrator)
pytest tests/smoke_test_gui.py -v          # GUI components + headless page renders
```

## What each file tests

| File | Tests |
|---|---|
| `smoke_test_config.py` | project.yaml validation: YAML parsing, required keys, gate/panel consistency, phenotype marker references, `load_config()` dual-format detection, stage-aware `validate_project()` |
| `smoke_test_gating.py` | `normalize_data()`/`apply_gates()` on synthetic AnnData, `normalize_column()`/`apply_gate()` helpers, `load_project_config()` dict overrides |
| `smoke_test_spatial.py` | `PhenotypeBuilder`, `MetadataManager`, `PopulationDynamics`/`DistanceAnalysis` init, `load_config()` both formats |
| `smoke_test_rawdata.py` | `rawdata/<sample>/` discovery, channel-name markers csv, derived DAPI index, dry-run per-sample command construction, stacked-mode fallback |
| `smoke_test_gui.py` | Component logic (config I/O, GMM, plots, h5ad discovery); soft channel→file matching incl. missing channels and case-insensitivity; every page rendered headlessly via `streamlit.testing.v1.AppTest` against a provisioned project dir, including regression tests for the audited GUI bugs and the page-4 raw-data matching pre-flight |

## Notes

- Tests create synthetic data — no real .h5ad or image files required
- Analysis class init tests only check instantiation; `.run()` is not called (avoids heavy I/O and full dependency chain)
- SpatialCells is NOT required for these smoke tests — tests only cover code paths that don't invoke SpatialCells directly
- GUI page tests seed `session_state.project_dir` with a temp project so pages render past their prerequisite guards
