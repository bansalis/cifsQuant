# cifsQuant GUI

Optional browser-based interface for the cifsQuant pipeline. The GUI is a wrapper — all core functionality remains available via the CLI and no pipeline files are modified.

## Launch

```bash
conda activate cifsquant
streamlit run gui/app.py
```

Browser opens at `http://localhost:8501`.

## Requirements

Install alongside the existing cifsquant conda environment:

```bash
pip install "streamlit>=1.35" "plotly>=5.20"
```

Or reinstall the full environment (these packages are in `environment.yaml`):

```bash
conda env update -f environment.yaml
```

## Pages

| Page | What it does |
|---|---|
| **1 · Panel Setup** | Map channel names to display names, set marker hierarchy and segmentation parameters |
| **2 · Interactive Gating** | Drag-to-adjust thresholds with live histogram + spatial scatter. Requires `normalized_data.h5ad` (run Stage 2 first or via CLI). |
| **3 · Spatial Config** | Define cell phenotypes, enable/disable analysis modules, edit sample metadata |
| **4 · Run Pipeline** | Validate config, dry-run, and execute one or more stages with live log output |
| **5 · Results Browser** | Browse output plots (PNG gallery) and tables (CSV viewer) |

## CLI alternative (no GUI required)

```bash
# Full pipeline
python run_cifsquant.py --project project.yaml

# Individual stages
python run_cifsquant.py --project project.yaml --stages gating spatial

# Validate without running
python run_cifsquant.py --project project.yaml --dry-run

# Gating only (fast re-gate after threshold changes)
python manual_gating.py --results_dir results --project project.yaml
```

## Notes

- The GUI reads and writes `project.yaml` in whichever directory is set via the sidebar.
- Interactive gating loads `manual_gating_output/normalized_data.h5ad` — run Stage 2 at least once from the CLI to generate this checkpoint.
- The "Apply Gates" button on Page 2 re-runs `manual_gating.py --skip_normalization` as a subprocess, so all tile-correction and hierarchy logic is preserved.
- Page 4 streams stdout live; long-running jobs (Stage 1 segmentation) may take hours.
