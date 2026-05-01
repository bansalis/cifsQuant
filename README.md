# cifsQuant

End-to-end spatial quantification pipeline for cyclic immunofluorescence (cIF) imaging data. cifsQuant takes raw multi-cycle tissue images and delivers quantitative spatial analysis of cell populations, their neighborhoods, and tissue architecture — with no programming required after initial setup.

---

## Pipeline Overview

```
Raw OME-TIFF images
       │
       ▼
┌─────────────────────────────┐
│  Stage 1: Segmentation      │  run_pipeline.sh
│  GPU-accelerated cell       │  Nextflow + Docker
│  detection via Cellpose     │
│  → per-cell marker counts   │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Stage 2: Cell Gating       │  manual_gating.py
│  Assign cell types from     │
│  marker expression profiles │
│  → gated_data.h5ad          │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Stage 3: Spatial Analysis  │  run_spatial_quantification.py
│  Population dynamics,       │
│  neighborhood analysis,     │
│  permutation testing, and   │
│  structure-level metrics    │
│  → CSV tables + figures     │
└─────────────────────────────┘
```

---

## Requirements

| Requirement | Version | Purpose |
|---|---|---|
| conda / mamba | any | Python environment management |
| Docker | ≥ 20.x | Stage 1 segmentation containers |
| Nextflow | ≥ 23.10 | Stage 1 pipeline orchestration |
| NVIDIA GPU | any | Recommended for Stage 1 (CPU fallback available) |
| RAM | ≥ 32 GB | Recommended for large datasets |

**One-command setup** (after installing conda, Docker, Nextflow):
```bash
bash setup_environment.sh
```
This creates the `cifsquant` conda environment with all Python dependencies and verifies the installation. See `environment.yaml` for the full dependency list.

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/bansalis/cifsQuant.git
cd cifsQuant
```

**2. Set up the Python environment**
```bash
bash setup_environment.sh
source activate_mcmicro.sh
```

**3. Configure your experiment**

Copy an example and edit it for your study:
```bash
cp configs/examples/batch25_tumor_kp/project.yaml project.yaml
```

Edit `project.yaml` — this single file configures all three stages:
- `markers:` — your imaging panel (channel names → display names)
- `gating:` — gate thresholds and normalization settings
- `spatial:` — phenotype definitions and which analyses to run

Also edit `sample_metadata.csv` with your sample IDs, groups, timepoints, and (optionally) treatment.

See `configs/examples/` for complete real-study configurations.

**4. Place your raw images**
```
rawdata/
├── SAMPLE1.ome.tif
├── SAMPLE2.ome.tif
└── ...
```

**5. Run the full pipeline**
```bash
python run_cifsquant.py --project project.yaml
```

Or run individual stages:
```bash
python run_cifsquant.py --project project.yaml --stages segmentation  # Stage 1 only
python run_cifsquant.py --project project.yaml --stages gating spatial # Stages 2+3
python run_cifsquant.py --project project.yaml --dry-run               # Validate config
```

Outputs:
- `manual_gating_output/gated_data.h5ad` — gated cell data
- `spatial_quantification_results/` — all analysis tables and figures

---

## Configuration Guide

All configuration lives in a single `project.yaml`. Key sections:

### Panel definition
```yaml
markers:
  Cy3_CD3: CD3        # channel_name: display_name
  Cy5_CD8: CD8
  DAPI: DAPI
```
`markers.csv` is auto-generated from this — you do not edit it directly.

### sample_metadata.csv
One row per sample. Required: `sample_id`, `group`. Optional: `timepoint`, `treatment`.
```csv
sample_id,group,treatment,timepoint
SAMPLE1,GroupA,treated,10
SAMPLE2,GroupB,untreated,10
```

### Phenotype definitions (`spatial.phenotypes`)
Define cell populations as marker combinations. The `base:` key restricts to a parent population:
```yaml
phenotypes:
  T_cells:
    positive: [CD3]
  CD8_T_cells:
    base: T_cells
    positive: [CD8]
```

### Analyses (`spatial.*`)
Each analysis has an `enabled: true/false` toggle. Key analyses:

- **`per_tumor_analysis`** — detect spatial structures (tumors, follicles) and measure infiltration per structure
- **`population_dynamics`** — compare cell frequencies across groups/timepoints
- **`spatial_permutation`** — test if spatial patterns exceed chance (500 permutations)
- **`distance_analysis`** — measure immune cell proximity to structure populations
- **`cellular_neighborhoods`** — classify cells by their local neighborhood composition
- **`temporal_analysis`** — longitudinal changes across timepoints
- **`tumor_microenvironment`** — zonal analysis at contact/close/distal distances from structure boundary

Start with `per_tumor_analysis` and `population_dynamics`, then enable more as needed.

---

## Example Configs

See `configs/examples/` for two complete working configurations:

- **`batch25_tumor_kp/`** — single-timepoint lung tumor with 23-marker panel
- **`batch6_treatment_validation/`** — longitudinal study with treated/untreated comparison

---

## Output Structure

```
spatial_quantification_results/
├── per_structure_analysis/        # Per-tumor/follicle metrics
│   ├── per_tumor_metrics.csv
│   └── plots/
├── population_dynamics/           # Cell frequency tables + plots
├── spatial_permutation/           # Permutation test results (z-scores, p-values)
├── distance_analysis/             # Inter-cell distance distributions
├── neighborhood_analysis/         # Neighborhood composition heatmaps
└── ...                            # One subdirectory per enabled analysis
```

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/realignscript.py` | Realign misaligned raw image channels before Stage 1 |
| `scripts/tile_artifact_correction.py` | Correct tile-boundary intensity artifacts |
| `scripts/partition_samples.py` | Split multi-sample slides into individual samples |
| `scripts/tile_from_channels.py` | Generate per-channel TIFF tiles |
| `scripts/plot_spatial_phenotypes.py` | Generate high-resolution spatial phenotype maps |
| `archive_cleanup.sh` | Compress intermediate segmentation files to save disk space |

---

## License

MIT License. See `LICENSE` for details.
