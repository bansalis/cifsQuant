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

| Requirement | Version |
|---|---|
| Docker | ≥ 20.x |
| Nextflow | ≥ 23.10 |
| Python | ≥ 3.10 |
| NVIDIA GPU | Recommended for Stage 1 |
| RAM | ≥ 32 GB recommended |

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

Edit these three files:
- `markers.csv` — list your imaging markers (cycle number + marker name)
- `sample_metadata.csv` — one row per sample with group, timepoint, and optionally treatment
- `spatial_quantification/config/spatial_config.yaml` — define phenotypes and enable analyses

See `configs/examples/` for complete real-study examples.

**4. Place your raw images**
```
rawdata/
├── SAMPLE1.ome.tif
├── SAMPLE2.ome.tif
└── ...
```

**5. Run Stage 1 — Segmentation**
```bash
bash run_pipeline.sh
```
This detects cells in each image and outputs per-cell marker intensities to `results/`.

**6. Run Stage 2 — Cell Gating**
```bash
python manual_gating.py --results_dir results --n_jobs 8
```
Output: `manual_gating_output/gated_data.h5ad`

**7. Run Stage 3 — Spatial Analysis**
```bash
python spatial_quantification/run_spatial_quantification.py \
  --config spatial_quantification/config/spatial_config.yaml
```
Output: `spatial_quantification_results/`

---

## Configuration Guide

### markers.csv
Define your imaging panel. One row per channel:
```csv
cycle,marker_name
1,DAPI
2,CD45
2,CD3
3,TUMOR_MARKER
```

### sample_metadata.csv
One row per sample. Required columns: `sample_id`, `group`. Optional: `timepoint`, `treatment`.
```csv
sample_id,group,treatment,timepoint
SAMPLE1,GroupA,treated,10
SAMPLE2,GroupB,untreated,10
```

### spatial_config.yaml
The main analysis configuration. Key sections:

- **`phenotypes`** — define cell populations as combinations of positive/negative markers
- **`per_tumor_analysis`** — detect spatial structures (tumors, follicles) and measure infiltration
- **`population_dynamics`** — compare cell frequencies across groups/timepoints
- **`spatial_permutation`** — test if spatial patterns exceed chance (500 permutations)
- **`distance_analysis`** — measure immune cell proximity to structure populations
- **`cellular_neighborhoods`** — classify cells by their local neighborhood composition

Each analysis has an `enabled: true/false` toggle. Start with the analyses you need and enable more as required.

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
