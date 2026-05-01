# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What is cifsQuant

Cyclic immunofluorescence (cycIF) spatial quantification pipeline for immunology labs. Three stages:

1. **Segmentation** (`mcmicro-tiled.nf`) — Nextflow + Cellpose, produces per-cell marker intensity CSVs in `results/`
2. **Gating** (`manual_gating.py`) → `manual_gating_output/gated_data.h5ad`
3. **Spatial analysis** (`spatial_quantification/run_spatial_quantification.py`) → CSVs + figures in `spatial_quantification_results/`

## How to Run

**Recommended — via orchestrator (all stages from one file):**
```bash
python run_cifsquant.py --project project.yaml              # full pipeline
python run_cifsquant.py --project project.yaml --stages gating spatial
python run_cifsquant.py --project project.yaml --dry-run    # validate only
```

**Legacy — stage by stage:**
```bash
nextflow run mcmicro-tiled.nf -c nextflow.config -params-file project.yaml
python manual_gating.py --results_dir results --n_jobs 16
python spatial_quantification/run_spatial_quantification.py --config project.yaml
```

No test suite. No build step. No linting config.

---

## Core Data Structure

**`gated_data.h5ad`** (AnnData) is the central artifact passed from Stage 2 → Stage 3.

| Slot | Content |
|---|---|
| `adata.obs` | Per-cell metadata: `sample_id`, marker-level `is_{marker}` gate booleans, phenotype `is_{name}` booleans, spatial groupings |
| `adata.var` | Marker names |
| `adata.X` | Raw marker intensities (normalized) |
| `adata.layers['gated']` | Binary gate matrix (cells × markers) |
| `adata.obsm['spatial']` | (X, Y) pixel coordinates — also stored as `adata.obs['X_centroid']`/`Y_centroid` |

---

## Stage 2: manual_gating.py

**Config lives at the top of the file** — study-specific dicts that must be edited per project:

- `MARKERS` — `{channel_name: display_name}` mapping from the cycIF panel
- `MARKER_HIERARCHY` — biological parent-child relationships (e.g. `'BCL6': 'B220'` means BCL6+ cells must be subset of B220+)
- `GATES` — per-marker threshold values (0–1 on normalized scale); `None` = auto-calculate
- `TILE_CORRECTION_CONFIG` — UniFORM tile-artifact correction for nuclear/dim markers
- `LIBERAL_GATING_CONFIG` — markers where sensitivity > specificity (rare GC populations)

**Key flow:**
1. Load all per-cell CSVs from `results/` → one combined DataFrame
2. 99th-percentile normalization across all samples (shared gates enabled by `USE_SHARED_GATES = True`)
3. Tile artifact correction (DBSCAN-detected tile grid → per-tile quantile normalization)
4. GMM-based threshold suggestion for each marker
5. Apply gates → write `is_{marker}` booleans to `adata.obs`
6. Enforce hierarchy (child cannot exceed parent count)
7. Save `gated_data.h5ad`

---

## Stage 3: Spatial Quantification Architecture

```
spatial_quantification/
├── run_spatial_quantification.py   # Orchestrator — reads config, runs all enabled analyses
├── config/spatial_config.yaml      # Single config file for all Stage 3 settings
├── core/
│   ├── data_loader.py              # Loads .h5ad, validates is_* columns, extracts coords
│   ├── metadata_manager.py         # Reads sample_metadata.csv, merges into adata.obs
│   ├── phenotype_builder.py        # Derives is_{phenotype} columns from config definitions
│   └── spatial_region_detector.py  # SpatialCells-based structure boundary detection
├── analyses/                       # 20+ analysis modules (all same interface)
├── stats/                          # comparisons.py, tests.py, temporal.py, plot_stats.py
└── visualization/                  # Plotters (one per analysis type)
```

### Orchestration in run_spatial_quantification.py

Steps in order:
1. `DataLoader.load()` → `(adata, metadata_df)`
2. `MetadataManager.process()` + `merge_with_adata(adata)` — adds group/timepoint/treatment cols to `adata.obs`
3. `PhenotypeBuilder.build_all_phenotypes()` — adds `is_{name}` boolean cols from `phenotypes:` block
4. Each enabled analysis block runs its class → appends to `all_results` dict
5. `per_structure_analysis` **must run first** if enabled — downstream analyses reuse `tumor_structures` and `region_detector` objects

### Analysis Module Pattern

Every analysis module is a self-contained class:

```python
class SomeAnalysis:
    def __init__(self, adata, config, output_dir): ...
    def run(self) -> dict: ...   # returns results dict, writes CSVs/plots to output_dir/subdir
```

No shared base class. Each module creates its own subdirectory under `output_dir`.

### Phenotype System

Defined in `spatial_config.yaml` under `phenotypes:`:

```yaml
phenotypes:
  T_cells:
    positive: ['CD3']
    negative: []
  CD8_T_cells:
    base: 'T_cells'       # must be subset of T_cells
    positive: ['CD8']
```

`PhenotypeBuilder` converts these to `is_{name}` boolean columns in `adata.obs`. Marker names in `positive`/`negative` must exactly match `var_names` in the .h5ad. Silently skips (with warning) if a marker is missing.

### SpatialCells Library (`import spatialcells as spc`)

Used in `spatial_region_detector.py` for structure (tumor/follicle) boundary detection:

| Function | Purpose |
|---|---|
| `spc.spatial.getCommunities` | DBSCAN-based cluster detection on (X,Y) coords |
| `spc.spa.getBoundary` | Alpha-shape polygon boundary from cluster cells |
| `spc.spa.pruneSmallComponents` | Remove small spurious boundary components |
| `spc.spatial.assignPointsToRegions` | Assign each cell to a structure region |
| `spc.msmt.getDistanceFromObject` | Distance from each cell to nearest structure boundary |
| `spc.msmt.getSlidingWindowsComposition` | Sliding-window immune composition across tissue |
| `spc.spatial.getRegionArea` | Area (µm²) of a bounded region |

Structure detection config lives under `tumor_definition:` in spatial_config.yaml (eps, min_samples, alpha, boundary_buffer, etc.).

---

## Config System

**Single-file approach (recommended):** Edit `project.yaml` — it drives all three stages.

| Section | Used by | Purpose |
|---|---|---|
| `markers:` | All stages | Panel: channel → display name mapping. Auto-generates `markers.csv`. |
| `marker_hierarchy:` | Stage 2 | Parent/child gate enforcement |
| Top-level keys (`dapi_channel`, `nuc_diameter`, etc.) | Stage 1 | Nextflow params via `-params-file` |
| `gating:` | Stage 2 | Gate thresholds, tile correction, liberal gating settings |
| `spatial:` | Stage 3 | Everything in spatial_config.yaml |

**How `run_spatial_quantification.py` handles both formats:** `load_config()` checks for a `spatial:` key — if found, it extracts that subsection (project.yaml format); otherwise it uses the file as-is (legacy standalone spatial_config.yaml).

**`nextflow.config`** contains only infrastructure (containers, compute resources, executor) — never edited between studies.

**`spatial_config.yaml`** (`spatial_quantification/config/`) still works as a standalone fallback for Stage-3-only use.

---

## Key Configs: Examples

Study-specific real configs are in `configs/examples/`:
- `batch25_tumor_kp/` — KPT/KPNT lung tumor, single timepoint, full 23-marker panel
- `batch6_treatment_validation/` — treatment + timepoint dual-dimension study

Use these as starting points by copying to the project root and editing.

---

## Metadata Dimensions

`sample_metadata.csv` must have `sample_id` (uppercase enforced). Optional extra columns:
- `group` — primary experimental grouping (genotype, condition)
- `timepoint` — numeric (weeks/days); enables temporal analyses
- `treatment` — optional; auto-detected if column exists; enables treatment comparisons

`MetadataManager` also supports derived groupings: if `additional_groupings: ['genotype', 'main_group']` is set in config, it auto-extracts KPT/KPNT and cis/trans from the group string.

---

## Statistics

Default throughout: **Mann-Whitney U** (non-parametric) + **Benjamini-Hochberg FDR** correction. Controlled via `statistics:` block in spatial_config.yaml. Effect sizes and bootstrap confidence intervals optional.

Permutation testing (analyses 3, 10, 13): shuffle `is_{phenotype}` labels 500× → z-score observed metric against null. Used for clustering, colocalization, and distance analyses.

---

## Scripts Directory

Utility scripts outside the main pipeline:

| Script | Stage | Purpose |
|---|---|---|
| `realignscript.py` | 0 (pre-pipeline) | Fix misaligned raw CyCIF channels before segmentation |
| `tile_from_channels.py` | 0 | Tile multi-cycle images into pipeline-ready format |
| `partition_samples.py` | 0 | Split large image batches across runs |
| `tile_artifact_correction.py` | 2 | Standalone tile correction (used internally by manual_gating.py) |
| `plot_spatial_phenotypes.py` | 3 | Quick spatial scatter plots of phenotype distributions |
| `visual_threshold_validation.py` | 2 | Interactive validation of gate thresholds |

`scripts/archive/phenotyping.py` — older post-gating phenotyping approach, not used by current pipeline.
