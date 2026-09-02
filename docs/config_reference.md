# cifsQuant Configuration Reference

Complete parameter reference for `project.yaml` — the single config file that drives all three pipeline stages.

---

## Panel Definition

```yaml
markers:
  channel_name: display_name
```

| Key | Type | Description |
|---|---|---|
| `markers` | dict | Maps raw cycIF channel names (from the OME-TIFF/CSV) to human-readable display names used everywhere else in the pipeline |

**Rules:**
- Keys are channel names exactly as they appear in segmentation output CSVs
- Values are display names — these are what you use in `gates`, `phenotypes`, `positive`/`negative` lists
- DAPI is always included but not gated (set to `null`)
- Auto-generates `markers.csv` for the Nextflow segmentation stage

```yaml
markers:
  DAPI: DAPI
  Cy3_CD3: CD3
  Cy5_CD8: CD8
```

---

## Marker Hierarchy

```yaml
marker_hierarchy:
  child_marker: parent_marker
```

| Key | Type | Description |
|---|---|---|
| `marker_hierarchy` | dict | Biological parent-child constraints: child marker+ cells are forced to be a subset of parent marker+ cells |

**Example:**
```yaml
marker_hierarchy:
  CD8: CD3    # CD8+ cells must be CD3+; enforced during gating
  FOXP3: CD3
  BCL6: B220
```

Set to `null` to disable hierarchy enforcement for that marker.

---

## Stage 1: Segmentation (Nextflow params)

These top-level keys are passed directly to Nextflow via `-params-file project.yaml`. They control input mode, tiling, and Cellpose segmentation behavior.

### Input mode — set exactly one

| Key | Type | Default | Description |
|---|---|---|---|
| `input_image` | path/null | `null` | **Stacked mode:** one multi-channel OME-TIFF per run |
| `rawdata_dir` | path/null | `null` | **Per-channel mode:** folder with one subfolder per sample, each holding per-channel `.ome.tif` files. If both keys are null and `./rawdata` contains sample folders, per-channel mode is auto-detected. All samples run in one command, each into `results/<sample>/` |

In per-channel mode, channel names from `markers:` are soft-matched against raw filenames (round / fluorophore / protein substrings); the match is printed per sample before running and shown on the GUI's Run Pipeline page. Unmatched channels are zero-filled. The DAPI channel for tile prescreening is derived from the `markers:` entry whose display name is `DAPI`.

### Tiling and segmentation

| Key | Type | Default | Description |
|---|---|---|---|
| `sample_name` | string | `'sample'` | Sample label (stacked mode; per-channel mode uses the folder name) |
| `outdir` | path | `./results` | Segmentation output directory |
| `tile_size` | int | `4096` | Tile edge length in pixels |
| `overlap` | int | `512` | Tile overlap in pixels |
| `pyramid_level` | int | `1` | Pyramid level to read from a stacked OME-TIFF (0 = full resolution) |
| `dapi_channel` | int | `0` | Zero-indexed DAPI channel in a stacked OME-TIFF |
| `nuc_diameter` | int | `12` | Expected nucleus diameter in pixels (Cellpose nuclear segmentation) |
| `cyto_model` | string | `'cyto2'` | Cellpose cytoplasm model (`cyto2` recommended for tissue CyCIF) |
| `cyto_diameter` | int | `24` | Expected cell diameter in pixels (Cellpose cytoplasmic expansion) |
| `custom_channel_weights` | string | `''` | Comma-separated `index:weight` pairs for the weighted cytoplasm composite (e.g. `'0:0.7,1:0.15,4:0.15'`). Empty = uniform. Indices refer to the panel's channel order |
| `skip_tiling` / `tiles_dir` | bool / path | `false` / `null` | Advanced: consume pre-generated tiles directly (the orchestrator sets these itself in per-channel mode) |

GPU use is auto-detected (`torch.cuda.is_available()`); there is no config key for it.

**Tuning notes:**
- `nuc_diameter`: measure a representative nucleus in FIJI/Napari; typical CyCIF values are 10–15 px at 20× magnification
- `custom_channel_weights`: use when a non-DAPI cytoplasmic marker (e.g. TOM reporter, EpCAM) reliably marks cell boundaries — never reuse weights across different panels
- `cyto_diameter`: set to ~2× `nuc_diameter` for most tissue types

---

## Stage 2: Gating

```yaml
gating:
  use_shared_gates: true
  normalization_method: percentile_99
  gates: {...}
  tile_correction: {...}
  liberal_gating: {...}
```

### Top-level gating params

| Key | Type | Default | Description |
|---|---|---|---|
| `use_shared_gates` | bool | `true` | `true` = one threshold per marker across all samples (recommended for cohorts). `false` = per-sample thresholds |
| `normalization_method` | string | `'percentile_99'` | Intensity normalization before gating. Options: `percentile_99`, `zscore`, `minmax` |

### gates

```yaml
gates:
  CD3: null       # null = auto-calculate via GMM
  CD8: 0.35       # explicit threshold on 0–1 normalized scale
```

| Value | Behavior |
|---|---|
| `null` | GMM auto-threshold. Fits a 2-component GMM to the intensity histogram and places the gate at the intersection |
| `float (0–1)` | Manual threshold on the 99th-percentile normalized scale |

**Setting gates manually:** Run the pipeline once with all `null`, review `tile_correction_diagnostics/` outputs and GMM fit plots, then override specific markers where the auto-threshold is incorrect.

### tile_correction

Corrects tile-boundary intensity artifacts introduced by the cycIF scanner. Uses DBSCAN to detect the tile grid and applies per-tile quantile normalization.

```yaml
tile_correction:
  enabled: true
  markers: [KI67, BCL6, PERK]   # markers affected by tile artifacts
  method: quantile               # only 'quantile' currently supported
  quantile: 0.95
```

| Key | Type | Description |
|---|---|---|
| `enabled` | bool | Whether to run tile correction |
| `markers` | list | Marker display names to correct. Typically nuclear/dim markers most affected by scanning artifacts |
| `method` | string | Correction method. `quantile`: normalizes each tile to a common quantile |
| `quantile` | float | Reference quantile for normalization (0–1) |

**When to enable:** If you see a visible grid pattern in dim markers or periodic intensity steps at tile boundaries in QC plots.

### liberal_gating

For rare populations where sensitivity matters more than specificity (e.g. low-abundance GC B cells, activation markers).

```yaml
liberal_gating:
  enabled: false
  liberal_markers: [GranzB, TIM3]
```

| Key | Type | Description |
|---|---|---|
| `enabled` | bool | Whether to apply liberal thresholds to specified markers |
| `liberal_markers` | list | Marker display names to gate liberally (lower threshold) |

---

## Stage 3: Spatial Analysis

All Stage 3 config lives under the `spatial:` key.

### input / output

```yaml
spatial:
  input:
    gated_data: manual_gating_output/gated_data.h5ad
    metadata: sample_metadata.csv
  output:
    base_directory: spatial_quantification_results
    save_intermediate: true
    formats: [png]
    dpi: 300
```

| Key | Type | Description |
|---|---|---|
| `input.gated_data` | path | Path to gated AnnData file from Stage 2 |
| `input.metadata` | path | Path to sample metadata CSV |
| `output.base_directory` | path | Root output directory for all analyses |
| `output.save_intermediate` | bool | Save per-analysis intermediate data files |
| `output.formats` | list | Plot formats: `png`, `pdf`, `svg` |
| `output.dpi` | int | Figure resolution (300 for publication) |

### metadata

```yaml
spatial:
  metadata:
    sample_column: sample_id
    group_column: group
    timepoint_column: timepoint        # optional
    treatment_column: treatment        # optional
    additional_groupings: []           # optional derived columns
```

| Key | Type | Description |
|---|---|---|
| `sample_column` | string | Column in metadata CSV with sample identifiers (must match sample IDs in .h5ad) |
| `group_column` | string | Primary experimental grouping (genotype, condition, stage) |
| `timepoint_column` | string | Numeric timepoint column. Enables temporal analyses when present |
| `treatment_column` | string | Treatment condition column. Enables treatment comparisons when present |
| `additional_groupings` | list | Extra metadata columns to merge into adata.obs |

### tumor_definition / structure_definition

Defines the cell population used as the spatial "structure" (tumor, B cell cluster, follicle, etc.). Used by `per_tumor_analysis`, `immune_infiltration`, `tumor_microenvironment`, and `enhanced_neighborhoods`.

```yaml
spatial:
  tumor_definition:
    base_phenotype: Tumor        # must match a key in phenotypes:
    required_positive: [TOM]
    required_negative: []
    structure_detection:
      method: DBSCAN
      eps: 1000
      min_samples: 500
      min_cluster_size: 250
      boundary_buffer: 100
      use_expanded_boundary: true
      alpha: 100
      min_edges: 20
      holes_min_edges: 200
```

| Key | Type | Description |
|---|---|---|
| `base_phenotype` | string | Phenotype key that defines structure cells |
| `required_positive` | list | Additional markers required to be positive (must match panel display names) |
| `required_negative` | list | Markers required to be negative |
| `structure_detection.method` | string | Clustering method. Only `DBSCAN` currently |
| `structure_detection.eps` | int | DBSCAN neighborhood radius in pixels. Larger = more cells merged into one structure |
| `structure_detection.min_samples` | int | DBSCAN minimum cells to form a cluster. Raise to require larger structures |
| `structure_detection.min_cluster_size` | int | Minimum cells per detected structure (post-DBSCAN filter) |
| `structure_detection.boundary_buffer` | int | Pixels to expand boundary outward from the cell cluster edge |
| `structure_detection.use_expanded_boundary` | bool | Use the buffered boundary for infiltration/zone analysis |
| `structure_detection.alpha` | int | Alpha-shape concavity parameter. Smaller = tighter fit to cluster shape |
| `structure_detection.min_edges` | int | Minimum polygon edges; structures with fewer are discarded |
| `structure_detection.holes_min_edges` | int | Minimum edges for internal hole polygons |

**Tuning `eps` and `min_samples`:**
- Small tumors / early timepoints: `eps=800, min_samples=100`
- Typical solid tumors: `eps=1000, min_samples=500`
- Lymphoid aggregates: `eps=250, min_samples=100`

### phenotypes

Defines cell populations as combinations of marker gates. PhenotypeBuilder converts these to `is_{name}` boolean columns in `adata.obs`.

```yaml
spatial:
  phenotypes:
    T_cells:
      positive: [CD3]
      negative: []
    CD8_T_cells:
      base: T_cells         # must be subset of T_cells
      positive: [CD8]
    Myeloid:
      anypos: [CD163, CD11c]   # OR logic: CD163+ OR CD11c+
```

| Key | Type | Description |
|---|---|---|
| `positive` | list | All listed markers must be positive (AND logic) |
| `negative` | list | All listed markers must be negative (AND logic) |
| `anypos` | list | At least one listed marker must be positive (OR logic) |
| `base` | string | Restricts to cells already positive for a parent phenotype |
| `description` | string | Optional documentation string (not used by pipeline) |

**Marker names** in `positive`/`negative`/`anypos` must exactly match display names in the `markers:` section.

### statistics

Global statistical settings (can be overridden per-analysis).

```yaml
spatial:
  statistics:
    test: mann_whitney           # default pairwise test
    fdr_method: benjamini-hochberg
    alpha: 0.05
    report_effect_sizes: true
    n_bootstrap: 1000
```

| Key | Type | Options | Description |
|---|---|---|---|
| `test` | string | `mann_whitney`, `kruskal`, `t_test` | Pairwise statistical test |
| `fdr_method` | string | `benjamini-hochberg`, `bonferroni` | Multiple comparison correction |
| `alpha` | float | 0–1 | Significance threshold |
| `report_effect_sizes` | bool | — | Include effect size (rank-biserial r) in output |
| `n_bootstrap` | int | — | Bootstrap iterations for confidence intervals |

### visualization / performance / advanced

```yaml
spatial:
  visualization:
    enabled: true
    style: publication      # 'publication' or 'exploratory'
    font_family: Arial
    font_size: 12

  performance:
    n_jobs: -1              # -1 = all cores
    use_kdtree: true        # fast spatial queries
    memory_efficient: true  # process samples sequentially

  advanced:
    random_seed: 42
    verbose: true
```

---

## Analysis Blocks

Every analysis has an `enabled: true/false` toggle. Set `enabled: false` for analyses you don't need.

**Run-order constraint:** `per_tumor_analysis` must be enabled and run before `cluster_composition_analysis`, `immune_infiltration` (SpatialCells mode), `enhanced_neighborhoods`, and `tumor_microenvironment` — these analyses receive the detected structure objects from `per_tumor_analysis`.

See `docs/analysis_guide.md` for detailed per-analysis parameter documentation.

### Quick reference

| Config key | Analysis | Requires `per_tumor_analysis` |
|---|---|---|
| `per_tumor_analysis` | Detect structures + per-structure metrics | No (is the source) |
| `population_dynamics` | Cell frequency comparisons across groups | No |
| `distance_analysis` | Nearest-neighbor distances between populations | No |
| `immune_infiltration` | Infiltration counts inside/outside structures | Yes (SpatialCells mode) |
| `cellular_neighborhoods` | K-means on windowed neighborhood composition | No |
| `spatial_permutation` | Permutation test for spatial colocalization | No |
| `distance_permutation_testing` | Permutation test for distance patterns | No |
| `neighborhood_permutation_testing` | Permutation test for neighborhood enrichment | No |
| `tumor_microenvironment` | Zone analysis at contact/close/distal distances | Yes |
| `enhanced_neighborhoods` | Marker-stratified neighborhood composition | Yes |
| `marker_region_analysis` | Alpha-shape regions of marker+ cells + immune enrichment | No |
| `cluster_composition_analysis` | Stacked bar immune composition per structure | Yes |
| `temporal_analysis` | Structure-level metrics across timepoints | No |
| `lda_neighborhood_analysis` | LDA recurrent cellular neighborhoods (Nirmal 2022) | No |
| `spatial_lag_analysis` | Spatial lag vectors → tumor cell communities (Nirmal 2022) | No |
| `shift_plot_analysis` | Harrell-Davis shift plots for distance distributions | No |
| `coexpression_analysis` | Pairwise phenotype co-occurrence | No |
| `spatial_overlap_analysis` | Spatial overlap between phenotype populations | No |
| `perk_mfi_analysis` | pERK MFI per phenotype and structure | No |
| `kpnt_correlation_analysis` | KPT vs KPNT tumor correlation analysis | No |
| `pseudotime_analysis` | Diffusion map pseudotime ordering | No |
| `marker_clustering_analysis` | Unsupervised marker clustering | No |
