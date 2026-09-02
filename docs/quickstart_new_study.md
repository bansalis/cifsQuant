# Quickstart: Setting Up a New Study

Step-by-step guide for configuring cifsQuant for a new experiment from scratch.

---

## Prerequisites

Run once after installing conda, Docker, and Nextflow:
```bash
bash setup_environment.sh
conda activate cifsquant
```

See `README.md` for system requirements.

**Prefer a browser?** Every step below can also be done in the GUI (`streamlit run gui/app.py`): Panel Setup = Steps 2–4, Gating = Step 5, Spatial Config = Steps 6–9, Run Pipeline = Steps 10–11 with a raw-data matching pre-flight.

---

## Step 1: Copy an example config

Choose the closest example to your study design:

```bash
# Solid tumor (single timepoint, 1+ genotypes)
cp configs/examples/batch25_tumor_kp/project.yaml project.yaml

# Longitudinal / treatment study
cp configs/examples/batch6_treatment_validation/project.yaml project.yaml

# Lymphoid aggregate / TLS / follicle analysis
cp configs/examples/flutls_balt_tls/project.yaml project.yaml

# Melanoma or validation against Nirmal et al. 2022
cp configs/examples/mel_val_nirmal2022/project.yaml project.yaml
```

---

## Step 2: Define your panel (`markers:`)

Replace the example `markers:` block with your actual channel names and display names.

Channel names come from your OME-TIFF metadata or the segmentation CSV column headers. Display names are what you use everywhere else in the config.

```yaml
markers:
  DAPI: DAPI           # always include nuclear stain; not gated
  Cy3_CD3: CD3
  Cy5_CD8: CD8
  FITC_B220: B220
  # ... one line per channel in your panel
```

**Check:** Open a segmentation CSV from `results/` and verify the column names exactly match your `markers:` keys.

---

## Step 3: Set marker hierarchy (`marker_hierarchy:`)

Define biological parent-child constraints. The pipeline enforces that child+ cells ≤ parent+ cells in count.

```yaml
marker_hierarchy:
  CD8: CD3       # CD8+ T cells must be CD3+
  FOXP3: CD3
  BCL6: B220     # BCL6+ GC B cells must be B220+
```

Set `null` for markers with no biological constraint (e.g. KI67 can mark any cell type):
```yaml
marker_hierarchy:
  KI67: null
```

---

## Step 4: Point the pipeline at your raw images (Stage 1 input)

**Per-channel mode (recommended):** one folder per sample under `rawdata/`, each with per-channel `.ome.tif` files. Leave `input_image: null` — `./rawdata` is auto-detected (or set `rawdata_dir:`).

```
rawdata/
└── JL216/
    ├── R1_DAPI.ome.tif
    ├── R1_Cy3_CD3.ome.tif
    └── ...
```

Channel names from `markers:` are **soft-matched** to filenames by round / fluorophore / protein substrings (`Cy3_CD3` matches `R1_Cy3_CD3.ome.tif`). Always confirm the match before a long run:
- CLI: `python run_cifsquant.py --dry-run` prints `✓ channel -> file` per sample
- GUI: the Run Pipeline page shows the same table with warnings for unmatched channels (which are zero-filled, not errors)

**Stacked mode:** a single multi-channel OME-TIFF per run via `input_image:`.

Then edit the flat top-level keys for Cellpose:

```yaml
dapi_channel: 0       # which channel index is DAPI (0-indexed)
nuc_diameter: 12      # measure representative nucleus in FIJI
cyto_diameter: 24     # typically 2x nucleus diameter
```

If you have a bright cytoplasmic marker (e.g. TOM reporter, EpCAM), use weighted channels:
```yaml
custom_channel_weights: '0:0.7,3:0.3'  # index:weight pairs
```

---

## Step 5: Set gating thresholds (`gating:`)

Start with all `null` (auto-calculate via GMM):

```yaml
gating:
  gates:
    CD3: null
    CD8: null
    B220: null
    # ...
```

Run the pipeline through Stage 2, review the gate diagnostic plots in `manual_gating_output/gating_diagnostics/` (or adjust thresholds live on the GUI's Gating page), then override specific markers where the auto-threshold is wrong:

```yaml
gating:
  gates:
    CD3: null
    CD8: 0.35     # manual threshold for dim markers
```

**Enable tile correction** for nuclear or dim markers that show scanner tile artifacts:
```yaml
gating:
  tile_correction:
    enabled: true
    markers: [KI67, BCL6, PERK]   # display names of affected markers
```

---

## Step 6: Define your sample metadata

Edit `sample_metadata.csv` (create it if it doesn't exist):

```csv
sample_id,group,timepoint,treatment
SAMPLE1,KPT,10,none
SAMPLE2,KPNT,10,none
SAMPLE3,KPT,4,treated
```

**Required:** `sample_id` — must match the base filename of each raw image (without extension).

**Optional columns:**
- `group` — primary experimental grouping
- `timepoint` — numeric (weeks/days); enables temporal analyses
- `treatment` — treatment condition; enables treatment comparisons

Update `spatial.metadata` in `project.yaml` to point to the correct columns:
```yaml
spatial:
  metadata:
    group_column: group
    timepoint_column: timepoint     # omit if single-timepoint
    treatment_column: treatment     # omit if no treatment arm
```

---

## Step 7: Define phenotypes (`spatial.phenotypes`)

Define all cell populations you want to analyze. Build from simple to complex:

```yaml
spatial:
  phenotypes:
    # Tier 1: broad lineages
    T_cells:
      positive: [CD3]

    B_cells:
      positive: [B220]

    # Tier 2: subtypes (using base: to enforce parent-child)
    CD8_T_cells:
      base: T_cells
      positive: [CD8]

    Tregs:
      base: T_cells
      positive: [FOXP3]

    # Tier 3: functional states
    PD1_positive_CD8:
      base: CD8_T_cells
      positive: [PD1]
```

**Rules:**
- Marker names in `positive`/`negative` must exactly match display names in `markers:`
- `base:` must reference a key defined earlier in `phenotypes:`
- Population names become `is_{name}` boolean columns in the gated data

---

## Step 8: Configure your spatial structure

If your study has a primary spatial structure (tumor, B cell cluster, follicle), configure `tumor_definition`:

```yaml
spatial:
  tumor_definition:
    base_phenotype: T_cells       # phenotype key that defines the structure
    required_positive: [CD3]
    structure_detection:
      eps: 1000           # DBSCAN radius — increase for large/dense structures
      min_samples: 500    # minimum cells — decrease for early-stage small clusters
```

**DBSCAN `eps` guidance:**
- Large solid tumors: `eps=1000`
- Small/early tumors: `eps=800`
- Lymphoid aggregates/TLS: `eps=250`

---

## Step 9: Enable analyses

Start with the two most informative analyses:

```yaml
spatial:
  per_tumor_analysis:
    enabled: true    # detects structures; required for infiltration/zone analyses

  population_dynamics:
    enabled: true    # cell frequency comparisons across groups
    populations: [T_cells, CD8_T_cells, B_cells, Macrophages]
    comparisons:
      - {name: Group_comparison, groups: [ControlA, ControlB, Treatment]}
```

Then enable additional analyses as needed (see `docs/analysis_guide.md`):
- `distance_analysis` — proximity between immune cells and structure cells
- `immune_infiltration` — infiltration counts per structure (requires `per_tumor_analysis`)
- `spatial_permutation` — are patterns statistically significant?
- `temporal_analysis` — longitudinal changes (requires `timepoint` column)
- `cellular_neighborhoods` — unsupervised spatial neighborhood classification

---

## Step 10: Validate your config

```bash
python run_cifsquant.py --project project.yaml --dry-run
```

This checks:
- Gate keys match panel display names
- Phenotype marker references exist in the panel
- Hierarchy parent/child relationships are valid
- Required config sections are present

Fix any reported errors before running.

---

## Step 11: Run the pipeline

```bash
# Full pipeline (segmentation → gating → spatial)
python run_cifsquant.py --project project.yaml

# Or stages individually:
python run_cifsquant.py --project project.yaml --stages segmentation
python run_cifsquant.py --project project.yaml --stages gating spatial

# Control parallelism:
python run_cifsquant.py --project project.yaml --n-jobs 16
```

---

## Common pitfalls

| Problem | Fix |
|---|---|
| `KeyError: 'is_CD8'` in spatial analysis | Marker name in `phenotypes` doesn't match display name in `markers:` |
| Gate auto-threshold too high/low | Override with explicit value in `gates:` after reviewing diagnostic plots |
| No structures detected | Lower `eps` or `min_samples` in `tumor_definition.structure_detection` |
| `per_tumor_analysis` required by downstream step | Ensure `per_tumor_analysis.enabled: true` runs before `immune_infiltration`, `cluster_composition_analysis`, `enhanced_neighborhoods`, `tumor_microenvironment` |
| Segmentation produces very small cells | Decrease `nuc_diameter` or adjust `custom_channel_weights` |
| SpatialCells import fails | Run `conda activate cifsquant && pip install -e /path/to/SpatialCells` |

---

## Output locations

```
manual_gating_output/
└── gated_data.h5ad              # Stage 2 output → input to Stage 3

spatial_quantification_results/  # (or configured output directory)
├── per_structure_analysis/      # per_tumor_analysis output
├── population_dynamics/
├── distance_analysis/
├── infiltration_analysis/
├── neighborhood_analysis/
└── ...                          # one subdirectory per enabled analysis
```
