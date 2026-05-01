# Example Configurations

This directory contains complete configuration sets from real studies run with cifsQuant. Use them as starting points for your own analysis.

## Available Examples

### `batch25_tumor_kp/`
**Study:** Single-timepoint lung tumor microenvironment (KP mouse model)
**Samples:** 4 samples — KPT cis/trans, KPNT cis/trans (JL216–219)
**Panel:** 23 markers across 8 imaging cycles
**Key features:**
- TOM+ tumor detection via DBSCAN (eps=1000µm)
- pERK, NINJA, Ki67 tumor phenotypes
- CD45+, T cells, CD8/CD4 immune populations
- Per-tumor SpatialCells analysis enabled

### `batch6_treatment_validation/`
**Study:** Longitudinal treatment response with treated/untreated comparison
**Key features:**
- `treatment` metadata column used as comparison dimension
- Multi-timepoint with `test_per_timepoint: true`
- Treatment-stratified comparisons in population dynamics

## How to Use

1. Copy the example `spatial_config.yaml` to `spatial_quantification/config/spatial_config.yaml`
2. Copy `markers.csv` and `sample_metadata.csv` to the project root
3. Update input paths to point to your data
4. Adjust `phenotypes` to match your marker panel
5. Set `enabled: true` for the analyses you want to run

All paths in `spatial_config.yaml` are relative to the project root.
