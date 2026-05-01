# Example Configurations

Real study configurations from cifsQuant analyses. Use them as starting points for your own project.

Each example now ships with a `project.yaml` — the recommended single-config format that drives all three pipeline stages.

---

## Available Examples

### `batch25_tumor_kp/`
**Study:** Single-timepoint lung tumor microenvironment (KP mouse model)
**Samples:** 4 samples — KPT cis/trans, KPNT cis/trans (JL216–219), week 10
**Panel:** 24 markers across 8 imaging cycles (TOM, CD45, NINJA, pERK, CD4, EPCAM, B220, CD3E, F480, TTF1, PDL1, CD8A, ASMA, GZMB, KLRG1, FOXP3, PD1, NAK, KI67, MHCII, BCL6, CC3, CD103 + DAPI)
**Key features:**
- TOM+ tumor detection, eps=1000µm (established week 10 tumors)
- Full tumor phenotyping: pERK, NINJA, Ki67, PDL1, EPCAM, TTF1, CC3, MHCII
- Full immune panel: CD8/CD4 T cells, Tregs, macrophages, B cells
- Per-tumor SpatialCells analysis + pERK MFI on CD8 T cells

### `batch6_treatment_validation/`
**Study:** Longitudinal T-cell adoptive transfer treatment (3, 6, 8 wk timepoints)
**Samples:** 13 samples — KPT and KPNT cis/trans, p14 transfer vs untreated
**Panel:** 13 markers (TOM, CD3, CC3, B220, Thy1, pERK, CD8, MHC1, ASMA, IFNy, EPCAM, CD4, DAPI)
**Key features:**
- `treatment` column (p14 vs none) as a comparison dimension alongside timepoint
- Thy1+ marks transferred p14 cells specifically
- IFNy+CD8 T cells as effector readout
- Smaller tumor eps=800µm to detect early 3wk tumors
- `test_per_timepoint: true` for temporal comparisons

---

## How to Use

**Recommended (unified config):**
```bash
# Copy the example project.yaml and edit it for your study
cp configs/examples/batch25_tumor_kp/project.yaml project.yaml

# Run the full pipeline
python run_cifsquant.py --project project.yaml

# Or spatial analysis only (if you already have gated_data.h5ad)
python run_cifsquant.py --project project.yaml --stages spatial
```

**Legacy (standalone spatial config):**
```bash
cp configs/examples/batch25_tumor_kp/spatial_config.yaml spatial_quantification/config/spatial_config.yaml
python spatial_quantification/run_spatial_quantification.py
```

---

## Key Differences Between Examples

| Feature | batch25 | batch6 |
|---|---|---|
| Timepoints | 1 (week 10) | 3 (3, 6, 8 wk) |
| Treatment dimension | No | Yes (p14 vs none) |
| Temporal analysis | `enabled: false` | `enabled: true` |
| `test_per_timepoint` | false | true |
| Macrophages | Yes (F480) | No (not in panel) |
| Proliferation | Yes (KI67) | No |
| Tumor size at detection | Large (eps=1000µm) | Small-medium (eps=800µm) |
