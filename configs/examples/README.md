# Example Configurations

Real study configurations from cifsQuant analyses. Use them as starting points for your own project.

Each example now ships with a `project.yaml` — the recommended single-config format that drives all three pipeline stages.

---

## Available Examples

### `flutls_balt_tls/`
**Study:** Flu-induced BALT / tertiary lymphoid structure (TLS) kinetics
**Samples:** Longitudinal lung tissue, 2, 4, 8, 16 weeks post-infection
**Panel:** 14 markers (B220, BCL6, GL7, IgD, CD21, CD23, AID, KI67, CD3, CD4, CD8B, PD1, PNAD + DAPI)
**Key features:**
- B cell clusters as the primary spatial structure (eps=250µm, vs tumor eps=1000µm)
- Full GC B cell hierarchy: Naive → Follicular → GL7+GC → Proliferating/Non-proliferating
- Tfh, FDC, HEV populations with per-structure distance and infiltration analysis
- GL7+ marker region analysis for GC zone quantification
- Temporal analysis for structure kinetics across timepoints

### `mel_val_nirmal2022/`
**Study:** Melanoma validation reproducing Nirmal et al. 2022 (Cancer Discovery)
**Samples:** 8 patients, 6 AJCC disease stages (Stage IA → IV), primary melanoma
**Panel:** 41 markers (full melanoma CyCIF panel including pERK, SOX10, MART1, HLADPB1, CD3d, CD8a, CD163, CD11c, TIM3, LAG3, FOXP3, GranzB, etc.)
**Key features:**
- LDA-based Recurrent Cellular Neighborhoods (Fig 2D/E, 3A-D from paper)
- Spatial Lag Tumor Cell Communities / TCC analysis (Fig 5C-F)
- Harrell-Davis shift plot distance comparisons (Fig 3F)
- pERK+/HLADPB1+ tumor state spatial zones and immune exclusion/engagement hypotheses

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

| Feature | batch25 | batch6 | flutls | mel_val |
|---|---|---|---|---|
| Tissue | Lung tumor | Lung tumor | Lung BALT/TLS | Skin melanoma |
| Structure type | Tumor (TOM+) | Tumor (TOM+) | B cell cluster (B220+) | Melanoma (SOX10+) |
| Timepoints | 1 (wk 10) | 3 (3,6,8 wk) | 4 (2,4,8,16 wk) | N/A (stage) |
| Treatment dim | No | Yes (p14) | No | No |
| Temporal analysis | No | Yes | Yes | No |
| Panel size | 24 markers | 13 markers | 14 markers | 41 markers |
| Structure eps | 1000µm | 800µm | 250µm | 1000µm |
| LDA/RCN analysis | No | No | No | Yes |
| Shift plot analysis | No | No | No | Yes |
| Spatial lag TCC | No | No | No | Yes |
