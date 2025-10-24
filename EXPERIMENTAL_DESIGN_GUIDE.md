# CORRECTED: Multi-Level Experimental Design Analysis

## Your Experimental Design (Now Properly Handled!)

You have a **2 × 2 factorial design**:

```
          cis        trans
KPT    KPT Het cis  KPT Het trans
KPNT   KPNT cis     KPNT trans
```

**Total: 4 distinct groups**

---

## What Was Fixed

### Before (WRONG):
❌ Treated as simple "cis vs trans" comparison
❌ Ignored KPT vs KPNT distinction
❌ No multi-level analysis

### After (CORRECT): ✅
✅ Model level: **KPT vs KPNT** (main effect)
✅ Genotype level: **cis vs trans** (within models)
✅ 4-way level: **All 4 groups** compared
✅ Proper factorial analysis

---

## Metadata Parsing (FIXED)

Your metadata:
```csv
sample_id,group,treatment,timepoint
GUEST29,KPNT cis,none,8
GUEST30,KPNT trans,none,8
Guest33,KPT Het trans,none,10
Guest36,KPT Het cis,none,12
...
```

Now parsed as:
- `model`: KPT or KPNT
- `genotype`: cis or trans
- `model_genotype`: KPT_cis, KPT_trans, KPNT_cis, KPNT_trans
- `group_full`: Original group name

---

## Analysis Levels

### LEVEL 1: KPT vs KPNT (Main Effect)

**Compares the two main groups**, collapsing across cis/trans.

**Questions answered:**
- Do KPT and KPNT tumors grow differently?
- Do they have different marker expression?
- Do they differ in immune infiltration?

**Outputs:**
- `statistics/tumor_size_temporal_KPT_vs_KPNT.csv`
  * Temporal trends for each model
  * Spearman ρ, p-values, FDR correction

- `statistics/tumor_size_KPT_vs_KPNT_by_timepoint.csv`
  * Model comparison at each timepoint
  * Mann-Whitney U tests
  * Fold changes

- `figures/temporal/tumor_size/KPT_vs_KPNT_temporal.png`
  * Line plots showing KPT vs KPNT growth
  * Boxplots by timepoint

- `figures/temporal/marker_expression/{marker}_KPT_vs_KPNT.png`
  * For each marker (TOM, AGFP, PERK, KI67, CD45, CD3, CD8B)
  * Line plots + boxplots

### LEVEL 2: All 4 Groups (Detailed Analysis)

**Compares all 4 groups** individually.

**Questions answered:**
- How does KPT cis compare to KPT trans?
- How does KPNT cis compare to KPNT trans?
- Do cis/trans effects differ between KPT and KPNT?
- Which group has the largest tumors?

**Outputs:**
- `statistics/tumor_size_temporal_4way.csv`
  * Temporal trends for each of 4 groups
  * Spearman ρ, p-values, FDR correction

- `statistics/tumor_size_4way_by_timepoint.csv`
  * Pairwise comparisons between all 4 groups
  * At each timepoint
  * 6 comparisons per timepoint:
    - KPT_cis vs KPT_trans
    - KPT_cis vs KPNT_cis
    - KPT_cis vs KPNT_trans
    - KPT_trans vs KPNT_cis
    - KPT_trans vs KPNT_trans
    - KPNT_cis vs KPNT_trans

- `figures/temporal/tumor_size/4way_groups_temporal.png`
  * Line plots for all 4 groups
  * Boxplots by timepoint
  * Color-coded:
    - KPT_cis: Red
    - KPT_trans: Blue
    - KPNT_cis: Green
    - KPNT_trans: Purple

- `figures/temporal/tumor_size/4way_violin.png`
  * Violin plots showing full distributions

- `figures/temporal/marker_expression/{marker}_4way.png`
  * For each marker
  * All 4 groups compared
  * Line plots + boxplots

---

## Complete Output Structure

```
comprehensive_spatial_analysis/
├── data/
│   ├── tumor_size_by_sample.csv
│   │   Columns: sample_id, timepoint, model, genotype,
│   │            model_genotype, total_tumor_cells, mean_structure_size,
│   │            n_structures, total_area, mean_area
│   │
│   └── marker_expression_temporal.csv
│       Columns: marker, sample_id, timepoint, model, genotype,
│                model_genotype, pct_positive, n_cells, n_positive
│
├── statistics/
│   ├── LEVEL 1 (KPT vs KPNT):
│   ├── tumor_size_temporal_KPT_vs_KPNT.csv
│   ├── tumor_size_KPT_vs_KPNT_by_timepoint.csv
│   │
│   └── LEVEL 2 (4-way):
│       ├── tumor_size_temporal_4way.csv
│       └── tumor_size_4way_by_timepoint.csv
│
├── figures/
│   ├── spatial_maps/
│   │   ├── by_sample/ (19 maps)
│   │   ├── by_timepoint/ (7 maps)
│   │   ├── by_genotype/
│   │   │   ├── KPT_spatial.png
│   │   │   ├── KPNT_spatial.png
│   │   │   ├── KPT_cis_spatial.png
│   │   │   ├── KPT_trans_spatial.png
│   │   │   ├── KPNT_cis_spatial.png
│   │   │   └── KPNT_trans_spatial.png
│   │
│   └── temporal/
│       ├── tumor_size/
│       │   ├── KPT_vs_KPNT_temporal.png      (LEVEL 1)
│       │   ├── 4way_groups_temporal.png      (LEVEL 2)
│       │   └── 4way_violin.png               (LEVEL 2)
│       │
│       └── marker_expression/
│           For EACH of 7 markers:
│           ├── {marker}_KPT_vs_KPNT.png      (LEVEL 1)
│           └── {marker}_4way.png             (LEVEL 2)
│           Total: 14 plots
```

---

## Running the Analysis

### Single Command:

```bash
python run_comprehensive_analysis.py \
  --config configs/comprehensive_config.yaml \
  --metadata sample_metadata.csv
```

### Expected Terminal Output:

```
PARSED SAMPLE METADATA - EXPERIMENTAL DESIGN
==================================================================
Total samples: 19

Main groups (model):
  {'KPT': 6, 'KPNT': 13}

Genotypes:
  {'cis': 11, 'trans': 8}

4-way groups:
  {'KPT_cis': 3, 'KPT_trans': 3, 'KPNT_cis': 8, 'KPNT_trans': 5}

Timepoints: [8, 10, 12, 14, 16, 18, 20]

Sample breakdown by model, genotype, and timepoint:
                   8   10   12   14   16   18   20
model genotype
KPT   cis         0    0    1    1    0    0    1
      trans       0    1    1    0    0    0    0
KPNT  cis         1    1    1    1    1    1    2
      trans       1    1    1    1    1    1    1

Full group names:
  KPT Het cis: 3 samples
  KPT Het trans: 3 samples
  KPNT cis: 8 samples
  KPNT trans: 5 samples
==================================================================
```

---

## Key Results Files Explained

### `statistics/tumor_size_temporal_KPT_vs_KPNT.csv`

| Column | Description |
|--------|-------------|
| `model` | KPT or KPNT |
| `metric` | What was tested (total_tumor_cells or mean_structure_size) |
| `spearman_rho` | Correlation coefficient (-1 to 1) |
| `p_value` | Uncorrected p-value |
| `p_adjusted` | FDR-corrected p-value |
| `significant` | True if p_adjusted < 0.05 |

**Interpretation:**
- If KPT row has `spearman_rho > 0` and `significant = True`:
  **KPT tumors are growing over time**

- If KPNT row has `spearman_rho < 0` and `significant = True`:
  **KPNT tumors are shrinking over time**

### `statistics/tumor_size_KPT_vs_KPNT_by_timepoint.csv`

| Column | Description |
|--------|-------------|
| `timepoint` | Timepoint tested |
| `comparison` | KPT_vs_KPNT |
| `KPT_mean` | Mean tumor cells in KPT |
| `KPNT_mean` | Mean tumor cells in KPNT |
| `fold_change` | KPNT / KPT ratio |
| `p_adjusted` | FDR-corrected p-value |
| `significant` | True if p_adjusted < 0.05 |

**Interpretation:**
- If timepoint 20 row has `fold_change = 2.5` and `significant = True`:
  **At timepoint 20, KPNT has 2.5× more tumor cells than KPT (significant)**

### `statistics/tumor_size_temporal_4way.csv`

Same as KPT vs KPNT but for all 4 groups:
- KPT_cis
- KPT_trans
- KPNT_cis
- KPNT_trans

Shows which groups are growing/shrinking independently.

### `statistics/tumor_size_4way_by_timepoint.csv`

Pairwise comparisons between all 4 groups.

| Column | Description |
|--------|-------------|
| `timepoint` | Timepoint |
| `group_1` | First group (e.g., KPT_cis) |
| `group_2` | Second group (e.g., KPNT_cis) |
| `mean_1` | Mean for group 1 |
| `mean_2` | Mean for group 2 |
| `fold_change` | mean_2 / mean_1 |
| `p_adjusted` | FDR-corrected p-value |
| `significant` | True if significant |

**Example interpretation:**
```
timepoint: 20
group_1: KPT_cis
group_2: KPNT_cis
fold_change: 3.2
p_adjusted: 0.001
significant: True
```

**Means**: At timepoint 20, KPNT cis tumors are 3.2× larger than KPT cis tumors (highly significant)

---

## Scientific Questions You Can Now Answer

### 1. Main Effects

**Q: Do KPT and KPNT differ overall?**

**A:** Check `statistics/tumor_size_KPT_vs_KPNT_by_timepoint.csv`
- Look for `significant = True` rows
- Check `fold_change` to see magnitude
- Plot: `figures/temporal/tumor_size/KPT_vs_KPNT_temporal.png`

### 2. Genotype Effects

**Q: Does cis vs trans matter within each model?**

**A:** Check `statistics/tumor_size_4way_by_timepoint.csv`
- Find rows comparing `KPT_cis` vs `KPT_trans`
- Find rows comparing `KPNT_cis` vs `KPNT_trans`
- If both are significant, genotype matters

### 3. Interaction Effects

**Q: Does the cis/trans effect depend on the model (KPT vs KPNT)?**

**A:** Compare:
- KPT_cis vs KPT_trans fold change
- KPNT_cis vs KPNT_trans fold change

If they're different directions or magnitudes → **interaction effect**

### 4. Temporal Dynamics

**Q: Which groups are growing fastest?**

**A:** Check `statistics/tumor_size_temporal_4way.csv`
- Sort by `spearman_rho` (descending)
- Highest ρ = fastest growth
- Only consider `significant = True` rows

### 5. Marker Dynamics

**Q: How does AGFP expression change over time in each group?**

**A:**
- Data: `data/marker_expression_temporal.csv`
- Plot: `figures/temporal/marker_expression/AGFP_4way.png`
- Statistics: Can add to future versions

---

## Color Coding in Plots

**Consistent across all figures:**

| Group | Color | Hex Code |
|-------|-------|----------|
| KPT_cis | **Red** | #E41A1C |
| KPT_trans | **Blue** | #377EB8 |
| KPNT_cis | **Green** | #4DAF4A |
| KPNT_trans | **Purple** | #984EA3 |

---

## Example Workflow

### 1. Run Analysis
```bash
python run_comprehensive_analysis.py \
  --config configs/comprehensive_config.yaml \
  --metadata sample_metadata.csv
```

### 2. Check Main Effect (KPT vs KPNT)
```bash
head statistics/tumor_size_KPT_vs_KPNT_by_timepoint.csv
```

Look for:
- Significant differences at late timepoints?
- KPT or KPNT larger?

### 3. Check 4-Way Comparisons
```bash
grep "KPT_cis.*KPNT_cis" statistics/tumor_size_4way_by_timepoint.csv
```

See if cis groups differ between models.

### 4. View Plots
```bash
open figures/temporal/tumor_size/4way_groups_temporal.png
```

Visual confirmation of statistical findings.

### 5. Check Markers
```bash
open figures/temporal/marker_expression/AGFP_4way.png
```

See marker expression dynamics.

---

## Summary

**You now get:**

✅ **2-level factorial analysis** (KPT vs KPNT, cis vs trans)
✅ **All pairwise comparisons** with FDR correction
✅ **Temporal trends** for each group
✅ **Multiple visualization formats** (line, box, violin)
✅ **Marker expression tracking** at both levels
✅ **Spatial maps** for all groupings
✅ **Publication-ready outputs** (300 DPI, proper colors)

**All analyses account for your experimental design!**

Run the analysis and you'll get comprehensive results for both:
1. **Main model effect** (KPT vs KPNT)
2. **Detailed 4-way comparison** (all groups)

Everything is properly color-coded and statistically tested with multiple hypothesis correction.

**Ready to discover which groups differ and how!**
