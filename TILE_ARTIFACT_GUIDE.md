# Tile Artifact Correction & Hierarchical Gating Guide

## Overview

This pipeline now includes **marker-specific** tile artifact correction and hierarchical gating. These features only apply to markers you specify, leaving well-functioning markers (like TOM, CD45) completely untouched.

---

## 1. Tile Artifact Correction (Per-Marker)

### Configuration Location
Edit the `TILE_ARTIFACT_CORRECTION` dictionary in `manual_gating.py` (lines 70-85)

### Two Types of Artifacts

#### **Type 1: Baseline Tile Brightness**
- **Problem**: Clusters of tiles are uniformly brighter or dimmer than others
- **Cause**: Uneven illumination, sample positioning, or instrument drift
- **Solution**: Detects outlier tiles using Coefficient of Variation (CV) and Median Absolute Deviation (MAD), then normalizes tile medians to global median
- **When to Use**: Dim/rare markers with visible "checkerboard" patterns in spatial plots

#### **Type 2: Edge Artifacts**
- **Problem**: Tile edges have sharp brightness transitions, creating grid-line artifacts
- **Cause**: Tiling algorithm creates artificial boundaries, edge cells behave differently
- **Solution**: Compares edge region (outer 10%) to center region intensities, corrects edges to match center
- **When to Use**: Markers showing sharp lines/grids at tile boundaries in spatial plots

### Configuration Options

```python
TILE_ARTIFACT_CORRECTION = {
    # Disable correction (marker works well)
    'TOM': {'enabled': False},
    'CD45': {'enabled': False},

    # Type 1 only (baseline brightness issues)
    'AGFP': {'enabled': True, 'type1': True, 'type2': False, 'sensitivity': 'medium'},
    'PERK': {'enabled': True, 'type1': True, 'type2': False, 'sensitivity': 'medium'},

    # Type 2 only (edge artifacts)
    'CD8B': {'enabled': True, 'type1': False, 'type2': True, 'sensitivity': 'medium'},

    # Both types (severe artifacts)
    'CD3': {'enabled': True, 'type1': True, 'type2': True, 'sensitivity': 'high'},
}
```

### Sensitivity Levels

| Sensitivity | Tile CV Threshold | Edge Gradient Threshold | When to Use |
|-------------|-------------------|-------------------------|-------------|
| **low**     | 0.30              | 0.25                    | Mild artifacts, abundant marker |
| **medium**  | 0.20              | 0.20                    | Moderate artifacts (default) |
| **high**    | 0.15              | 0.15                    | Severe artifacts, rare/dim marker |

**Higher sensitivity** = more aggressive correction (catches more tiles but may over-correct)

### How It Works

1. **Run AFTER** hierarchical normalization but **BEFORE** gating
2. For each marker with `enabled: True`:
   - **Type 1**: Calculate median intensity per tile → detect outlier tiles (z-score > 2.0) → normalize to global median
   - **Type 2**: Compare edge vs center intensities → correct edges if relative difference > threshold
3. Corrections are **capped** at 0.5x-2.0x to prevent over-correction
4. Prints diagnostic info showing which tiles/samples were corrected

### Output Example
```
MARKER-SPECIFIC TILE ARTIFACT CORRECTION
======================================================================

  AGFP (sensitivity=medium):
    Type 1: GUEST43 - corrected 12/45 tiles (CV=0.245)
    Type 1: GUEST47 - corrected 8/45 tiles (CV=0.212)

  CD3 (sensitivity=high):
    Type 1: GUEST43 - corrected 18/45 tiles (CV=0.168)
    Type 2: GUEST43 - corrected edges in 23/45 tiles

  ✓ Corrected: AGFP, PERK, KI67, CD3, CD8B
  ○ Skipped (working well): TOM, CD45
======================================================================
```

---

## 2. Hierarchical Marker Gating

### Configuration Location
Edit the `MARKER_HIERARCHY` dictionary in `manual_gating.py` (lines 100-106)

### Purpose
Ensures that **rare/functional markers** are only positive in cells that are positive for their **parent marker**.

### Use Cases

| Child Marker | Parent Marker | Biological Rationale |
|--------------|---------------|----------------------|
| FOXP3        | CD3           | FOXP3 (Treg marker) only in T cells (CD3+) |
| CD8B         | CD3           | CD8B only in T cells |
| GZMB         | CD45          | Granzyme B only in immune cells |
| PD1          | CD45          | PD1 immune checkpoint only in immune cells |
| Ki67         | CD45 or TOM   | Proliferation marker restricted to specific lineage |

### Configuration

```python
MARKER_HIERARCHY = {
    # Format: 'child_marker': 'parent_marker'
    'FOXP3': 'CD3',      # FOXP3+ cells must be CD3+
    'GZMB': 'CD45',      # GZMB+ cells must be CD45+
    'CD8B': 'CD3',       # CD8B+ cells must be CD3+
}
```

### How It Works

1. **Run AFTER** initial gating but **BEFORE** saving results
2. For each parent-child pair:
   - Identify cells that are `child+ but parent-` (biologically impossible)
   - Reclassify these cells as `child-` by setting intensity to `gate * 0.9`
3. Prints statistics showing how many cells were adjusted
4. **Does NOT change gate values**, only cell classifications

### Output Example
```
HIERARCHICAL MARKER GATING
======================================================================
  Enforcing parent-child relationships:
    FOXP3 ⊂ CD3: 1,245 cells adjusted (8.2% of FOXP3+ cells)
    CD8B ⊂ CD3: 892 cells adjusted (3.1% of CD8B+ cells)
    GZMB ⊂ CD45: 0 cells adjusted (0.0% of GZMB+ cells)
======================================================================
```

---

## 3. How to Use

### Step 1: Identify Problematic Markers

Run the pipeline once with default settings:
```bash
python manual_gating.py --results_dir results --n_jobs 16
```

Check diagnostic plots in `output/tile_artifact_diagnosis/`:
- Look for "checkerboard" patterns → Type 1 artifacts
- Look for grid lines at tile edges → Type 2 artifacts

### Step 2: Configure Tile Correction

Edit `TILE_ARTIFACT_CORRECTION` in `manual_gating.py`:
- Set `enabled: True` for problematic markers
- Choose Type 1, Type 2, or both
- Start with `sensitivity: 'medium'`, increase to `'high'` if needed

### Step 3: Configure Hierarchical Relationships

Edit `MARKER_HIERARCHY` in `manual_gating.py`:
- Add parent-child relationships based on biology
- Example: If you see FOXP3+ but CD3- cells (impossible), add `'FOXP3': 'CD3'`

### Step 4: Re-run Pipeline

Force re-normalization to apply tile corrections:
```bash
python manual_gating.py --results_dir results --force_normalization --n_jobs 16
```

Or skip normalization if you only changed hierarchical relationships:
```bash
python manual_gating.py --results_dir results --skip_normalization
```

### Step 5: Validate Results

Check:
1. **Spatial plots** - Tile artifacts should be reduced
2. **Hierarchical gating output** - See how many cells were adjusted
3. **Histograms** - Gates should be more accurate for corrected markers

---

## 4. Important Notes

### ✅ Safe for Working Markers
- Markers with `enabled: False` are **completely untouched**
- TOM, CD45, and other working markers remain unchanged
- Only specified markers are corrected

### ⚠️ Correction Limits
- Tile corrections are capped at 0.5x-2.0x to prevent over-correction
- Edge corrections only apply to outer 10% of tiles
- Corrections only trigger if artifacts exceed threshold

### 🔄 Pipeline Order
1. Load data
2. Hierarchical UniFORM normalization
3. **→ Per-marker tile artifact correction** (NEW)
4. Gating (density-based)
5. **→ Hierarchical gating enforcement** (NEW)
6. Save results

### 💾 Checkpoint Behavior
- Tile corrections run every time (after loading checkpoint)
- Use `--force_normalization` to re-run full pipeline with new tile correction settings
- Use `--skip_normalization` to only re-run gating with new hierarchical relationships

---

## 5. Troubleshooting

### Problem: Tile artifacts still visible after correction

**Solution**: Increase sensitivity from `'medium'` to `'high'`

### Problem: Over-correction (marker looks too flat)

**Solution**: Decrease sensitivity from `'medium'` to `'low'`, or disable specific type

### Problem: Child marker still has many parent-negative cells

**Solution**: Check if parent gate is too strict, or if biology is more complex (e.g., need multiple parents)

### Problem: Pipeline runs slow

**Solution**: Tile correction adds ~10-30 seconds per marker. Only enable for markers that need it.

---

## Contact

For issues or questions about this implementation, check the git commit history or ask the developer.
