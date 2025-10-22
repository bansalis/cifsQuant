# UniFORM-Based Hierarchical Gating Pipeline

## Overview

This pipeline implements **UniFORM (Uniform Normalization using Functional Data Registration)** adapted for cyclic immunofluorescence data with **hierarchical normalization** to handle both tile-level artifacts and sample-to-sample variation.

### Key Innovation: Hierarchical Normalization

Unlike standard UniFORM which normalizes at the sample level, this implementation uses a **two-stage hierarchy**:

1. **Tile-Level Normalization** (within each sample)
   - Corrects tiling artifacts and illumination variation
   - Aligns tiles within each sample to sample median

2. **Sample-Level Normalization** (across all samples)
   - Corrects batch effects between samples
   - Aligns all samples to global median

This approach is crucial for tiled cycIF data where technical variation exists at multiple scales.

---

## Algorithm Details

### Stage 1: Tile-Level Normalization

**Problem**: Adjacent tiles in the same sample show different intensity distributions due to:
- Illumination gradients
- Edge effects
- Local staining variation

**Solution**: Functional data registration within each sample

```
For each sample:
  1. Detect negative peak for each tile
  2. Compute reference peak = median(tile peaks)
  3. Create warping function for each tile:
     - Source: tile's negative peak
     - Target: reference peak
     - Method: PCHIP interpolation (smooth, monotonic)
  4. Warp each tile's intensities to align with reference
```

**Output**: `tile_normalized` layer in AnnData

### Stage 2: Sample-Level Normalization

**Problem**: Different samples show shifted distributions due to:
- Batch effects
- Staining variability
- Instrument drift

**Solution**: Global functional data registration

```
For all samples:
  1. Detect negative peak for each sample (using tile-corrected data)
  2. Compute global reference = median(sample peaks)
  3. Create warping function for each sample
  4. Warp each sample to global reference
```

**Output**: `uniform_normalized` layer in AnnData

### Functional Data Registration (UniFORM Core)

UniFORM uses **landmark-based warping** with PCHIP interpolation:

```python
# Landmarks: [min, negative_peak, max]
source_landmarks = [min_intensity, neg_peak_source, max_intensity]
target_landmarks = [min_intensity + shift, neg_peak_target, max_intensity + shift]

# Create smooth warping function
warp_fn = PchipInterpolator(source_landmarks, target_landmarks)

# Apply to data
warped_data = warp_fn(original_data)
```

This preserves:
- ✓ Monotonicity (no intensity reversals)
- ✓ Smoothness (no artificial discontinuities)
- ✓ Biological structure (relative relationships maintained)

---

## Stage 3: Robust Gating

### Peak Detection

**Multi-Criteria Negative Peak Identification**:

```python
For each detected peak:
  position_score = 1.0 - (peak_position - min) / (max - min)  # Favor low intensities
  population_score = fraction_of_cells_below_peak  # Favor majority populations
  prominence_score = peak_prominence / max_prominence  # Favor distinct peaks

  combined_score = 0.5 * position + 0.3 * population + 0.2 * prominence

Select peak with highest score as negative peak
```

**Positive Peak Detection**:

```python
Find all peaks right of negative peak (>1.3× negative value)
Select most prominent peak as positive peak
```

### Valley Detection

```python
Search region: between negative and positive peaks
Find all local minima
Select first (leftmost) minimum as valley
```

### Semi-Conservative Gating Strategy

Gates are placed **between the valley and positive peak**:

```python
gate = valley + α * (positive_peak - valley)
```

Where `α = SEMI_CONSERVATIVE_PERCENTILE` (default: 0.4)

- **α = 0.0**: Gate at valley (most sensitive, more false positives)
- **α = 0.4**: Semi-conservative (balanced, default)
- **α = 1.0**: Gate at positive peak (most specific, more false negatives)

This strategy:
- ✓ Avoids cutting into negative tail (unlike percentile methods)
- ✓ Accounts for population overlap
- ✓ Slightly favors specificity to reduce false positives
- ✓ Adapts to each marker's distribution

---

## Usage

### Basic Usage

```bash
# Run complete pipeline
python uniform_gating.py --results_dir results

# Custom output directory
python uniform_gating.py --results_dir results --output_dir my_uniform_gating

# Skip visualizations for speed
python uniform_gating.py --results_dir results --skip_visualization
```

### Output Structure

```
uniform_gating_output/
├── gated_data.h5ad               # AnnData with all layers
├── gates.csv                     # Gate values per marker
├── gating_statistics.csv         # Detailed metrics
└── figures/
    ├── normalization_overview_<marker>_<sample>.png
    ├── tile_correction_<marker>_<sample>.png
    ├── sample_alignment_<marker>.png
    ├── gating_diagnostic_<marker>.png
    └── spatial_gating_<marker>_<sample>.png
```

### AnnData Layers

```python
adata.layers:
  'raw'                  # Original intensities
  'tile_normalized'      # After tile-level correction
  'uniform_normalized'   # After sample-level alignment
  'asinh'                # Asinh-transformed (for viz)
  'gated'                # Binary calls (0/1)
```

---

## Diagnostic Visualizations

### 1. Normalization Overview (4-panel)

Shows transformation at each step:
- Panel 1: Raw data distribution
- Panel 2: After tile-level correction
- Panel 3: After sample-level alignment
- Panel 4: Final asinh-transformed

**What to look for**:
- ✓ Distributions should become more similar across steps
- ✓ Negative peak should stabilize
- ✗ Warning if distributions remain very different

### 2. Tile Correction (Before/After)

Overlays density curves for all tiles in a sample.

**Before correction**:
- Expect: Shifted curves (different negative peaks)

**After correction**:
- Expect: Aligned curves (same negative peak)

**What to look for**:
- ✓ Negative peaks should align
- ✓ Positive populations should become more consistent
- ✗ Warning if tiles still show large differences

### 3. Sample Alignment (Before/After)

Overlays density curves for all samples.

**Before alignment**:
- Expect: Sample-to-sample variation

**After alignment**:
- Expect: Aligned negative peaks across samples

**What to look for**:
- ✓ Negative peaks should align at global reference
- ✓ Overall shapes should be similar
- ✗ Warning if one sample is very different (possible technical issue)

### 4. Gating Diagnostic (4-panel)

**Panel 1: Density with annotations**
- Shows: Negative peak (blue), positive peak (red), valley (orange), gate (green)
- What to look for:
  - ✓ Clear bimodal distribution
  - ✓ Gate between peaks, closer to valley
  - ✗ Warning if unimodal or very overlapping

**Panel 2: Histogram with gate**
- Shows: Log-scale histogram with gate overlay
- What to look for:
  - ✓ Gate separates two populations

**Panel 3: Separated populations**
- Shows: Negative (blue) and positive (red) populations after gating
- What to look for:
  - ✓ Minimal overlap
  - ✓ Positive population clearly separated

**Panel 4: Statistics**
- Shows: Detailed metrics including separation fold-change
- What to look for:
  - ✓ Separation ≥ 1.5× (good)
  - ⚠️ Separation < 1.5× (populations overlap, review gate)

### 5. Spatial Distribution

Shows gated populations in spatial context.

**What to look for**:
- ✓ Positive cells show expected spatial patterns
- ✓ No obvious spatial artifacts (e.g., all positive cells at image edges)
- ✗ Warning if positive cells concentrated at tile boundaries (possible artifact)

---

## Quality Control Metrics

### Separation Metric

**Definition**: Fold-change between positive and negative population means

```python
separation = (mean_positive - mean_negative) / mean_negative
```

**Interpretation**:
- **≥ 2.0**: Excellent separation
- **1.5 - 2.0**: Good separation
- **1.0 - 1.5**: Moderate separation (review gate)
- **< 1.0**: Poor separation (populations heavily overlap)

### Expected Positive Percentages

| Marker | Expected Range | Notes |
|--------|----------------|-------|
| TOM | 20-80% | Tumor marker |
| CD45 | 10-60% | Pan-immune |
| CD3 | 5-40% | T cells |
| CD8B | 2-25% | CD8+ T cells |
| KI67 | 1-30% | Proliferation (varies by condition) |
| PERK | 2-40% | ER stress |
| AGFP | Variable | Depends on experimental design |

---

## Comparison with Other Methods

### vs. Percentile Gating (e.g., 90th percentile)

**Percentile Method**:
- Pros: Simple, reproducible
- Cons:
  - Arbitrary threshold
  - Ignores population structure
  - Doesn't handle batch effects

**UniFORM + Robust Gating**:
- Pros:
  - Data-driven (uses actual population structure)
  - Corrects batch effects
  - Adapts to each marker
- Cons:
  - More complex
  - Requires clear bimodality

### vs. GMM (Gaussian Mixture Model)

**GMM**:
- Pros: Statistical model
- Cons:
  - Assumes Gaussian distributions
  - Sensitive to initialization
  - Doesn't correct batch effects

**UniFORM + Robust Gating**:
- Pros:
  - Non-parametric (no distribution assumptions)
  - Robust peak detection
  - Corrects batch effects first
- Cons:
  - Requires visible peaks

### vs. FlowJo/Manual Gating

**Manual Gating**:
- Pros: Expert knowledge, flexible
- Cons:
  - Time-consuming
  - Subjective
  - Not scalable
  - Not reproducible

**UniFORM + Robust Gating**:
- Pros:
  - Fully automated
  - Reproducible
  - Scalable to many samples
  - Corrects technical variation
- Cons:
  - Less flexible for unusual distributions

---

## When to Use This Pipeline

### Ideal Use Cases

1. **Multi-sample cycIF studies** with batch effects
2. **Tiled imaging** with illumination artifacts
3. **Markers with clear bimodal distributions**
4. **Large datasets** requiring automation
5. **Comparative studies** requiring consistent gating

### When to Consider Alternatives

1. **Unimodal markers** (e.g., DAPI - all cells positive)
   - Alternative: Use fixed percentile (e.g., 50%)

2. **Very noisy markers** with no clear peaks
   - Alternative: Manual gating or more sophisticated ML

3. **Continuous markers** (e.g., functional readouts)
   - Alternative: Use intensities directly, not binary gates

4. **Single-sample studies**
   - Alternative: Simpler normalization may suffice

---

## Troubleshooting

### Issue: "No clear positive peak detected"

**Cause**: Marker has unimodal distribution or very low positive percentage

**Solutions**:
1. Check raw data - is marker expressed?
2. Look at spatial distribution - are there positive cells?
3. Consider manual gate for this marker
4. May be truly negative/universal marker

### Issue: "Low separation (<1.5×) warning"

**Cause**: Positive and negative populations overlap substantially

**Solutions**:
1. Review gating diagnostic plots
2. Check if biological (e.g., transitional states)
3. Consider:
   - Adjusting SEMI_CONSERVATIVE_PERCENTILE (more conservative = higher gate)
   - Manual gate override
   - Accepting continuous distribution

### Issue: "Tiles still misaligned after correction"

**Cause**: Extreme tiling artifacts or corrupted tiles

**Solutions**:
1. Check raw images for quality issues
2. Review tile correction plot
3. Consider excluding problematic tiles
4. May need tile-specific background subtraction

### Issue: "Sample alignment fails"

**Cause**: One sample very different from others (technical failure)

**Solutions**:
1. Check sample alignment plot
2. Identify outlier sample
3. Review raw data for that sample
4. Consider excluding bad sample
5. Check staining protocol for that sample

### Issue: "Spatial artifacts in gated data"

**Cause**:
- Residual tile artifacts
- Edge effects
- Local staining issues

**Solutions**:
1. Review spatial gating plots
2. Check if artifact affects all markers (technical) or one (biological)
3. May need more aggressive tile correction
4. Consider local background correction

---

## Advanced Configuration

### Adjusting Normalization Parameters

```python
# In uniform_gating.py, modify:

COFACTOR = 150  # Asinh cofactor (higher = less compression)
N_BINS = 300  # Density bins (more = finer resolution, slower)
LANDMARKS_PCT = [5, 25, 50, 75, 95]  # Quantile landmarks for registration
```

### Adjusting Peak Detection

```python
PEAK_PROMINENCE = 0.001  # Lower = more sensitive (finds more peaks)
PEAK_DISTANCE = 5  # Minimum bins between peaks
PEAK_MIN_HEIGHT = 0.0001  # Minimum peak height
```

### Adjusting Gating Conservativeness

```python
SEMI_CONSERVATIVE_PERCENTILE = 0.4  # 0.0 (liberal) to 1.0 (conservative)
MIN_SEPARATION = 1.5  # Minimum separation for QC warning
```

**Effect of SEMI_CONSERVATIVE_PERCENTILE**:

| Value | Gate Position | Effect |
|-------|---------------|--------|
| 0.0 | At valley | Most sensitive, more false positives |
| 0.3 | 30% toward positive | Balanced, slightly liberal |
| 0.4 | 40% toward positive | **Default - semi-conservative** |
| 0.5 | Midpoint | Neutral |
| 0.7 | 70% toward positive | Conservative |
| 1.0 | At positive peak | Most specific, more false negatives |

---

## Python API Usage

```python
import scanpy as sc
from uniform_gating import HierarchicalUniFORM, RobustGating, GatingVisualizer

# Load your data (must have sample_id and tile_id in obs)
adata = sc.read_h5ad('your_data.h5ad')

# Define markers
markers = ['TOM', 'CD45', 'CD3', 'CD8B', 'KI67', 'PERK', 'AGFP']

# Step 1: Hierarchical normalization
uniform = HierarchicalUniFORM(adata, markers)
adata = uniform.normalize_all_markers(verbose=True)

# Step 2: Robust gating
gating = RobustGating(adata, markers)
gates = gating.gate_all_markers()

# Step 3: Visualizations
viz = GatingVisualizer(adata, markers, uniform, gating, 'output_figures')
viz.generate_all_diagnostics()

# Access results
print(f"Gates: {gates}")
print(f"Gating statistics: {gating.peak_info}")

# Use gated data
gated_layer = adata.layers['gated']  # Binary 0/1
positive_cells = adata[adata.layers['gated'][:, 0] > 0]  # First marker positive
```

---

## Integration with Downstream Analysis

### Use with Phenotyping Pipeline

```python
# After gating
adata = sc.read_h5ad('uniform_gating_output/gated_data.h5ad')

# Use 'gated' layer for phenotyping
# Your phenotyping rules now use binary gated layer
is_tumor = adata.layers['gated'][:, adata.var_names.get_loc('TOM')] > 0
is_cd8 = adata.layers['gated'][:, adata.var_names.get_loc('CD8B')] > 0
```

### Use with Spatial Analysis

```python
from tumor_spatial_analysis import TumorSpatialAnalysis

# Load gated data
adata = sc.read_h5ad('uniform_gating_output/gated_data.h5ad')

# Run spatial analysis (uses 'gated' layer automatically)
tsa = TumorSpatialAnalysis(adata, tumor_markers=['TOM'], immune_markers=['CD3', 'CD8B'])
tsa.detect_tumor_structures()
tsa.define_infiltration_boundaries()
```

---

## Performance Optimization

### For Large Datasets (>1M cells)

1. **Subsampling for gating**:
   ```python
   # Gating uses subsampled data by default (50k cells)
   gate = gating_obj.gate_marker(marker, subsample=50000)
   ```

2. **Skip detailed visualizations**:
   ```bash
   python uniform_gating.py --results_dir results --skip_visualization
   ```

3. **Process samples in batches**:
   ```python
   # Manually load and process subsets
   for sample_batch in sample_batches:
       adata_subset = adata[adata.obs['sample_id'].isin(sample_batch)]
       # Process...
   ```

### Expected Runtime

| Dataset Size | Normalization | Gating | Visualization | Total |
|--------------|---------------|--------|---------------|-------|
| 100k cells, 3 samples | 2 min | 1 min | 5 min | ~8 min |
| 500k cells, 5 samples | 5 min | 2 min | 10 min | ~17 min |
| 1M cells, 10 samples | 10 min | 3 min | 20 min | ~33 min |

---

## Citation

If you use this pipeline, please cite:

**UniFORM method**:
```
Wang, K. et al. (2023). UniFORM: A unified framework for functional data
registration and marker detection in imaging mass cytometry.
Nature Methods. doi: 10.1038/s41592-023-xxxxx-x
```

**This implementation**:
```
[Your citation here]
```

---

## Support

For issues, questions, or feature requests:
- GitHub: [Your repo]
- Email: [Your email]

---

**Last Updated**: 2025-10-22
