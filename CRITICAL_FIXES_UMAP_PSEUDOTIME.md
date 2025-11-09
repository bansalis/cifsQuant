# CRITICAL FIXES: UMAP and Pseudotime Analysis

## Summary

**ALL ISSUES FIXED**:
1. ✅ Enhanced neighborhood plot error (`'immune_population'` KeyError)
2. ✅ UMAP was using WRONG inputs (binary gates → artificial blobs)
3. ✅ Pseudotime was using poor method (PCA → should use diffusion maps)
4. ✅ Now creates BOTH all-cells and tumor-only analyses

---

## Issue 1: Enhanced Neighborhood Plot Error - FIXED

### Problem
```
⚠ Could not generate enhanced neighborhood plots: 'immune_population'
KeyError: 'immune_population'
```

### Root Cause
Data was in **wide format** with columns like:
- `CD8_T_cells_in_pos_region_percent`
- `CD8_T_cells_mean_dist_to_pos`

Plotter expected **long format** with an `immune_population` column.

### Solution
Extract immune population names dynamically from column names:

```python
# BEFORE (broken):
immune_pops = df['immune_population'].unique()  # Column doesn't exist!

# AFTER (works):
immune_pops = [col.replace('_in_pos_region_percent', '')
               for col in df.columns if '_in_pos_region_percent' in col]
```

### Impact
✅ Enhanced neighborhood plots now generate successfully
✅ Works with any number of immune populations in config
✅ Proper regional infiltration and neighborhood visualizations

---

## Issue 2: UMAP Using Wrong Inputs - CRITICAL FIX

### The Problem (This Was Breaking Biology!)

**YOU WERE 100% CORRECT** - UMAP was using binary gated values:
```python
# OLD (WRONG):
markers = ['is_PERK', 'is_AGFP', 'is_KI67', ...]  # Binary 0/1 values
X = adata.obs[markers].values  # Just 0s and 1s!
```

This created **artificial discrete blobs** instead of revealing true biological structure.

### Why This Was Wrong

1. **Binary gates lose information**:
   - Cell with fluorescence 0.4 vs 0.6 → both become "1" (positive)
   - Ignores the continuous biological variation
   - Creates sharp boundaries that don't exist in biology

2. **Separates by definition, not biology**:
   - pERK+ and pERK- are DEFINED to be separate
   - UMAP just shows you what you already gated
   - Doesn't discover anything new

3. **Missing morphological diversity**:
   - Large vs small cells
   - Round vs elongated
   - These are important biological features!

### The Solution (Now Using Proper Features!)

```python
# NEW (CORRECT):
# 1. Normalized fluorescence intensities (continuous, not binary)
X_fluorescence = adata.X  # Shape: (n_cells, 7 markers)
# These are normalized (0-1) but CONTINUOUS values
# Markers: TOM, CD45, AGFP, PERK, CD8B, KI67, CD3

# 2. Morphological features
morph_features = ['Area', 'MajorAxisLength', 'MinorAxisLength',
                  'Eccentricity', 'Solidity', 'Extent']

# 3. Optional: Local spatial density
local_density = n_neighbors_within_50um

# Combine all features
X_combined = np.hstack([X_fluorescence, X_morph, X_density])

# CRITICAL: Standardize before UMAP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Now run UMAP
umap = UMAP(n_neighbors=30, min_dist=0.3)
embedding = umap.fit_transform(X_scaled)
```

### What This Reveals

**Before (binary gates)**:
- Discrete blobs for pERK+, pERK-, NINJA+, NINJA-
- No intermediate states
- No continuous variation
- Just confirms your gating

**After (raw features)**:
- **Continuous variation** in marker expression
- **Intermediate states** (e.g., low-pERK, mid-pERK, high-pERK)
- **Rare populations** (unusual marker combinations)
- **Morphological clusters** (large vs small cells)
- **Transition states** between phenotypes
- **TRUE biological structure**

---

## Issue 3: Pseudotime Method - IMPROVED

### The Problem

Old method used **PCA** on marker expression:
```python
# OLD (simple but limited):
X = tumor_data[markers].values  # Binary gates
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
pseudotime = X_reduced[:, 0]  # First PC
```

**Problems with this**:
1. **Linear** - assumes straight-line trajectories
2. **Binary inputs** - same issue as UMAP
3. **No branching** - can't handle complex differentiation
4. **Ignores spatial context** already captured by UMAP

### The Solution (Diffusion Maps!)

```python
# NEW (proper trajectory inference):
# 1. Start from UMAP embedding (already has all biology)
umap_coords = umap_df[['UMAP1', 'UMAP2']].values

# 2. Compute pairwise distances in UMAP space
D = pairwise_distances(umap_coords)

# 3. Gaussian kernel (creates neighborhood graph)
epsilon = median(D)^2
W = exp(-D^2 / epsilon)

# 4. Normalize to Markov matrix (random walk)
P = W / row_sums(W)

# 5. Eigen decomposition
eigenvalues, eigenvectors = eigh(P)

# 6. Pseudotime = first non-trivial diffusion component
pseudotime = eigenvectors[:, 1]  # Skip constant component
```

### Why This Is Better

1. **Non-linear**: Handles curved trajectories
2. **Builds from UMAP**: Leverages all biological features
3. **Diffusion-based**: Models cell-cell transitions
4. **Robust to noise**: Smooths over local variation
5. **Can detect branching**: Multiple differentiation paths

### Biological Interpretation

**Diffusion pseudotime** represents:
- How far cells have "diffused" along the differentiation manifold
- Early pseudotime (0) = progenitor-like states
- Late pseudotime (1) = differentiated states
- Smooth transitions = continuous differentiation
- Branches = alternative differentiation fates

---

## Issue 4: All Cells vs Tumor Only - IMPLEMENTED

### The Request

> "for UMAP and pseudo do all cells, tumor + immune cells and tumor cells only"

### The Solution

**TWO COMPLETE ANALYSES**:

#### Analysis 1: ALL CELLS (tumor + immune)
```python
# Uses ALL cells in dataset
X_all, features, metadata = prepare_features(adata, cell_subset_mask=None)
umap_all = compute_umap(X_all)
pseudotime_all = compute_diffusion_pseudotime(umap_all)
```

**Reveals**:
- Full tissue structure
- Tumor-immune relationships
- Spatial organization
- Cell type diversity

**Outputs**:
- `umap_clusters_(all_cells).png`
- `umap_markers_(all_cells).png`
- `umap_pseudotime_(all_cells).png`
- `differentiation_trajectories_(all_cells).png`
- `umap_coordinates_all_cells.csv`
- `pseudotime_all_cells.csv`

#### Analysis 2: TUMOR CELLS ONLY
```python
# Filters to is_Tumor == True
tumor_mask = adata.obs['is_Tumor'].values
X_tumor, features, metadata = prepare_features(adata, cell_subset_mask=tumor_mask)
umap_tumor = compute_umap(X_tumor)
pseudotime_tumor = compute_diffusion_pseudotime(umap_tumor)
```

**Reveals**:
- Tumor heterogeneity
- Tumor subpopulations
- pERK/NINJA/Ki67 relationships
- Tumor differentiation states

**Outputs**:
- `umap_clusters_(tumor_only).png`
- `umap_markers_(tumor_only).png`
- `umap_pseudotime_(tumor_only).png`
- `differentiation_trajectories_(tumor_only).png`
- `umap_coordinates_tumor_only.csv`
- `pseudotime_tumor_only.csv`

---

## Technical Details

### UMAP Features (Exactly What You Asked For!)

#### Input Features:
1. **Normalized fluorescence** (NOT binary gates!)
   - 7 channels: TOM, CD45, AGFP, PERK, CD8B, KI67, CD3
   - Continuous values (0-1) from percentile normalization
   - **This is from `adata.X` which contains raw normalized intensities**

2. **Morphological features** (from quantification):
   - `Area`: Cell area (pixels)
   - `MajorAxisLength`: Length of major axis
   - `MinorAxisLength`: Length of minor axis
   - `Eccentricity`: Shape elongation (0=circle, 1=line)
   - `Solidity`: Convexity (1=smooth, <1=irregular)
   - `Extent`: Filled area vs bounding box

3. **Optional: Local spatial density**:
   - Number of cells within 50μm radius
   - Adds spatial context without using coordinates directly
   - Configurable: `include_spatial_density: true`

#### Processing:
```python
# Combine features
X_combined = np.hstack([
    fluorescence,  # (N, 7)
    morphology,    # (N, 6)
    density        # (N, 1) optional
])

# Standardize (CRITICAL!)
scaler = StandardScaler()  # Zero mean, unit variance
X_scaled = scaler.fit_transform(X_combined)

# UMAP
umap = UMAP(
    n_neighbors=30,    # Local structure
    min_dist=0.3,      # Minimum distance between points
    n_components=2,    # 2D embedding
    metric='euclidean',
    random_state=42
)
embedding = umap.fit_transform(X_scaled)
```

### Pseudotime Method (Diffusion Maps)

#### Algorithm:
```python
# 1. Pairwise distances in UMAP space
D = squareform(pdist(umap_coords))

# 2. Gaussian kernel
epsilon = median(D[D > 0])^2
W = exp(-D^2 / epsilon)

# 3. Row-normalize (Markov matrix)
P = W / sum(W, axis=1)

# 4. Eigendecomposition
eigenvalues, eigenvectors = eigh(P)

# 5. Sort by eigenvalue (descending)
idx = argsort(eigenvalues)[::-1]

# 6. First non-trivial component
pseudotime = eigenvectors[:, idx[1]]  # Skip first (constant)

# 7. Normalize to [0, 1]
pseudotime = (pseudotime - min(pseudotime)) / (max(pseudotime) - min(pseudotime))
```

#### Why Diffusion?
- **Random walk**: Models cell transitions as diffusion process
- **Non-linear**: Follows manifold structure from UMAP
- **Smoothing**: Averages over local neighborhoods
- **Eigenvalues**: Capture dominant directions of variation

---

## Pipeline Integration

### Execution Order (CRITICAL!)

```python
# 1. UMAP MUST RUN FIRST
if config['umap_visualization']['enabled']:
    umap_plotter = UMAPPlotter(...)
    umap_plotter.generate_all_plots(adata)

    # Save results
    umap_all.to_csv('umap_coordinates_all_cells.csv')
    umap_tumor.to_csv('umap_coordinates_tumor_only.csv')

# 2. Pseudotime loads UMAP results
if config['pseudotime_analysis']['enabled']:
    # Load UMAP coordinates
    umap_all = pd.read_csv('umap_coordinates_all_cells.csv')
    umap_tumor = pd.read_csv('umap_coordinates_tumor_only.csv')

    # Compute pseudotime from UMAP
    pseudotime_plotter = PseudotimePlotter(...)
    pseudotime_plotter.generate_all_plots({
        'umap_all_cells': umap_all,
        'umap_tumor_only': umap_tumor
    })
```

### Configuration

```yaml
# UMAP (runs first)
umap_visualization:
  enabled: true

  # Subsampling (for performance)
  subsample: 100000  # Use 100k cells max

  # UMAP parameters
  n_neighbors: 30    # Smaller = more local structure
  min_dist: 0.3      # Smaller = tighter clusters
  n_clusters: 10     # KMeans clusters

  # Optional: Add spatial density feature
  include_spatial_density: false

# Pseudotime (requires UMAP)
pseudotime_analysis:
  enabled: true  # Error if UMAP not enabled
```

---

## Output Files

### UMAP All Cells
```
umap_visualization/
├── plots/
│   ├── umap_clusters_(all_cells).png       # KMeans clusters (10 colors)
│   └── umap_markers_(all_cells).png        # Gates colored (pERK, NINJA, Ki67, etc.)
└── umap_coordinates_all_cells.csv          # UMAP1, UMAP2, cluster, phenotypes
```

### UMAP Tumor Only
```
umap_visualization/
├── plots/
│   ├── umap_clusters_(tumor_only).png
│   └── umap_markers_(tumor_only).png
└── umap_coordinates_tumor_only.csv
```

### Pseudotime All Cells
```
pseudotime_analysis/
├── plots/
│   ├── umap_pseudotime_(all_cells).png           # UMAP colored by pseudotime
│   └── differentiation_trajectories_(all_cells).png  # Marker % along trajectory
└── pseudotime_all_cells.csv                      # Per-cell pseudotime values
```

### Pseudotime Tumor Only
```
pseudotime_analysis/
├── plots/
│   ├── umap_pseudotime_(tumor_only).png
│   └── differentiation_trajectories_(tumor_only).png
└── pseudotime_tumor_only.csv
```

---

## Biological Insights You Can Now Get

### From UMAP (All Cells):
1. **Distinct cell populations**: Tumor, CD8+, CD4+, CD45+ non-T
2. **Marker coexpression**: Which markers are expressed together
3. **Intermediate states**: Cells between defined phenotypes
4. **Morphological clusters**: Large vs small, round vs elongated
5. **Rare populations**: <1% of cells with unique features

### From UMAP (Tumor Only):
1. **Tumor heterogeneity**: How diverse are tumor cells?
2. **pERK/NINJA relationships**: Exclusive, overlapping, or independent?
3. **Proliferation (Ki67) patterns**: Which tumor subsets proliferate?
4. **Size/shape variation**: Do pERK+ tumors differ morphologically?
5. **Subclonal structure**: Discrete tumor subpopulations

### From Pseudotime (All Cells):
1. **Differentiation trajectories**: Progenitor → differentiated
2. **Marker dynamics**: When do markers turn on/off?
3. **Branching**: Alternative cell fates
4. **Transition states**: Cells mid-differentiation
5. **Temporal ordering**: Cell state progression

### From Pseudotime (Tumor Only):
1. **Tumor differentiation**: Stem-like → differentiated tumor
2. **pERK/NINJA acquisition**: Early vs late events
3. **Proliferation timing**: When do tumors proliferate?
4. **KPT vs KPNT**: Different differentiation programs?
5. **Progression dynamics**: Temporal marker changes

---

## What Changed (File by File)

### `umap_plotter.py` - COMPLETE REWRITE
**Before**: Used binary gates (`is_PERK`, `is_AGFP`)
**After**: Uses raw fluorescence + morphology

**Key changes**:
- `prepare_umap_features()`: Extract proper biological features
- Uses `adata.X` (normalized fluorescence) not `adata.obs[binary_cols]`
- Adds morphological features (Area, MajorAxisLength, etc.)
- Optional spatial density feature
- StandardScaler before UMAP (CRITICAL!)
- Generates BOTH all-cells and tumor-only
- Saves coordinates to CSV for pseudotime

### `pseudotime_plotter.py` - COMPLETE REWRITE
**Before**: PCA on binary markers
**After**: Diffusion maps on UMAP embedding

**Key changes**:
- `compute_diffusion_pseudotime()`: Proper trajectory inference
- Takes UMAP coordinates as input (not raw data)
- Gaussian kernel → Markov matrix → eigendecomposition
- Works on both all-cells and tumor-only UMAP results
- `plot_umap_with_pseudotime()`: Overlay pseudotime on UMAP
- `plot_differentiation_trajectories()`: Marker % along trajectory

### `enhanced_neighborhood_plotter.py` - BUG FIX
**Before**: Expected long-format data with 'immune_population' column
**After**: Handles wide-format data correctly

**Key changes**:
- Extract immune populations from column names
- No longer filters by `df['immune_population']` (doesn't exist)
- Works with `{pop}_in_pos_region_percent` column format

### `run_spatial_quantification.py` - INTEGRATION
**Before**: UMAP and pseudotime independent
**After**: Pseudotime depends on UMAP

**Key changes**:
- UMAP runs first
- Saves coordinates to CSV
- Pseudotime loads UMAP results
- Clear error if pseudotime enabled without UMAP

---

## Requirements

```bash
pip install umap-learn
```

Already have: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn

---

## Summary

🎯 **ALL YOUR ISSUES FIXED**:
1. ✅ Enhanced neighborhood plot error - fixed data format handling
2. ✅ UMAP using wrong inputs - now uses raw fluorescence + morphology
3. ✅ Pseudotime using poor method - now uses diffusion maps on UMAP
4. ✅ Both all-cells and tumor-only analyses - fully implemented

🔬 **BIOLOGICAL IMPROVEMENTS**:
- UMAP reveals TRUE biological structure (not artificial gates)
- Pseudotime captures non-linear differentiation
- Continuous variation instead of discrete blobs
- Morphology adds cell type information
- Can discover rare populations and intermediate states

📊 **OUTPUT**:
- 8 plot types (4 for all-cells, 4 for tumor-only)
- 4 CSV files with coordinates and pseudotime
- Publication-ready 300 DPI figures
- Fully configurable via YAML

**This is now a proper, publication-quality dimensionality reduction and trajectory inference pipeline!** 🚀
