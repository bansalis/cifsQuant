#!/usr/bin/env python3
"""
Manual Gating Pipeline with Config-Based Gates
Percentile normalization enables shared gates across samples
python manual_gating.py --results_dir results
"""

import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
# Add at top after imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no GUI needed
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ============================================================================
# CONFIGURATION
# ============================================================================

MARKERS = {
    'Channel_2': 'TOM', 
    'Channel_3': 'CD45',
    'Channel_4': 'AGFP',
    'Channel_6': 'PERK',
    'Channel_7': 'CD8B',
    'Channel_8': 'KI67',
    'Channel_10': 'CD3'
}

# GATE VALUES (normalized 0-1 scale)
# After 99th percentile normalization, gates can be shared across samples
# Set to None to auto-calculate suggestions

GATES = {
    'TOM': 0.228,#0.06,      # Example: manually set
    'CD45': 0.350,#0.32,
    'AGFP': 0.475,#0.60,    
    'PERK': 0.572,#0.80,
    'CD8B': 0.486,#0.60,
    'KI67': 0.175,#0.20,
    'CD3': 0.377#0.30
}

# Advanced options
USE_SHARED_GATES = True  # True = one gate per marker; False = per-sample gates
NORMALIZATION_METHOD = 'percentile_99'  # 'percentile_99' or 'zscore' or 'minmax'

# ============================================================================

def load_and_combine(results_dir):
    """Load all samples into single AnnData"""
    all_data = []
    
    for sample_dir in Path(results_dir).iterdir():
        if not sample_dir.is_dir():
            continue
            
        quant_file = sample_dir / "final" / "combined_quantification.csv"
        if not quant_file.exists():
            continue
            
        df = pd.read_csv(quant_file)
        df = df.rename(columns=MARKERS)
        df['sample_id'] = sample_dir.name
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    marker_cols = list(MARKERS.values())
    
    adata = ad.AnnData(X=combined[marker_cols].values, 
                       obs=combined.drop(columns=marker_cols))
    adata.var_names = marker_cols
    adata.obsm['spatial'] = combined[['X_centroid', 'Y_centroid']].values
    
    print(f"Loaded {len(adata):,} cells from {adata.obs['sample_id'].nunique()} samples")
    return adata

def correct_illumination(adata):
    """
    BaSiC illumination correction for cyclic IF.
    Must be done BEFORE outlier removal.
    Based on: Peng et al., Nat Commun 2017 (MCMICRO uses this)
    """
    from scipy.ndimage import gaussian_filter
    
    print("\nCorrecting illumination bias...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            coords = adata.obsm['spatial'][mask]
            
            # Create 2D intensity map by binning spatial coordinates
            x_bins = 50
            y_bins = 50
            
            x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), x_bins)
            y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), y_bins)
            
            # Calculate median intensity per spatial bin
            background_map = np.zeros((y_bins-1, x_bins-1))
            
            for i in range(len(x_edges)-1):
                for j in range(len(y_edges)-1):
                    bin_mask = ((coords[:, 0] >= x_edges[i]) & 
                               (coords[:, 0] < x_edges[i+1]) &
                               (coords[:, 1] >= y_edges[j]) & 
                               (coords[:, 1] < y_edges[j+1]))
                    
                    if bin_mask.sum() > 10:
                        # Use 25th percentile (robust to bright cells)
                        background_map[j, i] = np.percentile(vals[bin_mask], 25)
            
            # Smooth background map (large-scale illumination pattern)
            background_smooth = gaussian_filter(background_map, sigma=3)
            
            # Interpolate back to cell coordinates
            from scipy.interpolate import RectBivariateSpline
            
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            
            interp = RectBivariateSpline(y_centers, x_centers, background_smooth)
            background_at_cells = interp(coords[:, 1], coords[:, 0], grid=False)
            
            # Subtract background (additive correction)
            corrected = vals - background_at_cells + np.median(background_at_cells)
            corrected = np.maximum(corrected, 0)  # No negative values
            
            adata.X[mask, marker_idx] = corrected
            
            print(f"  {sample} {marker}: background range {background_at_cells.min():.0f}-{background_at_cells.max():.0f}")
    
    return adata

def normalize_tiles_by_background(adata):
    """
    Normalize each tile to have comparable background levels.
    Uses tile_y and tile_x from your segmentation output.
    """
    print("\nTile-based background normalization...")
    
    # Create unique tile identifier
    adata.obs['tile_id'] = adata.obs['tile_y'].astype(str) + '_' + adata.obs['tile_x'].astype(str)
    
    print(f"Found {adata.obs['tile_id'].nunique()} unique tiles")
    
    adata.layers['raw_prenorm'] = adata.X.copy()
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        # Identify background cells (low total intensity)
        total_intensity = adata.X.sum(axis=1)
        background_threshold = np.percentile(total_intensity, 20)
        is_background = total_intensity < background_threshold
        
        # Global background reference
        global_background = np.median(adata.X[is_background, marker_idx])
        print(f"  Global background: {global_background:.1f}")
        
        # Per-tile correction
        for tile_id in adata.obs['tile_id'].unique():
            tile_mask = adata.obs['tile_id'] == tile_id
            tile_bg_mask = tile_mask & is_background
            
            if tile_bg_mask.sum() < 10:
                continue
            
            tile_background = np.median(adata.X[tile_bg_mask, marker_idx])
            correction = global_background - tile_background
            
            adata.X[tile_mask, marker_idx] += correction
            adata.X[tile_mask, marker_idx] = np.maximum(adata.X[tile_mask, marker_idx], 0)
            
            n_cells = tile_mask.sum()
            print(f"  Tile {tile_id}: {n_cells:5,} cells, bg={tile_background:.1f} → {correction:+.1f}")
    
    return adata

def normalize_samples_after_tiles(adata):
    """
    After tile correction, normalize samples to each other.
    Uses same background-anchoring approach.
    """
    print("\nSample-level normalization...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        # Global background reference (all samples)
        total_intensity = adata.X.sum(axis=1)
        is_background = total_intensity < np.percentile(total_intensity, 20)
        global_background = np.median(adata.X[is_background, marker_idx])
        
        print(f"  Global background: {global_background:.1f}")
        
        # Normalize each sample
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            sample_bg_mask = sample_mask & is_background
            
            if sample_bg_mask.sum() < 50:
                continue
            
            sample_background = np.median(adata.X[sample_bg_mask, marker_idx])
            correction = global_background - sample_background
            
            adata.X[sample_mask, marker_idx] += correction
            adata.X[sample_mask, marker_idx] = np.maximum(adata.X[sample_mask, marker_idx], 0)
            
            print(f"  {sample}: bg={sample_background:.1f} → correction={correction:+.1f}")
    
    return adata

def normalize_data(adata, method='percentile_99'):
    """
    Per-marker 99th percentile normalization across ALL samples.
    Preserves relative marker intensities (CD3 dim stays dim, CD8B bright stays bright).
    """
    adata.layers['raw'] = adata.X.copy()
    adata.layers['normalized'] = adata.X.copy()
    
    if method == 'percentile_99':
        # Per-MARKER normalization across all samples
        for i in range(adata.n_vars):
            marker = adata.var_names[i]
            vals = adata.X[:, i]  # All cells, all samples for this marker
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) > 100:
                p99 = np.percentile(pos_vals, 99)
                if p99 > 0:
                    adata.layers['normalized'][:, i] = np.clip(vals / p99, 0, 1)
                    print(f"  {marker:8s}: p99={p99:.0f} → normalized to 0-1")
            else:
                print(f"  {marker:8s}: insufficient positive values, using raw")
        
        print("\n✓ Per-marker 99th percentile normalization")
        print("✓ Markers retain relative intensity differences")
        print("✓ Samples are now comparable for each marker")
    
    adata.X = adata.layers['normalized'].copy()
    return adata

def rolling_ball_background(adata, dim_markers=['CD3', 'CD8B', 'CD45', 'PERK'], 
                            radius=500, edge_buffer=25):
    """
    Rolling ball with edge-specific handling.
    """
    from scipy.ndimage import minimum_filter, maximum_filter, gaussian_filter
    
    print("\nRolling ball background subtraction...")
    
    for marker in dim_markers:
        if marker not in adata.var_names:
            continue
            
        marker_idx = adata.var_names.get_loc(marker)
        
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            coords = adata.obsm['spatial'][sample_mask]
            vals = adata.X[sample_mask, marker_idx]
            
            # Identify tile edges
            tile_boundaries = []
            for tile_id in adata.obs.loc[sample_mask, 'tile_id'].unique():
                tile_cells = adata.obs.loc[sample_mask & (adata.obs['tile_id'] == tile_id)]
                y_min, y_max = tile_cells['Y_centroid'].min(), tile_cells['Y_centroid'].max()
                x_min, x_max = tile_cells['X_centroid'].min(), tile_cells['X_centroid'].max()
                tile_boundaries.append((y_min, y_max, x_min, x_max))
            
            # Flag cells near tile edges
            is_edge_cell = np.zeros(len(vals), dtype=bool)
            for i, (y, x) in enumerate(coords):
                for y_min, y_max, x_min, x_max in tile_boundaries:
                    if (abs(y - y_min) < edge_buffer or abs(y - y_max) < edge_buffer or
                        abs(x - x_min) < edge_buffer or abs(x - x_max) < edge_buffer):
                        is_edge_cell[i] = True
                        break
            
            # Create background map
            grid_size = 50
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            x_bins = np.linspace(x_min, x_max, grid_size)
            y_bins = np.linspace(y_min, y_max, grid_size)
            
            intensity_grid = np.zeros((grid_size-1, grid_size-1))
            
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    mask = ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                           (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1]))
                    if mask.sum() > 0:
                        # Use 3rd percentile for edges (more conservative)
                        pct = 3 if is_edge_cell[mask].any() else 5
                        intensity_grid[j, i] = np.percentile(vals[mask], pct)
            
            # INCREASED smoothing for edge regions
            ball_radius = int(radius / ((x_max - x_min) / grid_size))
            
            # Apply rolling ball
            background = minimum_filter(intensity_grid, size=ball_radius)
            background = maximum_filter(background, size=ball_radius)
            
            # ADDITIONAL Gaussian smoothing (critical for edges)
            background = gaussian_filter(background, sigma=ball_radius*0.3)
            
            # Interpolate
            from scipy.interpolate import RectBivariateSpline
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            
            interp = RectBivariateSpline(y_centers, x_centers, background)
            background_at_cells = interp(coords[:, 1], coords[:, 0], grid=False)
            
            # Extra smoothing for edge cells specifically
            for i in np.where(is_edge_cell)[0]:
                # Average with neighbors
                distances = np.sqrt(((coords - coords[i])**2).sum(axis=1))
                neighbors = distances < edge_buffer
                background_at_cells[i] = np.median(background_at_cells[neighbors])
            
            # Subtract
            corrected = vals - background_at_cells
            corrected = np.maximum(corrected, 0)
            
            adata.X[sample_mask, marker_idx] = corrected
            
            print(f"  {sample} {marker}: {is_edge_cell.sum()} edge cells smoothed")
    
    return adata

def remove_tiling_artifacts(adata):
    """
    Remove tiling artifacts using spatial + intensity filters.
    Based on FLOWCUT (Aghaeepour et al. 2012) + spatial extension.
    """
    from scipy.stats import median_abs_deviation
    from scipy.spatial import cKDTree
    
    print("\nRemoving tiling artifacts...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        vals = adata.X[:, marker_idx]
        
        # Method 1: Intensity outliers (FLOWCUT standard)
        median = np.median(vals)
        mad = median_abs_deviation(vals)
        
        # Flag extreme outliers (>5 MAD)
        intensity_outliers = np.abs(vals - median) > 5 * mad
        
        # Method 2: Spatial outliers (NEW - for tiling artifacts)
        # Find cells with intensities very different from spatial neighbors
        coords = adata.obsm['spatial']
        tree = cKDTree(coords)
        
        spatial_outliers = np.zeros(len(vals), dtype=bool)
        
        for i in range(len(vals)):
            # Find 50 nearest neighbors
            distances, indices = tree.query(coords[i], k=51)  # k=51 includes self
            neighbors = indices[1:]  # Exclude self
            
            neighbor_vals = vals[neighbors]
            neighbor_median = np.median(neighbor_vals)
            neighbor_mad = median_abs_deviation(neighbor_vals)
            
            # Cell is outlier if intensity differs >3 MAD from neighbors
            if neighbor_mad > 0:
                if np.abs(vals[i] - neighbor_median) > 3 * neighbor_mad:
                    spatial_outliers[i] = True
        
        # Combine both outlier types
        all_outliers = intensity_outliers | spatial_outliers
        
        if all_outliers.sum() > 0:
            # Replace outliers with local median
            for i in np.where(all_outliers)[0]:
                distances, indices = tree.query(coords[i], k=51)
                neighbors = indices[1:]
                adata.X[i, marker_idx] = np.median(vals[neighbors])
            
            print(f"  {marker}: {all_outliers.sum():,} artifacts corrected "
                  f"({intensity_outliers.sum()} intensity + {spatial_outliers.sum()} spatial)")
    
    return adata

def stratified_subsample(adata, n_per_sample):
    """Subsample equal number of cells per sample"""
    indices = []
    for sample in adata.obs['sample_id'].unique():
        sample_idx = np.where(adata.obs['sample_id'] == sample)[0]
        if len(sample_idx) > n_per_sample:
            sample_idx = np.random.choice(sample_idx, n_per_sample, replace=False)
        indices.extend(sample_idx)
    return np.array(indices)

# Replace auto_suggest_gates function:
def auto_suggest_gates(adata):
    """Auto-suggest gates for all markers (for reference)"""
    suggestions = {}
    
    print("\n" + "="*70)
    print("AUTO-SUGGESTED GATES (for reference)")
    print("="*70)
    
    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)
        
        if USE_SHARED_GATES:
            vals = adata.X[:, marker_idx]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 100:
                suggestions[marker] = 0.15
                print(f"{marker:8s} | insufficient data → 0.150")
                continue
            
            # Multiple methods
            from scipy.signal import find_peaks
            hist, bins = np.histogram(pos_vals, bins=100)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Method 1: Valley between peaks
            peaks, _ = find_peaks(hist, distance=10)
            if len(peaks) >= 2:
                valley_region = hist[peaks[0]:peaks[1]]
                if len(valley_region) > 0:
                    valley_idx = peaks[0] + np.argmin(valley_region)
                    gate_valley = float(bin_centers[valley_idx])
                else:
                    gate_valley = float(np.percentile(pos_vals, 75))
            else:
                gate_valley = float(np.percentile(pos_vals, 75))
            
            # Method 2: Otsu
            from skimage.filters import threshold_otsu
            try:
                gate_otsu = float(threshold_otsu(pos_vals))
            except:
                gate_otsu = gate_valley
            
            # Method 3: Percentiles
            gate_p75 = float(np.percentile(pos_vals, 75))
            gate_p90 = float(np.percentile(pos_vals, 90))
            
            # Method 4: Rosin (maximum distance from peak-to-tail line)
            hist, bins = np.histogram(pos_vals, bins=256)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            peak_idx = np.argmax(hist)
            last_idx = len(hist) - 1
            while last_idx > peak_idx and hist[last_idx] < hist[peak_idx] * 0.01:
                last_idx -= 1

            if last_idx > peak_idx:
                x1, y1 = peak_idx, hist[peak_idx]
                x2, y2 = last_idx, hist[last_idx]
                
                distances = []
                for i in range(peak_idx, last_idx):
                    d = abs((y2-y1)*i - (x2-x1)*hist[i] + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    distances.append(d)
                
                if distances:
                    rosin_idx = peak_idx + np.argmax(distances)
                    gate_rosin = float(bin_centers[rosin_idx])
                else:
                    gate_rosin = gate_valley
            else:
                gate_rosin = gate_valley

            # Method 5: GMM (2-4 components, threshold between sorted means)
            best_gmm = None
            best_bic = np.inf
            best_n = 2

            for n_components in [2, 3, 4]:
                try:
                    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
                    gmm.fit(pos_vals.reshape(-1, 1))
                    bic = gmm.bic(pos_vals.reshape(-1, 1))
                    if bic < best_bic:
                        best_bic = bic
                        best_gmm = gmm
                        best_n = n_components
                except:
                    continue

            if best_gmm is not None:
                means = np.sort(best_gmm.means_.flatten())
                
                # Threshold between 2nd-3rd component (if 3+) or 3rd-4th (if 4)
                if best_n == 4 and len(means) >= 4:
                    gate_gmm = float((means[2] + means[3]) / 2)  # Between 3rd-4th
                elif best_n >= 3 and len(means) >= 3:
                    gate_gmm = float((means[1] + means[2]) / 2)  # Between 2nd-3rd
                else:
                    gate_gmm = float((means[0] + means[1]) / 2)  # Between 1st-2nd
            else:
                gate_gmm = gate_valley

            suggestions[marker] = {
                'valley': gate_valley,
                'otsu': gate_otsu,
                'rosin': gate_rosin,
                'gmm': gate_gmm,  # ADD THIS
                'p75': gate_p75,
                'p90': gate_p90,
                'recommended': gate_valley
            }

            manual = GATES.get(marker)
            if manual is not None:
                print(f"{marker:8s} | valley={gate_valley:.3f} otsu={gate_otsu:.3f} rosin={gate_rosin:.3f} "
                    f"gmm={gate_gmm:.3f} p75={gate_p75:.3f} p90={gate_p90:.3f} | USING: {manual:.3f} (manual)")
            else:
                print(f"{marker:8s} | valley={gate_valley:.3f} otsu={gate_otsu:.3f} rosin={gate_rosin:.3f} "
                    f"gmm={gate_gmm:.3f} p75={gate_p75:.3f} p90={gate_p90:.3f} | USING: {gate_valley:.3f} (auto)")

        else:
            suggestions[marker] = {}
            for sample in adata.obs['sample_id'].unique():
                mask = adata.obs['sample_id'] == sample
                vals = adata.X[mask, marker_idx]
                pos_vals = vals[vals > 0]
                suggestions[marker][sample] = float(np.percentile(pos_vals, 75)) if len(pos_vals) > 100 else 0.15
                print(f"{marker:8s} | {sample:12s} | suggested={suggestions[marker][sample]:.3f}")
    
    return suggestions

def finalize_gates(suggestions):
    """Combine manual gates with auto-suggestions"""
    final_gates = {}
    
    for marker in MARKERS.values():
        if GATES.get(marker) is not None:
            if USE_SHARED_GATES:
                final_gates[marker] = GATES[marker]
            else:
                final_gates[marker] = {sample: GATES[marker] 
                                       for sample in adata.obs['sample_id'].unique()}
        else:
            # Use recommended value from suggestions dict
            if USE_SHARED_GATES:
                final_gates[marker] = suggestions[marker]['recommended']
            else:
                final_gates[marker] = suggestions[marker]
    
    return final_gates

def apply_gates(adata, gates):
    """Apply gates to create binary layer"""
    print("\nApplying gates...")
    adata.layers['gated'] = np.zeros_like(adata.X)
    
    for i, marker in enumerate(adata.var_names):
        if USE_SHARED_GATES:
            gate = gates[marker]
            adata.layers['gated'][:, i] = (adata.X[:, i] > gate).astype(float)
        else:
            for sample in adata.obs['sample_id'].unique():
                mask = adata.obs['sample_id'] == sample
                gate = gates[marker][sample]
                adata.layers['gated'][mask, i] = (adata.X[mask, i] > gate).astype(float)
    
    print("\n" + "="*70)
    print("GATING SUMMARY")
    print("="*70)
    for marker in adata.var_names:
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            pos = (adata.layers['gated'][mask, adata.var_names.get_loc(marker)] > 0).sum()
            total = mask.sum()
            gate_val = gates[marker] if USE_SHARED_GATES else gates[marker][sample]
            print(f"{marker:8s} | {sample:12s} | gate={gate_val:.3f} | {pos:8,}/{total:8,} ({pos/total*100:5.1f}%)")
    print("="*70 + "\n")
    
    return adata

def create_validation_plots(adata, gates, output_dir):
    """Optimized validation plots with progress tracking"""
    plots_dir = Path(output_dir) / "gating_validation"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n[1/3] Creating histograms...")
    # 1. Histograms - ONE PLOT PER MARKER (not all samples in one figure)
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        
        for sample in adata.obs['sample_id'].unique():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'{marker} - {sample}', fontsize=12)
            
            mask = adata.obs['sample_id'] == sample
            norm_vals = adata.X[mask, adata.var_names.get_loc(marker)]
            gate = gates[marker] if USE_SHARED_GATES else gates[marker][sample]
            
            # Linear
            axes[0].hist(norm_vals[norm_vals > 0], bins=100, alpha=0.7, color='steelblue')
            axes[0].axvline(gate, color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Normalized (0-1)')
            axes[0].set_xlim(-0.05, 1.05)
            pos_pct = (norm_vals > gate).mean() * 100
            axes[0].text(0.95, 0.95, f'{pos_pct:.1f}%', transform=axes[0].transAxes, 
                        ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            
            # Log
            log_norm = np.log10(norm_vals[norm_vals > 0] + 0.001)
            axes[1].hist(log_norm, bins=100, alpha=0.7, color='steelblue')
            axes[1].axvline(np.log10(gate + 0.001), color='red', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Log10(Norm)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{marker}_{sample}_hist.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print("\n[2/3] Creating spatial plots...")
    # 2. Spatial - subsample to 100k per sample
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        
        for sample in adata.obs['sample_id'].unique():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            # SUBSAMPLE
            if len(sample_data) > 100000:
                idx = np.random.choice(len(sample_data), 100000, replace=False)
                sample_data = sample_data[idx]
            
            norm_vals = sample_data.X[:, adata.var_names.get_loc(marker)]
            gate = gates[marker] if USE_SHARED_GATES else gates[marker][sample]
            pos_mask = norm_vals > gate
            
            ax.scatter(sample_data.obsm['spatial'][:, 0], 
                      sample_data.obsm['spatial'][:, 1],
                      s=0.3, alpha=0.3, c='lightgray', rasterized=True)
            ax.scatter(sample_data.obsm['spatial'][pos_mask, 0],
                      sample_data.obsm['spatial'][pos_mask, 1],
                      s=0.5, alpha=0.8, c='red', rasterized=True)
            ax.set_title(f'{marker} - {sample}\n{pos_mask.sum():,} pos ({pos_mask.mean()*100:.1f}%)')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{marker}_{sample}_spatial.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print("\n[3/3] Creating 2D FACS plots...")
    # 3. FACS plots - stratified subsample
    pairs = [('CD3', 'CD8B'), ('CD45', 'CD3'), ('TOM', 'AGFP'), ('CD3', 'KI67')]
    n_samples = adata.obs['sample_id'].nunique()
    target_per_sample = 750000 // n_samples
    
    for pair_idx, pair in enumerate(pairs):
        if not all(m in adata.var_names for m in pair):
            continue
        print(f"  {pair_idx+1}/{len(pairs)}: {pair[0]} vs {pair[1]}")
        
        for sample in adata.obs['sample_id'].unique():
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            mask = adata.obs['sample_id'] == sample
            x_idx = adata.var_names.get_loc(pair[0])
            y_idx = adata.var_names.get_loc(pair[1])
            
            x_vals = adata.X[mask, x_idx]
            y_vals = adata.X[mask, y_idx]
            
            # SUBSAMPLE
            if len(x_vals) > target_per_sample:
                idx = np.random.choice(len(x_vals), target_per_sample, replace=False)
                x_vals, y_vals = x_vals[idx], y_vals[idx]
            
            gate_x = gates[pair[0]] if USE_SHARED_GATES else gates[pair[0]][sample]
            gate_y = gates[pair[1]] if USE_SHARED_GATES else gates[pair[1]][sample]
            
            ax.scatter(x_vals, y_vals, s=0.5, alpha=0.3, c='gray', rasterized=True)
            ax.axvline(gate_x, color='red', linestyle='--', linewidth=1)
            ax.axhline(gate_y, color='red', linestyle='--', linewidth=1)
            
            q1 = ((x_vals > gate_x) & (y_vals > gate_y)).mean() * 100
            ax.text(0.95, 0.95, f'{q1:.1f}%', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=12, color='red')
            ax.set_xlabel(f'{pair[0]}')
            ax.set_ylabel(f'{pair[1]}')
            ax.set_title(f'{sample}')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{pair[0]}_vs_{pair[1]}_{sample}.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Validation plots saved to {plots_dir}")

def create_spatial_fluorescence_plots(adata, gates, output_dir):
    """Optimized spatial fluorescence with progress"""
    plots_dir = Path(output_dir) / "spatial_fluorescence"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating spatial fluorescence plots...")
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        marker_idx_loc = adata.var_names.get_loc(marker)
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            # SUBSAMPLE
            if len(sample_data) > 100000:
                idx = np.random.choice(len(sample_data), 100000, replace=False)
                sample_data = sample_data[idx]
            
            raw_vals = sample_data.layers['raw'][:, marker_idx_loc]
            norm_vals = sample_data.X[:, marker_idx_loc]
            gate = gates[marker] if USE_SHARED_GATES else gates[marker][sample]
            gated_vals = (norm_vals > gate).astype(int)
            
            x_coords = sample_data.obsm['spatial'][:, 0]
            y_coords = sample_data.obsm['spatial'][:, 1]
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'{sample} - {marker}', fontsize=14)
            
            axes[0].scatter(x_coords, y_coords, c=raw_vals, s=0.5, alpha=0.8, 
                           cmap='viridis', rasterized=True)
            axes[0].set_title('Raw')
            axes[0].set_aspect('equal')
            plt.colorbar(axes[0].collections[0], ax=axes[0])
            
            axes[1].scatter(x_coords, y_coords, c=norm_vals, s=0.5, alpha=0.8,
                           cmap='viridis', vmin=0, vmax=1, rasterized=True)
            axes[1].set_title(f'Normalized (gate={gate:.3f})')
            axes[1].set_aspect('equal')
            plt.colorbar(axes[1].collections[0], ax=axes[1])
            
            axes[2].scatter(x_coords, y_coords, c=gated_vals, s=0.5, alpha=0.8,
                           cmap='RdGy_r', vmin=0, vmax=1, rasterized=True)
            axes[2].set_title(f'{gated_vals.sum():,} pos ({gated_vals.mean()*100:.1f}%)')
            axes[2].set_aspect('equal')
            plt.colorbar(axes[2].collections[0], ax=axes[2], ticks=[0, 1])
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{sample}_{marker}_spatial.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Spatial fluorescence saved to {plots_dir}")

def create_intensity_scatterplots(adata, gates, output_dir):
    """Optimized intensity scatterplots with progress"""
    plots_dir = Path(output_dir) / "intensity_scatterplots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    pairs = [('CD3', 'CD8B'), ('CD45', 'CD3'), ('TOM', 'AGFP'), ('TOM', 'PERK'), ('CD3', 'KI67')]
    
    print("\nCreating intensity scatterplots...")
    for pair_idx, pair in enumerate(pairs):
        if not all(m in adata.var_names for m in pair):
            continue
        print(f"  {pair_idx+1}/{len(pairs)}: {pair[0]} vs {pair[1]}")
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            # SUBSAMPLE
            if len(sample_data) > 50000:
                idx = np.random.choice(len(sample_data), 50000, replace=False)
                sample_data = sample_data[idx]
            
            x_idx = adata.var_names.get_loc(pair[0])
            y_idx = adata.var_names.get_loc(pair[1])
            
            raw_x = sample_data.layers['raw'][:, x_idx]
            raw_y = sample_data.layers['raw'][:, y_idx]
            norm_x = sample_data.X[:, x_idx]
            norm_y = sample_data.X[:, y_idx]
            
            gate_x = gates[pair[0]] if USE_SHARED_GATES else gates[pair[0]][sample]
            gate_y = gates[pair[1]] if USE_SHARED_GATES else gates[pair[1]][sample]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{sample}: {pair[0]} vs {pair[1]}', fontsize=12)
            
            # Raw
            axes[0].scatter(raw_x, raw_y, s=0.5, alpha=0.3, c='gray', rasterized=True)
            axes[0].set_title('Raw')
            
            # Normalized linear
            axes[1].scatter(norm_x, norm_y, s=0.5, alpha=0.3, c='gray', rasterized=True)
            axes[1].axvline(gate_x, color='red', linestyle='--')
            axes[1].axhline(gate_y, color='red', linestyle='--')
            axes[1].set_xlim(-0.05, 1.05)
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].set_title('Normalized')
            
            # Normalized log
            axes[2].scatter(np.log10(norm_x + 0.001), np.log10(norm_y + 0.001), 
                           s=0.5, alpha=0.3, c='gray', rasterized=True)
            axes[2].axvline(np.log10(gate_x + 0.001), color='red', linestyle='--')
            axes[2].axhline(np.log10(gate_y + 0.001), color='red', linestyle='--')
            axes[2].set_title('Log')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{sample}_{pair[0]}_vs_{pair[1]}.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Intensity scatterplots saved to {plots_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='manual_gating_output')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("MANUAL GATING PIPELINE")
    print("="*70)
    print(f"Normalization: {NORMALIZATION_METHOD}")
    print(f"Shared gates: {USE_SHARED_GATES}")
    
    global adata
    adata = load_and_combine(args.results_dir)
    adata = normalize_tiles_by_background(adata)
    adata = normalize_samples_after_tiles(adata)
    #adata = rolling_ball_background(adata, 
    #                                 dim_markers=['CD3', 'CD8B', 'CD45', 'PERK'])
    ####adata = remove_tiling_artifacts(adata)
    adata = normalize_data(adata, method=NORMALIZATION_METHOD)
    
    suggestions = auto_suggest_gates(adata)
    gates = finalize_gates(suggestions)
    
    # Save gates
    with open(output_dir / 'gates.json', 'w') as f:
        json.dump(gates, f, indent=2)
    
    adata = apply_gates(adata, gates)
    create_validation_plots(adata, gates, output_dir)
    create_spatial_fluorescence_plots(adata, gates, output_dir)
    create_intensity_scatterplots(adata, gates, output_dir)  
    adata.write(output_dir / 'gated_data.h5ad')
    
    print(f"\n✅ Complete! Output: {output_dir}")
    print(f"   - gated_data.h5ad: gated binary layer")
    print(f"   - gates.json: gate values")
    print(f"   - gating_validation/: plots")

if __name__ == '__main__':
    main()