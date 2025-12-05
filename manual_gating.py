#!/usr/bin/env python3
"""
Manual Gating Pipeline with Config-Based Gates
Percentile normalization enables shared gates across samples

slow complete
python manual_gating.py --results_dir results --n_jobs 16

Or explicitly skip normalization
python manual_gating.py --results_dir results --skip_normalization

rerun all
python manual_gating.py --results_dir results --force_normalization --n_jobs 15
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
import argparse
from scipy.signal import argrelextrema

# ============================================================================
# CONFIGURATION
# ============================================================================

MARKERS = {
    'R1.0.1_CY3': 'TOM',
  'R1.0.4_CY5_CD45': 'CD45',
  'R1.0.4_CY7_AGFP': 'NINJA',
  'R2.0.4_CY3_PERK': 'PERK',
  'R2.0.4_CY5_CD4': 'CD4',
  'R2.0.4_CY7_EPCAM': 'EPCAM',
  'R2.0.4_FITC_B220': 'B220',
  'R3.0.4_CY3_CD3E': 'CD3E',
  'R3.0.4_CY5_F480': 'F480',
  'R3.0.4_CY7_TTF1': 'TTF1',
  'R4.0.4_CY3_PD': 'PDL1',
  'R4.0.4_CY5_CD8A': 'CD8A',
  'R4.0.4_FITC_ASMA': 'ASMA',
  'R5.0.4_CY3_GZMB': 'GZMB',
  'R5.0.4_CY5_KLRG1': 'KLRG1',
  'R5.0.4_FITC_FOXP3': 'FOXP3',
  'R6.0.4_CY3_PD1': 'PD1',
  'R6.0.4_CY5_NAK': 'NAK',
  'R6.0.4_CY7_KI67': 'KI67',
  'R6.0.4_FITC_MHCII': 'MHCII',
  'R7.0.4_CY3_BCL6': 'BCL6',
  'R7.0.4_CY5_CC3': 'CC3',
  'R7.0.4_FITC_CD103': 'CD103'
}

# MARKER HIERARCHY for rare marker validation
# Common markers are abundant phenotypes that should have HIGH positive %
# Rare markers are subsets/functional markers that should have LOWER positive % than common markers
OLD_MARKER_HIERARCHY = {
    # Common markers (high abundance expected)
    'common': ['CD45', 'TOM', 'EPCAM', 'CD3E'],

    # Rare/functional markers that must have lower % than common markers
    'rare': ['GZMB', 'FOXP3', 'KLRG1', 'PD1', 'BCL6', 'CC3', 'PDL1'],

    # Intermediate markers (not strictly enforced)
    'intermediate': ['CD4', 'CD8A', 'B220', 'F480', 'TTF1', 'ASMA', 'MHCII', 'NINJA', 'PERK', 'CD103', 'NAK', 'KI67']
}

# GATE VALUES (normalized 0-1 scale)
# After 99th percentile normalization, gates can be shared across samples
# Set to None to auto-calculate suggestions

GATES = {
    'TOM': None,
  'CD45': None,
  'NINJA': None,
  'PERK': None,
  'CD4': None,
  'EPCAM': None,
  'B220': None,
  'CD3E': None,
  'F480': None,
  'TTF1': None,
  'PDL1': None,
  'CD8A': None,
  'ASMA': None,
  'GZMB': None,
  'KLRG1': None,
  'FOXP3': None,
  'PD1': None,
  'NAK': None,
  'KI67': None,
  'MHCII': None,
  'BCL6': None,
  'CC3': None,
  'CD103': None
}

# Advanced options
USE_SHARED_GATES = True  # True = one gate per marker; False = per-sample gates
NORMALIZATION_METHOD = 'percentile_99'  # 'percentile_99' or 'zscore' or 'minmax'

# ============================================================================
# TILE ARTIFACT CORRECTION (microscope grid detection + UniFORM normalization)
# ============================================================================
# Configuration for detecting actual microscope tile grid and applying UniFORM
# Detects regular grid of microscope tiles (not MCMICRO tiles) using intensity
# patterns, then normalizes dimmer tiles to match normal tiles

TILE_CORRECTION_CONFIG = {
    'enabled': True,
    'markers': ['GZMB', 'FOXP3', 'KLRG1', 'PD1', 'BCL6', 'CC3', 'PDL1', 'PERK'],

    # Grid detection parameters
    'bin_size': 400,                   # Spatial binning for heatmap (pixels)
    'peak_distance': 10,               # Minimum distance between grid lines (bins)
    'peak_height_percentile': 60,      # Peak detection threshold (percentile)
    'min_tiles': 4,                    # Minimum number of tiles to proceed
    'min_tile_size': 500,              # Minimum cells per tile
    'outlier_threshold': 2.0,          # MAD units for classifying dimmer/brighter tiles

    # UniFORM normalization parameters
    'n_quantiles': 75,                # Number of quantiles for UniFORM
    'correction_strength': 1.0,        # Dim tile correction strength (0-1, 1.0=full correction)
    'bright_correction_strength': 1.0, # Bright tile correction strength (REDUCED to minimize bright tile normalization)

    # Radial artifact correction parameters (within-tile vignetting)
    'radial_correction': True,         # Enable radial artifact correction
    'radial_bins': 5,                  # Number of radial zones (center to edge)
    'radial_threshold': 0.12,          # Max deviation to trigger correction (15%)
}

# ============================================================================
# HIERARCHICAL MARKER RELATIONSHIPS
# ============================================================================
# Define parent-child relationships: child markers must be subset of parent
# Format: 'child_marker': 'parent_marker'
# Child will only be positive if parent is also positive
MARKER_HIERARCHY = {
    'FOXP3': 'CD4',
    'GZMB': 'CD8A',
    'CD8A': 'CD3E',
    'CD4': 'CD3E',
    'PD1': 'CD3E',
    'KLRG1': 'CD8A',
    'CD103': 'CD3E',
    'B220': 'CD45',
    'F480': 'CD45',
    'MHCII': 'CD45',
    'BCL6': 'B220',
    'NINJA': 'TOM'
}

# ============================================================================
# LIBERAL GATING CONFIGURATION
# ============================================================================
# Specify markers where you want to OVERESTIMATE positive populations
# (i.e., be less conservative, allow more cells to be called positive)
# These markers will use relaxed thresholds to enrich positive populations

LIBERAL_GATING_CONFIG = {
    'enabled': True,

    # List markers that should use liberal (less conservative) gating
    # Example: markers where you want to capture more positive cells
    'liberal_markers': ['NINJA', 'PERK', 'GZMB', 'FOXP3', 'KLRG1', 'PD1'],

    # Liberal gating parameters (less stringent than default)
    # Default values: PEAK_MULTIPLIER=2.0, VALLEY_MAX_HEIGHT=0.20, MIN_PERCENTILE=85
    'liberal_peak_multiplier': 1.9,      # Reduced from 2.0 (allows gates closer to negative peak)
    'liberal_valley_max_height': 0.22,   # Increased from 0.20 (tolerates shallower valleys)
    'liberal_min_percentile': 82,        # Reduced from 85 (allows lower percentile gates)
    'liberal_min_absolute_gate': 0.15,   # Reduced from 0.15 (allows lower absolute gates)
}

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
    GPU-accelerated BaSiC illumination correction for cyclic IF.
    Must be done BEFORE outlier removal.
    Based on: Peng et al., Nat Commun 2017 (MCMICRO uses this)
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
        from cupyx.scipy.interpolate import RegularGridInterpolator
        use_gpu = True
        print("✓ Using GPU acceleration (CuPy)")
    except ImportError:
        import numpy as cp
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import RectBivariateSpline
        use_gpu = False
        print("⚠ GPU not available, using CPU")
    
    print("\nCorrecting illumination bias...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = cp.asarray(adata.X[mask, marker_idx]) if use_gpu else adata.X[mask, marker_idx]
            coords = cp.asarray(adata.obsm['spatial'][mask]) if use_gpu else adata.obsm['spatial'][mask]
            
            # Binning parameters
            x_bins, y_bins = 50, 50
            x_edges = cp.linspace(coords[:, 0].min(), coords[:, 0].max(), x_bins)
            y_edges = cp.linspace(coords[:, 1].min(), coords[:, 1].max(), y_bins)
            
            # Vectorized 2D histogram for background map
            x_idx = cp.searchsorted(x_edges[:-1], coords[:, 0], side='right') - 1
            y_idx = cp.searchsorted(y_edges[:-1], coords[:, 1], side='right') - 1
            x_idx = cp.clip(x_idx, 0, x_bins - 2)
            y_idx = cp.clip(y_idx, 0, y_bins - 2)
            
            background_map = cp.zeros((y_bins-1, x_bins-1))
            
            # Compute 25th percentile per bin (vectorized)
            for i in range(x_bins-1):
                for j in range(y_bins-1):
                    bin_mask = (x_idx == i) & (y_idx == j)
                    if cp.sum(bin_mask) > 10:
                        background_map[j, i] = cp.percentile(vals[bin_mask], 25)
            
            # Smooth background map
            background_smooth = gaussian_filter(background_map, sigma=3)
            
            # Fast bilinear interpolation
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            
            if use_gpu:
                interp = RegularGridInterpolator(
                    (y_centers, x_centers), 
                    background_smooth, 
                    method='linear',
                    bounds_error=False,
                    fill_value=cp.median(background_smooth)
                )
                background_at_cells = interp(cp.stack([coords[:, 1], coords[:, 0]], axis=1))
            else:
                interp = RectBivariateSpline(y_centers, x_centers, background_smooth)
                background_at_cells = interp(coords[:, 1], coords[:, 0], grid=False)
            
            # Subtract background
            corrected = vals - background_at_cells + cp.median(background_at_cells)
            corrected = cp.maximum(corrected, 0)
            
            if use_gpu:
                adata.X[mask, marker_idx] = cp.asnumpy(corrected)
                bg_min, bg_max = float(cp.asnumpy(background_at_cells.min())), float(cp.asnumpy(background_at_cells.max()))
            else:
                adata.X[mask, marker_idx] = corrected
                bg_min, bg_max = background_at_cells.min(), background_at_cells.max()
            
            print(f"  {sample} {marker}: background range {bg_min:.0f}-{bg_max:.0f}")
    
    return adata

def normalize_tiles_by_background(adata, n_jobs=16):
    """
    Parallelized tile-based background normalization.
    """
    from joblib import Parallel, delayed
    import time
    
    print(f"\nTile-based background normalization (using {n_jobs} cores)...")
    start = time.time()
    
    # Create unique tile identifier
    adata.obs['tile_id'] = adata.obs['tile_y'].astype(str) + '_' + adata.obs['tile_x'].astype(str)
    print(f"Found {adata.obs['tile_id'].nunique()} unique tiles")
    
    adata.layers['raw_prenorm'] = adata.X.copy()
    
    # Identify background cells once (low total intensity)
    total_intensity = adata.X.sum(axis=1)
    background_threshold = np.percentile(total_intensity, 20)
    is_background = total_intensity < background_threshold
    
    def process_marker(marker_idx, marker_name, X_data, obs_data, is_bg):
        """Process one marker across all tiles"""
        X_marker = X_data[:, marker_idx].copy()
        
        # Global background reference
        global_background = np.median(X_marker[is_bg])
        
        # Calculate corrections for all tiles
        tile_corrections = {}
        for tile_id in obs_data['tile_id'].unique():
            tile_mask = obs_data['tile_id'] == tile_id
            tile_bg_mask = tile_mask & is_bg
            
            if tile_bg_mask.sum() >= 10:
                tile_background = np.median(X_marker[tile_bg_mask])
                tile_corrections[tile_id] = global_background - tile_background
        
        # Apply corrections vectorized
        for tile_id, correction in tile_corrections.items():
            tile_mask = obs_data['tile_id'] == tile_id
            X_marker[tile_mask] += correction
        
        X_marker = np.maximum(X_marker, 0)
        
        return marker_name, X_marker, global_background, len(tile_corrections)
    
    # Parallel processing across markers
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_marker)(i, marker, adata.X, adata.obs, is_background)
        for i, marker in enumerate(adata.var_names)
    )
    
    # Update adata with results
    for marker_name, X_corrected, global_bg, n_tiles in results:
        marker_idx = list(adata.var_names).index(marker_name)
        adata.X[:, marker_idx] = X_corrected
        print(f"  {marker_name}: global_bg={global_bg:.1f}, corrected {n_tiles} tiles")
    
    elapsed = time.time() - start
    print(f"✓ Completed in {elapsed:.1f}s ({elapsed/len(adata.var_names):.1f}s per marker)")
    
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

def landmark_quantile_normalization(adata):
    """
    Landmark-based quantile normalization per marker.
    Based on UniFORM/CytoNorm principles: align negative populations.
    """
    from scipy.interpolate import PchipInterpolator
    
    print("\nLandmark quantile normalization...")
    
    # Define landmark percentiles (including negatives)
    landmarks = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        # Compute landmarks for each sample
        sample_landmarks = {}
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 1000:
                continue
            
            sample_landmarks[sample] = np.percentile(pos_vals, landmarks)
        
        # Compute reference (median across samples)
        ref_landmarks = np.median([lm for lm in sample_landmarks.values()], axis=0)
        
        print(f"  Reference landmarks: {ref_landmarks}")
        
        # Normalize each sample using spline interpolation
        for sample, source_lm in sample_landmarks.items():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            
            # Create spline mapping source -> reference
            # Add endpoints for extrapolation
            source_extended = np.concatenate([[0], source_lm, [source_lm[-1] * 1.5]])
            ref_extended = np.concatenate([[0], ref_landmarks, [ref_landmarks[-1] * 1.5]])
            
            spline = PchipInterpolator(source_extended, ref_extended)
            
            # Apply transformation
            vals_normalized = spline(vals)
            vals_normalized = np.maximum(vals_normalized, 0)  # No negatives
            
            adata.X[mask, marker_idx] = vals_normalized
            
            print(f"  {sample}: aligned {source_lm[0]:.0f}->{ref_landmarks[0]:.0f} (bg), "
                  f"{source_lm[-1]:.0f}->{ref_landmarks[-1]:.0f} (p99)")
    
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

def integrated_normalization(adata):
    """
    Integrated normalization with saturation handling.
    """
    from scipy.interpolate import PchipInterpolator
    
    print("\nIntegrated landmark normalization...")
    
    adata.layers['raw'] = adata.X.copy()
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        sample_references = {}
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 1000:
                continue
            
            # Check for saturation
            max_val = pos_vals.max()
            is_saturated = max_val > 65000
            
            # Use percentiles that avoid saturation
            if is_saturated:
                landmarks_pct = [5, 25, 50, 75, 90, 95, 98]  # Stop at 98th
            else:
                landmarks_pct = [5, 25, 50, 75, 90, 95, 99]
            
            landmarks = np.percentile(pos_vals, landmarks_pct)
            
            # Ensure strictly increasing
            landmarks = np.maximum.accumulate(landmarks)
            
            neg_ref = landmarks[0]  # 5th percentile
            
            sample_references[sample] = {
                'negative': neg_ref,
                'landmarks': landmarks,
                'saturated': is_saturated
            }
        
        # Global references
        global_neg = np.median([s['negative'] for s in sample_references.values()])
        global_landmarks = np.median([s['landmarks'] for s in sample_references.values()], axis=0)
        global_landmarks = np.maximum.accumulate(global_landmarks)  # Ensure increasing
        
        print(f"  Global negative reference: {global_neg:.1f}")
        
        # Normalize each sample
        for sample, refs in sample_references.items():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx].copy()
            
            # Build strictly increasing landmark pairs
            source_lm = refs['landmarks'].copy()
            target_lm = global_landmarks.copy()
            
            # Prepend zero
            source_lm = np.concatenate([[0], source_lm])
            target_lm = np.concatenate([[0], target_lm])
            
            # Ensure strict monotonicity by adding tiny increments
            for i in range(1, len(source_lm)):
                if source_lm[i] <= source_lm[i-1]:
                    source_lm[i] = source_lm[i-1] + 1e-6
                if target_lm[i] <= target_lm[i-1]:
                    target_lm[i] = target_lm[i-1] + 1e-6
            
            # Create spline
            try:
                spline = PchipInterpolator(source_lm, target_lm, extrapolate=True)
                vals_normalized = spline(vals)
                vals_normalized = np.maximum(vals_normalized, 0)
                
                adata.X[mask, marker_idx] = vals_normalized
                
                sat_flag = " [SATURATED]" if refs['saturated'] else ""
                print(f"  {sample}: neg {refs['negative']:.0f}→{global_neg:.0f}, "
                      f"p95 {refs['landmarks'][-2]:.0f}→{global_landmarks[-2]:.0f}{sat_flag}")
            
            except Exception as e:
                print(f"  {sample}: FAILED - {e}, using identity")
                continue
    
    # Final 0-1 normalization
    print("\nFinal 0-1 scaling...")
    adata.layers['normalized'] = adata.X.copy()
    for i, marker in enumerate(adata.var_names):
        vals = adata.X[:, i]
        pos_vals = vals[vals > 0]
        
        if len(pos_vals) > 100:
            # Use 98th to avoid saturation artifacts
            p98 = np.percentile(pos_vals, 98)
            if p98 > 0:
                adata.X[:, i] = np.clip(vals / p98, 0, 1)
                print(f"  {marker}: scaled by p98={p98:.0f}")
    
    return adata

def uniform_normalization(adata):
    """
    UniFORM-style normalization with log transformation (SCIMAP standard).
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    print("\nUniFORM-style functional data registration...")
    
    adata.layers['raw'] = adata.X.copy()
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        # Step 1: Compute smooth density curves
        sample_densities = {}
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 1000:
                continue
            
            # Log-spaced bins for right-skewed distributions
            max_val = min(pos_vals.max(), 65000)
            bins = np.logspace(np.log10(1), np.log10(max_val), 256)
            hist, bin_edges = np.histogram(pos_vals, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Smooth density
            hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
            density = hist_smooth / (hist_smooth.sum() * np.diff(bin_edges))
            
            # Find negative peak
            peaks, _ = find_peaks(density, prominence=0.01)
            negative_peak = bin_centers[peaks[0]] if len(peaks) > 0 else np.percentile(pos_vals, 5)
            
            sample_densities[sample] = {
                'negative_peak': negative_peak,
                'raw_vals': pos_vals
            }
        
        # Step 2: Aggregate reference
        all_vals = np.concatenate([s['raw_vals'] for s in sample_densities.values()])
        ref_negative_peak = np.median([s['negative_peak'] for s in sample_densities.values()])
        
        print(f"  Reference negative peak: {ref_negative_peak:.1f}")
        
        # Step 3: Rigid landmark registration
        for sample, samp_dens in sample_densities.items():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx].copy()
            
            # Source and target landmarks
            src_neg = samp_dens['negative_peak']
            src_percentiles = np.percentile(samp_dens['raw_vals'], [25, 50, 75, 90, 95, 98])
            src_landmarks = np.concatenate([[0, src_neg], src_percentiles])
            
            tgt_neg = ref_negative_peak
            tgt_percentiles = np.percentile(all_vals, [25, 50, 75, 90, 95, 98])
            tgt_landmarks = np.concatenate([[0, tgt_neg], tgt_percentiles])
            
            # Ensure monotonicity
            src_landmarks = np.maximum.accumulate(src_landmarks)
            tgt_landmarks = np.maximum.accumulate(tgt_landmarks)
            
            for i in range(1, len(src_landmarks)):
                if src_landmarks[i] <= src_landmarks[i-1]:
                    src_landmarks[i] = src_landmarks[i-1] * 1.001
                if tgt_landmarks[i] <= tgt_landmarks[i-1]:
                    tgt_landmarks[i] = tgt_landmarks[i-1] * 1.001
            
            try:
                warp_func = PchipInterpolator(src_landmarks, tgt_landmarks, extrapolate=True)
                vals_warped = warp_func(vals)
                vals_warped = np.maximum(vals_warped, 0)
                
                adata.X[mask, marker_idx] = vals_warped
                print(f"  {sample}: aligned (neg {src_neg:.0f}→{tgt_neg:.0f})")
                
            except Exception as e:
                print(f"  {sample}: FAILED - {e}")
                continue
    
    # Step 4: Log transformation (SCIMAP standard)
    print("\nApplying log transformation (SCIMAP standard)...")
    adata.layers['aligned'] = adata.X.copy()
    
    # Asinh transformation (handles zeros better than log)
    # asinh(x/c) where c is cofactor (typically 5 or 150 for IF)
    cofactor = 150  # Standard for immunofluorescence
    
    adata.layers['log'] = np.arcsinh(adata.X / cofactor)
    adata.X = adata.layers['log'].copy()
    
    for marker in adata.var_names:
        print(f"  {marker}: asinh(x/{cofactor})")
    
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

def slow_remove_tiling_artifacts(adata):
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

def remove_tiling_artifacts(adata):
    """
    Fast per-sample artifact removal using vectorized operations.
    """
    from scipy.stats import median_abs_deviation
    
    print("\nRemoving artifacts (per-sample)...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.X[mask, marker_idx]
            
            # Global intensity outliers (5 MAD)
            median = np.median(vals)
            mad = median_abs_deviation(vals)
            outliers = np.abs(vals - median) > 5 * mad
            
            if outliers.sum() > 0:
                # Replace with median
                adata.X[mask, marker_idx][outliers] = median
                print(f"    {sample}: corrected {outliers.sum()} outliers")
    
    return adata

def spatial_local_background_correction(adata, grid_size=20):
    """
    Correct residual tile-level illumination artifacts.
    Handles sparse grids with empty cells.
    """
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.interpolate import griddata
    
    print(f"\nSpatial background correction (grid_size={grid_size})...")
    
    adata.layers['raw_pre_spatial'] = adata.X.copy()
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            coords = adata.obsm['spatial'][sample_mask]
            vals = adata.X[sample_mask, marker_idx].copy()
            
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            x_bins = np.linspace(x_min, x_max, grid_size)
            y_bins = np.linspace(y_min, y_max, grid_size)
            
            # Calculate background per grid cell
            background_points = []  # Store (x, y, background_value)
            
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    mask = ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                           (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1]))
                    
                    if mask.sum() > 20:  # Only non-empty cells
                        x_center = (x_bins[i] + x_bins[i+1]) / 2
                        y_center = (y_bins[j] + y_bins[j+1]) / 2
                        bg_value = np.percentile(vals[mask], 5)
                        background_points.append([x_center, y_center, bg_value])
            
            if len(background_points) < 10:
                print(f"  {sample}: Insufficient grid points, skipping")
                continue
            
            background_points = np.array(background_points)
            grid_coords = background_points[:, :2]
            grid_values = background_points[:, 2]
            
            print(f"  {sample}: {len(background_points)} non-empty grid cells")
            
            # METHOD 1: Smooth the sparse grid points (preferred for coarse grids)
            # Uses radial basis function - robust to missing data
            from scipy.interpolate import Rbf
            
            rbf = Rbf(grid_coords[:, 0], grid_coords[:, 1], grid_values,
                     function='multiquadric', smooth=10)  # smooth=10 for heavy smoothing
            
            background_at_cells = rbf(coords[:, 0], coords[:, 1])
            
            # Normalize by background ratio
            global_background = np.median(background_at_cells)
            
            if global_background > 0:
                correction_factor = global_background / (background_at_cells + 1)
                vals_corrected = vals * correction_factor
                vals_corrected = np.maximum(vals_corrected, 0)
                
                adata.X[sample_mask, marker_idx] = vals_corrected
                
                print(f"    background range {background_at_cells.min():.0f}-"
                      f"{background_at_cells.max():.0f} → {global_background:.0f}")
    
    return adata

def two_stage_spatial_correction(adata, tile_grid_size=20, vignetting_grid_size=100):
    """
    Two-stage spatial correction:
    Stage 1: Coarse (tile-to-tile normalization)
    Stage 2: Fine (within-tile vignetting removal)
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    
    print(f"\n=== TWO-STAGE SPATIAL CORRECTION ===")
    print(f"Stage 1 grid: {tile_grid_size}x{tile_grid_size} (tile-to-tile)")
    print(f"Stage 2 grid: {vignetting_grid_size}x{vignetting_grid_size} (vignetting)")
    
    adata.layers['raw_pre_spatial'] = adata.X.copy()
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            coords = adata.obsm['spatial'][sample_mask]
            vals = adata.X[sample_mask, marker_idx].copy()
            
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            # ====================================================================
            # STAGE 1: TILE-TO-TILE CORRECTION (coarse, preserves high values)
            # ====================================================================
            x_bins_coarse = np.linspace(x_min, x_max, tile_grid_size)
            y_bins_coarse = np.linspace(y_min, y_max, tile_grid_size)
            
            background_points_coarse = []
            
            for i in range(len(x_bins_coarse)-1):
                for j in range(len(y_bins_coarse)-1):
                    mask = ((coords[:, 0] >= x_bins_coarse[i]) & 
                           (coords[:, 0] < x_bins_coarse[i+1]) &
                           (coords[:, 1] >= y_bins_coarse[j]) & 
                           (coords[:, 1] < y_bins_coarse[j+1]))
                    
                    if mask.sum() > 50:
                        x_center = (x_bins_coarse[i] + x_bins_coarse[i+1]) / 2
                        y_center = (y_bins_coarse[j] + y_bins_coarse[j+1]) / 2
                        
                        # Use 25th percentile (not 5th) to preserve high values
                        bg_value = np.percentile(vals[mask], 25)
                        background_points_coarse.append([x_center, y_center, bg_value])
            
            if len(background_points_coarse) < 10:
                print(f"  {sample}: Insufficient data, skipping")
                continue
            
            background_points_coarse = np.array(background_points_coarse)
            grid_coords_coarse = background_points_coarse[:, :2]
            grid_values_coarse = background_points_coarse[:, 2]
            
            # Interpolate tile-level background (smooth, large-scale)
            tree_coarse = cKDTree(grid_coords_coarse)
            k_coarse = min(9, len(grid_coords_coarse))  # Use 9 neighbors for smoothing
            distances_coarse, indices_coarse = tree_coarse.query(coords, k=k_coarse)
            
            weights_coarse = 1 / (distances_coarse + 1)
            weights_coarse = weights_coarse / weights_coarse.sum(axis=1, keepdims=True)
            
            background_coarse = (grid_values_coarse[indices_coarse] * weights_coarse).sum(axis=1)
            
            # Apply tile-level correction (multiplicative)
            global_bg_coarse = np.median(background_coarse)
            correction_factor_coarse = global_bg_coarse / (background_coarse + 1)
            vals_coarse_corrected = vals * correction_factor_coarse
            
            print(f"  Stage 1: tile-to-tile, bg range {background_coarse.min():.0f}-{background_coarse.max():.0f}")
            
            # ====================================================================
            # STAGE 2: WITHIN-TILE VIGNETTING CORRECTION (fine, only low percentile)
            # ====================================================================
            x_bins_fine = np.linspace(x_min, x_max, vignetting_grid_size)
            y_bins_fine = np.linspace(y_min, y_max, vignetting_grid_size)
            
            background_points_fine = []
            
            for i in range(len(x_bins_fine)-1):
                for j in range(len(y_bins_fine)-1):
                    mask = ((coords[:, 0] >= x_bins_fine[i]) & 
                           (coords[:, 0] < x_bins_fine[i+1]) &
                           (coords[:, 1] >= y_bins_fine[j]) & 
                           (coords[:, 1] < y_bins_fine[j+1]))
                    
                    if mask.sum() > 20:
                        x_center = (x_bins_fine[i] + x_bins_fine[i+1]) / 2
                        y_center = (y_bins_fine[j] + y_bins_fine[j+1]) / 2
                        
                        # Use 10th percentile for vignetting (finer-scale gradient)
                        bg_value = np.percentile(vals_coarse_corrected[mask], 10)
                        background_points_fine.append([x_center, y_center, bg_value])
            
            if len(background_points_fine) < 50:
                # Not enough points for fine correction, skip stage 2
                adata.X[sample_mask, marker_idx] = vals_coarse_corrected
                print(f"  Stage 2: skipped (insufficient points)")
                continue
            
            background_points_fine = np.array(background_points_fine)
            grid_coords_fine = background_points_fine[:, :2]
            grid_values_fine = background_points_fine[:, 2]
            
            # Interpolate fine-scale background
            tree_fine = cKDTree(grid_coords_fine)
            k_fine = min(5, len(grid_coords_fine))  # Fewer neighbors = preserve gradients
            distances_fine, indices_fine = tree_fine.query(coords, k=k_fine)
            
            weights_fine = 1 / (distances_fine + 1)
            weights_fine = weights_fine / weights_fine.sum(axis=1, keepdims=True)
            
            background_fine = (grid_values_fine[indices_fine] * weights_fine).sum(axis=1)
            
            # Apply vignetting correction (ADDITIVE for fine-scale, more conservative)
            global_bg_fine = np.median(background_fine)
            correction_additive = global_bg_fine - background_fine
            
            # Limit correction magnitude to avoid overcorrection
            correction_additive = np.clip(correction_additive, -500, 500)
            
            vals_final = vals_coarse_corrected + correction_additive
            vals_final = np.maximum(vals_final, 0)
            
            adata.X[sample_mask, marker_idx] = vals_final
            
            print(f"  Stage 2: vignetting, correction range {correction_additive.min():.0f} to {correction_additive.max():.0f}")
    
    return adata

def detect_physical_tiles(adata, sample):
    """
    Detect microscope tile boundaries by finding discontinuities in local background.
    """
    from scipy.ndimage import sobel
    
    sample_mask = adata.obs['sample_id'] == sample
    coords = adata.obsm['spatial'][sample_mask]
    
    # Create coarse intensity grid
    grid_size = 200
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid_size)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid_size)
    
    # Sum all markers to find intensity gradients
    total_intensity = adata.X[sample_mask].sum(axis=1)
    
    intensity_grid = np.zeros((grid_size-1, grid_size-1))
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                   (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1]))
            if mask.sum() > 0:
                intensity_grid[j, i] = np.median(total_intensity[mask])
    
    # Detect edges using Sobel
    edges_x = sobel(intensity_grid, axis=1)
    edges_y = sobel(intensity_grid, axis=0)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Threshold to find strong edges (tile boundaries)
    edge_threshold = np.percentile(edges[edges > 0], 90)
    tile_boundaries = edges > edge_threshold
    
    return tile_boundaries, x_bins, y_bins

def visualize_tile_artifacts(adata, output_dir):
    """Create diagnostic plots showing tile artifacts before/after correction."""
    plots_dir = Path(output_dir) / "tile_artifact_diagnosis"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating tile artifact diagnostic plots...")
    
    # Check which raw layer exists
    if 'raw' not in adata.layers:
        print("  WARNING: No 'raw' layer found, skipping visualization")
        return
    
    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        sample_data = adata[sample_mask]
        
        # Subsample for plotting
        if len(sample_data) > 50000:
            idx = np.random.choice(len(sample_data), 50000, replace=False)
            sample_data = sample_data[idx]
        
        coords = sample_data.obsm['spatial']
        
        fig, axes = plt.subplots(len(adata.var_names), 3, 
                                figsize=(18, 4*len(adata.var_names)))
        fig.suptitle(f'{sample} - Tile Artifact Diagnosis', fontsize=14)
        
        for marker_idx, marker in enumerate(adata.var_names):
            ax_row = axes[marker_idx] if len(adata.var_names) > 1 else axes
            
            # Raw (before hierarchical normalization)
            raw_vals = sample_data.layers['raw'][:, marker_idx]
            sc1 = ax_row[0].scatter(coords[:, 0], coords[:, 1], 
                                   c=np.log10(raw_vals + 1), s=0.5, 
                                   cmap='viridis', vmin=0, vmax=3, rasterized=True)
            ax_row[0].set_title(f'{marker} - Raw (tile artifacts visible)')
            ax_row[0].set_aspect('equal')
            plt.colorbar(sc1, ax=ax_row[0], fraction=0.046)
            
            # After hierarchical normalization
            corrected_vals = sample_data.layers['aligned'][:, marker_idx]
            sc2 = ax_row[1].scatter(coords[:, 0], coords[:, 1], 
                                   c=np.log10(corrected_vals + 1), s=0.5,
                                   cmap='viridis', vmin=0, vmax=3, rasterized=True)
            ax_row[1].set_title(f'{marker} - After Hierarchical Normalization')
            ax_row[1].set_aspect('equal')
            plt.colorbar(sc2, ax=ax_row[1], fraction=0.046)
            
            # Histogram comparison (filter out NaN/inf values)
            raw_log = np.log10(raw_vals + 1)
            corrected_log = np.log10(corrected_vals + 1)
            
            # Remove NaN and inf values
            raw_log = raw_log[np.isfinite(raw_log)]
            corrected_log = corrected_log[np.isfinite(corrected_log)]
            
            if len(raw_log) > 0 and len(corrected_log) > 0:
                ax_row[2].hist(raw_log, bins=100, alpha=0.5, 
                              label='Raw', density=True)
                ax_row[2].hist(corrected_log, bins=100, alpha=0.5,
                              label='Corrected', density=True)
                ax_row[2].set_xlabel('Log10(Intensity + 1)')
                ax_row[2].set_ylabel('Density')
                ax_row[2].legend()
                ax_row[2].set_title('Distribution Comparison')
            else:
                ax_row[2].text(0.5, 0.5, 'No valid data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{sample}_tile_artifacts.png', 
                   dpi=150, bbox_inches='tight')
        plt.close('all')
    
    print(f"✓ Tile artifact diagnostics saved to {plots_dir}")

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

def correct_tile_artifacts_per_marker(adata):
    """
    Detect dimmer tile regions and apply UniFORM normalization.

    This method:
    1. Uses Sobel edge detection to identify tile boundaries
    2. Segments the image into tile regions
    3. Classifies tiles as "dimmer" or "normal" based on intensity
    4. Applies UniFORM (quantile-based) normalization to align dimmer tiles to normal tiles

    Only corrects markers specified in TILE_CORRECTION_CONFIG.
    Runs BEFORE hierarchical UniFORM normalization.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
    from tile_artifact_correction import MicroscopeTileDetector, TileArtifactCorrector
    from tile_artifact_correction import create_diagnostic_plots, save_correction_report

    print("\n" + "="*70)
    print("TILE DETECTION & UniFORM CORRECTION")
    print("="*70)

    if not TILE_CORRECTION_CONFIG.get('enabled', False):
        print("  ⚠️  Tile correction disabled in config")
        return adata

    # Get markers to correct
    markers_to_correct = TILE_CORRECTION_CONFIG.get('markers', [])
    if not markers_to_correct:
        print("  ⚠️  No markers specified for correction")
        return adata

    # Filter to markers that exist in data
    markers_to_correct = [m for m in markers_to_correct if m in adata.var_names]
    if not markers_to_correct:
        print("  ⚠️  None of the specified markers found in data")
        return adata

    print(f"  Correcting {len(markers_to_correct)} markers: {', '.join(markers_to_correct)}")
    print(f"  Method: Gradient-based tile detection + UniFORM normalization")
    print()

    # Initialize detector and corrector with config parameters
    detector = MicroscopeTileDetector(
        bin_size=TILE_CORRECTION_CONFIG.get('bin_size', 50),
        peak_distance=TILE_CORRECTION_CONFIG.get('peak_distance', 20),
        peak_height_percentile=TILE_CORRECTION_CONFIG.get('peak_height_percentile', 75),
        min_tiles=TILE_CORRECTION_CONFIG.get('min_tiles', 4),
        min_tile_size=TILE_CORRECTION_CONFIG.get('min_tile_size', 100),
        outlier_threshold=TILE_CORRECTION_CONFIG.get('outlier_threshold', 2.0)
    )

    corrector = TileArtifactCorrector(
        n_quantiles=TILE_CORRECTION_CONFIG.get('n_quantiles', 100),
        correction_strength=TILE_CORRECTION_CONFIG.get('correction_strength', 1.0),
        bright_correction_strength=TILE_CORRECTION_CONFIG.get('bright_correction_strength', 1.0),
        radial_correction=TILE_CORRECTION_CONFIG.get('radial_correction', True),
        radial_bins=TILE_CORRECTION_CONFIG.get('radial_bins', 3),
        radial_threshold=TILE_CORRECTION_CONFIG.get('radial_threshold', 0.15)
    )

    # Store original data for visualization
    if 'raw_precorrection' not in adata.layers:
        adata.layers['raw_precorrection'] = adata.X.copy()

    # Correction report
    correction_report = {}

    # Create output directory for diagnostic plots
    diagnostic_dir = Path(__file__).parent / 'tile_correction_diagnostics'
    diagnostic_dir.mkdir(exist_ok=True, parents=True)

    # Process each marker
    for marker in markers_to_correct:
        print(f"  Processing {marker}...")
        marker_idx = adata.var_names.get_loc(marker)
        marker_report = {}

        # Process each sample separately
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            sample_data = adata[sample_mask]

            # Get coordinates and intensities
            x_coords = sample_data.obsm['spatial'][:, 0]
            y_coords = sample_data.obsm['spatial'][:, 1]
            intensities = sample_data.X[:, marker_idx].copy()

            # Phase 1-6: Detect tiles and classify them
            detection_results = detector.detect(x_coords, y_coords, intensities)

            if not detection_results['detected']:
                reason = detection_results.get('reason', 'Detection failed')
                print(f"    {sample}: {reason}")
                marker_report[sample] = {'detected': False, 'reason': reason}
                continue

            n_dimmer = detection_results['n_dimmer_cells']
            n_brighter = detection_results['n_brighter_cells']
            n_normal = detection_results['n_normal_cells']
            n_dimmer_tiles = len(detection_results['dimmer_tiles'])
            n_brighter_tiles = len(detection_results['brighter_tiles'])
            n_normal_tiles = len(detection_results['normal_tiles'])

            print(f"    {sample}: Detected {n_dimmer_tiles} dimmer tiles ({n_dimmer:,} cells), "
                  f"{n_brighter_tiles} brighter tiles ({n_brighter:,} cells), "
                  f"{n_normal_tiles} normal tiles ({n_normal:,} cells)")

            # Apply UniFORM normalization and radial correction
            corrected_intensities, stats = corrector.correct(x_coords, y_coords, intensities, detection_results)

            # Update adata
            sample_indices = np.where(sample_mask)[0]
            adata.X[sample_indices, marker_idx] = corrected_intensities

            if stats['n_corrected'] > 0:
                n_radial = stats.get('n_radial_corrected', 0)
                radial_msg = f", radial: {n_radial} tiles" if n_radial > 0 else ""
                print(f"    {sample}: Corrected {stats['n_corrected']:,} cells "
                      f"(dimmer: {stats['n_dimmer_corrected']:,}, brighter: {stats['n_brighter_corrected']:,})"
                      f"{radial_msg}")
            else:
                print(f"    {sample}: No abnormal tiles detected, skipping correction")

            # Store stats
            marker_report[sample] = stats

            # Create diagnostic plots for this marker/sample combination
            sample_diagnostic_dir = diagnostic_dir / sample
            sample_diagnostic_dir.mkdir(exist_ok=True, parents=True)

            create_diagnostic_plots(
                marker=f"{marker}_{sample}",
                x_coords=x_coords,
                y_coords=y_coords,
                original_intensities=intensities,
                corrected_intensities=corrected_intensities,
                detection_results=detection_results,
                output_dir=sample_diagnostic_dir
            )

        correction_report[marker] = marker_report

    print(f"\n  ✓ Tile correction complete for {len(markers_to_correct)} markers")
    print(f"  ✓ Diagnostic plots saved to: {diagnostic_dir}")
    print("="*70)

    # Save correction report
    save_correction_report(correction_report, diagnostic_dir)

    # Save correction layer
    adata.layers['tile_corrected'] = adata.X.copy()

    return adata


def apply_hierarchical_gating(adata, gates):
    """
    Enforce hierarchical marker relationships after initial gating.

    Child markers can only be positive if their parent marker is also positive.
    This is applied at the cell classification level, not during gate calculation.

    Example: If FOXP3 parent is CD3, then any cell with FOXP3+ but CD3- will be
             reclassified as FOXP3-.
    """
    if not MARKER_HIERARCHY:
        return adata

    print("\n" + "="*70)
    print("HIERARCHICAL MARKER GATING")
    print("="*70)
    print("  Enforcing parent-child relationships:")

    # Create gated boolean matrix
    gated_bool = {}
    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)
        gate = gates.get(marker, np.percentile(adata.X[:, marker_idx], 95))
        gated_bool[marker] = adata.X[:, marker_idx] > gate

    # Apply hierarchy constraints
    adjustments_made = {}
    for child_marker, parent_marker in MARKER_HIERARCHY.items():
        if child_marker not in adata.var_names or parent_marker not in adata.var_names:
            print(f"  ⚠️  Skipping {child_marker} → {parent_marker} (marker not found)")
            continue

        # Find cells that are child+ but parent-
        child_pos = gated_bool[child_marker]
        parent_neg = ~gated_bool[parent_marker]
        invalid_cells = child_pos & parent_neg

        n_invalid = invalid_cells.sum()
        n_child_pos = child_pos.sum()

        if n_invalid > 0:
            # Reclassify these cells as child-negative by setting intensity below gate
            child_idx = adata.var_names.get_loc(child_marker)
            gate = gates[child_marker]

            # Set to gate * 0.9 to ensure they're below threshold
            adata.X[invalid_cells, child_idx] = gate * 0.9

            # Update the gated boolean
            gated_bool[child_marker] = adata.X[:, child_idx] > gate

            pct_adjusted = (n_invalid / n_child_pos * 100) if n_child_pos > 0 else 0
            adjustments_made[child_marker] = (n_invalid, n_child_pos, pct_adjusted)

            print(f"    {child_marker} ⊂ {parent_marker}: {n_invalid:,} cells adjusted "
                  f"({pct_adjusted:.1f}% of {child_marker}+ cells)")

    if not adjustments_made:
        print("    ✓ No adjustments needed - all markers respect hierarchy")

    print("="*70)

    return adata


def gmm_gating(adata):
    """
    2-component GMM gating following UniFORM paper.
    Applied to normalized intensity data (NOT log-transformed).
    """
    from sklearn.mixture import GaussianMixture
    
    print("\n" + "="*70)
    print("GMM-BASED GATING (auto-calculated)")
    print("="*70)
    
    gates = {}
    
    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)
        
        # Global GMM across all samples (on aligned intensity scale)
        all_vals = adata.layers['aligned'][:, marker_idx]  # Use aligned, not log
        pos_vals = all_vals[all_vals > 0].reshape(-1, 1)
        
        if len(pos_vals) < 1000:
            gates[marker] = np.percentile(pos_vals, 50)
            print(f"  {marker:8s}: insufficient data, using median={gates[marker]:.1f}")
            continue
        
        # Fit 2-component GMM
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(pos_vals)
        
        # Sort components by mean (low=negative, high=positive)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        
        # Threshold = midpoint between means
        gate = float((means[order[0]] + means[order[1]]) / 2)
        gates[marker] = gate
        
        # Calculate positive percentage
        pos_pct = (all_vals > gate).mean() * 100
        
        print(f"  {marker:8s}: gate={gate:8.1f} | "
              f"neg_μ={means[order[0]]:7.1f} (σ={stds[order[0]]:.1f}) | "
              f"pos_μ={means[order[1]]:7.1f} (σ={stds[order[1]]:.1f}) | "
              f"{pos_pct:5.1f}% positive")
    
    print("="*70)
    return gates

def density_based_gating(adata):
    """
    Fixed density gating: Valley detection between negative and positive peaks.
    Uses biological constraints to avoid cutting into negative tails.
    Supports liberal gating mode for markers specified in LIBERAL_GATING_CONFIG.
    """
    from scipy.signal import find_peaks, argrelextrema
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import median_abs_deviation
    from sklearn.mixture import GaussianMixture

    print("\n" + "="*70)
    print("DENSITY-BASED GATING (valley detection)")
    print("="*70)

    # Default (conservative) parameters
    PEAK_MULTIPLIER = 2.0      # Was 1.5 - increase to 2.0, 2.5, 3.0
    MIN_ABSOLUTE_GATE = 0.15   # NEW - hard minimum (prevents gates below this)
    MIN_PERCENTILE = 85        # NEW - gate must be at least p85 of positive cells
    VALLEY_MAX_HEIGHT = 0.20   # Was 0.30 - decrease to 0.20, 0.15 for stricter

    # Liberal gating configuration
    liberal_enabled = LIBERAL_GATING_CONFIG.get('enabled', False)
    liberal_markers = LIBERAL_GATING_CONFIG.get('liberal_markers', [])

    if liberal_enabled and liberal_markers:
        print(f"Liberal gating ENABLED for markers: {', '.join(liberal_markers)}")
        print(f"  Liberal params: peak_mult={LIBERAL_GATING_CONFIG['liberal_peak_multiplier']}, "
              f"valley_height={LIBERAL_GATING_CONFIG['liberal_valley_max_height']}, "
              f"min_pct=p{LIBERAL_GATING_CONFIG['liberal_min_percentile']}")

    print(f"Default (conservative) parameters: peak_mult={PEAK_MULTIPLIER}, "
        f"min_gate={MIN_ABSOLUTE_GATE}, min_pct=p{MIN_PERCENTILE}")

    gates = {}

    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)

        # Check if this marker should use liberal gating
        use_liberal = liberal_enabled and marker in liberal_markers
        if use_liberal:
            # Use liberal parameters for this marker
            peak_mult = LIBERAL_GATING_CONFIG['liberal_peak_multiplier']
            valley_max_height = LIBERAL_GATING_CONFIG['liberal_valley_max_height']
            min_percentile = LIBERAL_GATING_CONFIG['liberal_min_percentile']
            min_absolute_gate = LIBERAL_GATING_CONFIG['liberal_min_absolute_gate']
        else:
            # Use default conservative parameters
            peak_mult = PEAK_MULTIPLIER
            valley_max_height = VALLEY_MAX_HEIGHT
            min_percentile = MIN_PERCENTILE
            min_absolute_gate = MIN_ABSOLUTE_GATE

        # Step 1: Stratified subsample
        sampled_vals = []
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            sample_vals = adata.layers['aligned'][mask, marker_idx]
            sample_vals = sample_vals[sample_vals > 0]
            
            if len(sample_vals) > 10000:
                sample_vals = np.random.choice(sample_vals, 10000, replace=False)
            sampled_vals.extend(sample_vals)
        
        sampled_vals = np.array(sampled_vals)

        gating_mode = "LIBERAL" if use_liberal else "conservative"
        print(f"\n  {marker} ({gating_mode}): {len(sampled_vals):,} cells sampled")
        if use_liberal:
            print(f"    Using liberal params: peak_mult={peak_mult}, valley_max={valley_max_height}, "
                  f"min_pct=p{min_percentile}, min_gate={min_absolute_gate}")
        
        # Step 2: Remove artifacts
        if sampled_vals.max() > 65000:
            n_saturated = (sampled_vals > 65000).sum()
            sampled_vals = sampled_vals[sampled_vals <= 65000]
            print(f"    Removed {n_saturated:,} saturated")
        
        median = np.median(sampled_vals)
        mad = median_abs_deviation(sampled_vals)
        outlier_mask = np.abs(sampled_vals - median) < 5 * mad
        sampled_vals = sampled_vals[outlier_mask]
        
        # Check data range
        val_min = sampled_vals.min()
        val_max = sampled_vals.max()
        
        if val_max <= val_min * 1.1:
            gate = float(np.percentile(sampled_vals, 90))
            gates[marker] = gate
            print(f"    Insufficient range → p90 gate={gate:.3f}")
            continue
        
        if len(sampled_vals) < 100:
            gate = float(np.percentile(sampled_vals, 90))
            gates[marker] = gate
            print(f"    Insufficient data → p90 gate={gate:.3f}")
            continue
        
        # Step 3: Build smooth density
        bins = np.logspace(np.log10(max(1, val_min)),
                          np.log10(val_max), 300)

        if not np.all(np.diff(bins) > 0):
            bins = np.linspace(val_min, val_max, 300)

        hist, bin_edges = np.histogram(sampled_vals, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        density = hist_smooth / (hist_smooth.sum() + 1e-10)

        # Step 4: Find negative peak (ROBUST - not just leftmost)
        # OLD: peaks, properties = find_peaks(density, prominence=0.001, distance=5, height=0.0001)
        # NEW: More sensitive peak detection, then intelligent selection
        peaks, properties = find_peaks(density, prominence=0.0005, distance=5, height=0.00005)

        if len(peaks) == 0:
            gate = float(np.percentile(sampled_vals, 90))
            print(f"    No peaks → p90 gate={gate:.3f}")
            gates[marker] = gate
            continue

        # ROBUSTNESS IMPROVEMENT: Identify true negative peak using multiple criteria
        # Criterion 1: Negative peak should be in lower half of intensity distribution
        median_intensity = np.median(sampled_vals)

        # Criterion 2: Calculate cumulative fraction at each peak (population size)
        cumulative_fractions = []
        for peak_idx in peaks:
            peak_val = bin_centers[peak_idx]
            cum_frac = (sampled_vals <= peak_val).mean()
            cumulative_fractions.append(cum_frac)

        # Criterion 3: Peak prominence (how distinct is the peak)
        prominences = properties['prominences']

        # Score each peak as a candidate negative peak
        peak_scores = []
        for i, peak_idx in enumerate(peaks):
            peak_val = bin_centers[peak_idx]
            peak_height = density[peak_idx]

            # Favor peaks in lower intensity range (negative = low signal)
            # Score from 0 (at max) to 1 (at min)
            position_score = 1.0 - (peak_val - val_min) / (val_max - val_min + 1e-10)

            # Favor peaks with large populations before them (negative is usually majority)
            # Score from 0 (few cells) to 1 (most cells)
            population_score = cumulative_fractions[i]

            # Favor prominent peaks
            prominence_score = prominences[i] / (prominences.max() + 1e-10)

            # Combined score (weighted average)
            # Position is most important (0.5), then population (0.3), then prominence (0.2)
            combined_score = (0.5 * position_score +
                            0.3 * population_score +
                            0.2 * prominence_score)

            peak_scores.append(combined_score)

        # Select peak with highest score as negative peak
        best_neg_idx = np.argmax(peak_scores)
        neg_peak_idx = peaks[best_neg_idx]
        neg_peak_val = bin_centers[neg_peak_idx]
        neg_peak_height = density[neg_peak_idx]

        print(f"    Found {len(peaks)} peaks, selected peak #{best_neg_idx+1} as negative:")
        print(f"    Negative peak: position={neg_peak_val:.3f}, height={neg_peak_height:.4f}, "
              f"score={peak_scores[best_neg_idx]:.3f}")

        # Validation: Check if there's a second peak (bimodality check)
        if len(peaks) >= 2:
            # Find the most prominent positive peak (should be right of negative)
            pos_candidates = [i for i, p in enumerate(peaks) if bin_centers[p] > neg_peak_val * 1.2]
            if pos_candidates:
                pos_peak_idx = peaks[pos_candidates[0]]
                pos_peak_val = bin_centers[pos_peak_idx]
                print(f"    Positive peak detected at: position={pos_peak_val:.3f}, "
                      f"separation={pos_peak_val/neg_peak_val:.2f}×")
        
        # Step 5: METHOD 1 - Valley detection (PRIMARY - IMPROVED)
        search_start = neg_peak_idx
        search_end = min(neg_peak_idx + 150, len(density))
        right_side = density[search_start:search_end]

        # IMPROVEMENT: Find multiple local minima and score them
        local_mins = argrelextrema(right_side, np.less, order=5)[0]

        if len(local_mins) > 0:
            # Score each valley candidate
            valley_scores = []
            for valley_idx_local in local_mins:
                valley_height = right_side[valley_idx_local]
                valley_position = bin_centers[search_start + valley_idx_local]

                # Criterion 1: Prefer lower valleys (deeper separation)
                depth_score = 1.0 - (valley_height / (neg_peak_height + 1e-10))

                # Criterion 2: Prefer valleys closer to negative peak (avoid going too far right)
                # But not TOO close (need some separation)
                distance_from_neg = valley_position - neg_peak_val
                ideal_distance = neg_peak_val * 0.5  # Ideally ~0.5× the negative peak value away
                distance_score = np.exp(-((distance_from_neg - ideal_distance)**2) / (ideal_distance**2 + 1e-10))

                # Criterion 3: Check if valley is between two peaks (true valley vs tail)
                # Look for a peak to the right of this valley
                remaining_region = right_side[valley_idx_local:]
                if len(remaining_region) > 10:
                    has_peak_right = np.any(remaining_region > valley_height * 1.2)
                    bimodal_score = 1.0 if has_peak_right else 0.3
                else:
                    bimodal_score = 0.5

                # Combined score
                combined_score = 0.5 * depth_score + 0.3 * distance_score + 0.2 * bimodal_score
                valley_scores.append(combined_score)

            # Select best valley
            best_valley_idx = np.argmax(valley_scores)
            valley_idx_local = local_mins[best_valley_idx]
            gate_valley = float(bin_centers[search_start + valley_idx_local])
            valley_height = right_side[valley_idx_local]

            print(f"    Valley method: found {len(local_mins)} valleys, selected #{best_valley_idx+1}")
            print(f"    Valley: gate={gate_valley:.3f}, height={valley_height:.4f}, score={valley_scores[best_valley_idx]:.3f}")
        else:
            # Fallback: global minimum in search region
            valley_idx_local = np.argmin(right_side)
            gate_valley = float(bin_centers[search_start + valley_idx_local])
            valley_height = right_side[valley_idx_local]
            print(f"    Valley method (global min): gate={gate_valley:.3f}, height={valley_height:.4f}")
        
        # Step 6: METHOD 2 - GMM intersection (BACKUP)
        gate_gmm = gate_valley  # Default to valley
        
        if len(peaks) >= 2:
            # Try GMM on region between first two peaks
            peak1_val = bin_centers[peaks[0]]
            peak2_val = bin_centers[peaks[1]]
            
            between_peaks_mask = (sampled_vals >= peak1_val * 0.8) & \
                                (sampled_vals <= peak2_val * 1.2)
            between_data = sampled_vals[between_peaks_mask]
            
            if len(between_data) > 200:
                try:
                    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100)
                    gmm.fit(between_data.reshape(-1, 1))
                    
                    means = np.sort(gmm.means_.flatten())
                    stds = np.sqrt(gmm.covariances_.flatten())
                    order = np.argsort(means)
                    
                    # Weighted midpoint (accounts for different variances)
                    if stds[order[0]] + stds[order[1]] > 0:
                        gate_gmm = float((means[order[0]] * stds[order[1]] + 
                                         means[order[1]] * stds[order[0]]) / 
                                        (stds[order[0]] + stds[order[1]]))
                    else:
                        gate_gmm = float((means[order[0]] + means[order[1]]) / 2)
                    
                    print(f"    GMM method: gate={gate_gmm:.3f} (means: {means[order[0]]:.3f}, {means[order[1]]:.3f})")
                
                except Exception as e:
                    print(f"    GMM failed: {e}")
                    gate_gmm = gate_valley
        
        # Step 7: Apply biological constraints

        # Constraint 1: Gate must be at least peak_mult× negative peak position
        # (ensures we're clearly right of negative mode)
        min_gate_biological = neg_peak_val * peak_mult

        min_gate_absolute = min_absolute_gate

        # Constraint 2: Gate must be in low-density region
        # (valley height should be below max threshold)
        min_gate_percentile = float(np.percentile(sampled_vals, min_percentile))
        max_valley_height = neg_peak_height * valley_max_height
        
        # Choose gate
        if valley_height < max_valley_height:
            # Valley is clear - use it
            gate_candidate = max(gate_valley, gate_gmm)
        else:
            # No clear valley - valley too high, use GMM or wider search
            print(f"    WARNING: Valley not clear (height {valley_height:.4f} > {max_valley_height:.4f})")
            
            # Try wider search for valley
            search_end_wide = min(neg_peak_idx + 250, len(density))
            right_side_wide = density[search_start:search_end_wide]
            local_mins_wide = argrelextrema(right_side_wide, np.less, order=10)[0]
            
            if len(local_mins_wide) > 0:
                # Use first clear minimum
                for min_idx in local_mins_wide:
                    if right_side_wide[min_idx] < max_valley_height:
                        gate_candidate = float(bin_centers[search_start + min_idx])
                        print(f"    Found clearer valley at {gate_candidate:.3f}")
                        break
                else:
                    # No clear valley found, use GMM
                    gate_candidate = gate_gmm
            else:
                gate_candidate = gate_gmm
        
        # Apply minimum threshold
        # Apply ALL constraints (take maximum = most stringent)
        gate = max(gate_candidate, min_gate_biological, min_gate_absolute, min_gate_percentile)

        print(f"    Constraints: valley={gate_candidate:.3f}, "
            f"peak×{peak_mult}={min_gate_biological:.3f}, "
            f"abs={min_gate_absolute:.3f}, "
            f"p{min_percentile}={min_gate_percentile:.3f} "
            f"→ FINAL={gate:.3f}")
        
        if gate != gate_candidate:
            print(f"    Adjusted {gate_candidate:.3f} → {gate:.3f} (min threshold)")
        
        gates[marker] = gate
        
        # Step 8: Calculate statistics on full dataset
        all_vals = adata.layers['aligned'][:, marker_idx]
        pos_pct = (all_vals > gate).mean() * 100
        
        neg_vals = all_vals[(all_vals > 0) & (all_vals <= gate)]
        pos_vals = all_vals[all_vals > gate]
        
        neg_mean = np.mean(neg_vals) if len(neg_vals) > 0 else 0
        pos_mean = np.mean(pos_vals) if len(pos_vals) > 0 else 0
        
        separation = (pos_mean - neg_mean) / (neg_mean + 1) if neg_mean > 0 else 0
        
        print(f"    FINAL: gate={gate:.3f} | "
              f"neg_μ={neg_mean:.3f} | pos_μ={pos_mean:.3f} | "
              f"separation={separation:.2f}× | {pos_pct:.1f}% positive")
        
        # Warning if separation is poor
        if separation < 1.5:
            print(f"    ⚠️  LOW SEPARATION - populations may overlap")
    
    print("="*70)
    return gates

def quantile_normalize_tiles(adata):
    """
    Quantile normalization per tile (UniFORM-inspired).
    Aligns landmarks across tiles within each sample.
    """
    from scipy.interpolate import PchipInterpolator
    
    print("\nQuantile normalizing tiles within samples...")
    
    adata.layers['pre_tile_norm'] = adata.X.copy()
    
    landmarks_pct = [5, 25, 50, 75, 95]
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{marker}:")
        
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            
            # Get tile IDs for this sample
            if 'tile_id' not in adata.obs.columns:
                adata.obs['tile_id'] = (adata.obs['tile_y'].astype(str) + '_' + 
                                        adata.obs['tile_x'].astype(str))
            
            tiles = adata.obs.loc[sample_mask, 'tile_id'].unique()
            
            # Compute landmarks per tile
            tile_landmarks = {}
            for tile_id in tiles:
                tile_mask = sample_mask & (adata.obs['tile_id'] == tile_id)
                vals = adata.X[tile_mask, marker_idx]
                pos_vals = vals[vals > 0]
                
                if len(pos_vals) < 100:
                    continue
                
                tile_landmarks[tile_id] = np.percentile(pos_vals, landmarks_pct)
            
            if len(tile_landmarks) < 2:
                continue
            
            # Reference = median across tiles
            ref_landmarks = np.median(list(tile_landmarks.values()), axis=0)
            
            # Normalize each tile
            for tile_id, src_lm in tile_landmarks.items():
                tile_mask = sample_mask & (adata.obs['tile_id'] == tile_id)
                vals = adata.X[tile_mask, marker_idx].copy()
                
                # Spline mapping
                src_extended = np.concatenate([[0], src_lm, [src_lm[-1] * 1.5]])
                ref_extended = np.concatenate([[0], ref_landmarks, [ref_landmarks[-1] * 1.5]])
                
                # Ensure monotonic
                for i in range(1, len(src_extended)):
                    if src_extended[i] <= src_extended[i-1]:
                        src_extended[i] = src_extended[i-1] * 1.001
                
                try:
                    spline = PchipInterpolator(src_extended, ref_extended, extrapolate=True)
                    vals_norm = spline(vals)
                    vals_norm = np.maximum(vals_norm, 0)
                    adata.X[tile_mask, marker_idx] = vals_norm
                except Exception as e:
                    print(f"    {tile_id}: FAILED - {e}")
                    continue
            
            print(f"  {sample}: {len(tile_landmarks)} tiles normalized")
    
    return adata

def create_spatial_triple_panel(adata, gates, output_dir):
    """Triple panel: Raw | Aligned | Gated"""
    plots_dir = Path(output_dir) / "spatial_validation"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating spatial triple panels...")
    
    # Check required layers exist
    if 'raw' not in adata.layers:
        print("  WARNING: No 'raw' layer, skipping")
        return
    if 'aligned' not in adata.layers:
        print("  WARNING: No 'aligned' layer, skipping")
        return
    if 'gated' not in adata.layers:
        print("  WARNING: No 'gated' layer, skipping")
        return
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        marker_idx_loc = adata.var_names.get_loc(marker)
        gate = gates[marker]
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            if len(sample_data) > 50000:
                idx = np.random.choice(len(sample_data), 50000, replace=False)
                sample_data = sample_data[idx]
            
            coords = sample_data.obsm['spatial']
            raw_vals = sample_data.layers['raw'][:, marker_idx_loc]
            aligned_vals = sample_data.layers['aligned'][:, marker_idx_loc]
            gated_vals = sample_data.layers['gated'][:, marker_idx_loc]
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'{marker} - {sample}', fontsize=14, fontweight='bold')
            
            # Panel 1: Raw
            raw_log = np.log10(raw_vals + 1)
            sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], 
                                 c=raw_log, s=0.5, alpha=0.6, 
                                 cmap='viridis', rasterized=True)
            axes[0].set_title('Raw Intensity (log10)')
            axes[0].set_aspect('equal')
            plt.colorbar(sc1, ax=axes[0], fraction=0.046)
            
            # Panel 2: Aligned
            aligned_log = np.log10(aligned_vals + 1)
            sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], 
                                 c=aligned_log, s=0.5, alpha=0.6,
                                 cmap='viridis', rasterized=True)
            axes[1].set_title('Aligned Intensity (log10)')
            axes[1].set_aspect('equal')
            plt.colorbar(sc2, ax=axes[1], fraction=0.046)
            
            # Panel 3: Binary gated
            pos_mask = gated_vals > 0
            axes[2].scatter(coords[~pos_mask, 0], coords[~pos_mask, 1],
                          s=0.3, alpha=0.2, c='lightgray', rasterized=True,
                          label=f'Neg: {(~pos_mask).sum():,}')
            axes[2].scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                          s=0.5, alpha=0.8, c='red', rasterized=True,
                          label=f'Pos: {pos_mask.sum():,} ({pos_mask.mean()*100:.1f}%)')
            axes[2].set_title(f'Gated (threshold={gate:.1f})')
            axes[2].set_aspect('equal')
            axes[2].legend(loc='upper right')
            
            for ax in axes:
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{marker}_{sample}_spatial_triple.png', 
                       dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Spatial triple panels saved to {plots_dir}")

def create_normalization_kde_comparison(adata, output_dir):
    """
    Create KDE density comparison plots showing per-sample distributions.
    For each marker, shows three panels with per-sample KDE curves:
    1. Before normalization (raw) - shows sample misalignment
    2. After hierarchical UniFORM normalization (aligned) - shows samples aligned
    3. After 99th percentile normalization - shows poor alignment
    """
    plots_dir = Path(output_dir) / "normalization_comparison"
    plots_dir.mkdir(exist_ok=True, parents=True)

    print("\nCreating normalization KDE comparison plots (per-sample)...")

    from scipy.ndimage import gaussian_filter1d
    import matplotlib.cm as cm

    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)

        print(f"  Processing {marker}...")

        # Get list of samples
        samples = sorted(adata.obs['sample_id'].unique())
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(samples)))

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(f'{marker} - Normalization Method Comparison (Per-Sample KDE)', fontsize=16, fontweight='bold')

        # Helper function to compute KDE-like density for a single sample
        def compute_kde_density(vals, n_bins=200):
            if len(vals) < 10:
                return None, None

            bins = np.logspace(np.log10(max(1, vals.min())),
                              np.log10(vals.max()), n_bins)
            hist, bin_edges = np.histogram(vals, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
            density = hist_smooth / (hist_smooth.sum() + 1e-10)

            return bin_centers, density

        # Plot per-sample KDE curves for each normalization method
        for sample_idx, sample in enumerate(samples):
            mask = adata.obs['sample_id'] == sample
            color = colors[sample_idx]

            # Raw (before normalization)
            raw_sample = adata.layers['raw'][mask, marker_idx]
            raw_sample = raw_sample[raw_sample > 0]
            if len(raw_sample) > 5000:
                raw_sample = np.random.choice(raw_sample, 5000, replace=False)
            raw_sample = raw_sample[raw_sample <= 65000]

            if len(raw_sample) >= 10:
                bin_centers, density = compute_kde_density(raw_sample)
                if density is not None:
                    axes[0].plot(bin_centers, density, linewidth=2, label=sample, color=color, alpha=0.8)

            # Aligned (after hierarchical UniFORM)
            aligned_sample = adata.layers['aligned'][mask, marker_idx]
            aligned_sample = aligned_sample[aligned_sample > 0]
            if len(aligned_sample) > 5000:
                aligned_sample = np.random.choice(aligned_sample, 5000, replace=False)
            aligned_sample = aligned_sample[aligned_sample <= 65000]

            if len(aligned_sample) >= 10:
                bin_centers, density = compute_kde_density(aligned_sample)
                if density is not None:
                    axes[1].plot(bin_centers, density, linewidth=2, label=sample, color=color, alpha=0.8)

            # 99th percentile normalization
            raw_for_percentile = adata.layers['raw'][mask, marker_idx]
            raw_for_percentile = raw_for_percentile[raw_for_percentile > 0]
            p99 = np.percentile(raw_for_percentile, 99) if len(raw_for_percentile) > 0 else 1.0
            percentile_sample = raw_for_percentile / p99 * 1000  # Scale to 0-1000 range
            percentile_sample = percentile_sample[percentile_sample > 0]
            if len(percentile_sample) > 5000:
                percentile_sample = np.random.choice(percentile_sample, 5000, replace=False)
            percentile_sample = percentile_sample[percentile_sample <= 5000]

            if len(percentile_sample) >= 10:
                bin_centers, density = compute_kde_density(percentile_sample)
                if density is not None:
                    axes[2].plot(bin_centers, density, linewidth=2, label=sample, color=color, alpha=0.8)

        # Style plot 1: Raw
        axes[0].set_xlabel('Intensity', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Before Normalization (Raw)', fontsize=13, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].legend(fontsize=9, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Style plot 2: UniFORM
        axes[1].set_xlabel('Intensity', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('After Hierarchical UniFORM (Aligned)', fontsize=13, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].legend(fontsize=9, loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Style plot 3: 99th percentile
        axes[2].set_xlabel('Intensity (scaled to 0-1000)', fontsize=12)
        axes[2].set_ylabel('Density', fontsize=12)
        axes[2].set_title('99th Percentile Normalization', fontsize=13, fontweight='bold')
        axes[2].set_xscale('log')
        axes[2].legend(fontsize=9, loc='upper right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f'{marker}_normalization_comparison.png', dpi=150, bbox_inches='tight')
        plt.close('all')

    print(f"✓ Normalization comparison plots saved to {plots_dir}")

def create_diagnostic_plots(adata, gates, output_dir):
    """Create diagnostic plots showing gating quality"""
    plots_dir = Path(output_dir) / "gating_diagnostics"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating diagnostic plots...")
    
    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)
        gate = gates[marker]
        
        # Subsample for plotting
        plot_vals = []
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.layers['aligned'][mask, marker_idx]
            vals = vals[vals > 0]
            if len(vals) > 5000:
                vals = np.random.choice(vals, 5000, replace=False)
            plot_vals.extend(vals)
        
        plot_vals = np.array(plot_vals)
        
        # Remove saturation for plotting
        plot_vals = plot_vals[plot_vals <= 65000]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{marker} - Gating Diagnostics', fontsize=14, fontweight='bold')
        
        # Linear histogram
        axes[0].hist(plot_vals, bins=100, alpha=0.7, color='steelblue')
        axes[0].axvline(gate, color='red', linestyle='--', linewidth=2, label=f'Gate: {gate:.1f}')
        axes[0].set_xlabel('Aligned Intensity')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Linear Scale')
        axes[0].legend()
        
        # Log histogram
        axes[1].hist(np.log10(plot_vals + 1), bins=100, alpha=0.7, color='steelblue')
        axes[1].axvline(np.log10(gate + 1), color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Log10(Intensity + 1)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Log Scale')
        
        # Density with peaks
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        bins = np.logspace(np.log10(max(1, plot_vals.min())), 
                          np.log10(plot_vals.max()), 200)
        hist, bin_edges = np.histogram(plot_vals, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        density = hist_smooth / hist_smooth.sum()
        
        axes[2].plot(bin_centers, density, 'b-', linewidth=2, label='Density')
        axes[2].axvline(gate, color='red', linestyle='--', linewidth=2, label=f'Gate: {gate:.1f}')
        
        peaks, _ = find_peaks(density, prominence=0.01)
        if len(peaks) > 0:
            axes[2].plot(bin_centers[peaks], density[peaks], 'go', markersize=8, label='Peaks')
        
        axes[2].set_xlabel('Aligned Intensity')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Density with Peak Detection')
        axes[2].legend()
        axes[2].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{marker}_diagnostic.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    print(f"✓ Diagnostics saved to {plots_dir}")

def finalize_gates_with_override(gmm_gates):
    """Use manual gates where specified, otherwise use GMM gates"""
    final_gates = {}
    
    print("\n" + "="*70)
    print("FINAL GATES (manual override + GMM)")
    print("="*70)
    
    for marker in MARKERS.values():
        manual_gate = GATES.get(marker)
        
        if manual_gate is not None and manual_gate > 0:
            # Manual override
            final_gates[marker] = manual_gate
            print(f"  {marker:8s}: {manual_gate:8.1f} (MANUAL OVERRIDE)")
        else:
            # Use GMM
            final_gates[marker] = gmm_gates[marker]
            print(f"  {marker:8s}: {gmm_gates[marker]:8.1f} (GMM auto)")
    
    print("="*70)
    return final_gates

def apply_gates(adata, gates):
    """Apply gates to aligned intensity data (NOT log scale)"""
    print("\nApplying gates to aligned data...")
    
    # Create gated layer on aligned data
    adata.layers['gated'] = np.zeros_like(adata.layers['aligned'])
    
    for i, marker in enumerate(adata.var_names):
        gate = gates[marker]
        adata.layers['gated'][:, i] = (adata.layers['aligned'][:, i] > gate).astype(float)
    
    # Summary
    print("\n" + "="*70)
    print("GATING SUMMARY")
    print("="*70)
    for marker in adata.var_names:
        marker_idx = adata.var_names.get_loc(marker)
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            pos = (adata.layers['gated'][mask, marker_idx] > 0).sum()
            total = mask.sum()
            gate_val = gates[marker]
            print(f"{marker:8s} | {sample:12s} | gate={gate_val:8.1f} | "
                  f"{pos:8,}/{total:8,} ({pos/total*100:5.1f}%)")
    print("="*70 + "\n")
    
    return adata

def create_validation_plots(adata, gates, output_dir):
    """
    Create merged histograms and density curves per marker.
    Works on aligned intensity scale (not 0-1).
    """
    plots_dir = Path(output_dir) / "gating_validation"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating validation plots...")
    
    from scipy.stats import gaussian_kde
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        
        gate = gates[marker]
        marker_idx_loc = adata.var_names.get_loc(marker)
        
        # Create figure with 3 subplots: histogram, density curves, spatial
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        fig.suptitle(f'{marker} - All Samples', fontsize=14, fontweight='bold')
        
        # === Panel 1: Merged Histogram (all samples overlaid) ===
        colors = plt.cm.tab20(np.linspace(0, 1, adata.obs['sample_id'].nunique()))
        
        for i, sample in enumerate(adata.obs['sample_id'].unique()):
            mask = adata.obs['sample_id'] == sample
            vals = adata.layers['aligned'][mask, marker_idx_loc]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 100:
                continue
            
            # Histogram with alpha for overlay
            ax1.hist(pos_vals, bins=100, alpha=0.4, label=sample, 
                    color=colors[i], density=True)
        
        ax1.axvline(gate, color='red', linestyle='--', linewidth=2, 
                   label=f'Gate: {gate:.1f}')
        ax1.set_xlabel('Aligned Intensity')
        ax1.set_ylabel('Density')
        ax1.set_title('Overlaid Histograms')
        ax1.legend(fontsize=6, loc='upper right')
        ax1.set_xlim(0, np.percentile(adata.layers['aligned'][:, marker_idx_loc][
            adata.layers['aligned'][:, marker_idx_loc] > 0], 99))
        
        # === Panel 2: Density Curves (KDE) ===
        x_range = np.linspace(0, np.percentile(adata.layers['aligned'][:, marker_idx_loc][
            adata.layers['aligned'][:, marker_idx_loc] > 0], 99), 500)
        
        for i, sample in enumerate(adata.obs['sample_id'].unique()):
            mask = adata.obs['sample_id'] == sample
            vals = adata.layers['aligned'][mask, marker_idx_loc]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 100:
                continue
            
            # Subsample for KDE efficiency
            if len(pos_vals) > 50000:
                pos_vals = np.random.choice(pos_vals, 50000, replace=False)
            
            # KDE
            try:
                kde = gaussian_kde(pos_vals, bw_method=0.1)
                density = kde(x_range)
                ax2.plot(x_range, density, label=sample, color=colors[i], linewidth=1.5)
            except:
                continue
        
        ax2.axvline(gate, color='red', linestyle='--', linewidth=2, 
                   label=f'Gate: {gate:.1f}')
        ax2.set_xlabel('Aligned Intensity')
        ax2.set_ylabel('Density')
        ax2.set_title('Density Curves (KDE)')
        ax2.legend(fontsize=6, loc='upper right')
        ax2.set_xlim(0, np.percentile(adata.layers['aligned'][:, marker_idx_loc][
            adata.layers['aligned'][:, marker_idx_loc] > 0], 99))
        
        # === Panel 3: GMM Fit ===
        # Fit GMM on all data for visualization
        all_vals = adata.layers['aligned'][:, marker_idx_loc]
        pos_vals_all = all_vals[all_vals > 0]
        
        if len(pos_vals_all) > 100000:
            pos_vals_all = np.random.choice(pos_vals_all, 100000, replace=False)
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(pos_vals_all.reshape(-1, 1))
        
        # Plot histogram
        ax3.hist(pos_vals_all, bins=100, alpha=0.5, density=True, color='gray', label='Data')
        
        # Plot GMM components
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        
        for i in order:
            x = np.linspace(means[i] - 3*stds[i], means[i] + 3*stds[i], 100)
            y = gmm.weights_[i] * (1/(stds[i] * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - means[i])/stds[i])**2)
            ax3.plot(x, y, linewidth=2, label=f'Component {i+1}')
        
        ax3.axvline(gate, color='red', linestyle='--', linewidth=2, 
                   label=f'Gate: {gate:.1f}')
        ax3.set_xlabel('Aligned Intensity')
        ax3.set_ylabel('Density')
        ax3.set_title('GMM Fit (2 components)')
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, np.percentile(pos_vals_all, 99))
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{marker}_merged.png', dpi=200, bbox_inches='tight')
        plt.close('all')
    
    print(f"✓ Merged histograms saved to {plots_dir}")

def create_per_sample_histograms(adata, gates, output_dir):
    """Individual histograms per sample (for detailed inspection)"""
    plots_dir = Path(output_dir) / "per_sample_histograms"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating per-sample histograms...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        marker_idx_loc = adata.var_names.get_loc(marker)
        gate = gates[marker]
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            vals = adata.layers['aligned'][mask, marker_idx_loc]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) < 100:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'{marker} - {sample}', fontsize=12)
            
            # Linear scale
            axes[0].hist(pos_vals, bins=100, alpha=0.7, color='steelblue')
            axes[0].axvline(gate, color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Aligned Intensity')
            axes[0].set_ylabel('Count')
            pos_pct = (vals > gate).mean() * 100
            axes[0].text(0.95, 0.95, f'{pos_pct:.1f}%', 
                        transform=axes[0].transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat'))
            
            # Log scale
            axes[1].hist(np.log10(pos_vals + 1), bins=100, alpha=0.7, color='steelblue')
            axes[1].axvline(np.log10(gate + 1), color='red', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Log10(Aligned Intensity + 1)')
            axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{marker}_{sample}_hist.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Per-sample histograms saved to {plots_dir}")

def create_spatial_plots(adata, gates, output_dir):
    """Spatial plots using aligned intensity scale"""
    plots_dir = Path(output_dir) / "spatial_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating spatial plots...")
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"  {marker_idx+1}/{len(adata.var_names)}: {marker}")
        marker_idx_loc = adata.var_names.get_loc(marker)
        gate = gates[marker]
        
        for sample in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample
            sample_data = adata[mask]
            
            # Subsample
            if len(sample_data) > 100000:
                idx = np.random.choice(len(sample_data), 100000, replace=False)
                sample_data = sample_data[idx]
            
            vals = sample_data.layers['aligned'][:, marker_idx_loc]
            pos_mask = vals > gate
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # All cells gray
            ax.scatter(sample_data.obsm['spatial'][:, 0], 
                      sample_data.obsm['spatial'][:, 1],
                      s=0.3, alpha=0.3, c='lightgray', rasterized=True)
            
            # Positive cells red
            ax.scatter(sample_data.obsm['spatial'][pos_mask, 0],
                      sample_data.obsm['spatial'][pos_mask, 1],
                      s=0.5, alpha=0.8, c='red', rasterized=True)
            
            ax.set_title(f'{marker} - {sample}\n{pos_mask.sum():,} pos ({pos_mask.mean()*100:.1f}%)')
            ax.set_aspect('equal')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{marker}_{sample}_spatial.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    print(f"✓ Spatial plots saved to {plots_dir}")

def fast_detect_tile_size(adata, sample, max_cells=50000):
    """
    Fast tile detection using subsampled cells and autocorrelation.
    """
    import time
    from scipy.signal import correlate2d, find_peaks
    from scipy.spatial import cKDTree
    
    start = time.time()
    
    sample_mask = adata.obs['sample_id'] == sample
    coords = adata.obsm['spatial'][sample_mask]
    
    # Subsample for speed
    if len(coords) > max_cells:
        idx = np.random.choice(len(coords), max_cells, replace=False)
        coords = coords[idx]
        vals = adata.X[sample_mask][idx].sum(axis=1)
    else:
        vals = adata.X[sample_mask].sum(axis=1)
    
    print(f"    Using {len(coords):,} cells for detection")
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # COARSE grid (50×50 instead of 300×300)
    grid_size = 50
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    print(f"    Building {grid_size}×{grid_size} intensity map...")
    
    # Fast binning using digitize
    x_indices = np.digitize(coords[:, 0], x_bins) - 1
    y_indices = np.digitize(coords[:, 1], y_bins) - 1
    
    # Clip to valid range
    x_indices = np.clip(x_indices, 0, grid_size-2)
    y_indices = np.clip(y_indices, 0, grid_size-2)
    
    # Build intensity map using bincount (very fast)
    intensity_map = np.zeros((grid_size-1, grid_size-1))
    
    for i in range(grid_size-1):
        for j in range(grid_size-1):
            mask = (x_indices == i) & (y_indices == j)
            if mask.sum() > 0:
                intensity_map[j, i] = np.median(vals[mask])
    
    # Autocorrelation to find repeating pattern (tile size)
    from scipy.ndimage import gaussian_filter
    intensity_smooth = gaussian_filter(intensity_map, sigma=1)
    
    # 1D autocorrelation along x and y
    x_profile = intensity_smooth.mean(axis=0)
    y_profile = intensity_smooth.mean(axis=1)
    
    # Find peaks in autocorrelation
    x_autocorr = np.correlate(x_profile, x_profile, mode='full')[len(x_profile)-1:]
    y_autocorr = np.correlate(y_profile, y_profile, mode='full')[len(y_profile)-1:]
    
    # Peaks in autocorrelation = tile spacing
    x_peaks, _ = find_peaks(x_autocorr[1:], distance=3, prominence=x_autocorr.max()*0.1)
    y_peaks, _ = find_peaks(y_autocorr[1:], distance=3, prominence=y_autocorr.max()*0.1)
    
    if len(x_peaks) > 0 and len(y_peaks) > 0:
        # Convert from grid units to pixels
        x_tile_spacing_bins = x_peaks[0]
        y_tile_spacing_bins = y_peaks[0]
        
        x_tile_size = int((x_max - x_min) / (grid_size-1) * x_tile_spacing_bins)
        y_tile_size = int((y_max - y_min) / (grid_size-1) * y_tile_spacing_bins)
        
        # Average and round to common sizes
        avg_tile_size = int((x_tile_size + y_tile_size) / 2)
        
        # Round to nearest common tile size
        common_sizes = [512, 1024, 2048, 2560, 4096]
        tile_size = min(common_sizes, key=lambda x: abs(x - avg_tile_size))
        
        elapsed = time.time() - start
        print(f"    Detected tile size: {tile_size}×{tile_size} pixels ({elapsed:.1f}s)")
        
        return tile_size
    
    else:
        print(f"    Could not detect tiles, using default 2048")
        return 2048

def load_or_detect_tile_config(adata, config_file='tile_config.json'):
    """
    Load tile size from config, or detect and save.
    """
    from pathlib import Path
    import json
    
    config_path = Path(config_file)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        tile_size = config.get('microscope_tile_size')
        print(f"\n✓ Loaded tile size from config: {tile_size}×{tile_size} pixels")
        print(f"  (Delete {config_file} to re-detect)")
        return tile_size
    
    else:
        print("\nNo tile config found, detecting...")
        first_sample = adata.obs['sample_id'].unique()[0]
        
        tile_size = fast_detect_tile_size(adata, first_sample)
        
        # Save config
        config = {
            'microscope_tile_size': tile_size,
            'detected_from_sample': first_sample,
            'detection_date': str(pd.Timestamp.now())
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Saved tile config to: {config_file}")
        print(f"  Tile size: {tile_size}×{tile_size} pixels")
        print(f"  This will be reused for all future runs")
        
        return tile_size

def assign_tiles_fast(adata, tile_size):
    """
    Fast tile assignment using simple grid (no detection needed).
    """
    print(f"\nAssigning tiles ({tile_size}×{tile_size})...")
    
    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        coords = adata.obsm['spatial'][sample_mask]
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Simple grid assignment
        x_tile_ids = ((coords[:, 0] - x_min) / tile_size).astype(int)
        y_tile_ids = ((coords[:, 1] - y_min) / tile_size).astype(int)
        
        tile_ids = [f"{y}_{x}" for y, x in zip(y_tile_ids, x_tile_ids)]
        adata.obs.loc[sample_mask, 'tile_id'] = tile_ids
        
        n_tiles = len(np.unique(tile_ids))
        n_cells = sample_mask.sum()
        print(f"  {sample}: {n_tiles} tiles ({n_cells/n_tiles:.0f} cells/tile)")
    
    print("✓ Tile assignment complete")

def hierarchical_uniform_normalization(adata, autodetect_tiles=True, n_jobs=8,
                                      config_file='tile_config.json',
                                      skip_within_tile=False, skip_cross_sample=False):
    """
    Hierarchical UniFORM with optimized parallelization.
    
    Parameters:
    -----------
    skip_within_tile : bool
        If True, skip level 1 (within-tile). Faster but less accurate.
    """
    from scipy.interpolate import PchipInterpolator
    from joblib import Parallel, delayed, parallel_backend
    import time
    
    # Manual skip list for Level 2 normalization
    SKIP_LEVEL2 = {
        'TOM': ['GUEST43', 'GUEST45', 'GUEST46', 'GUEST47'],  # Late-stage samples
        # Add more as needed:
        # 'CD45': ['Guest47'],
        # 'PERK': [],
    }

    print("\n=== HIERARCHICAL UNIFORM NORMALIZATION ===")
    print(f"Using {n_jobs} parallel jobs (multiprocessing backend)")
    
    adata.layers['raw'] = adata.X.copy()
    
    # Load or detect tile size
    if autodetect_tiles:
        tile_size = load_or_detect_tile_config(adata, config_file)
        assign_tiles_fast(adata, tile_size)
    
    # Count tiles
    if 'tile_id' in adata.obs.columns:
        total_tiles = 0
        tile_stats = []
        for sample in adata.obs['sample_id'].unique():
            sample_mask = adata.obs['sample_id'] == sample
            n_tiles = adata.obs.loc[sample_mask, 'tile_id'].nunique()
            n_cells = sample_mask.sum()
            total_tiles += n_tiles
            tile_stats.append((sample, n_tiles, n_cells))
            print(f"  {sample}: {n_tiles} tiles, {n_cells:,} cells")
        print(f"  TOTAL: {total_tiles} tiles")
        
        if skip_within_tile:
            print("  ⚠️  Skipping within-tile normalization (--skip_within_tile)")
    
    landmarks_pct = [5, 10, 25, 40, 50, 60, 75, 90, 95, 99]#[5, 25, 50, 75, 95]
    
    for marker_idx, marker in enumerate(adata.var_names):
        print(f"\n{'='*70}")
        print(f"{marker} ({marker_idx+1}/{len(adata.var_names)})")
        print('='*70)
        
        # ====================================================================
        # LEVEL 1: WITHIN-TILE NORMALIZATION (PARALLELIZED)
        # ====================================================================
        print("  Level 1: Background-anchored tile correction...")
        start_time = time.time()
        
        def process_sample_tiles(sample, sample_mask, marker_idx, X_data, obs_data):
            """Process all tiles for one sample"""
            tiles = obs_data.loc[sample_mask, 'tile_id'].unique()
            
            # Global reference = 5th percentile across all tiles
            sample_vals = X_data[sample_mask, marker_idx]
            pos_sample_vals = sample_vals[sample_vals > 0]
            if len(pos_sample_vals) < 100:
                return None
            global_bg = np.percentile(pos_sample_vals, 5)
            
            corrections = {}
            for tile_id in tiles:
                tile_mask = sample_mask & (obs_data['tile_id'] == tile_id)
                vals = X_data[tile_mask, marker_idx]
                
                pos_vals_tile = vals[vals > 0]
                if len(pos_vals_tile) < 10:
                    continue
                tile_bg = np.percentile(pos_vals_tile, 5)
                
                if tile_bg > 0:
                    corrections[tile_id] = global_bg / tile_bg
            
            return sample, corrections
        if not skip_within_tile:
            # Parallel processing across samples
            sample_masks = {sample: adata.obs['sample_id'] == sample 
                        for sample in adata.obs['sample_id'].unique()}
            
            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(process_sample_tiles)(sample, mask, marker_idx, adata.X, adata.obs)
                for sample, mask in sample_masks.items()
            )
            
            # Apply corrections
            for result in results:
                if result is None:
                    continue
                sample, corrections = result
                sample_mask = sample_masks[sample]
                
                for tile_id, factor in corrections.items():
                    tile_mask = sample_mask & (adata.obs['tile_id'] == tile_id)
                    adata.X[tile_mask, marker_idx] *= factor
        
        elapsed = time.time() - start_time
        print(f"    ✓ Level 1 complete in {elapsed:.1f}s")
        
        # ====================================================================
        # LEVEL 2: ACROSS TILES WITHIN SAMPLE (fast, serial is fine)
        # ====================================================================
        print("\n  Level 2: Across tiles within sample...")
        start_time = time.time()
        
        # Check manual skip list
        skip_samples = SKIP_LEVEL2.get(marker, [])
        if skip_samples:
            print(f"    Manual skip enabled for: {skip_samples}")
        
        sample_references = {}
        
        for sample in adata.obs['sample_id'].unique():
            # Skip if in manual list
            if sample in skip_samples:
                print(f"    {sample}: SKIPPED (manual override)")
                continue
            
            sample_mask = adata.obs['sample_id'] == sample
            vals = adata.X[sample_mask, marker_idx]
            pos_vals = vals[vals > 0]
            
            if len(pos_vals) >= 1000:
                sample_landmarks = np.percentile(pos_vals, landmarks_pct)
                sample_references[sample] = sample_landmarks
        
        if len(sample_references) < 2:
            print("    Only 1 sample, skipping")
        else:
            global_sample_ref = np.median(list(sample_references.values()), axis=0)
            
            for sample, src_lm in sample_references.items():
                sample_mask = adata.obs['sample_id'] == sample
                vals = adata.X[sample_mask, marker_idx].copy()
                
                src_extended = np.concatenate([[0], src_lm, [src_lm[-1] * 1.5]])
                ref_extended = np.concatenate([[0], global_sample_ref, [global_sample_ref[-1] * 1.5]])
                
                for i in range(1, len(src_extended)):
                    if src_extended[i] <= src_extended[i-1]:
                        src_extended[i] = src_extended[i-1] * 1.001
                
                try:
                    spline = PchipInterpolator(src_extended, ref_extended, 
                                              extrapolate=True)
                    vals_norm = spline(vals)
                    vals_norm = np.maximum(vals_norm, 0)
                    adata.X[sample_mask, marker_idx] = vals_norm
                    
                    before = np.median(vals[vals > 0])
                    after = np.median(vals_norm[vals_norm > 0])
                    print(f"      {sample}: {before:.0f} → {after:.0f}")
                except Exception as e:
                    print(f"      {sample}: FAILED - {e}")
            
            elapsed = time.time() - start_time
            print(f"    ✓ Level 2 complete in {elapsed:.1f}s")
        
        # Measure dynamic range
        vals = adata.X[:, marker_idx]
        pos_vals = vals[vals > 0]
        p5 = np.percentile(pos_vals, 5)
        p95 = np.percentile(pos_vals, 95)
        dynamic_range = p95 / (p5 + 1)

        # Adaptive percentiles
        if dynamic_range > 50:  # TOM, CD8B
            top_pct = 99.7
            landmarks_weight = [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1]  # Weight extremes
        elif dynamic_range > 20:  # CD45, KI67
            top_pct = 99.3
            landmarks_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:  # PERK, AGFP
            top_pct = 99.0
            landmarks_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        print(f"    Dynamic range: {dynamic_range:.1f}x → using p{top_pct:.1f}")

        # ====================================================================
        # LEVEL 3: FINAL SCALING
        # ====================================================================
        print("\n  Level 3: Final 0-1 scaling...")
        
        vals = adata.X[:, marker_idx]
        pos_vals = vals[vals > 0]
        
        if len(pos_vals) > 100:
            # Contrast enhancement with background removal
            p10 = np.percentile(pos_vals, 10)  # New "zero point"
            p_top = np.percentile(pos_vals, top_pct)  # From Fix 2

            if p_top > p10:
                # Subtract background, then scale
                vals_bg_removed = vals - p10
                vals_bg_removed = np.maximum(vals_bg_removed, 0)
                
                # Scale to 0-1
                vals_scaled = vals_bg_removed / (p_top - p10)
                vals_scaled = np.clip(vals_scaled, 0, 1)
                
                # GAMMA correction for middle-range boost
                gamma = 0.8 if dynamic_range > 50 else 0.9  # <1 = boost mid-range
                vals_gamma = np.power(vals_scaled, gamma)
                
                adata.X[:, marker_idx] = vals_gamma
                
                print(f"    Contrast: p10={p10:.0f}→0, p{top_pct:.1f}={p_top:.0f}→1, γ={gamma}")
    

    # Marker-specific asinh (OPTIONAL - only if still too compressed)
    if dynamic_range > 30:
        # Calculate marker-specific cofactor
        cofactor = max(50, p5)  # 5th percentile of positives
        
        # Convert back to intensity scale
        vals_intensity = adata.X[:, marker_idx] * (p_top - p10) + p10
        
        # Asinh transform
        vals_asinh = np.arcsinh(vals_intensity / cofactor)
        
        # Renormalize to 0-1
        vals_asinh = vals_asinh / np.percentile(vals_asinh[vals_asinh > 0], 99.5)
        vals_asinh = np.clip(vals_asinh, 0, 1)
        
        adata.X[:, marker_idx] = vals_asinh
        
        print(f"    Asinh: cofactor={cofactor:.0f}")

    adata.layers['aligned'] = adata.X.copy()
    
    print("\n" + "="*70)
    print("HIERARCHICAL NORMALIZATION COMPLETE")
    print("="*70)
    
    return adata

"""
def detect_and_assign_tiles(adata, tile_size_estimate=2048, use_metadata=False):
    
    Autodetect tile boundaries.
    
    Parameters:
    -----------
    use_metadata : bool
        If False, always use edge detection (ignores tile_x/tile_y)
    
    from scipy.ndimage import sobel, gaussian_filter
    import time
    
    print("\n" + "="*70)
    print("TILE DETECTION")
    print("="*70)
    
    start_time = time.time()
    
    for sample_idx, sample in enumerate(adata.obs['sample_id'].unique()):
        print(f"\n[{sample_idx+1}/{adata.obs['sample_id'].nunique()}] {sample}")
        sample_mask = adata.obs['sample_id'] == sample
        coords = adata.obsm['spatial'][sample_mask]
        n_cells = sample_mask.sum()
        
        print(f"  Cells: {n_cells:,}")
        print(f"  Spatial extent: {coords[:, 0].min():.0f}-{coords[:, 0].max():.0f} x "
              f"{coords[:, 1].min():.0f}-{coords[:, 1].max():.0f} pixels")
        
        # Check if metadata exists
        has_metadata = 'tile_x' in adata.obs.columns and 'tile_y' in adata.obs.columns
        
        if has_metadata and use_metadata:
            print(f"  WARNING: Using computational tiles (tile_x/tile_y)")
            print(f"  These may NOT match microscope acquisition tiles!")
            tile_ids = (adata.obs.loc[sample_mask, 'tile_y'].astype(str) + '_' + 
                       adata.obs.loc[sample_mask, 'tile_x'].astype(str))
            adata.obs.loc[sample_mask, 'tile_id'] = tile_ids
            n_tiles = tile_ids.nunique()
            cells_per_tile = n_cells / n_tiles
            print(f"  ✓ Method: Metadata (computational tiles)")
            print(f"  ✓ Tiles: {n_tiles} ({cells_per_tile:.0f} cells/tile avg)")
            continue
        
        # Edge detection (finds real microscope tiles)
        print(f"  Method: Edge detection (finding microscope tiles)...")
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Fine grid for edge detection
        grid_size = 300  # Increased from 200 for better resolution
        x_bins = np.linspace(x_min, x_max, grid_size)
        y_bins = np.linspace(y_min, y_max, grid_size)
        
        print(f"    Building intensity map ({grid_size}×{grid_size})...")
        
        intensity_map = np.zeros((grid_size-1, grid_size-1))
        
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                       (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1]))
                if mask.sum() > 0:
                    # Use total intensity across all channels
                    intensity_map[j, i] = np.median(adata.X[sample_mask][mask].sum(axis=1))
        
        # Smooth to reduce noise
        intensity_smooth = gaussian_filter(intensity_map, sigma=2)
        
        # Edge detection
        edges_x = sobel(intensity_smooth, axis=1)
        edges_y = sobel(intensity_smooth, axis=0)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Find strong vertical and horizontal edges separately
        edge_threshold = np.percentile(edges[edges > 0], 85)  # Lowered from 90 for sensitivity
        
        # Sum edges along axes to find tile boundaries
        x_edge_strength = edges.sum(axis=0)
        y_edge_strength = edges.sum(axis=1)
        
        # Find peaks in edge strength (tile boundaries)
        from scipy.signal import find_peaks
        
        x_peaks, _ = find_peaks(x_edge_strength, height=edge_threshold * grid_size / 15, distance=5)
        y_peaks, _ = find_peaks(y_edge_strength, height=edge_threshold * grid_size / 15, distance=5)
        
        print(f"    Detected {len(x_peaks)} vertical edges, {len(y_peaks)} horizontal edges")
        
        if len(x_peaks) > 1 and len(y_peaks) > 1:
            x_edges = x_bins[x_peaks]
            y_edges = y_bins[y_peaks]
            
            # Show estimated tile size
            avg_x_spacing = np.mean(np.diff(x_edges)) if len(x_edges) > 1 else 0
            avg_y_spacing = np.mean(np.diff(y_edges)) if len(y_edges) > 1 else 0
            print(f"    Estimated microscope tile size: {avg_x_spacing:.0f} × {avg_y_spacing:.0f} pixels")
            
            x_tile_ids = np.digitize(coords[:, 0], x_edges)
            y_tile_ids = np.digitize(coords[:, 1], y_edges)
            tile_ids = [f"{y}_{x}" for y, x in zip(y_tile_ids, x_tile_ids)]
            adata.obs.loc[sample_mask, 'tile_id'] = tile_ids
            n_tiles = len(np.unique(tile_ids))
            cells_per_tile = n_cells / n_tiles
            print(f"  ✓ Microscope tiles: {n_tiles} ({cells_per_tile:.0f} cells/tile avg)")
            
        else:
            # Fallback: estimate based on known tile size
            print(f"    Edge detection failed, using grid fallback (tile_size={tile_size_estimate})")
            n_tiles_x = max(2, int((x_max - x_min) / tile_size_estimate))
            n_tiles_y = max(2, int((y_max - y_min) / tile_size_estimate))
            
            x_tile_ids = ((coords[:, 0] - x_min) / (x_max - x_min) * n_tiles_x).astype(int)
            y_tile_ids = ((coords[:, 1] - y_min) / (y_max - y_min) * n_tiles_y).astype(int)
            
            tile_ids = [f"{y}_{x}" for y, x in zip(y_tile_ids, x_tile_ids)]
            adata.obs.loc[sample_mask, 'tile_id'] = tile_ids
            n_tiles = n_tiles_x * n_tiles_y
            cells_per_tile = n_cells / n_tiles
            print(f"  ✓ Grid fallback: {n_tiles_x}×{n_tiles_y} = {n_tiles} tiles")
            print(f"  ✓ Estimated tile size: {tile_size_estimate}×{tile_size_estimate} pixels")
            print(f"  ✓ Cells/tile: {cells_per_tile:.0f} avg")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TILE DETECTION COMPLETE in {elapsed:.1f}s")
    print("="*70)
    
    return adata
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='manual_gating_output')
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--tile_config', default='tile_config.json')
    parser.add_argument('--redetect_tiles', action='store_true')
    parser.add_argument('--skip_normalization', action='store_true')
    parser.add_argument('--force_normalization', action='store_true')
    parser.add_argument('--skip_within_tile', action='store_true',
                       help='Skip within-tile normalization (faster, less accurate)')
    parser.add_argument('--skip_cross_sample', action='store_true',
                   help='Skip cross-sample normalization (preserve biological differences)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Delete config if re-detection requested
    if args.redetect_tiles and Path(args.tile_config).exists():
        Path(args.tile_config).unlink()
        print(f"Deleted {args.tile_config} - will re-detect")
    
    # Checkpoint file
    checkpoint_file = output_dir / 'normalized_data.h5ad'
    
    print("="*70)
    print("MANUAL GATING PIPELINE")
    print("="*70)
    
    # ====================================================================
    # LOAD OR NORMALIZE
    # ====================================================================
    if checkpoint_file.exists() and not args.force_normalization:
        print(f"\n✓ Found cached normalized data: {checkpoint_file}")
        print("  Loading from checkpoint (use --force_normalization to re-run)")
        
        import time
        start = time.time()
        adata = ad.read_h5ad(checkpoint_file)
        elapsed = time.time() - start
        
        print(f"✓ Loaded in {elapsed:.1f}s")
        print(f"  {len(adata):,} cells, {len(adata.var_names)} markers")
        print(f"  Layers: {list(adata.layers.keys())}")
        
    else:
        if args.skip_normalization and not checkpoint_file.exists():
            print("❌ ERROR: --skip_normalization specified but no checkpoint found")
            print(f"Expected: {checkpoint_file}")
            exit(1)
        
        print("\nRunning hierarchical normalization...")

        adata = load_and_combine(args.results_dir)

        # ====================================================================
        # TILE BOUNDARY CORRECTION (before normalization)
        # ====================================================================
        # Apply gradient-based tile boundary correction BEFORE UniFORM normalization
        # This ensures the normalization works on already-corrected data
        adata = correct_tile_artifacts_per_marker(adata)

        # Hierarchical normalization
        adata = hierarchical_uniform_normalization(
            adata,
            autodetect_tiles=True,
            n_jobs=args.n_jobs,
            config_file=args.tile_config,
            skip_within_tile=args.skip_within_tile,
            skip_cross_sample=args.skip_cross_sample
        )

        # Save checkpoint
        print(f"\n💾 Saving normalized data to: {checkpoint_file}")
        adata.write(checkpoint_file)
        print("✓ Checkpoint saved")

    # ====================================================================
    # GATING (always runs)
    # ====================================================================
    print("\n" + "="*70)
    print("GATING WORKFLOW")
    print("="*70)

    # Normalization comparison KDE plots (FIRST - can resume from checkpoint after this)
    create_normalization_kde_comparison(adata, output_dir)

    # Visualize tile correction
    visualize_tile_artifacts(adata, output_dir)

    # Density-based gating
    density_gates = density_based_gating(adata)
    gates = finalize_gates_with_override(density_gates)

    # Apply hierarchical marker relationships (child markers must be subset of parent)
    adata = apply_hierarchical_gating(adata, gates)

    # Diagnostic plots
    create_diagnostic_plots(adata, gates, output_dir)

    # Save gates
    with open(output_dir / 'gates.json', 'w') as f:
        json.dump({k: float(v) for k, v in gates.items()}, f, indent=2)

    # Apply gates
    adata = apply_gates(adata, gates)
    
    # Validation plots
    create_validation_plots(adata, gates, output_dir)
    create_per_sample_histograms(adata, gates, output_dir)
    create_spatial_triple_panel(adata, gates, output_dir)
    
    # Save final gated data
    adata.write(output_dir / 'gated_data.h5ad')
    
    print(f"\n✅ Complete! Output: {output_dir}")
    print(f"   - normalized_data.h5ad: Checkpoint (reuse with --skip_normalization)")
    print(f"   - gated_data.h5ad: Final output")
    print(f"   - gates.json: Gate values")

if __name__ == '__main__':
    main()