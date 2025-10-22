#!/usr/bin/env python3
"""
UniFORM-Based Hierarchical Gating Pipeline
===========================================

Implements UniFORM (Uniform Normalization using Functional Data Registration)
adapted for cyclic immunofluorescence data with hierarchical normalization:

1. TILE-LEVEL NORMALIZATION: Correct tiling artifacts within each sample
2. SAMPLE-LEVEL NORMALIZATION: Align samples globally
3. ROBUST GATING: Detect negative and positive peaks, semi-conservative gating

Based on:
Wang et al. "UniFORM: A unified framework for functional data registration and
marker detection in imaging mass cytometry" (Nature Methods, 2023)
https://github.com/kunlunW/UniFORM

Usage:
    python uniform_gating.py --results_dir results --output_dir uniform_gating_output

Author: Automated generation
Date: 2025-10-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import anndata as ad
from scipy import signal, interpolate, stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
from tqdm import tqdm

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

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

# Normalization parameters
COFACTOR = 150  # asinh transformation cofactor
N_BINS = 300  # Number of bins for density estimation
LANDMARKS_PCT = [5, 25, 50, 75, 95]  # Landmark quantiles for registration

# Peak detection parameters
PEAK_PROMINENCE = 0.001  # Minimum prominence for peak detection
PEAK_DISTANCE = 5  # Minimum distance between peaks (in bins)
PEAK_MIN_HEIGHT = 0.0001  # Minimum peak height

# Gating parameters
SEMI_CONSERVATIVE_PERCENTILE = 0.4  # Gate at 40% between peaks (semi-conservative)
MIN_SEPARATION = 1.5  # Minimum fold-change separation for valid gating


# ============================================================================
# HIERARCHICAL UNIFORM NORMALIZATION
# ============================================================================

class HierarchicalUniFORM:
    """
    Hierarchical UniFORM normalization: Tile-level then Sample-level.

    UniFORM uses functional data registration to align negative peaks
    across samples by warping the intensity distributions.
    """

    def __init__(self, adata: ad.AnnData, markers: List[str],
                 tile_col: str = 'tile_id', sample_col: str = 'sample_id'):
        """
        Initialize hierarchical normalization.

        Parameters
        ----------
        adata : AnnData
            Annotated data with raw intensities in adata.X
        markers : list
            List of marker names (columns in adata.var_names)
        tile_col : str
            Column name in adata.obs containing tile IDs
        sample_col : str
            Column name in adata.obs containing sample IDs
        """
        self.adata = adata
        self.markers = markers
        self.tile_col = tile_col
        self.sample_col = sample_col

        # Storage for normalization artifacts
        self.tile_warps = {}  # Warping functions per tile
        self.sample_warps = {}  # Warping functions per sample
        self.negative_peaks = {}  # Detected negative peaks

        print(f"Initialized HierarchicalUniFORM")
        print(f"  Markers: {len(markers)}")
        print(f"  Samples: {adata.obs[sample_col].nunique()}")
        if tile_col in adata.obs:
            print(f"  Tiles: {adata.obs[tile_col].nunique()}")


    def compute_smooth_density(self, values: np.ndarray,
                               bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute smooth density estimate in log-space.

        Parameters
        ----------
        values : array
            Intensity values
        bins : array, optional
            Bin edges. If None, create log-spaced bins.

        Returns
        -------
        bin_centers : array
            Bin centers
        density : array
            Smoothed density
        bins : array
            Bin edges used
        """
        # Remove zeros and negatives
        values = values[values > 0]

        if len(values) < 100:
            return None, None, None

        # Create log-spaced bins
        if bins is None:
            val_min = max(1, values.min())
            val_max = values.max()
            bins = np.logspace(np.log10(val_min), np.log10(val_max), N_BINS)

        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth with Gaussian filter
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)

        # Normalize to density
        density = hist_smooth / (hist_smooth.sum() + 1e-10)

        return bin_centers, density, bins


    def detect_negative_peak(self, bin_centers: np.ndarray,
                            density: np.ndarray) -> Tuple[float, int, float]:
        """
        Robustly detect the negative (background) peak.

        Uses multi-criteria scoring:
        - Position: Favor lower intensities
        - Population: Favor peaks with more cells before them
        - Prominence: Favor distinct peaks

        Returns
        -------
        peak_value : float
            Intensity value at negative peak
        peak_idx : int
            Index of peak in density array
        peak_height : float
            Height (density) of peak
        """
        # Find all peaks
        peaks, properties = find_peaks(density,
                                      prominence=PEAK_PROMINENCE,
                                      distance=PEAK_DISTANCE,
                                      height=PEAK_MIN_HEIGHT)

        if len(peaks) == 0:
            # No peaks found, use mode
            peak_idx = np.argmax(density)
            return bin_centers[peak_idx], peak_idx, density[peak_idx]

        # Score each peak as candidate negative peak
        val_min = bin_centers.min()
        val_max = bin_centers.max()

        peak_scores = []
        for i, peak_idx in enumerate(peaks):
            peak_val = bin_centers[peak_idx]

            # Position score: favor lower intensities
            position_score = 1.0 - (peak_val - val_min) / (val_max - val_min + 1e-10)

            # Population score: favor peaks with large population before them
            cum_frac = (bin_centers <= peak_val).sum() / len(bin_centers)
            population_score = cum_frac

            # Prominence score
            prominence_score = properties['prominences'][i] / (properties['prominences'].max() + 1e-10)

            # Combined score (weighted)
            combined_score = (0.5 * position_score +
                            0.3 * population_score +
                            0.2 * prominence_score)
            peak_scores.append(combined_score)

        # Select best peak
        best_idx = np.argmax(peak_scores)
        neg_peak_idx = peaks[best_idx]
        neg_peak_val = bin_centers[neg_peak_idx]
        neg_peak_height = density[neg_peak_idx]

        return neg_peak_val, neg_peak_idx, neg_peak_height


    def register_to_reference(self, source_centers: np.ndarray,
                             source_density: np.ndarray,
                             reference_peak: float) -> interpolate.PchipInterpolator:
        """
        Create warping function to align source to reference negative peak.

        Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for
        smooth, monotonic warping.

        Parameters
        ----------
        source_centers : array
            Bin centers for source distribution
        source_density : array
            Density values for source
        reference_peak : float
            Target negative peak position

        Returns
        -------
        warp_fn : PchipInterpolator
            Warping function: source_intensity → aligned_intensity
        """
        # Detect source negative peak
        source_peak, _, _ = self.detect_negative_peak(source_centers, source_density)

        # Create landmark-based warping
        # Use negative peak as anchor, preserve overall range

        # Source landmarks
        source_min = source_centers.min()
        source_max = source_centers.max()

        # Target landmarks (shift by difference in peaks)
        shift = reference_peak - source_peak

        # Create warping: rigid shift centered on negative peak
        # y = x + shift, but preserve boundaries
        source_landmarks = np.array([source_min, source_peak, source_max])
        target_landmarks = np.array([source_min + shift, reference_peak, source_max + shift])

        # Create smooth interpolator
        warp_fn = interpolate.PchipInterpolator(source_landmarks, target_landmarks)

        return warp_fn


    def normalize_tile_level(self, marker: str, verbose: bool = True) -> np.ndarray:
        """
        Tile-level normalization: Align tiles within each sample.

        For each sample, aligns all tiles to the median tile negative peak.

        Parameters
        ----------
        marker : str
            Marker name to normalize
        verbose : bool
            Print progress

        Returns
        -------
        normalized_data : array
            Tile-normalized intensities
        """
        if verbose:
            print(f"\n  Tile-level normalization: {marker}")

        marker_idx = self.adata.var_names.get_loc(marker)
        raw_data = self.adata.X[:, marker_idx].copy()
        normalized_data = np.zeros_like(raw_data)

        # Process each sample separately
        for sample_id in self.adata.obs[self.sample_col].unique():
            sample_mask = self.adata.obs[self.sample_col] == sample_id

            # Get tiles in this sample
            if self.tile_col not in self.adata.obs:
                # No tiles, treat entire sample as one tile
                normalized_data[sample_mask] = raw_data[sample_mask]
                continue

            tiles = self.adata.obs.loc[sample_mask, self.tile_col].unique()

            if len(tiles) <= 1:
                # Only one tile, no normalization needed
                normalized_data[sample_mask] = raw_data[sample_mask]
                continue

            # Detect negative peak for each tile
            tile_peaks = {}
            tile_densities = {}

            for tile_id in tiles:
                tile_mask = sample_mask & (self.adata.obs[self.tile_col] == tile_id)
                tile_values = raw_data[tile_mask]

                bin_centers, density, bins = self.compute_smooth_density(tile_values)

                if bin_centers is not None:
                    neg_peak, _, _ = self.detect_negative_peak(bin_centers, density)
                    tile_peaks[tile_id] = neg_peak
                    tile_densities[tile_id] = (bin_centers, density)

            if len(tile_peaks) == 0:
                normalized_data[sample_mask] = raw_data[sample_mask]
                continue

            # Reference peak: median across tiles
            reference_peak = np.median(list(tile_peaks.values()))

            if verbose:
                print(f"    Sample {sample_id}: {len(tiles)} tiles, reference peak = {reference_peak:.1f}")

            # Warp each tile to reference
            for tile_id in tiles:
                tile_mask = sample_mask & (self.adata.obs[self.tile_col] == tile_id)
                tile_values = raw_data[tile_mask]

                if tile_id in tile_densities:
                    bin_centers, density = tile_densities[tile_id]

                    # Create warping function
                    warp_fn = self.register_to_reference(bin_centers, density, reference_peak)

                    # Apply warping
                    warped_values = warp_fn(tile_values)
                    warped_values = np.clip(warped_values, 0, None)  # Ensure non-negative

                    normalized_data[tile_mask] = warped_values

                    # Store warping function
                    if marker not in self.tile_warps:
                        self.tile_warps[marker] = {}
                    self.tile_warps[marker][(sample_id, tile_id)] = warp_fn
                else:
                    # Couldn't create density, use raw
                    normalized_data[tile_mask] = tile_values

        return normalized_data


    def normalize_sample_level(self, tile_normalized_data: np.ndarray,
                               marker: str, verbose: bool = True) -> np.ndarray:
        """
        Sample-level normalization: Align samples globally.

        Aligns all samples to the global median negative peak.

        Parameters
        ----------
        tile_normalized_data : array
            Data after tile-level normalization
        marker : str
            Marker name
        verbose : bool
            Print progress

        Returns
        -------
        normalized_data : array
            Fully normalized intensities
        """
        if verbose:
            print(f"\n  Sample-level normalization: {marker}")

        normalized_data = np.zeros_like(tile_normalized_data)

        # Detect negative peak for each sample
        sample_peaks = {}
        sample_densities = {}

        for sample_id in self.adata.obs[self.sample_col].unique():
            sample_mask = self.adata.obs[self.sample_col] == sample_id
            sample_values = tile_normalized_data[sample_mask]

            bin_centers, density, bins = self.compute_smooth_density(sample_values)

            if bin_centers is not None:
                neg_peak, _, _ = self.detect_negative_peak(bin_centers, density)
                sample_peaks[sample_id] = neg_peak
                sample_densities[sample_id] = (bin_centers, density)

        if len(sample_peaks) == 0:
            return tile_normalized_data

        # Global reference peak: median across samples
        reference_peak = np.median(list(sample_peaks.values()))

        if verbose:
            print(f"    Global reference peak = {reference_peak:.1f}")
            print(f"    Sample peaks: {dict(sorted(sample_peaks.items()))}")

        # Warp each sample to global reference
        for sample_id in self.adata.obs[self.sample_col].unique():
            sample_mask = self.adata.obs[self.sample_col] == sample_id
            sample_values = tile_normalized_data[sample_mask]

            if sample_id in sample_densities:
                bin_centers, density = sample_densities[sample_id]

                # Create warping function
                warp_fn = self.register_to_reference(bin_centers, density, reference_peak)

                # Apply warping
                warped_values = warp_fn(sample_values)
                warped_values = np.clip(warped_values, 0, None)

                normalized_data[sample_mask] = warped_values

                # Store warping function
                if marker not in self.sample_warps:
                    self.sample_warps[marker] = {}
                self.sample_warps[marker][sample_id] = warp_fn

                # Store negative peak
                if marker not in self.negative_peaks:
                    self.negative_peaks[marker] = {}
                self.negative_peaks[marker][sample_id] = sample_peaks[sample_id]
            else:
                normalized_data[sample_mask] = sample_values

        return normalized_data


    def normalize_all_markers(self, verbose: bool = True) -> ad.AnnData:
        """
        Apply hierarchical UniFORM normalization to all markers.

        Returns
        -------
        adata : AnnData
            Updated AnnData with normalized layers
        """
        print("\n" + "="*70)
        print("HIERARCHICAL UniFORM NORMALIZATION")
        print("="*70)

        # Store raw data
        self.adata.layers['raw'] = self.adata.X.copy()

        # Storage for normalized data
        tile_normalized = np.zeros_like(self.adata.X)
        fully_normalized = np.zeros_like(self.adata.X)

        for i, marker in enumerate(tqdm(self.markers, desc="Normalizing markers")):
            if marker not in self.adata.var_names:
                print(f"  WARNING: {marker} not found in data")
                continue

            marker_idx = self.adata.var_names.get_loc(marker)

            # Step 1: Tile-level normalization
            tile_norm = self.normalize_tile_level(marker, verbose=verbose)
            tile_normalized[:, marker_idx] = tile_norm

            # Step 2: Sample-level normalization
            full_norm = self.normalize_sample_level(tile_norm, marker, verbose=verbose)
            fully_normalized[:, marker_idx] = full_norm

        # Store in layers
        self.adata.layers['tile_normalized'] = tile_normalized
        self.adata.layers['uniform_normalized'] = fully_normalized

        # Apply asinh transformation for visualization
        self.adata.layers['asinh'] = np.arcsinh(fully_normalized / COFACTOR)

        print("\n" + "="*70)
        print("Normalization complete!")
        print("  Layers added: 'tile_normalized', 'uniform_normalized', 'asinh'")
        print("="*70)

        return self.adata


# ============================================================================
# ROBUST GATING
# ============================================================================

class RobustGating:
    """
    Robust gating algorithm with negative and positive peak detection.

    Uses semi-conservative gating strategy: gates between peaks,
    slightly favoring specificity over sensitivity.
    """

    def __init__(self, adata: ad.AnnData, markers: List[str]):
        """
        Initialize gating.

        Parameters
        ----------
        adata : AnnData
            Data with 'uniform_normalized' layer
        markers : list
            Marker names to gate
        """
        self.adata = adata
        self.markers = markers
        self.gates = {}
        self.peak_info = {}


    def detect_peaks(self, bin_centers: np.ndarray,
                    density: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Detect both negative and positive peaks.

        Returns
        -------
        negative_peak : dict
            {'value': float, 'index': int, 'height': float}
        positive_peak : dict or None
            {'value': float, 'index': int, 'height': float}
        """
        # Find all peaks
        peaks, properties = find_peaks(density,
                                      prominence=PEAK_PROMINENCE,
                                      distance=PEAK_DISTANCE,
                                      height=PEAK_MIN_HEIGHT)

        if len(peaks) == 0:
            # No peaks, use max as negative
            peak_idx = np.argmax(density)
            negative_peak = {
                'value': bin_centers[peak_idx],
                'index': peak_idx,
                'height': density[peak_idx]
            }
            return negative_peak, None

        # Score peaks as negative candidates (same as before)
        val_min = bin_centers.min()
        val_max = bin_centers.max()

        peak_scores = []
        for i, peak_idx in enumerate(peaks):
            peak_val = bin_centers[peak_idx]
            position_score = 1.0 - (peak_val - val_min) / (val_max - val_min + 1e-10)
            cum_frac = (bin_centers <= peak_val).sum() / len(bin_centers)
            population_score = cum_frac
            prominence_score = properties['prominences'][i] / (properties['prominences'].max() + 1e-10)
            combined_score = 0.5 * position_score + 0.3 * population_score + 0.2 * prominence_score
            peak_scores.append(combined_score)

        # Select negative peak
        neg_idx = np.argmax(peak_scores)
        neg_peak_idx = peaks[neg_idx]

        negative_peak = {
            'value': bin_centers[neg_peak_idx],
            'index': neg_peak_idx,
            'height': density[neg_peak_idx]
        }

        # Find positive peak: most prominent peak right of negative
        pos_candidates = [i for i, p in enumerate(peaks)
                         if bin_centers[p] > negative_peak['value'] * 1.3]

        if len(pos_candidates) == 0:
            return negative_peak, None

        # Select most prominent positive peak
        pos_prominences = [properties['prominences'][i] for i in pos_candidates]
        best_pos_idx = pos_candidates[np.argmax(pos_prominences)]
        pos_peak_idx = peaks[best_pos_idx]

        positive_peak = {
            'value': bin_centers[pos_peak_idx],
            'index': pos_peak_idx,
            'height': density[pos_peak_idx]
        }

        return negative_peak, positive_peak


    def find_valley(self, bin_centers: np.ndarray, density: np.ndarray,
                   neg_peak_idx: int, pos_peak_idx: Optional[int]) -> float:
        """
        Find valley (minimum) between negative and positive peaks.

        Parameters
        ----------
        bin_centers : array
            Bin centers
        density : array
            Density values
        neg_peak_idx : int
            Index of negative peak
        pos_peak_idx : int or None
            Index of positive peak (if found)

        Returns
        -------
        valley_value : float
            Intensity at valley
        """
        if pos_peak_idx is None:
            # No positive peak, search right of negative
            search_start = neg_peak_idx
            search_end = min(neg_peak_idx + 100, len(density))
        else:
            # Search between peaks
            search_start = neg_peak_idx
            search_end = pos_peak_idx

        search_region = density[search_start:search_end]

        if len(search_region) == 0:
            return bin_centers[neg_peak_idx] * 2.0

        # Find local minima
        local_mins = argrelextrema(search_region, np.less, order=3)[0]

        if len(local_mins) > 0:
            # Use first (leftmost) minimum
            valley_idx = search_start + local_mins[0]
        else:
            # No clear minimum, use global minimum in region
            valley_idx = search_start + np.argmin(search_region)

        return bin_centers[valley_idx]


    def gate_marker(self, marker: str, subsample: int = 50000) -> float:
        """
        Gate a single marker using semi-conservative strategy.

        Strategy:
        1. Detect negative and positive peaks
        2. Find valley between them
        3. Place gate at SEMI_CONSERVATIVE_PERCENTILE between valley and positive peak
           (default: 40% from valley toward positive = favors specificity)

        Parameters
        ----------
        marker : str
            Marker name
        subsample : int
            Number of cells to subsample for gating

        Returns
        -------
        gate : float
            Gate threshold
        """
        marker_idx = self.adata.var_names.get_loc(marker)

        # Get normalized data
        if 'uniform_normalized' not in self.adata.layers:
            print(f"ERROR: 'uniform_normalized' layer not found. Run normalization first.")
            return None

        values = self.adata.layers['uniform_normalized'][:, marker_idx]

        # Subsample for speed
        if len(values) > subsample:
            values = np.random.choice(values, subsample, replace=False)

        # Remove zeros
        values = values[values > 0]

        if len(values) < 100:
            return np.percentile(values, 90)

        # Compute density
        val_min = max(1, values.min())
        val_max = values.max()
        bins = np.logspace(np.log10(val_min), np.log10(val_max), N_BINS)

        hist, bin_edges = np.histogram(values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        density = hist_smooth / (hist_smooth.sum() + 1e-10)

        # Detect peaks
        neg_peak, pos_peak = self.detect_peaks(bin_centers, density)

        # Find valley
        valley_value = self.find_valley(bin_centers, density,
                                       neg_peak['index'],
                                       pos_peak['index'] if pos_peak else None)

        # Semi-conservative gating
        if pos_peak is not None:
            # Gate between valley and positive peak
            # SEMI_CONSERVATIVE_PERCENTILE = 0.4 means 40% from valley toward positive
            # (closer to valley = more conservative = fewer false positives)
            gate = valley_value + SEMI_CONSERVATIVE_PERCENTILE * (pos_peak['value'] - valley_value)

            print(f"  {marker}:")
            print(f"    Negative peak: {neg_peak['value']:.1f}")
            print(f"    Positive peak: {pos_peak['value']:.1f}")
            print(f"    Valley: {valley_value:.1f}")
            print(f"    Gate: {gate:.1f} (semi-conservative)")
        else:
            # No clear positive peak, use valley * multiplier
            gate = valley_value * 1.2
            print(f"  {marker}:")
            print(f"    Negative peak: {neg_peak['value']:.1f}")
            print(f"    No clear positive peak")
            print(f"    Gate: {gate:.1f} (valley-based)")

        # Calculate statistics
        all_values = self.adata.layers['uniform_normalized'][:, marker_idx]
        pos_pct = (all_values > gate).mean() * 100

        neg_vals = all_values[(all_values > 0) & (all_values <= gate)]
        pos_vals = all_values[all_values > gate]

        neg_mean = neg_vals.mean() if len(neg_vals) > 0 else 0
        pos_mean = pos_vals.mean() if len(pos_vals) > 0 else 0

        separation = (pos_mean - neg_mean) / (neg_mean + 1) if neg_mean > 0 else 0

        print(f"    Separation: {separation:.2f}× | {pos_pct:.1f}% positive")

        if separation < MIN_SEPARATION:
            print(f"    ⚠️  WARNING: Low separation (<{MIN_SEPARATION}×)")

        # Store information
        self.peak_info[marker] = {
            'negative_peak': neg_peak,
            'positive_peak': pos_peak,
            'valley': valley_value,
            'gate': gate,
            'separation': separation,
            'pct_positive': pos_pct
        }

        return gate


    def gate_all_markers(self) -> Dict[str, float]:
        """
        Gate all markers.

        Returns
        -------
        gates : dict
            Dictionary mapping marker names to gate values
        """
        print("\n" + "="*70)
        print("ROBUST GATING")
        print("="*70)

        for marker in self.markers:
            if marker not in self.adata.var_names:
                print(f"  WARNING: {marker} not found")
                continue

            gate = self.gate_marker(marker)
            self.gates[marker] = gate

        # Apply gates to create binary layer
        gated = np.zeros_like(self.adata.X)
        for i, marker in enumerate(self.adata.var_names):
            if marker in self.gates:
                marker_idx = self.adata.var_names.get_loc(marker)
                gated[:, marker_idx] = (self.adata.layers['uniform_normalized'][:, marker_idx] >
                                       self.gates[marker]).astype(int)

        self.adata.layers['gated'] = gated

        print("\n" + "="*70)
        print("Gating complete!")
        print(f"  Gates: {self.gates}")
        print("="*70)

        return self.gates


# ============================================================================
# VISUALIZATION
# ============================================================================

class GatingVisualizer:
    """Publication-quality diagnostic visualizations."""

    def __init__(self, adata: ad.AnnData, markers: List[str],
                 uniform_obj: HierarchicalUniFORM, gating_obj: RobustGating,
                 output_dir: str):
        self.adata = adata
        self.markers = markers
        self.uniform = uniform_obj
        self.gating = gating_obj
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def plot_normalization_overview(self, marker: str, sample_id: str = None):
        """
        4-panel figure showing normalization steps for a marker.

        Panels:
        1. Raw data (pre-normalization)
        2. After tile-level normalization
        3. After sample-level normalization
        4. Final asinh-transformed
        """
        if sample_id is None:
            sample_id = self.adata.obs['sample_id'].unique()[0]

        marker_idx = self.adata.var_names.get_loc(marker)
        sample_mask = self.adata.obs['sample_id'] == sample_id

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'UniFORM Normalization Pipeline: {marker} (Sample: {sample_id})',
                    fontsize=14, fontweight='bold')

        layers = ['raw', 'tile_normalized', 'uniform_normalized', 'asinh']
        titles = ['1. Raw Data', '2. Tile-Level Normalized',
                 '3. Sample-Level Normalized', '4. Asinh Transformed']

        for ax, layer, title in zip(axes.flat, layers, titles):
            if layer not in self.adata.layers:
                continue

            values = self.adata.layers[layer][sample_mask, marker_idx]
            values = values[values > 0]

            if len(values) == 0:
                continue

            # Plot histogram
            ax.hist(values, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Intensity', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_yscale('log')

            # Add statistics
            stats_text = f'Mean: {values.mean():.1f}\nMedian: {np.median(values):.1f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'normalization_overview_{marker}_{sample_id}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


    def plot_tile_correction(self, marker: str, sample_id: str = None):
        """
        Show tile-level correction for a marker.

        Before/after density curves overlaid by tile.
        """
        if sample_id is None:
            sample_id = self.adata.obs['sample_id'].unique()[0]

        if 'tile_id' not in self.adata.obs:
            print("  No tile information available")
            return

        marker_idx = self.adata.var_names.get_loc(marker)
        sample_mask = self.adata.obs['sample_id'] == sample_id
        tiles = self.adata.obs.loc[sample_mask, 'tile_id'].unique()

        if len(tiles) <= 1:
            print(f"  Sample {sample_id} has only 1 tile, skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Tile-Level Correction: {marker} (Sample: {sample_id})',
                    fontsize=14, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(tiles)))

        for ax, layer, title in zip(axes, ['raw', 'tile_normalized'],
                                   ['Before Correction', 'After Correction']):
            for tile_id, color in zip(tiles, colors):
                tile_mask = sample_mask & (self.adata.obs['tile_id'] == tile_id)
                values = self.adata.layers[layer][tile_mask, marker_idx]
                values = values[values > 0]

                if len(values) < 100:
                    continue

                # Compute density
                val_min = max(1, values.min())
                val_max = values.max()
                bins = np.logspace(np.log10(val_min), np.log10(val_max), 100)

                hist, bin_edges = np.histogram(values, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
                density = hist_smooth / (hist_smooth.sum() + 1e-10)

                ax.plot(bin_centers, density, linewidth=2, alpha=0.7,
                       color=color, label=f'Tile {tile_id}')

            ax.set_xlabel('Intensity', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'tile_correction_{marker}_{sample_id}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


    def plot_sample_alignment(self, marker: str):
        """
        Show sample-level alignment.

        Before/after density curves overlaid by sample.
        """
        marker_idx = self.adata.var_names.get_loc(marker)
        samples = self.adata.obs['sample_id'].unique()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Sample-Level Alignment: {marker}',
                    fontsize=14, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(samples)))

        for ax, layer, title in zip(axes, ['tile_normalized', 'uniform_normalized'],
                                   ['Before Alignment', 'After Alignment']):
            for sample_id, color in zip(samples, colors):
                sample_mask = self.adata.obs['sample_id'] == sample_id
                values = self.adata.layers[layer][sample_mask, marker_idx]
                values = values[values > 0]

                if len(values) < 100:
                    continue

                # Compute density
                val_min = max(1, values.min())
                val_max = values.max()
                bins = np.logspace(np.log10(val_min), np.log10(val_max), 100)

                hist, bin_edges = np.histogram(values, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
                density = hist_smooth / (hist_smooth.sum() + 1e-10)

                ax.plot(bin_centers, density, linewidth=2, alpha=0.7,
                       color=color, label=f'Sample {sample_id}')

            ax.set_xlabel('Intensity', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'sample_alignment_{marker}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


    def plot_gating_diagnostic(self, marker: str):
        """
        Detailed gating diagnostic for a marker.

        Shows:
        - Density curve with peaks, valley, and gate
        - Histogram with gate overlay
        - Positive/negative distributions
        """
        if marker not in self.gating.peak_info:
            print(f"  No gating info for {marker}")
            return

        marker_idx = self.adata.var_names.get_loc(marker)
        values = self.adata.layers['uniform_normalized'][:, marker_idx]
        values = values[values > 0]

        info = self.gating.peak_info[marker]
        gate = info['gate']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Gating Diagnostic: {marker}', fontsize=14, fontweight='bold')

        # Panel 1: Density curve with annotations
        ax = axes[0, 0]
        val_min = max(1, values.min())
        val_max = values.max()
        bins = np.logspace(np.log10(val_min), np.log10(val_max), N_BINS)

        hist, bin_edges = np.histogram(values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        density = hist_smooth / (hist_smooth.sum() + 1e-10)

        ax.plot(bin_centers, density, 'k-', linewidth=2, label='Density')

        # Mark negative peak
        ax.axvline(info['negative_peak']['value'], color='blue', linestyle='--',
                  linewidth=2, label=f"Negative peak ({info['negative_peak']['value']:.1f})")

        # Mark positive peak if exists
        if info['positive_peak']:
            ax.axvline(info['positive_peak']['value'], color='red', linestyle='--',
                      linewidth=2, label=f"Positive peak ({info['positive_peak']['value']:.1f})")

        # Mark valley
        ax.axvline(info['valley'], color='orange', linestyle=':',
                  linewidth=2, label=f"Valley ({info['valley']:.1f})")

        # Mark gate
        ax.axvline(gate, color='green', linestyle='-',
                  linewidth=3, label=f"GATE ({gate:.1f})")

        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Density Curve with Gate', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Histogram with gate
        ax = axes[0, 1]
        ax.hist(values, bins=100, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axvline(gate, color='green', linestyle='-', linewidth=3, label=f'Gate = {gate:.1f}')
        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Histogram with Gate', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 3: Separated populations
        ax = axes[1, 0]
        neg_vals = values[values <= gate]
        pos_vals = values[values > gate]

        ax.hist(neg_vals, bins=50, alpha=0.6, color='blue', label='Negative', edgecolor='black')
        ax.hist(pos_vals, bins=50, alpha=0.6, color='red', label='Positive', edgecolor='black')
        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Population Separation', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = f"""
Gating Statistics
{'='*40}

Negative Peak: {info['negative_peak']['value']:.1f}
Positive Peak: {info['positive_peak']['value']:.1f if info['positive_peak'] else 'N/A'}
Valley: {info['valley']:.1f}
Gate: {gate:.1f}

Separation: {info['separation']:.2f}×
Positive: {info['pct_positive']:.1f}%

Negative Population:
  N cells: {len(neg_vals):,}
  Mean: {neg_vals.mean():.1f}
  Median: {np.median(neg_vals):.1f}

Positive Population:
  N cells: {len(pos_vals):,}
  Mean: {pos_vals.mean():.1f}
  Median: {np.median(pos_vals):.1f}

{'⚠️ Low separation!' if info['separation'] < MIN_SEPARATION else '✓ Good separation'}
        """

        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / f'gating_diagnostic_{marker}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


    def plot_spatial_gating(self, marker: str, sample_id: str = None):
        """
        Spatial plot showing gated populations.
        """
        if sample_id is None:
            sample_id = self.adata.obs['sample_id'].unique()[0]

        if 'spatial' not in self.adata.obsm and 'X_centroid' not in self.adata.obs:
            print("  No spatial coordinates available")
            return

        marker_idx = self.adata.var_names.get_loc(marker)
        sample_mask = self.adata.obs['sample_id'] == sample_id

        # Get coordinates
        if 'spatial' in self.adata.obsm:
            coords = self.adata.obsm['spatial'][sample_mask]
        else:
            coords = np.column_stack([
                self.adata.obs.loc[sample_mask, 'X_centroid'],
                self.adata.obs.loc[sample_mask, 'Y_centroid']
            ])

        # Get gated values
        is_positive = self.adata.layers['gated'][sample_mask, marker_idx] > 0

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot negative cells
        ax.scatter(coords[~is_positive, 0], coords[~is_positive, 1],
                  c='lightgray', s=1, alpha=0.3, label='Negative')

        # Plot positive cells
        ax.scatter(coords[is_positive, 0], coords[is_positive, 1],
                  c='red', s=2, alpha=0.7, label='Positive')

        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_title(f'Spatial Distribution: {marker} (Sample: {sample_id})\n{is_positive.sum():,} positive / {len(is_positive):,} total',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, markerscale=5)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'spatial_gating_{marker}_{sample_id}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()


    def generate_all_diagnostics(self):
        """Generate all diagnostic plots."""
        print("\n" + "="*70)
        print("GENERATING DIAGNOSTIC VISUALIZATIONS")
        print("="*70)

        samples = self.adata.obs['sample_id'].unique()

        for marker in tqdm(self.markers, desc="Creating visualizations"):
            if marker not in self.adata.var_names:
                continue

            print(f"\n  {marker}:")

            # Normalization overview (first sample)
            print("    - Normalization overview")
            self.plot_normalization_overview(marker, samples[0])

            # Tile correction (if tiles exist)
            if 'tile_id' in self.adata.obs:
                print("    - Tile correction")
                self.plot_tile_correction(marker, samples[0])

            # Sample alignment
            print("    - Sample alignment")
            self.plot_sample_alignment(marker)

            # Gating diagnostic
            print("    - Gating diagnostic")
            self.plot_gating_diagnostic(marker)

            # Spatial gating (first sample)
            print("    - Spatial distribution")
            self.plot_spatial_gating(marker, samples[0])

        print("\n" + "="*70)
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_data(results_dir: str) -> ad.AnnData:
    """Load quantification data from MCMICRO results."""
    results_path = Path(results_dir)

    # Find all combined_quantification.csv files
    quant_files = list(results_path.glob("*/final/combined_quantification.csv"))

    if len(quant_files) == 0:
        raise FileNotFoundError(f"No quantification files found in {results_dir}")

    print(f"Found {len(quant_files)} samples")

    # Load and concatenate
    dfs = []
    for quant_file in quant_files:
        sample_name = quant_file.parent.parent.name
        df = pd.read_csv(quant_file)
        df['sample_id'] = sample_name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(combined_df):,} cells from {len(quant_files)} samples")

    # Extract intensity columns
    intensity_cols = [col for col in combined_df.columns if col.startswith('Channel_')]

    # Create AnnData
    X = combined_df[intensity_cols].values
    obs = combined_df[[col for col in combined_df.columns if col not in intensity_cols]]
    var = pd.DataFrame(index=intensity_cols)

    # Rename channels to markers
    var_names = [MARKERS.get(col, col) for col in intensity_cols]

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names = var_names

    # Add spatial coordinates if available
    if 'X_centroid' in obs.columns and 'Y_centroid' in obs.columns:
        adata.obsm['spatial'] = obs[['X_centroid', 'Y_centroid']].values

    # Detect tiles (if Tile_X and Tile_Y exist)
    if 'Tile_X' in obs.columns and 'Tile_Y' in obs.columns:
        adata.obs['tile_id'] = adata.obs['Tile_X'].astype(str) + '_' + adata.obs['Tile_Y'].astype(str)

    return adata


def main():
    parser = argparse.ArgumentParser(description='UniFORM-based hierarchical gating pipeline')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing MCMICRO results')
    parser.add_argument('--output_dir', type=str, default='uniform_gating_output',
                       help='Output directory')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip diagnostic visualizations (faster)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("UniFORM-BASED HIERARCHICAL GATING PIPELINE")
    print("="*70)

    # Load data
    print("\nLoading data...")
    adata = load_data(args.results_dir)

    # Get markers
    markers = [m for m in MARKERS.values() if m in adata.var_names]
    print(f"Markers to process: {markers}")

    # Step 1: Hierarchical UniFORM normalization
    uniform = HierarchicalUniFORM(adata, markers)
    adata = uniform.normalize_all_markers(verbose=True)

    # Step 2: Robust gating
    gating = RobustGating(adata, markers)
    gates = gating.gate_all_markers()

    # Step 3: Save results
    print("\nSaving results...")
    adata.write_h5ad(output_dir / 'gated_data.h5ad')
    print(f"  Saved: {output_dir / 'gated_data.h5ad'}")

    # Save gates
    gates_df = pd.DataFrame([{'marker': k, 'gate': v} for k, v in gates.items()])
    gates_df.to_csv(output_dir / 'gates.csv', index=False)
    print(f"  Saved: {output_dir / 'gates.csv'}")

    # Save peak info
    peak_info_df = []
    for marker, info in gating.peak_info.items():
        row = {
            'marker': marker,
            'negative_peak': info['negative_peak']['value'],
            'positive_peak': info['positive_peak']['value'] if info['positive_peak'] else None,
            'valley': info['valley'],
            'gate': info['gate'],
            'separation': info['separation'],
            'pct_positive': info['pct_positive']
        }
        peak_info_df.append(row)

    peak_info_df = pd.DataFrame(peak_info_df)
    peak_info_df.to_csv(output_dir / 'gating_statistics.csv', index=False)
    print(f"  Saved: {output_dir / 'gating_statistics.csv'}")

    # Step 4: Visualizations
    if not args.skip_visualization:
        print("\nGenerating visualizations...")
        viz = GatingVisualizer(adata, markers, uniform, gating, output_dir / 'figures')
        viz.generate_all_diagnostics()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - gated_data.h5ad: AnnData with all layers")
    print(f"  - gates.csv: Gate values per marker")
    print(f"  - gating_statistics.csv: Detailed gating metrics")
    if not args.skip_visualization:
        print(f"  - figures/: Diagnostic visualizations")
    print("\nLayers in AnnData:")
    print(f"  - raw: Original intensities")
    print(f"  - tile_normalized: After tile-level correction")
    print(f"  - uniform_normalized: After sample-level alignment")
    print(f"  - asinh: Asinh-transformed for visualization")
    print(f"  - gated: Binary gating calls (0/1)")


if __name__ == '__main__':
    main()
