"""
Gradient-Based Tile Boundary Correction for MCMICRO Data

This module implements a sophisticated tile artifact correction system using
gradient-based edge detection (Sobel operator) to identify and correct intensity
discontinuities at tile boundaries in multiplex imaging data.

Algorithm Overview:
1. Spatial Heatmap Generation: Convert sparse cell coordinates to dense 2D intensity map
2. Edge Detection: Apply Sobel gradients to detect sharp intensity discontinuities
3. Grid Line Identification: Validate detected edges form coherent grid structure
4. Boundary Region Mapping: Map detected edges to cell coordinates
5. Local Intensity Correction: Normalize boundary cells using KNN approach

Author: Claude Code
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage import morphology, filters
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class SharpEdgeTileDetector:
    """
    Detects tile regions using gradient-based edge detection (Sobel operator).

    This class identifies sharp intensity discontinuities that indicate tile boundaries,
    then segments the image into tile regions for classification and normalization.
    """

    def __init__(self,
                 bin_size: int = 50,
                 smooth_sigma: float = 2.0,
                 edge_threshold_percentile: int = 90,
                 min_edge_pixels: int = 10,
                 min_tile_size: int = 100):
        """
        Initialize the edge detector.

        Parameters
        ----------
        bin_size : int
            Spatial binning size in pixels (default: 50)
        smooth_sigma : float
            Gaussian smoothing kernel width (default: 2.0)
        edge_threshold_percentile : int
            Percentile threshold for edge detection (85-95, default: 90)
        min_edge_pixels : int
            Minimum number of edge pixels to proceed (default: 10)
        min_tile_size : int
            Minimum number of cells per tile region (default: 100)
        """
        self.bin_size = bin_size
        self.smooth_sigma = smooth_sigma
        self.edge_threshold_percentile = edge_threshold_percentile
        self.min_edge_pixels = min_edge_pixels
        self.min_tile_size = min_tile_size

    def create_spatial_heatmap(self,
                              x_coords: np.ndarray,
                              y_coords: np.ndarray,
                              intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert sparse single-cell coordinates to dense 2D intensity map.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Marker intensities for each cell

        Returns
        -------
        heatmap : np.ndarray
            2D array representing tissue intensity landscape
        x_edges : np.ndarray
            Bin edges for X axis
        y_edges : np.ndarray
            Bin edges for Y axis
        """
        # Filter out invalid values
        valid = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(intensities)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        intensities = intensities[valid]

        if len(x_coords) == 0:
            return np.zeros((10, 10)), np.linspace(0, 1000, 11), np.linspace(0, 1000, 11)

        # Calculate number of bins
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        n_x_bins = max(10, int((x_max - x_min) / self.bin_size))
        n_y_bins = max(10, int((y_max - y_min) / self.bin_size))

        # Create bins
        x_edges = np.linspace(x_min, x_max, n_x_bins + 1)
        y_edges = np.linspace(y_min, y_max, n_y_bins + 1)

        # Bin the data and calculate mean intensity per bin
        heatmap = np.zeros((n_y_bins, n_x_bins))
        counts = np.zeros((n_y_bins, n_x_bins))

        # Digitize coordinates
        x_idx = np.digitize(x_coords, x_edges) - 1
        y_idx = np.digitize(y_coords, y_edges) - 1

        # Clip to valid range
        x_idx = np.clip(x_idx, 0, n_x_bins - 1)
        y_idx = np.clip(y_idx, 0, n_y_bins - 1)

        # Accumulate intensities
        for i in range(len(x_coords)):
            heatmap[y_idx[i], x_idx[i]] += intensities[i]
            counts[y_idx[i], x_idx[i]] += 1

        # Calculate mean (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, 0)

        return heatmap, x_edges, y_edges

    def detect_sharp_edges(self, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify sharp intensity discontinuities using Sobel gradients.

        Parameters
        ----------
        heatmap : np.ndarray
            2D intensity heatmap

        Returns
        -------
        edge_mask : np.ndarray
            Binary mask of detected edges
        edge_strength : np.ndarray
            Gradient magnitude map
        """
        # Apply Gaussian smoothing to reduce noise while preserving edges
        smoothed = gaussian_filter(heatmap, sigma=self.smooth_sigma)

        # Compute Sobel gradients in X and Y directions
        grad_x = filters.sobel_h(smoothed)
        grad_y = filters.sobel_v(smoothed)

        # Calculate gradient magnitude
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold at specified percentile of non-zero gradients
        non_zero = edge_strength[edge_strength > 0]
        if len(non_zero) == 0:
            return np.zeros_like(heatmap, dtype=bool), edge_strength

        threshold = np.percentile(non_zero, self.edge_threshold_percentile)
        edge_mask = edge_strength > threshold

        # Apply morphological operations to clean up edges
        # Binary dilation to connect nearby edges
        edge_mask = binary_dilation(edge_mask, iterations=2)

        # Skeletonization to thin edges to single-pixel lines
        edge_mask = morphology.skeletonize(edge_mask)

        return edge_mask, edge_strength

    def detect_grid_lines(self, edge_mask: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Validate detected edges form coherent grid structure.

        Parameters
        ----------
        edge_mask : np.ndarray
            Binary edge mask

        Returns
        -------
        h_lines : List[int]
            Positions of horizontal lines (in bin coordinates)
        v_lines : List[int]
            Positions of vertical lines (in bin coordinates)
        """
        # Project edge mask onto axes
        h_projection = np.sum(edge_mask, axis=1)  # Horizontal projection
        v_projection = np.sum(edge_mask, axis=0)  # Vertical projection

        # Find peaks in projections
        h_threshold = np.percentile(h_projection[h_projection > 0], 75) if np.any(h_projection > 0) else 0
        v_threshold = np.percentile(v_projection[v_projection > 0], 75) if np.any(v_projection > 0) else 0

        h_peaks, _ = find_peaks(h_projection, height=h_threshold, distance=5)
        v_peaks, _ = find_peaks(v_projection, height=v_threshold, distance=5)

        return list(h_peaks), list(v_peaks)

    def segment_tiles(self, edge_mask: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Segment the image into tile regions using detected edges.

        Parameters
        ----------
        edge_mask : np.ndarray
            Binary edge mask
        heatmap : np.ndarray
            2D intensity heatmap

        Returns
        -------
        tile_labels : np.ndarray
            2D array with tile region labels (0 = edge, 1+ = tiles)
        tile_stats : Dict
            Statistics for each tile region
        """
        from scipy import ndimage

        # Invert edge mask (1 = interior, 0 = edge)
        interior_mask = ~edge_mask

        # Label connected regions
        tile_labels, n_tiles = ndimage.label(interior_mask)

        # Calculate statistics for each tile
        tile_stats = {}
        for tile_id in range(1, n_tiles + 1):
            tile_mask = tile_labels == tile_id
            tile_intensities = heatmap[tile_mask]

            # Filter out zero intensities
            tile_intensities = tile_intensities[tile_intensities > 0]

            if len(tile_intensities) > 0:
                tile_stats[tile_id] = {
                    'n_bins': int(np.sum(tile_mask)),
                    'median_intensity': float(np.median(tile_intensities)),
                    'mean_intensity': float(np.mean(tile_intensities)),
                    'std_intensity': float(np.std(tile_intensities))
                }
            else:
                tile_stats[tile_id] = {
                    'n_bins': int(np.sum(tile_mask)),
                    'median_intensity': 0.0,
                    'mean_intensity': 0.0,
                    'std_intensity': 0.0
                }

        return tile_labels, tile_stats

    def classify_tiles(self, tile_stats: Dict, outlier_threshold: float = 2.0) -> Tuple[List[int], List[int]]:
        """
        Classify tiles as dimmer (outlier) or normal based on intensity.

        Uses MAD (Median Absolute Deviation) to robustly identify outlier tiles.

        Parameters
        ----------
        tile_stats : Dict
            Statistics for each tile region
        outlier_threshold : float
            Number of MAD units for outlier detection (default: 2.0)

        Returns
        -------
        dimmer_tiles : List[int]
            IDs of tiles that are significantly dimmer
        normal_tiles : List[int]
            IDs of normal tiles
        """
        from scipy.stats import median_abs_deviation

        if len(tile_stats) < 3:
            # Not enough tiles to classify
            return [], list(tile_stats.keys())

        # Get median intensities for all tiles
        tile_ids = list(tile_stats.keys())
        medians = np.array([tile_stats[tid]['median_intensity'] for tid in tile_ids])

        # Filter out zero intensities
        valid_mask = medians > 0
        if np.sum(valid_mask) < 3:
            return [], tile_ids

        valid_tile_ids = [tid for tid, v in zip(tile_ids, valid_mask) if v]
        valid_medians = medians[valid_mask]

        # Calculate global median and MAD
        global_median = np.median(valid_medians)
        mad = median_abs_deviation(valid_medians)

        if mad == 0:
            # All tiles have same intensity
            return [], tile_ids

        # Classify tiles
        dimmer_tiles = []
        normal_tiles = []

        for tile_id, median in zip(valid_tile_ids, valid_medians):
            # Calculate MAD score (how many MADs away from global median)
            mad_score = (global_median - median) / mad

            if mad_score > outlier_threshold:  # Significantly dimmer
                dimmer_tiles.append(tile_id)
            else:
                normal_tiles.append(tile_id)

        # Add zero-intensity tiles to normal (they'll be skipped anyway)
        for tid in tile_ids:
            if tid not in valid_tile_ids:
                normal_tiles.append(tid)

        return dimmer_tiles, normal_tiles

    def assign_cells_to_tiles(self,
                             x_coords: np.ndarray,
                             y_coords: np.ndarray,
                             tile_labels: np.ndarray,
                             x_edges: np.ndarray,
                             y_edges: np.ndarray) -> np.ndarray:
        """
        Assign each cell to a tile region.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        tile_labels : np.ndarray
            2D array with tile region labels from segmentation
        x_edges : np.ndarray
            Bin edges for X axis
        y_edges : np.ndarray
            Bin edges for Y axis

        Returns
        -------
        cell_tile_ids : np.ndarray
            Tile ID for each cell (0 = edge/unassigned)
        """
        n_cells = len(x_coords)
        cell_tile_ids = np.zeros(n_cells, dtype=int)

        # Digitize coordinates to bin indices
        x_idx = np.digitize(x_coords, x_edges) - 1
        y_idx = np.digitize(y_coords, y_edges) - 1

        # Clip to valid range
        n_y_bins, n_x_bins = tile_labels.shape
        x_idx = np.clip(x_idx, 0, n_x_bins - 1)
        y_idx = np.clip(y_idx, 0, n_y_bins - 1)

        # Assign cells to tiles
        for i in range(n_cells):
            cell_tile_ids[i] = tile_labels[y_idx[i], x_idx[i]]

        return cell_tile_ids

    def detect(self,
              x_coords: np.ndarray,
              y_coords: np.ndarray,
              intensities: np.ndarray) -> Dict:
        """
        Full detection pipeline: detect tiles and classify them.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Marker intensities for each cell

        Returns
        -------
        results : Dict
            Detection results including tile segmentation and classification
        """
        # Phase 1: Create spatial heatmap
        heatmap, x_edges, y_edges = self.create_spatial_heatmap(x_coords, y_coords, intensities)

        # Phase 2: Detect edges
        edge_mask, edge_strength = self.detect_sharp_edges(heatmap)

        # Phase 3: Identify grid lines
        h_lines, v_lines = self.detect_grid_lines(edge_mask)

        # Count edge pixels
        n_edge_pixels = np.sum(edge_mask)

        # Check if detection was successful
        detected = n_edge_pixels >= self.min_edge_pixels

        if not detected:
            return {
                'detected': False,
                'reason': 'Insufficient edge pixels detected'
            }

        # Phase 4: Segment into tile regions
        tile_labels, tile_stats = self.segment_tiles(edge_mask, heatmap)

        # Filter tiles by minimum size
        valid_tiles = {tid: stats for tid, stats in tile_stats.items()
                      if stats['n_bins'] >= 5}  # At least 5 bins per tile

        if len(valid_tiles) < 2:
            return {
                'detected': False,
                'reason': 'Insufficient valid tile regions'
            }

        # Phase 5: Classify tiles as dimmer vs normal
        dimmer_tiles, normal_tiles = self.classify_tiles(valid_tiles)

        # Phase 6: Assign cells to tiles
        cell_tile_ids = self.assign_cells_to_tiles(x_coords, y_coords, tile_labels, x_edges, y_edges)

        # Count cells in each group
        dimmer_cell_mask = np.isin(cell_tile_ids, dimmer_tiles)
        normal_cell_mask = np.isin(cell_tile_ids, normal_tiles)

        n_dimmer_cells = np.sum(dimmer_cell_mask)
        n_normal_cells = np.sum(normal_cell_mask)

        # Need enough cells in both groups
        if n_dimmer_cells < self.min_tile_size or n_normal_cells < self.min_tile_size:
            return {
                'detected': False,
                'reason': f'Insufficient cells (dimmer={n_dimmer_cells}, normal={n_normal_cells})'
            }

        return {
            'detected': True,
            'edge_mask': edge_mask,
            'edge_strength': edge_strength,
            'heatmap': heatmap,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'n_edge_pixels': int(n_edge_pixels),
            'n_h_lines': len(h_lines),
            'n_v_lines': len(v_lines),
            'tile_labels': tile_labels,
            'tile_stats': tile_stats,
            'dimmer_tiles': dimmer_tiles,
            'normal_tiles': normal_tiles,
            'cell_tile_ids': cell_tile_ids,
            'n_dimmer_cells': int(n_dimmer_cells),
            'n_normal_cells': int(n_normal_cells)
        }


class TileArtifactCorrector:
    """
    Corrects tile artifacts using UniFORM normalization between dimmer and normal tiles.

    This class applies quantile-based normalization to align the intensity distributions
    of dimmer tiles to match normal tiles.
    """

    def __init__(self,
                 n_quantiles: int = 100,
                 correction_strength: float = 1.0):
        """
        Initialize the corrector.

        Parameters
        ----------
        n_quantiles : int
            Number of quantiles for UniFORM normalization (default: 100)
        correction_strength : float
            Strength of correction to apply (0-1, default: 1.0)
            1.0 = full UniFORM correction, <1.0 = partial correction
        """
        self.n_quantiles = n_quantiles
        self.correction_strength = correction_strength

    def uniform_normalize(self,
                         source_values: np.ndarray,
                         target_values: np.ndarray) -> np.ndarray:
        """
        Apply UniFORM-style quantile normalization.

        Maps the distribution of source_values to match target_values using
        quantile matching (CDF alignment).

        Parameters
        ----------
        source_values : np.ndarray
            Values to be normalized (dimmer tiles)
        target_values : np.ndarray
            Reference values (normal tiles)

        Returns
        -------
        normalized_values : np.ndarray
            Source values normalized to match target distribution
        """
        # Filter positive values only
        source_pos = source_values[source_values > 0]
        target_pos = target_values[target_values > 0]

        if len(source_pos) < 10 or len(target_pos) < 10:
            return source_values

        # Calculate quantiles
        quantiles = np.linspace(0, 1, self.n_quantiles)
        source_quantiles = np.quantile(source_pos, quantiles)
        target_quantiles = np.quantile(target_pos, quantiles)

        # Interpolate mapping function
        normalized = np.interp(source_values, source_quantiles, target_quantiles)

        # Apply correction strength
        if self.correction_strength < 1.0:
            normalized = source_values + self.correction_strength * (normalized - source_values)

        return normalized

    def correct(self,
               intensities: np.ndarray,
               detection_results: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Apply UniFORM normalization between dimmer and normal tiles.

        Parameters
        ----------
        intensities : np.ndarray
            Original marker intensities
        detection_results : Dict
            Results from SharpEdgeTileDetector.detect()

        Returns
        -------
        corrected_intensities : np.ndarray
            Corrected marker intensities
        stats : Dict
            Correction statistics
        """
        if not detection_results['detected']:
            return intensities, {
                'detected': False,
                'reason': detection_results.get('reason', 'Detection failed')
            }

        # Get cell tile assignments
        cell_tile_ids = detection_results['cell_tile_ids']
        dimmer_tiles = detection_results['dimmer_tiles']
        normal_tiles = detection_results['normal_tiles']

        # Create masks for dimmer and normal cells
        dimmer_mask = np.isin(cell_tile_ids, dimmer_tiles)
        normal_mask = np.isin(cell_tile_ids, normal_tiles)

        n_dimmer = np.sum(dimmer_mask)
        n_normal = np.sum(normal_mask)

        if n_dimmer == 0:
            # No dimmer tiles, no correction needed
            return intensities, {
                'detected': True,
                'n_dimmer_tiles': len(dimmer_tiles),
                'n_normal_tiles': len(normal_tiles),
                'n_dimmer_cells': 0,
                'n_normal_cells': int(n_normal),
                'n_corrected': 0,
                'mean_correction_pct': 0.0
            }

        # Apply UniFORM normalization: dimmer tiles -> normal tiles
        corrected = intensities.copy()
        dimmer_values = intensities[dimmer_mask]
        normal_values = intensities[normal_mask]

        # Normalize dimmer to match normal
        normalized_dimmer = self.uniform_normalize(dimmer_values, normal_values)

        # Update corrected values
        corrected[dimmer_mask] = normalized_dimmer

        # Calculate statistics
        mean_correction_pct = 0.0
        if n_dimmer > 0:
            changes = (normalized_dimmer - dimmer_values) / (dimmer_values + 1e-10)
            mean_correction_pct = np.mean(changes[np.isfinite(changes)]) * 100

        stats = {
            'detected': True,
            'n_dimmer_tiles': len(dimmer_tiles),
            'n_normal_tiles': len(normal_tiles),
            'n_dimmer_cells': int(n_dimmer),
            'n_normal_cells': int(n_normal),
            'n_corrected': int(n_dimmer),
            'mean_correction_pct': float(mean_correction_pct),
            'dimmer_tiles': dimmer_tiles,
            'normal_tiles': normal_tiles
        }

        return corrected, stats


def create_diagnostic_plots(marker: str,
                           x_coords: np.ndarray,
                           y_coords: np.ndarray,
                           original_intensities: np.ndarray,
                           corrected_intensities: np.ndarray,
                           detection_results: Dict,
                           output_dir: Path):
    """
    Create 6-panel validation figure showing tile detection and UniFORM correction.

    Parameters
    ----------
    marker : str
        Marker name
    x_coords : np.ndarray
        X coordinates of cells
    y_coords : np.ndarray
        Y coordinates of cells
    original_intensities : np.ndarray
        Original marker intensities
    corrected_intensities : np.ndarray
        Corrected marker intensities
    detection_results : Dict
        Results from SharpEdgeTileDetector.detect()
    output_dir : Path
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{marker} - Tile Detection & UniFORM Correction', fontsize=16, fontweight='bold')

    # Top Row
    # 1. Intensity heatmap
    heatmap = detection_results['heatmap']
    im1 = axes[0, 0].imshow(heatmap, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Intensity Heatmap (Binned)')
    axes[0, 0].set_xlabel('X Bin')
    axes[0, 0].set_ylabel('Y Bin')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Tile segmentation (show dimmer vs normal tiles)
    tile_labels = detection_results.get('tile_labels', np.zeros_like(heatmap))
    dimmer_tiles = detection_results.get('dimmer_tiles', [])
    normal_tiles = detection_results.get('normal_tiles', [])

    # Create classification map: 0=edge, 1=normal, 2=dimmer
    tile_classification = np.zeros_like(tile_labels)
    for tid in normal_tiles:
        tile_classification[tile_labels == tid] = 1
    for tid in dimmer_tiles:
        tile_classification[tile_labels == tid] = 2

    cmap = plt.cm.colors.ListedColormap(['black', 'blue', 'red'])
    im2 = axes[0, 1].imshow(tile_classification, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    axes[0, 1].set_title(f'Tile Classification\n({len(dimmer_tiles)} dimmer, {len(normal_tiles)} normal)')
    axes[0, 1].set_xlabel('X Bin')
    axes[0, 1].set_ylabel('Y Bin')
    cbar = plt.colorbar(im2, ax=axes[0, 1], ticks=[0, 1, 2])
    cbar.set_ticklabels(['Edge', 'Normal', 'Dimmer'])

    # 3. Edge detection overlay
    axes[0, 2].imshow(heatmap, cmap='gray', aspect='auto', alpha=0.5)
    edge_mask = detection_results.get('edge_mask', np.zeros_like(heatmap, dtype=bool))
    axes[0, 2].imshow(edge_mask, cmap='Reds', aspect='auto', alpha=0.7)
    axes[0, 2].set_title(f"Detected Tile Boundaries\n({detection_results.get('n_edge_pixels', 0)} edge pixels)")
    axes[0, 2].set_xlabel('X Bin')
    axes[0, 2].set_ylabel('Y Bin')

    # Bottom Row
    # Get tile assignments for cells
    cell_tile_ids = detection_results.get('cell_tile_ids', np.zeros(len(x_coords), dtype=int))
    dimmer_mask = np.isin(cell_tile_ids, dimmer_tiles)
    normal_mask = np.isin(cell_tile_ids, normal_tiles)
    edge_mask_cells = ~(dimmer_mask | normal_mask)

    # Subsample for plotting if too many cells
    if len(x_coords) > 50000:
        sample_idx = np.random.choice(len(x_coords), 50000, replace=False)
        x_plot = x_coords[sample_idx]
        y_plot = y_coords[sample_idx]
        orig_plot = original_intensities[sample_idx]
        corr_plot = corrected_intensities[sample_idx]
        dimmer_plot = dimmer_mask[sample_idx]
        normal_plot = normal_mask[sample_idx]
    else:
        x_plot = x_coords
        y_plot = y_coords
        orig_plot = original_intensities
        corr_plot = corrected_intensities
        dimmer_plot = dimmer_mask
        normal_plot = normal_mask

    # 4. Spatial scatter: Original intensities
    sc1 = axes[1, 0].scatter(x_plot, y_plot, c=np.log10(orig_plot + 1),
                            s=0.5, cmap='viridis', vmin=0, vmax=3, rasterized=True)
    axes[1, 0].set_title('Original Intensities')
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[1, 0], label='log10(intensity + 1)')

    # 5. Tile classification (spatial view)
    axes[1, 1].scatter(x_plot[normal_plot], y_plot[normal_plot],
                      c='blue', s=0.3, alpha=0.3, label='Normal', rasterized=True)
    axes[1, 1].scatter(x_plot[dimmer_plot], y_plot[dimmer_plot],
                      c='red', s=0.5, alpha=0.5, label='Dimmer', rasterized=True)
    axes[1, 1].set_title(f'Tile Classification\n({np.sum(dimmer_plot):,} dimmer, {np.sum(normal_plot):,} normal)')
    axes[1, 1].set_xlabel('X Coordinate')
    axes[1, 1].set_ylabel('Y Coordinate')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend(markerscale=5)

    # 6. Before/after histograms
    axes[1, 2].hist(np.log10(orig_plot[normal_plot] + 1), bins=50,
                   alpha=0.5, label='Normal (Before)', color='blue', density=True)
    axes[1, 2].hist(np.log10(orig_plot[dimmer_plot] + 1), bins=50,
                   alpha=0.5, label='Dimmer (Before)', color='red', density=True)
    axes[1, 2].hist(np.log10(corr_plot[dimmer_plot] + 1), bins=50,
                   alpha=0.5, label='Dimmer (After UniFORM)', color='green', density=True)
    axes[1, 2].set_xlabel('log10(Intensity + 1)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('UniFORM Normalization Effect')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{marker}_tile_correction.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved diagnostic plot: {output_file}")


def save_correction_report(report: Dict, output_dir: Path):
    """
    Save correction report as JSON.

    Parameters
    ----------
    report : Dict
        Per-marker correction statistics
    output_dir : Path
        Directory to save report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'tile_correction_report.json'

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved correction report: {output_file}")
