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
    Detects tile boundaries using gradient-based edge detection (Sobel operator).

    This class identifies sharp intensity discontinuities that indicate tile boundaries,
    without relying on pre-existing tile_id information.
    """

    def __init__(self,
                 bin_size: int = 50,
                 smooth_sigma: float = 2.0,
                 edge_threshold_percentile: int = 95,
                 min_edge_pixels: int = 10):
        """
        Initialize the edge detector.

        Parameters
        ----------
        bin_size : int
            Spatial binning size in pixels (default: 50)
        smooth_sigma : float
            Gaussian smoothing kernel width (default: 2.0)
        edge_threshold_percentile : int
            Percentile threshold for edge detection (90-99, default: 95)
        min_edge_pixels : int
            Minimum number of edge pixels to proceed (default: 10)
        """
        self.bin_size = bin_size
        self.smooth_sigma = smooth_sigma
        self.edge_threshold_percentile = edge_threshold_percentile
        self.min_edge_pixels = min_edge_pixels

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

    def detect(self,
              x_coords: np.ndarray,
              y_coords: np.ndarray,
              intensities: np.ndarray) -> Dict:
        """
        Full detection pipeline.

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
            Detection results including edge_mask, heatmap, bin_edges, grid_lines
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
        detected = (n_edge_pixels >= self.min_edge_pixels and
                   len(h_lines) >= 2 and len(v_lines) >= 2)

        return {
            'detected': detected,
            'edge_mask': edge_mask,
            'edge_strength': edge_strength,
            'heatmap': heatmap,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'n_edge_pixels': int(n_edge_pixels),
            'n_h_lines': len(h_lines),
            'n_v_lines': len(v_lines)
        }


class TileArtifactCorrector:
    """
    Corrects tile boundary artifacts using local KNN normalization.

    This class maps detected edges to cell coordinates and performs
    local intensity correction to normalize boundary cells to match
    interior cell intensities.
    """

    def __init__(self,
                 boundary_buffer: int = 50,
                 local_window: int = 200,
                 correction_strength: float = 0.8,
                 max_boundary_pct: float = 50.0):
        """
        Initialize the corrector.

        Parameters
        ----------
        boundary_buffer : int
            Width of correction zone around edges (pixels, default: 50)
        local_window : int
            Neighborhood radius for KNN (pixels, default: 200)
        correction_strength : float
            Damping factor for correction (0-1, default: 0.8)
        max_boundary_pct : float
            Safety threshold: warn if >X% cells flagged (default: 50%)
        """
        self.boundary_buffer = boundary_buffer
        self.local_window = local_window
        self.correction_strength = correction_strength
        self.max_boundary_pct = max_boundary_pct

    def create_boundary_mask(self,
                            x_coords: np.ndarray,
                            y_coords: np.ndarray,
                            edge_mask: np.ndarray,
                            x_edges: np.ndarray,
                            y_edges: np.ndarray) -> np.ndarray:
        """
        Convert detected edges from binned space to cell coordinates.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        edge_mask : np.ndarray
            Binary edge mask in binned space
        x_edges : np.ndarray
            Bin edges for X axis
        y_edges : np.ndarray
            Bin edges for Y axis

        Returns
        -------
        boundary_mask : np.ndarray
            Boolean array indicating boundary cells
        """
        n_cells = len(x_coords)
        boundary_mask = np.zeros(n_cells, dtype=bool)

        # Find edge pixels in binned space
        edge_y, edge_x = np.where(edge_mask)

        if len(edge_x) == 0:
            return boundary_mask

        # Convert edge pixels to coordinate ranges
        for ey, ex in zip(edge_y, edge_x):
            # Get coordinate range for this bin
            x_min = x_edges[ex]
            x_max = x_edges[ex + 1] if ex + 1 < len(x_edges) else x_edges[-1]
            y_min = y_edges[ey]
            y_max = y_edges[ey + 1] if ey + 1 < len(y_edges) else y_edges[-1]

            # Expand by buffer
            x_min -= self.boundary_buffer
            x_max += self.boundary_buffer
            y_min -= self.boundary_buffer
            y_max += self.boundary_buffer

            # Mark cells within this boundary region
            in_region = ((x_coords >= x_min) & (x_coords <= x_max) &
                        (y_coords >= y_min) & (y_coords <= y_max))
            boundary_mask |= in_region

        return boundary_mask

    def correct_boundary_artifacts(self,
                                  x_coords: np.ndarray,
                                  y_coords: np.ndarray,
                                  intensities: np.ndarray,
                                  boundary_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize boundary cells to match interior cell intensities using KNN.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Original marker intensities
        boundary_mask : np.ndarray
            Boolean array indicating boundary cells

        Returns
        -------
        corrected_intensities : np.ndarray
            Corrected marker intensities
        stats : Dict
            Correction statistics
        """
        corrected = intensities.copy()

        # Safety check
        pct_boundary = 100 * np.sum(boundary_mask) / len(boundary_mask)
        if pct_boundary > self.max_boundary_pct:
            warnings.warn(f"Warning: {pct_boundary:.1f}% cells flagged as boundary (>{self.max_boundary_pct}%). "
                         "May indicate oversensitive detection.")

        # Get boundary and interior cells
        boundary_idx = np.where(boundary_mask)[0]
        interior_idx = np.where(~boundary_mask)[0]

        if len(boundary_idx) == 0 or len(interior_idx) < 3:
            return corrected, {
                'n_corrected': 0,
                'mean_correction_pct': 0.0,
                'pct_boundary_cells': pct_boundary
            }

        # Build KNN tree for all cells
        coords = np.column_stack([x_coords, y_coords])
        nbrs = NearestNeighbors(radius=self.local_window, algorithm='ball_tree')
        nbrs.fit(coords)

        # Correct each boundary cell
        correction_factors = []
        n_corrected = 0

        for idx in boundary_idx:
            # Find neighbors within radius
            indices = nbrs.radius_neighbors([coords[idx]], return_distance=False)[0]

            if len(indices) < 4:  # Need at least some neighbors
                continue

            # Separate into boundary and interior neighbors
            neighbor_is_boundary = boundary_mask[indices]
            boundary_neighbors = indices[neighbor_is_boundary]
            interior_neighbors = indices[~neighbor_is_boundary]

            # Require at least 3 interior neighbors for statistical reliability
            if len(interior_neighbors) < 3:
                continue

            # Calculate correction factor
            interior_median = np.median(intensities[interior_neighbors])
            boundary_median = np.median(intensities[boundary_neighbors]) if len(boundary_neighbors) > 0 else intensities[idx]

            if boundary_median > 0:
                correction_factor = interior_median / boundary_median

                # Apply scaled correction
                corrected[idx] = intensities[idx] * (1 + self.correction_strength * (correction_factor - 1))

                correction_factors.append(correction_factor)
                n_corrected += 1

        # Calculate statistics
        mean_correction_pct = 0.0
        if len(correction_factors) > 0:
            mean_correction_pct = np.mean([(cf - 1) * 100 for cf in correction_factors])

        stats = {
            'n_corrected': n_corrected,
            'n_boundary_cells': int(np.sum(boundary_mask)),
            'pct_boundary_cells': pct_boundary,
            'mean_correction_pct': float(mean_correction_pct)
        }

        return corrected, stats

    def correct(self,
               x_coords: np.ndarray,
               y_coords: np.ndarray,
               intensities: np.ndarray,
               detection_results: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Full correction pipeline.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Original marker intensities
        detection_results : Dict
            Results from SharpEdgeTileDetector.detect()

        Returns
        -------
        corrected_intensities : np.ndarray
            Corrected marker intensities
        stats : Dict
            Combined detection and correction statistics
        """
        if not detection_results['detected']:
            return intensities, {
                'detected': False,
                'reason': 'No tile boundaries detected'
            }

        # Phase 4: Create boundary mask
        boundary_mask = self.create_boundary_mask(
            x_coords, y_coords,
            detection_results['edge_mask'],
            detection_results['x_edges'],
            detection_results['y_edges']
        )

        # Phase 5: Correct boundary artifacts
        corrected, correction_stats = self.correct_boundary_artifacts(
            x_coords, y_coords, intensities, boundary_mask
        )

        # Combine statistics
        stats = {
            **detection_results,
            **correction_stats,
            'boundary_regions': len(detection_results['h_lines']) * len(detection_results['v_lines'])
        }

        # Remove large arrays from stats
        stats.pop('edge_mask', None)
        stats.pop('edge_strength', None)
        stats.pop('heatmap', None)
        stats.pop('x_edges', None)
        stats.pop('y_edges', None)

        return corrected, stats


def create_diagnostic_plots(marker: str,
                           x_coords: np.ndarray,
                           y_coords: np.ndarray,
                           original_intensities: np.ndarray,
                           corrected_intensities: np.ndarray,
                           detection_results: Dict,
                           boundary_mask: np.ndarray,
                           output_dir: Path):
    """
    Create 6-panel validation figure for a marker.

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
    boundary_mask : np.ndarray
        Boolean array indicating boundary cells
    output_dir : Path
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{marker} - Tile Boundary Correction Diagnostics', fontsize=16, fontweight='bold')

    # Top Row
    # 1. Intensity heatmap
    heatmap = detection_results['heatmap']
    im1 = axes[0, 0].imshow(heatmap, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Intensity Heatmap (Binned)')
    axes[0, 0].set_xlabel('X Bin')
    axes[0, 0].set_ylabel('Y Bin')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Edge strength
    edge_strength = detection_results['edge_strength']
    im2 = axes[0, 1].imshow(edge_strength, cmap='hot', aspect='auto')
    axes[0, 1].set_title('Edge Strength (Gradient Magnitude)')
    axes[0, 1].set_xlabel('X Bin')
    axes[0, 1].set_ylabel('Y Bin')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Detected boundaries overlay with grid lines
    axes[0, 2].imshow(heatmap, cmap='gray', aspect='auto', alpha=0.5)
    edge_mask = detection_results['edge_mask']
    axes[0, 2].imshow(edge_mask, cmap='Reds', aspect='auto', alpha=0.7)

    # Draw grid lines
    for h_line in detection_results['h_lines']:
        axes[0, 2].axhline(h_line, color='blue', linestyle='--', linewidth=1, alpha=0.7)
    for v_line in detection_results['v_lines']:
        axes[0, 2].axvline(v_line, color='blue', linestyle='--', linewidth=1, alpha=0.7)

    axes[0, 2].set_title(f"Detected Boundaries\n({detection_results['n_h_lines']} H-lines, {detection_results['n_v_lines']} V-lines)")
    axes[0, 2].set_xlabel('X Bin')
    axes[0, 2].set_ylabel('Y Bin')

    # Bottom Row
    # Subsample for plotting if too many cells
    if len(x_coords) > 50000:
        sample_idx = np.random.choice(len(x_coords), 50000, replace=False)
        x_plot = x_coords[sample_idx]
        y_plot = y_coords[sample_idx]
        orig_plot = original_intensities[sample_idx]
        corr_plot = corrected_intensities[sample_idx]
        boundary_plot = boundary_mask[sample_idx]
    else:
        x_plot = x_coords
        y_plot = y_coords
        orig_plot = original_intensities
        corr_plot = corrected_intensities
        boundary_plot = boundary_mask

    # 4. Spatial scatter (all cells, colored by intensity)
    sc1 = axes[1, 0].scatter(x_plot, y_plot, c=np.log10(orig_plot + 1),
                            s=0.5, cmap='viridis', vmin=0, vmax=3, rasterized=True)
    axes[1, 0].set_title('Original Intensities (Spatial)')
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[1, 0], label='log10(intensity + 1)')

    # 5. Boundary cells highlighted
    interior_mask = ~boundary_plot
    axes[1, 1].scatter(x_plot[interior_mask], y_plot[interior_mask],
                      c='blue', s=0.3, alpha=0.3, label='Interior', rasterized=True)
    axes[1, 1].scatter(x_plot[boundary_plot], y_plot[boundary_plot],
                      c='red', s=0.5, alpha=0.5, label='Boundary', rasterized=True)
    axes[1, 1].set_title(f'Boundary Detection\n({np.sum(boundary_plot):,} boundary cells)')
    axes[1, 1].set_xlabel('X Coordinate')
    axes[1, 1].set_ylabel('Y Coordinate')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend(markerscale=5)

    # 6. Before/after histograms
    axes[1, 2].hist(np.log10(orig_plot[interior_mask] + 1), bins=50,
                   alpha=0.5, label='Original (Interior)', color='blue', density=True)
    axes[1, 2].hist(np.log10(orig_plot[boundary_plot] + 1), bins=50,
                   alpha=0.5, label='Original (Boundary)', color='red', density=True)
    axes[1, 2].hist(np.log10(corr_plot[boundary_plot] + 1), bins=50,
                   alpha=0.5, label='Corrected (Boundary)', color='green', density=True)
    axes[1, 2].set_xlabel('log10(Intensity + 1)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Intensity Distributions')
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
