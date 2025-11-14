"""
Microscope Tile Grid Detection and Correction for MCMICRO Data

This module detects actual microscope tile boundaries (not MCMICRO tiles) using
intensity pattern analysis and applies UniFORM normalization to correct dimmer tiles.

Algorithm:
1. Create fine-resolution intensity heatmap
2. Apply Sobel edge detection to find tile boundaries
3. Project edges to 1D and find peaks (boundary lines)
4. Estimate regular grid spacing from peak distances
5. Fit regular grid and assign cells to microscope tiles
6. Classify tiles as dimmer vs normal
7. Apply UniFORM normalization between populations

Expected microscope tile size: ~1024-2048 pixels (CellDIVE/Leica)

Author: Claude Code
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from skimage import filters
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class MicroscopeTileDetector:
    """
    Detects actual microscope tile grid using intensity pattern analysis.

    This class finds the regular grid structure of microscope tiles (not MCMICRO tiles)
    by detecting intensity discontinuities and fitting a regular grid.
    """

    def __init__(self,
                 bin_size: int = 50,
                 peak_distance: int = 20,
                 peak_height_percentile: int = 75,
                 min_tiles: int = 4,
                 min_tile_size: int = 100,
                 outlier_threshold: float = 2.0):
        """
        Initialize the microscope tile detector.

        Parameters
        ----------
        bin_size : int
            Spatial binning size in pixels (default: 50)
        peak_distance : int
            Minimum distance between peaks in bins (default: 20)
        peak_height_percentile : int
            Percentile for peak detection threshold (default: 75)
        min_tiles : int
            Minimum number of tiles to proceed (default: 4)
        min_tile_size : int
            Minimum number of cells per tile (default: 100)
        outlier_threshold : float
            MAD units for classifying dimmer tiles (default: 2.0)
        """
        self.bin_size = bin_size
        self.peak_distance = peak_distance
        self.peak_height_percentile = peak_height_percentile
        self.min_tiles = min_tiles
        self.min_tile_size = min_tile_size
        self.outlier_threshold = outlier_threshold

    def create_spatial_heatmap(self,
                              x_coords: np.ndarray,
                              y_coords: np.ndarray,
                              intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create fine-resolution intensity heatmap for tile boundary detection.

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
            2D array with median intensity per bin
        x_edges : np.ndarray
            Bin edges for X axis
        y_edges : np.ndarray
            Bin edges for Y axis
        """
        # Filter out invalid values
        valid = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(intensities) & (intensities > 0)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        intensities = intensities[valid]

        if len(x_coords) == 0:
            return np.zeros((10, 10)), np.linspace(0, 1000, 11), np.linspace(0, 1000, 11)

        # Create bins
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_edges = np.arange(x_min, x_max + self.bin_size, self.bin_size)
        y_edges = np.arange(y_min, y_max + self.bin_size, self.bin_size)

        n_x_bins = len(x_edges) - 1
        n_y_bins = len(y_edges) - 1

        # Bin the data and calculate median intensity per bin
        heatmap = np.zeros((n_y_bins, n_x_bins))

        # Digitize coordinates
        x_idx = np.digitize(x_coords, x_edges) - 1
        y_idx = np.digitize(y_coords, y_edges) - 1

        # Clip to valid range
        x_idx = np.clip(x_idx, 0, n_x_bins - 1)
        y_idx = np.clip(y_idx, 0, n_y_bins - 1)

        # Calculate median per bin
        for by in range(n_y_bins):
            for bx in range(n_x_bins):
                mask = (x_idx == bx) & (y_idx == by)
                if np.any(mask):
                    heatmap[by, bx] = np.median(intensities[mask])

        return heatmap, x_edges, y_edges

    def detect_tile_boundaries(self, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Detect microscope tile boundaries using Sobel edge detection.

        Parameters
        ----------
        heatmap : np.ndarray
            2D intensity heatmap

        Returns
        -------
        edges_y : np.ndarray
            Horizontal edge strength (for detecting horizontal tile boundaries)
        edges_x : np.ndarray
            Vertical edge strength (for detecting vertical tile boundaries)
        y_peaks : List[int]
            Y-coordinates of horizontal boundary lines (in bin indices)
        x_peaks : List[int]
            X-coordinates of vertical boundary lines (in bin indices)
        """
        # Apply Sobel to detect edges (tile boundaries)
        edges_y = filters.sobel(heatmap, axis=0)  # Horizontal lines
        edges_x = filters.sobel(heatmap, axis=1)  # Vertical lines

        # Project edges to 1D and find peaks (tile boundaries)
        y_projection = np.abs(edges_y).sum(axis=1)
        x_projection = np.abs(edges_x).sum(axis=0)

        # Find peaks with adaptive threshold
        y_threshold = np.percentile(y_projection[y_projection > 0], self.peak_height_percentile) if np.any(y_projection > 0) else 0
        x_threshold = np.percentile(x_projection[x_projection > 0], self.peak_height_percentile) if np.any(x_projection > 0) else 0

        y_peaks, _ = find_peaks(y_projection, height=y_threshold, distance=self.peak_distance)
        x_peaks, _ = find_peaks(x_projection, height=x_threshold, distance=self.peak_distance)

        return edges_y, edges_x, list(y_peaks), list(x_peaks)

    def fit_regular_grid(self,
                        y_peaks: List[int],
                        x_peaks: List[int],
                        y_edges: np.ndarray,
                        x_edges: np.ndarray,
                        max_y: float,
                        max_x: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Estimate grid spacing and fit regular grid.

        Parameters
        ----------
        y_peaks : List[int]
            Y-coordinates of detected boundaries (bin indices)
        x_peaks : List[int]
            X-coordinates of detected boundaries (bin indices)
        y_edges : np.ndarray
            Y bin edges (pixel coordinates)
        x_edges : np.ndarray
            X bin edges (pixel coordinates)
        max_y : float
            Maximum Y coordinate in data
        max_x : float
            Maximum X coordinate in data

        Returns
        -------
        y_grid : np.ndarray
            Y-coordinates of horizontal grid lines (pixel coordinates)
        x_grid : np.ndarray
            X-coordinates of vertical grid lines (pixel coordinates)
        y_spacing : float
            Vertical spacing between tiles (pixels)
        x_spacing : float
            Horizontal spacing between tiles (pixels)
        """
        # Convert peak indices to pixel coordinates
        y_peak_coords = y_edges[y_peaks] if len(y_peaks) > 0 else np.array([])
        x_peak_coords = x_edges[x_peaks] if len(x_peaks) > 0 else np.array([])

        # Estimate grid spacing (median of differences)
        if len(y_peaks) > 1:
            y_spacing = np.median(np.diff(y_peak_coords))
        else:
            y_spacing = 1500.0  # Default CellDIVE tile size

        if len(x_peaks) > 1:
            x_spacing = np.median(np.diff(x_peak_coords))
        else:
            x_spacing = 1500.0  # Default CellDIVE tile size

        # Fit regular grid starting from first detected boundary
        if len(y_peak_coords) > 0:
            y_start = y_peak_coords[0]
            y_grid = np.arange(y_start, max_y + y_spacing, y_spacing)
        else:
            y_grid = np.arange(0, max_y + y_spacing, y_spacing)

        if len(x_peak_coords) > 0:
            x_start = x_peak_coords[0]
            x_grid = np.arange(x_start, max_x + x_spacing, x_spacing)
        else:
            x_grid = np.arange(0, max_x + x_spacing, x_spacing)

        return y_grid, x_grid, y_spacing, x_spacing

    def assign_cells_to_grid_tiles(self,
                                   x_coords: np.ndarray,
                                   y_coords: np.ndarray,
                                   y_grid: np.ndarray,
                                   x_grid: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Assign cells to microscope grid tiles and calculate tile statistics.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        y_grid : np.ndarray
            Y-coordinates of horizontal grid lines
        x_grid : np.ndarray
            X-coordinates of vertical grid lines

        Returns
        -------
        cell_tile_ids : np.ndarray
            Tile ID for each cell (row * n_cols + col)
        tile_stats : Dict
            Statistics for each tile
        """
        n_cells = len(x_coords)
        cell_tile_ids = np.zeros(n_cells, dtype=int)

        # Assign cells to grid tiles
        y_tile_idx = np.digitize(y_coords, y_grid)
        x_tile_idx = np.digitize(x_coords, x_grid)

        # Create tile IDs (row * n_cols + col)
        n_cols = len(x_grid)
        cell_tile_ids = y_tile_idx * n_cols + x_tile_idx

        return cell_tile_ids

    def calculate_tile_statistics(self,
                                  cell_tile_ids: np.ndarray,
                                  intensities: np.ndarray) -> Dict:
        """
        Calculate median intensity for each tile.

        Parameters
        ----------
        cell_tile_ids : np.ndarray
            Tile ID for each cell
        intensities : np.ndarray
            Marker intensities for each cell

        Returns
        -------
        tile_stats : Dict
            Statistics for each tile
        """
        tile_stats = {}
        unique_tiles = np.unique(cell_tile_ids)

        for tile_id in unique_tiles:
            if tile_id == 0:  # Skip unassigned cells
                continue

            tile_mask = cell_tile_ids == tile_id
            tile_intensities = intensities[tile_mask]

            # Filter positive values
            tile_intensities_pos = tile_intensities[tile_intensities > 0]

            if len(tile_intensities_pos) > 0:
                tile_stats[int(tile_id)] = {
                    'n_cells': int(np.sum(tile_mask)),
                    'median_intensity': float(np.median(tile_intensities_pos)),
                    'mean_intensity': float(np.mean(tile_intensities_pos)),
                    'std_intensity': float(np.std(tile_intensities_pos))
                }

        return tile_stats

    def classify_tiles(self, tile_stats: Dict) -> Tuple[List[int], List[int], List[int]]:
        """
        Classify tiles as dimmer, brighter, or normal using MAD-based outlier detection.

        Parameters
        ----------
        tile_stats : Dict
            Statistics for each tile

        Returns
        -------
        dimmer_tiles : List[int]
            IDs of dimmer tiles
        brighter_tiles : List[int]
            IDs of brighter tiles
        normal_tiles : List[int]
            IDs of normal tiles
        """
        from scipy.stats import median_abs_deviation

        if len(tile_stats) < 3:
            return [], [], list(tile_stats.keys())

        # Get median intensities
        tile_ids = list(tile_stats.keys())
        medians = np.array([tile_stats[tid]['median_intensity'] for tid in tile_ids])

        # Calculate global median and MAD
        global_median = np.median(medians)
        mad = median_abs_deviation(medians)

        if mad == 0:
            return [], [], tile_ids

        # Classify tiles based on MAD score
        dimmer_tiles = []
        brighter_tiles = []
        normal_tiles = []

        for tile_id, median in zip(tile_ids, medians):
            mad_score = (global_median - median) / mad

            if mad_score > self.outlier_threshold:  # Significantly dimmer
                dimmer_tiles.append(tile_id)
            elif mad_score < -self.outlier_threshold:  # Significantly brighter
                brighter_tiles.append(tile_id)
            else:  # Normal
                normal_tiles.append(tile_id)

        return dimmer_tiles, brighter_tiles, normal_tiles

    def detect(self,
              x_coords: np.ndarray,
              y_coords: np.ndarray,
              intensities: np.ndarray) -> Dict:
        """
        Detect microscope tile grid and classify tiles as dimmer vs normal.

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
            Detection results including grid, tile assignments, and classification
        """
        # Step 1: Create fine-resolution intensity heatmap
        heatmap, x_edges, y_edges = self.create_spatial_heatmap(x_coords, y_coords, intensities)

        # Step 2: Detect tile boundaries using Sobel edge detection
        edges_y, edges_x, y_peaks, x_peaks = self.detect_tile_boundaries(heatmap)

        # Check if we detected enough boundaries
        if len(y_peaks) < 1 or len(x_peaks) < 1:
            return {
                'detected': False,
                'reason': f'Insufficient tile boundaries detected (y={len(y_peaks)}, x={len(x_peaks)})'
            }

        # Step 3: Fit regular grid and estimate tile spacing
        max_y = y_coords.max()
        max_x = x_coords.max()
        y_grid, x_grid, y_spacing, x_spacing = self.fit_regular_grid(
            y_peaks, x_peaks, y_edges, x_edges, max_y, max_x
        )

        print(f"      Grid: {len(y_grid)} rows × {len(x_grid)} cols, "
              f"spacing: {y_spacing:.0f}px × {x_spacing:.0f}px")

        # Step 4: Assign cells to grid tiles
        cell_tile_ids = self.assign_cells_to_grid_tiles(x_coords, y_coords, y_grid, x_grid)

        # Step 5: Calculate statistics for each tile
        tile_stats = self.calculate_tile_statistics(cell_tile_ids, intensities)

        # Filter tiles by minimum cell count
        valid_tiles = {tid: stats for tid, stats in tile_stats.items()
                      if stats['n_cells'] >= self.min_tile_size}

        if len(valid_tiles) < self.min_tiles:
            return {
                'detected': False,
                'reason': f'Insufficient valid tiles ({len(valid_tiles)} < {self.min_tiles})'
            }

        # Step 6: Classify tiles as dimmer, brighter, or normal
        dimmer_tiles, brighter_tiles, normal_tiles = self.classify_tiles(valid_tiles)

        # Count cells in each group
        dimmer_cell_mask = np.isin(cell_tile_ids, dimmer_tiles)
        brighter_cell_mask = np.isin(cell_tile_ids, brighter_tiles)
        normal_cell_mask = np.isin(cell_tile_ids, normal_tiles)

        n_dimmer_cells = np.sum(dimmer_cell_mask)
        n_brighter_cells = np.sum(brighter_cell_mask)
        n_normal_cells = np.sum(normal_cell_mask)
        n_abnormal_cells = n_dimmer_cells + n_brighter_cells

        # Need enough normal cells for UniFORM reference
        if n_normal_cells < self.min_tile_size:
            return {
                'detected': False,
                'reason': f'Insufficient normal cells ({n_normal_cells} < {self.min_tile_size})'
            }

        # Need at least some abnormal tiles to correct
        if n_abnormal_cells < self.min_tile_size:
            return {
                'detected': False,
                'reason': f'No abnormal tiles to correct (dimmer={n_dimmer_cells}, brighter={n_brighter_cells})'
            }

        # Create tile labels for visualization (grid-based)
        tile_labels = np.zeros(heatmap.shape, dtype=int)
        for by in range(heatmap.shape[0]):
            for bx in range(heatmap.shape[1]):
                y_coord = y_edges[by]
                x_coord = x_edges[bx]
                y_idx = np.digitize([y_coord], y_grid)[0]
                x_idx = np.digitize([x_coord], x_grid)[0]
                tile_labels[by, bx] = y_idx * len(x_grid) + x_idx

        return {
            'detected': True,
            'heatmap': heatmap,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'edges_y': edges_y,
            'edges_x': edges_x,
            'y_peaks': y_peaks,
            'x_peaks': x_peaks,
            'y_grid': y_grid,
            'x_grid': x_grid,
            'y_spacing': y_spacing,
            'x_spacing': x_spacing,
            'tile_labels': tile_labels,
            'tile_stats': valid_tiles,
            'dimmer_tiles': dimmer_tiles,
            'brighter_tiles': brighter_tiles,
            'normal_tiles': normal_tiles,
            'cell_tile_ids': cell_tile_ids,
            'n_dimmer_cells': int(n_dimmer_cells),
            'n_brighter_cells': int(n_brighter_cells),
            'n_normal_cells': int(n_normal_cells),
            'n_tiles': len(valid_tiles),
            'n_dimmer_tiles': len(dimmer_tiles),
            'n_brighter_tiles': len(brighter_tiles),
            'n_normal_tiles': len(normal_tiles)
        }


class TileArtifactCorrector:
    """
    Corrects tile artifacts using UniFORM normalization.

    This class applies:
    1. Tile-level correction: Normalizes dimmer/brighter tiles to match normal tiles
    2. Radial correction: Corrects circular/vignetting artifacts within tiles
    """

    def __init__(self,
                 n_quantiles: int = 100,
                 correction_strength: float = 1.0,
                 bright_correction_strength: float = None,
                 radial_correction: bool = True,
                 radial_bins: int = 3,
                 radial_threshold: float = 0.15):
        """
        Initialize the corrector.

        Parameters
        ----------
        n_quantiles : int
            Number of quantiles for UniFORM normalization (default: 100)
        correction_strength : float
            Strength of dim tile correction (0-1, default: 1.0)
        bright_correction_strength : float
            Strength of bright tile correction (0-1, default: same as correction_strength)
            Use lower values to reduce bright tile normalization
        radial_correction : bool
            Enable within-tile radial artifact correction (default: True)
        radial_bins : int
            Number of radial zones (center to edge) for analysis (default: 3)
        radial_threshold : float
            Relative intensity difference threshold for radial correction (default: 0.15)
        """
        self.n_quantiles = n_quantiles
        self.correction_strength = correction_strength
        self.bright_correction_strength = bright_correction_strength if bright_correction_strength is not None else correction_strength
        self.radial_correction = radial_correction
        self.radial_bins = radial_bins
        self.radial_threshold = radial_threshold

    def uniform_normalize(self,
                         source_values: np.ndarray,
                         target_values: np.ndarray,
                         strength: float = None) -> np.ndarray:
        """
        Apply UniFORM-style quantile normalization.

        Maps the distribution of source_values to match target_values using
        quantile matching (CDF alignment).

        Parameters
        ----------
        source_values : np.ndarray
            Values to be normalized (dimmer or brighter tiles)
        target_values : np.ndarray
            Reference values (normal tiles)
        strength : float, optional
            Correction strength override (default: use self.correction_strength)

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
        correction_strength = strength if strength is not None else self.correction_strength
        if correction_strength < 1.0:
            normalized = source_values + correction_strength * (normalized - source_values)

        return normalized

    def correct_radial_artifacts(self,
                                 x_coords: np.ndarray,
                                 y_coords: np.ndarray,
                                 intensities: np.ndarray,
                                 cell_tile_ids: np.ndarray,
                                 y_grid: np.ndarray,
                                 x_grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Correct circular/radial artifacts within tiles (vignetting).

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Cell intensities
        cell_tile_ids : np.ndarray
            Tile ID for each cell
        y_grid : np.ndarray
            Y grid lines
        x_grid : np.ndarray
            X grid lines

        Returns
        -------
        corrected : np.ndarray
            Corrected intensities
        n_tiles_corrected : int
            Number of tiles with radial correction applied
        """
        corrected = intensities.copy()
        n_tiles_corrected = 0

        # Get unique tiles
        unique_tiles = np.unique(cell_tile_ids)
        unique_tiles = unique_tiles[unique_tiles > 0]  # Skip unassigned

        # Calculate tile spacing (approximate)
        if len(y_grid) > 1 and len(x_grid) > 1:
            y_spacing = np.median(np.diff(y_grid))
            x_spacing = np.median(np.diff(x_grid))
        else:
            return corrected, 0

        for tile_id in unique_tiles:
            tile_mask = cell_tile_ids == tile_id
            if np.sum(tile_mask) < 100:  # Need enough cells
                continue

            # Get tile cells
            tile_x = x_coords[tile_mask]
            tile_y = y_coords[tile_mask]
            tile_intensities = intensities[tile_mask]

            # Calculate tile center
            tile_center_x = np.median(tile_x)
            tile_center_y = np.median(tile_y)

            # Calculate radial distance from center (normalized by tile size)
            dx = (tile_x - tile_center_x) / x_spacing
            dy = (tile_y - tile_center_y) / y_spacing
            radial_dist = np.sqrt(dx**2 + dy**2)

            # Divide into radial bins
            max_dist = radial_dist.max()
            if max_dist == 0:
                continue

            radial_edges = np.linspace(0, max_dist, self.radial_bins + 1)
            radial_bin_ids = np.digitize(radial_dist, radial_edges) - 1
            radial_bin_ids = np.clip(radial_bin_ids, 0, self.radial_bins - 1)

            # Calculate median intensity per radial bin
            bin_medians = []
            for bin_id in range(self.radial_bins):
                bin_mask = radial_bin_ids == bin_id
                if np.sum(bin_mask) > 10:
                    bin_vals = tile_intensities[bin_mask]
                    bin_vals_pos = bin_vals[bin_vals > 0]
                    if len(bin_vals_pos) > 0:
                        bin_medians.append(np.median(bin_vals_pos))
                    else:
                        bin_medians.append(np.nan)
                else:
                    bin_medians.append(np.nan)

            bin_medians = np.array(bin_medians)
            valid_bins = ~np.isnan(bin_medians)

            if np.sum(valid_bins) < 2:
                continue

            # Calculate overall tile median
            tile_median = np.median(bin_medians[valid_bins])

            # Check for significant radial variation
            max_deviation = np.max(np.abs(bin_medians[valid_bins] - tile_median)) / (tile_median + 1e-10)

            if max_deviation > self.radial_threshold:
                # Apply radial correction
                for bin_id in range(self.radial_bins):
                    if not valid_bins[bin_id]:
                        continue

                    bin_mask = radial_bin_ids == bin_id
                    bin_median = bin_medians[bin_id]

                    if bin_median > 0:
                        correction_factor = tile_median / bin_median
                        # Limit correction
                        correction_factor = np.clip(correction_factor, 0.7, 1.3)

                        # Apply to this radial bin
                        tile_indices = np.where(tile_mask)[0]
                        bin_indices = tile_indices[bin_mask]
                        corrected[bin_indices] *= correction_factor

                n_tiles_corrected += 1

        return corrected, n_tiles_corrected

    def correct(self,
               x_coords: np.ndarray,
               y_coords: np.ndarray,
               intensities: np.ndarray,
               detection_results: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Apply tile-level and radial artifact correction.

        Parameters
        ----------
        x_coords : np.ndarray
            X coordinates of cells
        y_coords : np.ndarray
            Y coordinates of cells
        intensities : np.ndarray
            Original marker intensities
        detection_results : Dict
            Results from MicroscopeTileDetector.detect()

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
        brighter_tiles = detection_results['brighter_tiles']
        normal_tiles = detection_results['normal_tiles']

        # Create masks
        dimmer_mask = np.isin(cell_tile_ids, dimmer_tiles)
        brighter_mask = np.isin(cell_tile_ids, brighter_tiles)
        normal_mask = np.isin(cell_tile_ids, normal_tiles)

        n_dimmer = np.sum(dimmer_mask)
        n_brighter = np.sum(brighter_mask)
        n_normal = np.sum(normal_mask)

        corrected = intensities.copy()
        normal_values = intensities[normal_mask]

        n_tile_corrected = 0
        dimmer_correction_pct = 0.0
        brighter_correction_pct = 0.0

        # Step 1: Normalize dimmer tiles to match normal (using dim correction strength)
        if n_dimmer > 0:
            dimmer_values = intensities[dimmer_mask]
            normalized_dimmer = self.uniform_normalize(dimmer_values, normal_values, strength=self.correction_strength)
            corrected[dimmer_mask] = normalized_dimmer
            n_tile_corrected += n_dimmer

            changes = (normalized_dimmer - dimmer_values) / (dimmer_values + 1e-10)
            dimmer_correction_pct = np.mean(changes[np.isfinite(changes)]) * 100

        # Step 2: Normalize brighter tiles to match normal (using bright correction strength)
        if n_brighter > 0:
            brighter_values = intensities[brighter_mask]
            normalized_brighter = self.uniform_normalize(brighter_values, normal_values, strength=self.bright_correction_strength)
            corrected[brighter_mask] = normalized_brighter
            n_tile_corrected += n_brighter

            changes = (normalized_brighter - brighter_values) / (brighter_values + 1e-10)
            brighter_correction_pct = np.mean(changes[np.isfinite(changes)]) * 100

        # Step 3: Apply radial correction within tiles
        n_radial_tiles = 0
        if self.radial_correction:
            y_grid = detection_results.get('y_grid', np.array([]))
            x_grid = detection_results.get('x_grid', np.array([]))

            if len(y_grid) > 0 and len(x_grid) > 0:
                corrected, n_radial_tiles = self.correct_radial_artifacts(
                    x_coords, y_coords, corrected, cell_tile_ids, y_grid, x_grid
                )

        stats = {
            'detected': True,
            'n_dimmer_tiles': len(dimmer_tiles),
            'n_brighter_tiles': len(brighter_tiles),
            'n_normal_tiles': len(normal_tiles),
            'n_dimmer_cells': int(n_dimmer),
            'n_brighter_cells': int(n_brighter),
            'n_normal_cells': int(n_normal),
            'n_tile_corrected': int(n_tile_corrected),
            'n_corrected': int(n_tile_corrected),  # Alias for compatibility
            'n_dimmer_corrected': int(n_dimmer),
            'n_brighter_corrected': int(n_brighter),
            'n_radial_tiles': int(n_radial_tiles),
            'n_radial_corrected': int(n_radial_tiles),  # Alias for compatibility
            'dimmer_correction_pct': float(dimmer_correction_pct),
            'brighter_correction_pct': float(brighter_correction_pct),
            'mean_correction_pct': float((dimmer_correction_pct + brighter_correction_pct) / 2),
            'dimmer_tiles': dimmer_tiles,
            'brighter_tiles': brighter_tiles,
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

    # 2. Tile segmentation (show dimmer/brighter/normal tiles)
    tile_labels = detection_results.get('tile_labels', np.zeros_like(heatmap))
    dimmer_tiles = detection_results.get('dimmer_tiles', [])
    brighter_tiles = detection_results.get('brighter_tiles', [])
    normal_tiles = detection_results.get('normal_tiles', [])

    # Create classification map: 0=edge, 1=normal, 2=dimmer, 3=brighter
    tile_classification = np.zeros_like(tile_labels)
    for tid in normal_tiles:
        tile_classification[tile_labels == tid] = 1
    for tid in dimmer_tiles:
        tile_classification[tile_labels == tid] = 2
    for tid in brighter_tiles:
        tile_classification[tile_labels == tid] = 3

    cmap = plt.cm.colors.ListedColormap(['black', 'blue', 'red', 'yellow'])
    im2 = axes[0, 1].imshow(tile_classification, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    axes[0, 1].set_title(f'Tile Classification\n({len(dimmer_tiles)} dimmer, {len(brighter_tiles)} brighter, {len(normal_tiles)} normal)')
    axes[0, 1].set_xlabel('X Bin')
    axes[0, 1].set_ylabel('Y Bin')
    cbar = plt.colorbar(im2, ax=axes[0, 1], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Edge', 'Normal', 'Dimmer', 'Brighter'])

    # 3. Detected grid overlay
    axes[0, 2].imshow(heatmap, cmap='gray', aspect='auto', alpha=0.7)

    # Draw detected grid lines
    x_edges_plot = detection_results.get('x_edges', np.array([]))
    y_edges_plot = detection_results.get('y_edges', np.array([]))
    x_grid = detection_results.get('x_grid', np.array([]))
    y_grid = detection_results.get('y_grid', np.array([]))

    # Convert grid coordinates to bin indices for plotting
    for x_line in x_grid:
        x_idx = np.argmin(np.abs(x_edges_plot - x_line))
        axes[0, 2].axvline(x_idx, color='red', linestyle='-', linewidth=2, alpha=0.8)

    for y_line in y_grid:
        y_idx = np.argmin(np.abs(y_edges_plot - y_line))
        axes[0, 2].axhline(y_idx, color='red', linestyle='-', linewidth=2, alpha=0.8)

    axes[0, 2].set_title(f"Detected Microscope Grid\n({len(y_grid)} rows × {len(x_grid)} cols)")
    axes[0, 2].set_xlabel('X Bin')
    axes[0, 2].set_ylabel('Y Bin')

    # Bottom Row
    # Get tile assignments for cells
    cell_tile_ids = detection_results.get('cell_tile_ids', np.zeros(len(x_coords), dtype=int))
    dimmer_mask = np.isin(cell_tile_ids, dimmer_tiles)
    brighter_mask = np.isin(cell_tile_ids, brighter_tiles)
    normal_mask = np.isin(cell_tile_ids, normal_tiles)

    # Subsample for plotting if too many cells
    if len(x_coords) > 50000:
        sample_idx = np.random.choice(len(x_coords), 50000, replace=False)
        x_plot = x_coords[sample_idx]
        y_plot = y_coords[sample_idx]
        orig_plot = original_intensities[sample_idx]
        corr_plot = corrected_intensities[sample_idx]
        dimmer_plot = dimmer_mask[sample_idx]
        brighter_plot = brighter_mask[sample_idx]
        normal_plot = normal_mask[sample_idx]
    else:
        x_plot = x_coords
        y_plot = y_coords
        orig_plot = original_intensities
        corr_plot = corrected_intensities
        dimmer_plot = dimmer_mask
        brighter_plot = brighter_mask
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
