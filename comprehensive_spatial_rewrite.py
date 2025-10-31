#!/usr/bin/env python3
"""
COMPREHENSIVE SPATIAL ANALYSIS - COMPLETE REWRITE
===================================================

This is a ground-up rewrite addressing all requirements:

1. ALL analyses at TWO levels: per-sample AND per-tumor-structure
2. Distance from EACH tumor cell type to EACH immune population type
3. Per-marker region analysis for EACH tumor marker over time
4. ONLY KPT vs KPNT comparisons (NEVER cis vs trans alone)
5. Every combo plot has individual component plots
6. Both raw scatter plots AND summary bar plots
7. Only create directories that will have content
8. Comprehensive validation of all outputs

Author: Complete pipeline rewrite
Date: 2025-10-31
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import DBSCAN
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveSpatialAnalysisRewrite:
    """
    Complete rewrite of spatial analysis with proper structure.

    Key Principles:
    - Dual-level analysis: per-sample AND per-tumor-structure
    - Comprehensive distance matrices: ALL tumor types to ALL immune types
    - Per-marker analysis for ALL markers
    - ONLY KPT vs KPNT comparisons
    - Individual + combo plots for everything
    - Raw data + summary plots for everything
    """

    def __init__(self, adata, sample_metadata: pd.DataFrame,
                 output_dir: str = 'comprehensive_spatial_output'):
        """
        Initialize comprehensive spatial analysis.

        Parameters
        ----------
        adata : AnnData
            Full annotated data with spatial coordinates and phenotypes
        sample_metadata : pd.DataFrame
            Metadata with sample_id, group, timepoint
            group MUST contain 'KPT' or 'KPNT' for main_group extraction
        output_dir : str
            Base output directory
        """
        self.adata = adata
        self.sample_metadata = self._parse_metadata(sample_metadata)
        self.output_dir = Path(output_dir)

        # Extract spatial coordinates
        self.coords = self._extract_coordinates()

        # Merge metadata into adata
        self._merge_metadata()

        # Detect all available populations
        self.populations = self._detect_populations()

        # Separate tumor and immune populations
        self.tumor_populations = {k: v for k, v in self.populations.items()
                                 if 'tumor' in k.lower() or 'tom' in k.lower()}
        self.immune_populations = {k: v for k, v in self.populations.items()
                                  if k not in self.tumor_populations}

        # Storage for results
        self.tumor_structures = None  # Will store per-structure index
        self.results = {}

        print("\n" + "="*80)
        print("COMPREHENSIVE SPATIAL ANALYSIS INITIALIZED")
        print("="*80)
        print(f"Total cells: {len(adata):,}")
        print(f"Samples: {adata.obs['sample_id'].nunique()}")
        print(f"Main groups: {sorted(self.adata.obs['main_group'].dropna().unique())}")
        print(f"Timepoints: {sorted(self.adata.obs['timepoint'].dropna().unique())}")
        print(f"\nPopulations detected:")
        print(f"  Tumor types: {len(self.tumor_populations)}")
        for name, col in self.tumor_populations.items():
            count = self.adata.obs[col].sum()
            print(f"    - {name}: {count:,} cells")
        print(f"  Immune types: {len(self.immune_populations)}")
        for name, col in self.immune_populations.items():
            count = self.adata.obs[col].sum()
            print(f"    - {name}: {count:,} cells")
        print(f"\nOutput directory: {output_dir}/")
        print("="*80 + "\n")


    def _parse_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Parse metadata to extract KPT vs KPNT groups."""
        metadata = metadata.copy()
        metadata['sample_id'] = metadata['sample_id'].str.upper()

        # Extract main group: KPT vs KPNT (PRIMARY comparison)
        metadata['main_group'] = metadata['group'].apply(
            lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
        )

        # Extract cis/trans for detailed tracking (but NEVER compared alone)
        metadata['genotype'] = metadata['group'].apply(
            lambda x: 'cis' if 'cis' in str(x).lower() else
                     ('trans' if 'trans' in str(x).lower() else 'unknown')
        )

        # Full genotype name for 4-way comparisons (KPT-cis, KPT-trans, KPNT-cis, KPNT-trans)
        metadata['genotype_full'] = metadata.apply(
            lambda row: f"{row['main_group']}-{row['genotype']}", axis=1
        )

        # Convert timepoint to numeric
        metadata['timepoint'] = pd.to_numeric(metadata['timepoint'])

        return metadata


    def _extract_coordinates(self) -> np.ndarray:
        """Extract spatial coordinates from adata."""
        if 'spatial' in self.adata.obsm:
            return self.adata.obsm['spatial']
        elif 'X_centroid' in self.adata.obs and 'Y_centroid' in self.adata.obs:
            coords = np.column_stack([
                self.adata.obs['X_centroid'].values,
                self.adata.obs['Y_centroid'].values
            ])
            self.adata.obsm['spatial'] = coords
            return coords
        else:
            raise ValueError("No spatial coordinates found in adata")


    def _merge_metadata(self):
        """Merge metadata into adata.obs."""
        if 'sample_id' not in self.adata.obs:
            raise ValueError("adata.obs must have 'sample_id' column")

        self.adata.obs['sample_id'] = self.adata.obs['sample_id'].str.upper()

        # Create mapping dictionaries
        for col in ['main_group', 'genotype', 'genotype_full', 'timepoint', 'treatment', 'group']:
            if col in self.sample_metadata.columns:
                mapping = dict(zip(
                    self.sample_metadata['sample_id'],
                    self.sample_metadata[col]
                ))
                self.adata.obs[col] = self.adata.obs['sample_id'].map(mapping)


    def _detect_populations(self) -> Dict[str, str]:
        """
        Detect all population columns in adata.obs.

        Returns
        -------
        dict
            Mapping of population names to column names
        """
        populations = {}

        # Find all is_* columns
        for col in self.adata.obs.columns:
            if col.startswith('is_'):
                # Extract clean name
                clean_name = col.replace('is_', '')
                if self.adata.obs[col].dtype == bool or self.adata.obs[col].dtype == 'bool':
                    count = self.adata.obs[col].sum()
                    if count > 0:
                        populations[clean_name] = col

        return populations


    def detect_tumor_structures(self,
                               tumor_population: str = 'Tumor',
                               eps: float = 30,
                               min_samples: int = 10,
                               min_cluster_size: int = 50):
        """
        Detect tumor structures using DBSCAN clustering.

        This creates a per-structure index for dual-level analysis.

        Parameters
        ----------
        tumor_population : str
            Name of tumor population (will look up column)
        eps : float
            DBSCAN epsilon parameter (distance threshold)
        min_samples : int
            DBSCAN min_samples parameter
        min_cluster_size : int
            Minimum cells to consider a valid structure
        """
        print("\n" + "="*80)
        print("DETECTING TUMOR STRUCTURES")
        print("="*80 + "\n")

        # Get tumor column
        if tumor_population not in self.tumor_populations:
            raise ValueError(f"Tumor population '{tumor_population}' not found")

        tumor_col = self.tumor_populations[tumor_population]
        tumor_mask = self.adata.obs[tumor_col].values

        print(f"Clustering {tumor_mask.sum():,} {tumor_population} cells...")
        print(f"Parameters: eps={eps}, min_samples={min_samples}, min_size={min_cluster_size}")
        print()

        # Initialize structure index
        structure_data = []
        global_structure_id = 0

        # Cluster per sample
        for sample_id in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample_id
            sample_tumor_mask = sample_mask & tumor_mask

            if sample_tumor_mask.sum() < min_cluster_size:
                continue

            # Get coordinates for this sample's tumor cells
            coords = self.coords[sample_tumor_mask]

            # DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(coords)

            # Process each cluster
            unique_labels = set(labels) - {-1}  # Exclude noise
            sample_structures = 0

            for label in unique_labels:
                cluster_mask_local = labels == label
                cluster_size = cluster_mask_local.sum()

                if cluster_size >= min_cluster_size:
                    # Get global cell indices for this cluster
                    global_indices = np.where(sample_tumor_mask)[0][cluster_mask_local]

                    # Calculate structure properties
                    struct_coords = coords[cluster_mask_local]
                    centroid = struct_coords.mean(axis=0)

                    # Store structure info
                    structure_data.append({
                        'structure_id': global_structure_id,
                        'sample_id': sample_id,
                        'n_cells': cluster_size,
                        'centroid_x': centroid[0],
                        'centroid_y': centroid[1],
                        'cell_indices': global_indices  # Store for later use
                    })

                    global_structure_id += 1
                    sample_structures += 1

            if sample_structures > 0:
                print(f"  {sample_id}: {sample_structures} structures")

        self.tumor_structures = pd.DataFrame(structure_data)

        print(f"\n✓ Detected {len(self.tumor_structures)} tumor structures")
        print(f"  Size range: {self.tumor_structures['n_cells'].min()} - {self.tumor_structures['n_cells'].max()} cells")
        print(f"  Median size: {self.tumor_structures['n_cells'].median():.0f} cells")
        print()

        return self.tumor_structures


    def analyze_distances_comprehensive(self) -> pd.DataFrame:
        """
        COMPREHENSIVE DISTANCE ANALYSIS
        ================================

        Calculate distances from EACH tumor cell type to EACH immune population,
        at BOTH per-sample AND per-tumor-structure levels.

        For each tumor-immune pair:
        1. Per-structure level: Mean distance within each tumor structure
        2. Per-sample level: Mean distance within each sample
        3. Over time: Temporal dynamics
        4. Between groups: KPT vs KPNT comparison

        Returns
        -------
        DataFrame with columns:
            - tumor_type: Which tumor population
            - immune_type: Which immune population
            - structure_id: Structure ID (for structure-level)
            - sample_id: Sample ID
            - main_group: KPT or KPNT
            - genotype_full: Full genotype (KPT-cis, etc.)
            - timepoint: Timepoint
            - level: 'structure' or 'sample'
            - mean_distance: Mean distance (μm)
            - median_distance: Median distance (μm)
            - min_distance: Minimum distance (μm)
            - n_tumor_cells: Number of tumor cells
            - n_immune_cells: Number of immune cells
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE DISTANCE ANALYSIS")
        print("="*80)
        print("\nCalculating distances from ALL tumor types to ALL immune types...")
        print(f"  Tumor types: {len(self.tumor_populations)}")
        print(f"  Immune types: {len(self.immune_populations)}")
        print(f"  Total combinations: {len(self.tumor_populations) * len(self.immune_populations)}")
        print()

        results = []
        total_combos = len(self.tumor_populations) * len(self.immune_populations)
        current_combo = 0

        for tumor_name, tumor_col in self.tumor_populations.items():
            for immune_name, immune_col in self.immune_populations.items():
                current_combo += 1
                print(f"  Progress: {current_combo}/{total_combos} - {tumor_name} → {immune_name}")

                # LEVEL 1: PER-TUMOR-STRUCTURE ANALYSIS
                if self.tumor_structures is not None:
                    structure_results = self._calculate_distances_per_structure(
                        tumor_name, tumor_col, immune_name, immune_col
                    )
                    results.extend(structure_results)

                # LEVEL 2: PER-SAMPLE ANALYSIS
                sample_results = self._calculate_distances_per_sample(
                    tumor_name, tumor_col, immune_name, immune_col
                )
                results.extend(sample_results)

        df = pd.DataFrame(results)

        print(f"\n✓ Distance analysis complete!")
        print(f"  Total measurements: {len(df):,}")
        print(f"  Structure-level: {len(df[df['level']=='structure']):,}")
        print(f"  Sample-level: {len(df[df['level']=='sample']):,}")
        print()

        # Save results
        output_dir = self.output_dir / 'distance_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'comprehensive_distances.csv', index=False)

        self.results['distances'] = df
        return df


    def _calculate_distances_per_structure(self, tumor_name: str, tumor_col: str,
                                          immune_name: str, immune_col: str) -> List[Dict]:
        """Calculate distances at per-structure level."""
        results = []

        for _, structure in self.tumor_structures.iterrows():
            structure_id = structure['structure_id']
            sample_id = structure['sample_id']
            cell_indices = structure['cell_indices']

            # Get tumor cells in this structure
            tumor_mask = np.zeros(len(self.adata), dtype=bool)
            tumor_mask[cell_indices] = True
            tumor_mask = tumor_mask & self.adata.obs[tumor_col].values

            # Get immune cells in this sample
            sample_mask = self.adata.obs['sample_id'] == sample_id
            immune_mask = sample_mask & self.adata.obs[immune_col].values

            if tumor_mask.sum() == 0 or immune_mask.sum() == 0:
                continue

            # Calculate distances
            tumor_coords = self.coords[tumor_mask]
            immune_coords = self.coords[immune_mask]

            # Build KDTree for immune cells
            tree = cKDTree(immune_coords)

            # Find nearest immune cell for each tumor cell
            distances, _ = tree.query(tumor_coords, k=1)

            # Get metadata
            sample_meta = self.sample_metadata[
                self.sample_metadata['sample_id'] == sample_id
            ].iloc[0]

            results.append({
                'tumor_type': tumor_name,
                'immune_type': immune_name,
                'structure_id': structure_id,
                'sample_id': sample_id,
                'main_group': sample_meta['main_group'],
                'genotype_full': sample_meta['genotype_full'],
                'timepoint': sample_meta['timepoint'],
                'level': 'structure',
                'mean_distance': float(np.mean(distances)),
                'median_distance': float(np.median(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'std_distance': float(np.std(distances)),
                'n_tumor_cells': int(tumor_mask.sum()),
                'n_immune_cells': int(immune_mask.sum())
            })

        return results


    def _calculate_distances_per_sample(self, tumor_name: str, tumor_col: str,
                                       immune_name: str, immune_col: str) -> List[Dict]:
        """Calculate distances at per-sample level."""
        results = []

        for sample_id in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample_id

            # Get tumor and immune cells in this sample
            tumor_mask = sample_mask & self.adata.obs[tumor_col].values
            immune_mask = sample_mask & self.adata.obs[immune_col].values

            if tumor_mask.sum() == 0 or immune_mask.sum() == 0:
                continue

            # Calculate distances
            tumor_coords = self.coords[tumor_mask]
            immune_coords = self.coords[immune_mask]

            # Build KDTree
            tree = cKDTree(immune_coords)
            distances, _ = tree.query(tumor_coords, k=1)

            # Get metadata
            sample_meta = self.sample_metadata[
                self.sample_metadata['sample_id'] == sample_id
            ].iloc[0]

            results.append({
                'tumor_type': tumor_name,
                'immune_type': immune_name,
                'structure_id': None,  # Sample-level, not structure-specific
                'sample_id': sample_id,
                'main_group': sample_meta['main_group'],
                'genotype_full': sample_meta['genotype_full'],
                'timepoint': sample_meta['timepoint'],
                'level': 'sample',
                'mean_distance': float(np.mean(distances)),
                'median_distance': float(np.median(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'std_distance': float(np.std(distances)),
                'n_tumor_cells': int(tumor_mask.sum()),
                'n_immune_cells': int(immune_mask.sum())
            })

        return results


    def analyze_marker_regions_temporal(self) -> pd.DataFrame:
        """
        PER-MARKER REGION ANALYSIS OVER TIME
        =====================================

        For EACH tumor marker (pERK+, Ki67+, NINJA+, etc.):
        1. Quantify fraction of tumor that is marker+
        2. Track over time
        3. Compare KPT vs KPNT
        4. At both sample and structure levels

        Returns
        -------
        DataFrame with:
            - marker_name: Which marker
            - marker_type: 'positive' or 'negative'
            - level: 'structure' or 'sample'
            - structure_id / sample_id
            - main_group, genotype_full, timepoint
            - n_cells: Number of cells in this marker region
            - fraction: Fraction of total tumor
            - density: Cells per area
        """
        print("\n" + "="*80)
        print("PER-MARKER REGION TEMPORAL ANALYSIS")
        print("="*80 + "\n")

        print(f"Analyzing temporal dynamics for {len(self.tumor_populations)} tumor markers...")
        print()

        results = []

        for marker_name, marker_col in self.tumor_populations.items():
            print(f"  Analyzing {marker_name}...")

            # Get marker+ and marker- populations
            marker_pos_mask = self.adata.obs[marker_col].values

            # Get base tumor population for normalization
            base_tumor_col = self.tumor_populations.get('Tumor', marker_col)
            base_tumor_mask = self.adata.obs[base_tumor_col].values

            # LEVEL 1: Per-structure analysis
            if self.tumor_structures is not None:
                for _, structure in self.tumor_structures.iterrows():
                    structure_id = structure['structure_id']
                    sample_id = structure['sample_id']
                    cell_indices = structure['cell_indices']

                    # Get cells in this structure
                    structure_mask = np.zeros(len(self.adata), dtype=bool)
                    structure_mask[cell_indices] = True

                    # Count marker+ and marker- in structure
                    structure_base = structure_mask & base_tumor_mask
                    structure_pos = structure_mask & marker_pos_mask
                    structure_neg = structure_base & ~marker_pos_mask

                    if structure_base.sum() == 0:
                        continue

                    # Get metadata
                    sample_meta = self.sample_metadata[
                        self.sample_metadata['sample_id'] == sample_id
                    ].iloc[0]

                    # Calculate area (convex hull)
                    coords = self.coords[structure_base]
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(coords)
                        area = hull.volume  # 2D area
                    except:
                        area = np.nan

                    # Store marker+ results
                    results.append({
                        'marker_name': marker_name,
                        'marker_type': 'positive',
                        'level': 'structure',
                        'structure_id': structure_id,
                        'sample_id': sample_id,
                        'main_group': sample_meta['main_group'],
                        'genotype_full': sample_meta['genotype_full'],
                        'timepoint': sample_meta['timepoint'],
                        'n_cells': int(structure_pos.sum()),
                        'fraction': float(structure_pos.sum() / structure_base.sum()),
                        'density': float(structure_pos.sum() / area) if not np.isnan(area) else np.nan,
                        'total_cells': int(structure_base.sum())
                    })

                    # Store marker- results
                    results.append({
                        'marker_name': marker_name,
                        'marker_type': 'negative',
                        'level': 'structure',
                        'structure_id': structure_id,
                        'sample_id': sample_id,
                        'main_group': sample_meta['main_group'],
                        'genotype_full': sample_meta['genotype_full'],
                        'timepoint': sample_meta['timepoint'],
                        'n_cells': int(structure_neg.sum()),
                        'fraction': float(structure_neg.sum() / structure_base.sum()),
                        'density': float(structure_neg.sum() / area) if not np.isnan(area) else np.nan,
                        'total_cells': int(structure_base.sum())
                    })

            # LEVEL 2: Per-sample analysis
            for sample_id in self.adata.obs['sample_id'].unique():
                sample_mask = self.adata.obs['sample_id'] == sample_id

                # Count marker+ and marker-
                sample_base = sample_mask & base_tumor_mask
                sample_pos = sample_mask & marker_pos_mask
                sample_neg = sample_base & ~marker_pos_mask

                if sample_base.sum() == 0:
                    continue

                # Get metadata
                sample_meta = self.sample_metadata[
                    self.sample_metadata['sample_id'] == sample_id
                ].iloc[0]

                # Store results
                results.append({
                    'marker_name': marker_name,
                    'marker_type': 'positive',
                    'level': 'sample',
                    'structure_id': None,
                    'sample_id': sample_id,
                    'main_group': sample_meta['main_group'],
                    'genotype_full': sample_meta['genotype_full'],
                    'timepoint': sample_meta['timepoint'],
                    'n_cells': int(sample_pos.sum()),
                    'fraction': float(sample_pos.sum() / sample_base.sum()),
                    'density': np.nan,  # Would need total sample area
                    'total_cells': int(sample_base.sum())
                })

                results.append({
                    'marker_name': marker_name,
                    'marker_type': 'negative',
                    'level': 'sample',
                    'structure_id': None,
                    'sample_id': sample_id,
                    'main_group': sample_meta['main_group'],
                    'genotype_full': sample_meta['genotype_full'],
                    'timepoint': sample_meta['timepoint'],
                    'n_cells': int(sample_neg.sum()),
                    'fraction': float(sample_neg.sum() / sample_base.sum()),
                    'density': np.nan,
                    'total_cells': int(sample_base.sum())
                })

        df = pd.DataFrame(results)

        print(f"\n✓ Marker region analysis complete!")
        print(f"  Total measurements: {len(df):,}")
        print(f"  Structure-level: {len(df[df['level']=='structure']):,}")
        print(f"  Sample-level: {len(df[df['level']=='sample']):,}")
        print()

        # Save results
        output_dir = self.output_dir / 'marker_regions'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'marker_regions_temporal.csv', index=False)

        self.results['marker_regions'] = df
        return df


    def run_all_analyses(self):
        """
        Run ALL comprehensive analyses.

        This executes:
        1. Tumor structure detection
        2. Comprehensive distance analysis (all tumor types to all immune types)
        3. Per-marker region temporal analysis
        4. Comprehensive infiltration analysis (all immune types at multiple scales)
        """
        print("\n" + "="*80)
        print("RUNNING ALL COMPREHENSIVE ANALYSES")
        print("="*80 + "\n")

        # Phase 1: Detect tumor structures
        self.detect_tumor_structures()

        # Phase 2: Comprehensive distance analysis
        self.analyze_distances_comprehensive()

        # Phase 3: Per-marker region analysis
        self.analyze_marker_regions_temporal()

        # Phase 4: Comprehensive infiltration analysis
        self.analyze_infiltration_comprehensive()

        print("\n" + "="*80)
        print("ALL ANALYSES COMPLETE")
        print("="*80 + "\n")

        return self.results


    def analyze_infiltration_comprehensive(self,
                                          boundary_widths: List[int] = [30, 100, 200]) -> pd.DataFrame:
        """
        COMPREHENSIVE INFILTRATION ANALYSIS
        ====================================

        Quantify immune infiltration into tumor structures at multiple scales.
        Analyzes at BOTH per-sample AND per-tumor-structure levels.

        For each immune population:
        1. Calculate infiltration at multiple boundary widths
        2. Track over time
        3. Compare KPT vs KPNT
        4. At both structure and sample levels

        Parameters
        ----------
        boundary_widths : list of int
            Boundary widths to analyze (in μm)

        Returns
        -------
        DataFrame with:
            - immune_type: Which immune population
            - boundary_width: Boundary width analyzed
            - level: 'structure' or 'sample'
            - structure_id / sample_id
            - main_group, genotype_full, timepoint
            - n_immune_in_boundary: Number of immune cells within boundary
            - n_immune_total: Total immune cells in sample
            - infiltration_density: Immune cells per μm² of boundary area
            - infiltration_fraction: Fraction of immune cells in boundary
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE INFILTRATION ANALYSIS")
        print("="*80 + "\n")

        print(f"Analyzing infiltration for {len(self.immune_populations)} immune populations...")
        print(f"At boundary widths: {boundary_widths} μm")
        print()

        results = []

        for immune_name, immune_col in self.immune_populations.items():
            print(f"  Analyzing {immune_name} infiltration...")

            # LEVEL 1: Per-structure infiltration
            if self.tumor_structures is not None:
                structure_results = self._calculate_infiltration_per_structure(
                    immune_name, immune_col, boundary_widths
                )
                results.extend(structure_results)

            # LEVEL 2: Per-sample infiltration
            sample_results = self._calculate_infiltration_per_sample(
                immune_name, immune_col, boundary_widths
            )
            results.extend(sample_results)

        df = pd.DataFrame(results)

        print(f"\n✓ Infiltration analysis complete!")
        print(f"  Total measurements: {len(df):,}")
        print(f"  Structure-level: {len(df[df['level']=='structure']):,}")
        print(f"  Sample-level: {len(df[df['level']=='sample']):,}")
        print()

        # Save results
        output_dir = self.output_dir / 'infiltration_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'comprehensive_infiltration.csv', index=False)

        self.results['infiltration'] = df
        return df


    def _calculate_infiltration_per_structure(self, immune_name: str, immune_col: str,
                                             boundary_widths: List[int]) -> List[Dict]:
        """Calculate infiltration at per-structure level."""
        from shapely.geometry import Point, MultiPoint
        from shapely.ops import unary_union
        import numpy as np

        results = []

        for _, structure in self.tumor_structures.iterrows():
            structure_id = structure['structure_id']
            sample_id = structure['sample_id']
            cell_indices = structure['cell_indices']

            # Get tumor cells in this structure
            tumor_coords = self.coords[cell_indices]

            # Get immune cells in this sample
            sample_mask = self.adata.obs['sample_id'] == sample_id
            immune_mask = sample_mask & self.adata.obs[immune_col].values
            immune_coords = self.coords[immune_mask]

            if len(immune_coords) == 0:
                continue

            # Get metadata
            sample_meta = self.sample_metadata[
                self.sample_metadata['sample_id'] == sample_id
            ].iloc[0]

            # For each boundary width
            for width in boundary_widths:
                # Create boundary region around tumor
                try:
                    # Create points for tumor boundary
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(tumor_coords)
                    hull_points = tumor_coords[hull.vertices]

                    # Create polygon and buffer
                    from shapely.geometry import Polygon
                    tumor_polygon = Polygon(hull_points)
                    boundary_region = tumor_polygon.buffer(width)

                    # Count immune cells in boundary
                    immune_in_boundary = 0
                    for ic in immune_coords:
                        point = Point(ic)
                        if boundary_region.contains(point):
                            immune_in_boundary += 1

                    # Calculate boundary area
                    boundary_area = boundary_region.area - tumor_polygon.area

                    results.append({
                        'immune_type': immune_name,
                        'boundary_width': width,
                        'level': 'structure',
                        'structure_id': structure_id,
                        'sample_id': sample_id,
                        'main_group': sample_meta['main_group'],
                        'genotype_full': sample_meta['genotype_full'],
                        'timepoint': sample_meta['timepoint'],
                        'n_immune_in_boundary': immune_in_boundary,
                        'n_immune_total': int(immune_mask.sum()),
                        'infiltration_density': immune_in_boundary / boundary_area if boundary_area > 0 else 0,
                        'infiltration_fraction': immune_in_boundary / immune_mask.sum() if immune_mask.sum() > 0 else 0,
                        'boundary_area': boundary_area,
                        'tumor_size': len(cell_indices)
                    })

                except Exception as e:
                    # If boundary calculation fails (e.g., too few points for hull)
                    continue

        return results


    def _calculate_infiltration_per_sample(self, immune_name: str, immune_col: str,
                                          boundary_widths: List[int]) -> List[Dict]:
        """Calculate infiltration at per-sample level (aggregate across all structures)."""
        results = []

        # Get base tumor population
        base_tumor_col = self.tumor_populations.get('Tumor')
        if base_tumor_col is None:
            return results

        for sample_id in self.adata.obs['sample_id'].unique():
            sample_mask = self.adata.obs['sample_id'] == sample_id

            # Get tumor and immune cells
            tumor_mask = sample_mask & self.adata.obs[base_tumor_col].values
            immune_mask = sample_mask & self.adata.obs[immune_col].values

            if tumor_mask.sum() == 0 or immune_mask.sum() == 0:
                continue

            tumor_coords = self.coords[tumor_mask]
            immune_coords = self.coords[immune_mask]

            # Get metadata
            sample_meta = self.sample_metadata[
                self.sample_metadata['sample_id'] == sample_id
            ].iloc[0]

            # For each boundary width
            for width in boundary_widths:
                try:
                    # Build KDTree on tumor cells
                    tree = cKDTree(tumor_coords)

                    # Find immune cells within boundary
                    immune_in_boundary = 0
                    for ic in immune_coords:
                        dist, _ = tree.query(ic, k=1)
                        if dist <= width:
                            immune_in_boundary += 1

                    results.append({
                        'immune_type': immune_name,
                        'boundary_width': width,
                        'level': 'sample',
                        'structure_id': None,
                        'sample_id': sample_id,
                        'main_group': sample_meta['main_group'],
                        'genotype_full': sample_meta['genotype_full'],
                        'timepoint': sample_meta['timepoint'],
                        'n_immune_in_boundary': immune_in_boundary,
                        'n_immune_total': int(immune_mask.sum()),
                        'infiltration_density': np.nan,  # Would need total sample area
                        'infiltration_fraction': immune_in_boundary / immune_mask.sum() if immune_mask.sum() > 0 else 0,
                        'boundary_area': np.nan,
                        'tumor_size': int(tumor_mask.sum())
                    })

                except Exception as e:
                    continue

        return results


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

class ComprehensivePlotGenerator:
    """
    Generates ALL required plots with proper organization.

    Rules:
    1. Every combo plot MUST have individual component plots
    2. Both scatter/raw data AND summary bar plots
    3. Only KPT vs KPNT comparisons (NEVER cis vs trans alone)
    4. Proper directory organization
    5. Only create directories with actual content
    """

    def __init__(self, results: Dict, output_dir: Path):
        """
        Initialize plot generator.

        Parameters
        ----------
        results : dict
            Dictionary of analysis results from ComprehensiveSpatialAnalysisRewrite
        output_dir : Path
            Base output directory
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'


    def plot_distance_analysis(self):
        """
        Generate ALL distance analysis plots.

        For each tumor-immune pair:
        1. Individual temporal plot (KPT vs KPNT over time)
        2. Individual boxplot (KPT vs KPNT comparison)
        3. Individual scatter plot (raw data)
        4. Summary heatmap (all pairs)
        5. Combo figure (3x3 grid for each pair)
        """
        print("\n" + "="*80)
        print("GENERATING DISTANCE ANALYSIS PLOTS")
        print("="*80 + "\n")

        if 'distances' not in self.results:
            print("No distance results found, skipping...")
            return

        df = self.results['distances']

        # Create output directories
        dist_dir = self.figures_dir / 'distance_analysis'
        dist_dir.mkdir(parents=True, exist_ok=True)

        individual_dir = dist_dir / 'individual_plots'
        individual_dir.mkdir(parents=True, exist_ok=True)

        combo_dir = dist_dir / 'combo_plots'
        combo_dir.mkdir(parents=True, exist_ok=True)

        # Get all tumor-immune pairs
        pairs = df[['tumor_type', 'immune_type']].drop_duplicates()

        print(f"Generating plots for {len(pairs)} tumor-immune pairs...")

        for _, row in pairs.iterrows():
            tumor_type = row['tumor_type']
            immune_type = row['immune_type']

            print(f"  {tumor_type} → {immune_type}")

            # Filter data for this pair
            pair_data = df[
                (df['tumor_type'] == tumor_type) &
                (df['immune_type'] == immune_type)
            ].copy()

            # Generate individual plots
            self._plot_distance_individual(
                pair_data, tumor_type, immune_type, individual_dir
            )

            # Generate combo plot
            self._plot_distance_combo(
                pair_data, tumor_type, immune_type, combo_dir
            )

        # Generate summary heatmap
        self._plot_distance_heatmap(df, dist_dir)

        print("\n✓ Distance analysis plots complete!")


    def _plot_distance_individual(self, df: pd.DataFrame,
                                  tumor_type: str, immune_type: str,
                                  output_dir: Path):
        """Generate individual plots for a tumor-immune pair."""
        safe_name = f"{tumor_type}_to_{immune_type}".replace(' ', '_').replace('+', 'pos')

        # Sample-level data only for cleaner temporal plots
        sample_data = df[df['level'] == 'sample'].copy()

        if len(sample_data) == 0:
            return

        # PLOT 1: Temporal plot - KPT vs KPNT over time
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['mean_distance'].mean()
            temporal_sem = group_data.groupby('timepoint')['mean_distance'].sem()

            ax.errorbar(temporal_mean.index, temporal_mean.values,
                       yerr=temporal_sem.values,
                       marker='o', label=group, linewidth=2.5,
                       markersize=8, capsize=5)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Distance (μm)', fontweight='bold', fontsize=12)
        ax.set_title(f'{tumor_type} to {immune_type}\nKPT vs KPNT Over Time',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 2: Boxplot - KPT vs KPNT comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(data=sample_data, x='main_group', y='mean_distance',
                   ax=ax, palette='Set2', linewidth=2)

        # Add p-value
        groups = sample_data['main_group'].unique()
        if len(groups) == 2:
            data1 = sample_data[sample_data['main_group'] == groups[0]]['mean_distance'].dropna()
            data2 = sample_data[sample_data['main_group'] == groups[1]]['mean_distance'].dropna()
            if len(data1) > 0 and len(data2) > 0:
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(data1, data2)
                p_text = f'p = {p:.4f}' if p >= 0.001 else 'p < 0.001'
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=11, fontweight='bold')

        ax.set_ylabel('Distance (μm)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Group', fontweight='bold', fontsize=12)
        ax.set_title(f'{tumor_type} to {immune_type}\nKPT vs KPNT',
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 3: Scatter plot - raw data with temporal color
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            scatter = ax.scatter(group_data['timepoint'], group_data['mean_distance'],
                               alpha=0.6, s=80, label=group)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Distance (μm)', fontweight='bold', fontsize=12)
        ax.set_title(f'{tumor_type} to {immune_type}\nRaw Data (Sample-level)',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_distance_combo(self, df: pd.DataFrame,
                            tumor_type: str, immune_type: str,
                            output_dir: Path):
        """Generate combo 3x3 plot for a tumor-immune pair."""
        safe_name = f"{tumor_type}_to_{immune_type}".replace(' ', '_').replace('+', 'pos')

        # Create 3x3 figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Sample-level data
        sample_data = df[df['level'] == 'sample'].copy()
        structure_data = df[df['level'] == 'structure'].copy()

        if len(sample_data) == 0:
            plt.close()
            return

        # ROW 1: Sample-level analysis
        # Plot 1: Temporal - KPT vs KPNT
        ax1 = fig.add_subplot(gs[0, 0])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['mean_distance'].mean()
            temporal_sem = group_data.groupby('timepoint')['mean_distance'].sem()
            ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                        marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
        ax1.set_xlabel('Timepoint', fontweight='bold')
        ax1.set_ylabel('Distance (μm)', fontweight='bold')
        ax1.set_title('Sample-level: KPT vs KPNT Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=sample_data, x='main_group', y='mean_distance',
                   ax=ax2, palette='Set2')
        ax2.set_ylabel('Distance (μm)', fontweight='bold')
        ax2.set_xlabel('Group', fontweight='bold')
        ax2.set_title('Sample-level: KPT vs KPNT', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            ax3.scatter(group_data['timepoint'], group_data['mean_distance'],
                       alpha=0.6, s=60, label=group)
        ax3.set_xlabel('Timepoint', fontweight='bold')
        ax3.set_ylabel('Distance (μm)', fontweight='bold')
        ax3.set_title('Sample-level: Raw Data', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ROW 2: Structure-level analysis (if available)
        if len(structure_data) > 0:
            ax4 = fig.add_subplot(gs[1, 0])
            for group in structure_data['main_group'].unique():
                group_data = structure_data[structure_data['main_group'] == group]
                temporal_mean = group_data.groupby('timepoint')['mean_distance'].mean()
                temporal_sem = group_data.groupby('timepoint')['mean_distance'].sem()
                ax4.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                            marker='s', label=group, linewidth=2, markersize=7, capsize=4)
            ax4.set_xlabel('Timepoint', fontweight='bold')
            ax4.set_ylabel('Distance (μm)', fontweight='bold')
            ax4.set_title('Structure-level: KPT vs KPNT Over Time', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs[1, 1])
            sns.boxplot(data=structure_data, x='main_group', y='mean_distance',
                       ax=ax5, palette='Set2')
            ax5.set_ylabel('Distance (μm)', fontweight='bold')
            ax5.set_xlabel('Group', fontweight='bold')
            ax5.set_title('Structure-level: KPT vs KPNT', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')

            ax6 = fig.add_subplot(gs[1, 2])
            for group in structure_data['main_group'].unique():
                group_data = structure_data[structure_data['main_group'] == group]
                ax6.scatter(group_data['timepoint'], group_data['mean_distance'],
                           alpha=0.4, s=40, label=group)
            ax6.set_xlabel('Timepoint', fontweight='bold')
            ax6.set_ylabel('Distance (μm)', fontweight='bold')
            ax6.set_title('Structure-level: Raw Data', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # ROW 3: 4-way genotype comparisons
        ax7 = fig.add_subplot(gs[2, 0])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            temporal_mean = gf_data.groupby('timepoint')['mean_distance'].mean()
            ax7.plot(temporal_mean.index, temporal_mean.values,
                    marker='o', label=genotype, linewidth=2, markersize=6)
        ax7.set_xlabel('Timepoint', fontweight='bold')
        ax7.set_ylabel('Distance (μm)', fontweight='bold')
        ax7.set_title('4-Way Genotype Comparison', fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        sns.violinplot(data=sample_data, x='genotype_full', y='mean_distance',
                      ax=ax8, palette='tab10')
        ax8.set_ylabel('Distance (μm)', fontweight='bold')
        ax8.set_xlabel('Genotype', fontweight='bold')
        ax8.set_title('4-Way Distribution', fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')

        ax9 = fig.add_subplot(gs[2, 2])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            ax9.scatter(gf_data['timepoint'], gf_data['mean_distance'],
                       alpha=0.5, s=50, label=genotype)
        ax9.set_xlabel('Timepoint', fontweight='bold')
        ax9.set_ylabel('Distance (μm)', fontweight='bold')
        ax9.set_title('4-Way Raw Data', fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)

        plt.suptitle(f'Comprehensive Distance Analysis\n{tumor_type} → {immune_type}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_dir / f'{safe_name}_combo.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_distance_heatmap(self, df: pd.DataFrame, output_dir: Path):
        """Generate summary heatmap of all tumor-immune distances."""
        # Sample-level data only
        sample_data = df[df['level'] == 'sample'].copy()

        if len(sample_data) == 0:
            return

        # Create matrix: rows = tumor types, cols = immune types
        # Values = mean distance (KPT vs KPNT difference)

        tumor_types = sorted(sample_data['tumor_type'].unique())
        immune_types = sorted(sample_data['immune_type'].unique())

        # KPT mean distances
        kpt_matrix = np.zeros((len(tumor_types), len(immune_types)))
        kpnt_matrix = np.zeros((len(tumor_types), len(immune_types)))
        diff_matrix = np.zeros((len(tumor_types), len(immune_types)))

        for i, tumor_type in enumerate(tumor_types):
            for j, immune_type in enumerate(immune_types):
                pair_data = sample_data[
                    (sample_data['tumor_type'] == tumor_type) &
                    (sample_data['immune_type'] == immune_type)
                ]

                kpt_mean = pair_data[pair_data['main_group'] == 'KPT']['mean_distance'].mean()
                kpnt_mean = pair_data[pair_data['main_group'] == 'KPNT']['mean_distance'].mean()

                kpt_matrix[i, j] = kpt_mean
                kpnt_matrix[i, j] = kpnt_mean
                diff_matrix[i, j] = kpt_mean - kpnt_mean

        # Create heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # KPT heatmap
        sns.heatmap(kpt_matrix, annot=True, fmt='.1f', cmap='viridis',
                   xticklabels=immune_types, yticklabels=tumor_types,
                   ax=axes[0], cbar_kws={'label': 'Distance (μm)'})
        axes[0].set_title('KPT: Mean Distances', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Immune Population', fontweight='bold')
        axes[0].set_ylabel('Tumor Population', fontweight='bold')

        # KPNT heatmap
        sns.heatmap(kpnt_matrix, annot=True, fmt='.1f', cmap='viridis',
                   xticklabels=immune_types, yticklabels=tumor_types,
                   ax=axes[1], cbar_kws={'label': 'Distance (μm)'})
        axes[1].set_title('KPNT: Mean Distances', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Immune Population', fontweight='bold')
        axes[1].set_ylabel('Tumor Population', fontweight='bold')

        # Difference heatmap
        sns.heatmap(diff_matrix, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                   xticklabels=immune_types, yticklabels=tumor_types,
                   ax=axes[2], cbar_kws={'label': 'Difference (μm)'})
        axes[2].set_title('KPT - KPNT Difference\n(Positive = KPT farther)',
                         fontweight='bold', fontsize=14)
        axes[2].set_xlabel('Immune Population', fontweight='bold')
        axes[2].set_ylabel('Tumor Population', fontweight='bold')

        plt.suptitle('Comprehensive Distance Summary Heatmaps',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'distance_summary_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Summary heatmap saved")


    def plot_marker_regions(self):
        """
        Generate ALL marker region plots.

        For each marker:
        1. Individual temporal plot (marker+ fraction over time, KPT vs KPNT)
        2. Individual boxplot (KPT vs KPNT comparison)
        3. Individual scatter plot (raw data)
        4. Combo figure (3x3 grid)
        """
        print("\n" + "="*80)
        print("GENERATING MARKER REGION PLOTS")
        print("="*80 + "\n")

        if 'marker_regions' not in self.results:
            print("No marker region results found, skipping...")
            return

        df = self.results['marker_regions']

        # Create output directories
        marker_dir = self.figures_dir / 'marker_regions'
        marker_dir.mkdir(parents=True, exist_ok=True)

        individual_dir = marker_dir / 'individual_plots'
        individual_dir.mkdir(parents=True, exist_ok=True)

        combo_dir = marker_dir / 'combo_plots'
        combo_dir.mkdir(parents=True, exist_ok=True)

        # Get all markers
        markers = df['marker_name'].unique()

        print(f"Generating plots for {len(markers)} markers...")

        for marker_name in markers:
            print(f"  {marker_name}")

            # Filter data for this marker (positive only)
            marker_data = df[
                (df['marker_name'] == marker_name) &
                (df['marker_type'] == 'positive')
            ].copy()

            if len(marker_data) == 0:
                continue

            # Generate individual plots
            self._plot_marker_individual(marker_data, marker_name, individual_dir)

            # Generate combo plot
            self._plot_marker_combo(marker_data, marker_name, combo_dir)

        print("\n✓ Marker region plots complete!")


    def _plot_marker_individual(self, df: pd.DataFrame, marker_name: str, output_dir: Path):
        """Generate individual plots for a marker."""
        safe_name = marker_name.replace(' ', '_').replace('+', 'pos')

        # Sample-level data
        sample_data = df[df['level'] == 'sample'].copy()

        if len(sample_data) == 0:
            return

        # PLOT 1: Temporal plot - KPT vs KPNT over time
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['fraction'].mean() * 100
            temporal_sem = group_data.groupby('timepoint')['fraction'].sem() * 100

            ax.errorbar(temporal_mean.index, temporal_mean.values,
                       yerr=temporal_sem.values,
                       marker='o', label=group, linewidth=2.5,
                       markersize=8, capsize=5)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'% {marker_name}+ Tumor Cells', fontweight='bold', fontsize=12)
        ax.set_title(f'{marker_name}+ Fraction\nKPT vs KPNT Over Time',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 2: Boxplot - KPT vs KPNT comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        sample_data_pct = sample_data.copy()
        sample_data_pct['fraction_pct'] = sample_data_pct['fraction'] * 100

        sns.boxplot(data=sample_data_pct, x='main_group', y='fraction_pct',
                   ax=ax, palette='Set2', linewidth=2)

        # Add p-value
        groups = sample_data_pct['main_group'].unique()
        if len(groups) == 2:
            data1 = sample_data_pct[sample_data_pct['main_group'] == groups[0]]['fraction_pct'].dropna()
            data2 = sample_data_pct[sample_data_pct['main_group'] == groups[1]]['fraction_pct'].dropna()
            if len(data1) > 0 and len(data2) > 0:
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(data1, data2)
                p_text = f'p = {p:.4f}' if p >= 0.001 else 'p < 0.001'
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=11, fontweight='bold')

        ax.set_ylabel(f'% {marker_name}+ Cells', fontweight='bold', fontsize=12)
        ax.set_xlabel('Group', fontweight='bold', fontsize=12)
        ax.set_title(f'{marker_name}+ Fraction\nKPT vs KPNT',
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 3: Scatter plot - raw data
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            ax.scatter(group_data['timepoint'], group_data['fraction'] * 100,
                      alpha=0.6, s=80, label=group)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'% {marker_name}+ Cells', fontweight='bold', fontsize=12)
        ax.set_title(f'{marker_name}+ Fraction\nRaw Data (Sample-level)',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_marker_combo(self, df: pd.DataFrame, marker_name: str, output_dir: Path):
        """Generate combo 3x3 plot for a marker."""
        safe_name = marker_name.replace(' ', '_').replace('+', 'pos')

        # Create 3x3 figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Sample and structure level data
        sample_data = df[df['level'] == 'sample'].copy()
        structure_data = df[df['level'] == 'structure'].copy()

        if len(sample_data) == 0:
            plt.close()
            return

        # Add percentage column
        sample_data['fraction_pct'] = sample_data['fraction'] * 100
        structure_data['fraction_pct'] = structure_data['fraction'] * 100

        # ROW 1: Sample-level analysis
        # Plot 1: Temporal - KPT vs KPNT
        ax1 = fig.add_subplot(gs[0, 0])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['fraction_pct'].mean()
            temporal_sem = group_data.groupby('timepoint')['fraction_pct'].sem()
            ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                        marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
        ax1.set_xlabel('Timepoint', fontweight='bold')
        ax1.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax1.set_title('Sample-level: KPT vs KPNT Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=sample_data, x='main_group', y='fraction_pct',
                   ax=ax2, palette='Set2')
        ax2.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax2.set_xlabel('Group', fontweight='bold')
        ax2.set_title('Sample-level: KPT vs KPNT', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            ax3.scatter(group_data['timepoint'], group_data['fraction_pct'],
                       alpha=0.6, s=60, label=group)
        ax3.set_xlabel('Timepoint', fontweight='bold')
        ax3.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax3.set_title('Sample-level: Raw Data', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ROW 2: Structure-level analysis (if available)
        if len(structure_data) > 0:
            ax4 = fig.add_subplot(gs[1, 0])
            for group in structure_data['main_group'].unique():
                group_data = structure_data[structure_data['main_group'] == group]
                temporal_mean = group_data.groupby('timepoint')['fraction_pct'].mean()
                temporal_sem = group_data.groupby('timepoint')['fraction_pct'].sem()
                ax4.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                            marker='s', label=group, linewidth=2, markersize=7, capsize=4)
            ax4.set_xlabel('Timepoint', fontweight='bold')
            ax4.set_ylabel(f'% {marker_name}+', fontweight='bold')
            ax4.set_title('Structure-level: KPT vs KPNT Over Time', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs[1, 1])
            sns.boxplot(data=structure_data, x='main_group', y='fraction_pct',
                       ax=ax5, palette='Set2')
            ax5.set_ylabel(f'% {marker_name}+', fontweight='bold')
            ax5.set_xlabel('Group', fontweight='bold')
            ax5.set_title('Structure-level: KPT vs KPNT', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')

            ax6 = fig.add_subplot(gs[1, 2])
            for group in structure_data['main_group'].unique():
                group_data = structure_data[structure_data['main_group'] == group]
                ax6.scatter(group_data['timepoint'], group_data['fraction_pct'],
                           alpha=0.4, s=40, label=group)
            ax6.set_xlabel('Timepoint', fontweight='bold')
            ax6.set_ylabel(f'% {marker_name}+', fontweight='bold')
            ax6.set_title('Structure-level: Raw Data', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # ROW 3: 4-way genotype comparisons
        ax7 = fig.add_subplot(gs[2, 0])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            temporal_mean = gf_data.groupby('timepoint')['fraction_pct'].mean()
            ax7.plot(temporal_mean.index, temporal_mean.values,
                    marker='o', label=genotype, linewidth=2, markersize=6)
        ax7.set_xlabel('Timepoint', fontweight='bold')
        ax7.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax7.set_title('4-Way Genotype Comparison', fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        sns.violinplot(data=sample_data, x='genotype_full', y='fraction_pct',
                      ax=ax8, palette='tab10')
        ax8.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax8.set_xlabel('Genotype', fontweight='bold')
        ax8.set_title('4-Way Distribution', fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')

        ax9 = fig.add_subplot(gs[2, 2])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            ax9.scatter(gf_data['timepoint'], gf_data['fraction_pct'],
                       alpha=0.5, s=50, label=genotype)
        ax9.set_xlabel('Timepoint', fontweight='bold')
        ax9.set_ylabel(f'% {marker_name}+', fontweight='bold')
        ax9.set_title('4-Way Raw Data', fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)

        plt.suptitle(f'Comprehensive Marker Analysis: {marker_name}+',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_dir / f'{safe_name}_combo.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_infiltration_analysis(self):
        """
        Generate ALL infiltration analysis plots.

        For each immune population and boundary width:
        1. Individual temporal plot (KPT vs KPNT over time)
        2. Individual boxplot (KPT vs KPNT comparison)
        3. Individual scatter plot (raw data)
        4. Combo figure (3x3 grid for each population)
        """
        print("\n" + "="*80)
        print("GENERATING INFILTRATION ANALYSIS PLOTS")
        print("="*80 + "\n")

        if 'infiltration' not in self.results:
            print("No infiltration results found, skipping...")
            return

        df = self.results['infiltration']

        # Create output directories
        infilt_dir = self.figures_dir / 'infiltration_analysis'
        infilt_dir.mkdir(parents=True, exist_ok=True)

        individual_dir = infilt_dir / 'individual_plots'
        individual_dir.mkdir(parents=True, exist_ok=True)

        combo_dir = infilt_dir / 'combo_plots'
        combo_dir.mkdir(parents=True, exist_ok=True)

        # Get all immune populations
        immune_pops = df['immune_type'].unique()

        print(f"Generating plots for {len(immune_pops)} immune populations...")

        for immune_pop in immune_pops:
            print(f"  {immune_pop}")

            # Filter data for this immune population
            pop_data = df[df['immune_type'] == immune_pop].copy()

            if len(pop_data) == 0:
                continue

            # For simplicity, use the middle boundary width for main plots
            boundary_widths = sorted(pop_data['boundary_width'].unique())
            if len(boundary_widths) > 0:
                main_width = boundary_widths[len(boundary_widths)//2]  # Middle width
                main_data = pop_data[pop_data['boundary_width'] == main_width].copy()

                # Generate individual plots
                self._plot_infiltration_individual(main_data, immune_pop, main_width, individual_dir)

                # Generate combo plot (with all widths)
                self._plot_infiltration_combo(pop_data, immune_pop, boundary_widths, combo_dir)

        print("\n✓ Infiltration analysis plots complete!")


    def _plot_infiltration_individual(self, df: pd.DataFrame, immune_pop: str,
                                     boundary_width: int, output_dir: Path):
        """Generate individual plots for an immune population."""
        safe_name = immune_pop.replace(' ', '_').replace('+', 'pos')

        # Sample-level data
        sample_data = df[df['level'] == 'sample'].copy()

        if len(sample_data) == 0:
            return

        # Convert fraction to percentage
        sample_data['infiltration_pct'] = sample_data['infiltration_fraction'] * 100

        # PLOT 1: Temporal plot - KPT vs KPNT over time
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['infiltration_pct'].mean()
            temporal_sem = group_data.groupby('timepoint')['infiltration_pct'].sem()

            ax.errorbar(temporal_mean.index, temporal_mean.values,
                       yerr=temporal_sem.values,
                       marker='o', label=group, linewidth=2.5,
                       markersize=8, capsize=5)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold', fontsize=12)
        ax.set_title(f'{immune_pop} Infiltration (Width={boundary_width}μm)\nKPT vs KPNT Over Time',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_w{boundary_width}_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 2: Boxplot - KPT vs KPNT comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(data=sample_data, x='main_group', y='infiltration_pct',
                   ax=ax, palette='Set2', linewidth=2)

        # Add p-value
        groups = sample_data['main_group'].unique()
        if len(groups) == 2:
            data1 = sample_data[sample_data['main_group'] == groups[0]]['infiltration_pct'].dropna()
            data2 = sample_data[sample_data['main_group'] == groups[1]]['infiltration_pct'].dropna()
            if len(data1) > 0 and len(data2) > 0:
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(data1, data2)
                p_text = f'p = {p:.4f}' if p >= 0.001 else 'p < 0.001'
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=11, fontweight='bold')

        ax.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold', fontsize=12)
        ax.set_xlabel('Group', fontweight='bold', fontsize=12)
        ax.set_title(f'{immune_pop} Infiltration (Width={boundary_width}μm)\nKPT vs KPNT',
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_w{boundary_width}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PLOT 3: Scatter plot - raw data
        fig, ax = plt.subplots(figsize=(10, 6))

        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            ax.scatter(group_data['timepoint'], group_data['infiltration_pct'],
                      alpha=0.6, s=80, label=group)

        ax.set_xlabel('Timepoint (weeks)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold', fontsize=12)
        ax.set_title(f'{immune_pop} Infiltration (Width={boundary_width}μm)\nRaw Data',
                    fontweight='bold', fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_w{boundary_width}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_infiltration_combo(self, df: pd.DataFrame, immune_pop: str,
                                boundary_widths: List[int], output_dir: Path):
        """Generate combo 3x3 plot for an immune population."""
        safe_name = immune_pop.replace(' ', '_').replace('+', 'pos')

        # Create 3x3 figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Use middle width for main plots
        main_width = boundary_widths[len(boundary_widths)//2]

        # Sample and structure level data
        sample_data = df[(df['level'] == 'sample') & (df['boundary_width'] == main_width)].copy()
        structure_data = df[(df['level'] == 'structure') & (df['boundary_width'] == main_width)].copy()

        if len(sample_data) == 0:
            plt.close()
            return

        # Add percentage column
        sample_data['infiltration_pct'] = sample_data['infiltration_fraction'] * 100
        structure_data['infiltration_pct'] = structure_data['infiltration_fraction'] * 100

        # ROW 1: Sample-level analysis
        # Plot 1: Temporal - KPT vs KPNT
        ax1 = fig.add_subplot(gs[0, 0])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            temporal_mean = group_data.groupby('timepoint')['infiltration_pct'].mean()
            temporal_sem = group_data.groupby('timepoint')['infiltration_pct'].sem()
            ax1.errorbar(temporal_mean.index, temporal_mean.values, yerr=temporal_sem.values,
                        marker='o', label=group, linewidth=2.5, markersize=8, capsize=5)
        ax1.set_xlabel('Timepoint', fontweight='bold')
        ax1.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax1.set_title(f'Sample-level: KPT vs KPNT Over Time\n(Width={main_width}μm)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=sample_data, x='main_group', y='infiltration_pct',
                   ax=ax2, palette='Set2')
        ax2.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax2.set_xlabel('Group', fontweight='bold')
        ax2.set_title('Sample-level: KPT vs KPNT', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        for group in sample_data['main_group'].unique():
            group_data = sample_data[sample_data['main_group'] == group]
            ax3.scatter(group_data['timepoint'], group_data['infiltration_pct'],
                       alpha=0.6, s=60, label=group)
        ax3.set_xlabel('Timepoint', fontweight='bold')
        ax3.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax3.set_title('Sample-level: Raw Data', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ROW 2: Multi-width comparison (sample-level)
        ax4 = fig.add_subplot(gs[1, 0])
        all_sample_data = df[df['level'] == 'sample'].copy()
        all_sample_data['infiltration_pct'] = all_sample_data['infiltration_fraction'] * 100
        for width in boundary_widths:
            width_data = all_sample_data[all_sample_data['boundary_width'] == width]
            if len(width_data) == 0:
                continue
            # KPT only for clarity
            kpt_data = width_data[width_data['main_group'] == 'KPT']
            temporal_mean = kpt_data.groupby('timepoint')['infiltration_pct'].mean()
            ax4.plot(temporal_mean.index, temporal_mean.values,
                    marker='o', label=f'{width}μm', linewidth=2, markersize=6)
        ax4.set_xlabel('Timepoint', fontweight='bold')
        ax4.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax4.set_title('KPT: Multiple Boundary Widths', fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        for width in boundary_widths:
            width_data = all_sample_data[all_sample_data['boundary_width'] == width]
            if len(width_data) == 0:
                continue
            # KPNT only
            kpnt_data = width_data[width_data['main_group'] == 'KPNT']
            temporal_mean = kpnt_data.groupby('timepoint')['infiltration_pct'].mean()
            ax5.plot(temporal_mean.index, temporal_mean.values,
                    marker='s', label=f'{width}μm', linewidth=2, markersize=6)
        ax5.set_xlabel('Timepoint', fontweight='bold')
        ax5.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax5.set_title('KPNT: Multiple Boundary Widths', fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Heatmap by width
        ax6 = fig.add_subplot(gs[1, 2])
        pivot_data = all_sample_data.groupby(['main_group', 'boundary_width'])['infiltration_pct'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax6, cbar_kws={'label': '% Infiltration'})
        ax6.set_xlabel('Boundary Width (μm)', fontweight='bold')
        ax6.set_ylabel('Group', fontweight='bold')
        ax6.set_title('Infiltration by Width', fontweight='bold')

        # ROW 3: 4-way genotype comparisons
        ax7 = fig.add_subplot(gs[2, 0])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            temporal_mean = gf_data.groupby('timepoint')['infiltration_pct'].mean()
            ax7.plot(temporal_mean.index, temporal_mean.values,
                    marker='o', label=genotype, linewidth=2, markersize=6)
        ax7.set_xlabel('Timepoint', fontweight='bold')
        ax7.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax7.set_title('4-Way Genotype Comparison', fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        sns.violinplot(data=sample_data, x='genotype_full', y='infiltration_pct',
                      ax=ax8, palette='tab10')
        ax8.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax8.set_xlabel('Genotype', fontweight='bold')
        ax8.set_title('4-Way Distribution', fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')

        ax9 = fig.add_subplot(gs[2, 2])
        for genotype in sample_data['genotype_full'].unique():
            gf_data = sample_data[sample_data['genotype_full'] == genotype]
            ax9.scatter(gf_data['timepoint'], gf_data['infiltration_pct'],
                       alpha=0.5, s=50, label=genotype)
        ax9.set_xlabel('Timepoint', fontweight='bold')
        ax9.set_ylabel(f'% {immune_pop} Infiltration', fontweight='bold')
        ax9.set_title('4-Way Raw Data', fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)

        plt.suptitle(f'Comprehensive Infiltration Analysis: {immune_pop}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_dir / f'{safe_name}_combo.png', dpi=300, bbox_inches='tight')
        plt.close()


    def generate_all_plots(self):
        """Generate ALL plots."""
        print("\n" + "="*80)
        print("GENERATING ALL COMPREHENSIVE PLOTS")
        print("="*80 + "\n")

        self.plot_distance_analysis()
        self.plot_marker_regions()
        self.plot_infiltration_analysis()

        print("\n" + "="*80)
        print("ALL PLOTS COMPLETE")
        print("="*80 + "\n")


if __name__ == '__main__':
    print("This is a module to be imported, not run directly.")
    print("See run_comprehensive_rewrite.py for usage example.")
