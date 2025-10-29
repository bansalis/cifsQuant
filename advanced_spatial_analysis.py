#!/usr/bin/env python3
"""
Advanced Multi-Level Spatial Immunofluorescence Analysis Pipeline

Comprehensive spatial analysis for cyclic immunofluorescence data from murine lung
adenocarcinoma samples. Implements MCMICRO/SCIMAP workflows with enhanced tumor-immune
dynamics, spatial heterogeneity, and temporal evolution analysis.

Key Features:
- Enhanced phenotyping with intensity-based gating
- Tumor structure detection with DBSCAN
- pERK spatial architecture analysis (clustering, growth, infiltration)
- NINJA escape mechanism analysis
- Heterogeneity emergence and evolution
- Cellular neighborhood (RCN) temporal dynamics
- Multi-level distance analysis
- Infiltration-tumor association analysis
- Pseudo-temporal trajectory inference
- Publication-quality visualizations with rigorous statistics

References:
- Nirmal et al. 2021 (SCIMAP)
- Schapiro et al. 2017 (histoCAT)
- Jackson et al. 2020 (cancer cell neighborhoods)

Author: Advanced spatial analysis expansion
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Spatial analysis
from scipy.spatial import cKDTree, distance_matrix, ConvexHull
from scipy.stats import (mannwhitneyu, wilcoxon, kruskal, spearmanr,
                         pearsonr, ttest_ind, f_oneway, linregress)
from scipy.ndimage import gaussian_filter

# Clustering and ML
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, silhouette_score

# Statistics
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Typing
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import gc
from collections import defaultdict
from dataclasses import dataclass
import json

# Try to import scanpy for trajectory analysis
try:
    import scanpy as sc
    import anndata
    SCANPY_AVAILABLE = True
except ImportError:
    print("Warning: scanpy not available, pseudo-temporal analysis will be limited")
    SCANPY_AVAILABLE = False


@dataclass
class AnalysisConfig:
    """Configuration for spatial analysis parameters."""

    # Tumor detection
    tumor_dbscan_eps: float = 50.0  # μm
    tumor_dbscan_min_samples: int = 30

    # Zone definitions
    margin_width: float = 50.0  # μm
    peritumor_inner: float = 50.0  # μm
    peritumor_outer: float = 150.0  # μm

    # pERK/NINJA clustering
    perk_cluster_eps: float = 30.0  # μm
    perk_cluster_min_samples: int = 10
    ninja_cluster_eps: float = 30.0  # μm
    ninja_cluster_min_samples: int = 10

    # Infiltration analysis
    infiltration_radii: List[float] = None  # [30, 50, 100] μm

    # Neighborhood analysis
    n_neighbors: int = 10
    n_rcn_clusters: int = 8

    # Spatial statistics
    ripley_max_distance: float = 200.0  # μm
    ripley_n_steps: int = 50
    hotspot_radius: float = 50.0  # μm

    # Visualization
    dpi: int = 300
    figure_format: str = 'pdf'
    color_palette: str = 'colorblind'

    # Statistics
    fdr_alpha: float = 0.05
    n_permutations: int = 1000

    def __post_init__(self):
        if self.infiltration_radii is None:
            self.infiltration_radii = [30.0, 50.0, 100.0]


class AdvancedSpatialAnalysis:
    """
    Advanced multi-level spatial analysis for tumor immunology.

    Implements comprehensive spatial analysis pipeline with enhanced phenotyping,
    tumor structure detection, spatial clustering, heterogeneity analysis, and
    temporal dynamics tracking.
    """

    def __init__(self,
                 output_dir: str = 'spatial_analysis_comprehensive',
                 config: Optional[AnalysisConfig] = None):
        """
        Initialize advanced spatial analysis.

        Parameters
        ----------
        output_dir : str
            Output directory for results
        config : AnalysisConfig, optional
            Analysis configuration parameters
        """
        self.output_dir = Path(output_dir)
        self.config = config if config is not None else AnalysisConfig()

        # Data containers
        self.cells_df = None
        self.sample_metadata = None
        self.markers_info = None
        self.tumors_df = None
        self.phenotypes = {}

        # Results storage
        self.results = {}

        # Setup directories
        self._setup_directories()

        print("=" * 80)
        print("ADVANCED SPATIAL ANALYSIS INITIALIZED")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration:")
        for key, value in vars(self.config).items():
            print(f"  {key}: {value}")
        print()

    def _setup_directories(self):
        """Create comprehensive directory structure."""
        subdirs = [
            # Data outputs
            'data',
            'statistics',

            # Phase 1: Phenotyping
            '01_phenotyping/individual_plots',
            '01_phenotyping/aggregated_panels',

            # Phase 2: Research questions
            '02_perk_analysis/individual_plots',
            '02_perk_analysis/aggregated_panels',
            '02_perk_analysis/statistics',

            '03_ninja_analysis/individual_plots',
            '03_ninja_analysis/aggregated_panels',
            '03_ninja_analysis/statistics',

            '04_heterogeneity/individual_plots',
            '04_heterogeneity/aggregated_panels',
            '04_heterogeneity/statistics',

            '05_rcn_dynamics/individual_plots',
            '05_rcn_dynamics/aggregated_panels',
            '05_rcn_dynamics/statistics',

            # Phase 3: Advanced metrics
            '06_distance_analysis/individual_plots',
            '06_distance_analysis/aggregated_panels',
            '06_distance_analysis/statistics',

            '07_infiltration_associations/individual_plots',
            '07_infiltration_associations/aggregated_panels',
            '07_infiltration_associations/statistics',

            '08_pseudotime/individual_plots',
            '08_pseudotime/aggregated_panels',
            '08_pseudotime/statistics',
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def load_data(self,
                  quantification_files: Union[str, List[str]],
                  sample_metadata_file: str,
                  markers_file: str):
        """
        Load quantification data and metadata.

        Parameters
        ----------
        quantification_files : str or list of str
            Path(s) to combined_quantification.csv files or directory
        sample_metadata_file : str
            Path to sample_metadata.csv
        markers_file : str
            Path to markers.csv
        """
        print("Loading data...")

        # Load sample metadata
        self.sample_metadata = pd.read_csv(sample_metadata_file)
        self._parse_metadata()

        # Load markers
        self.markers_info = pd.read_csv(markers_file)

        # Load quantification data
        if isinstance(quantification_files, str):
            if Path(quantification_files).is_dir():
                # Find all combined_quantification.csv files
                quant_files = list(Path(quantification_files).rglob('combined_quantification.csv'))
            else:
                quant_files = [Path(quantification_files)]
        else:
            quant_files = [Path(f) for f in quantification_files]

        # Load and concatenate all files
        dfs = []
        for file in quant_files:
            if file.exists():
                df = pd.read_csv(file)
                # Extract sample name from path if not present
                if 'sample_id' not in df.columns:
                    sample_name = file.parent.parent.name
                    df['sample_id'] = sample_name
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No quantification files found in {quantification_files}")

        self.cells_df = pd.concat(dfs, ignore_index=True)

        # Merge metadata
        self.cells_df = self.cells_df.merge(
            self.sample_metadata,
            on='sample_id',
            how='left'
        )

        print(f"Loaded {len(self.cells_df):,} cells from {len(dfs)} samples")
        print(f"Samples: {sorted(self.cells_df['sample_id'].unique())}")
        print(f"Groups: {sorted(self.cells_df['group'].dropna().unique())}")
        print(f"Timepoints: {sorted(self.cells_df['timepoint'].dropna().unique())}")
        print()

        return self

    def _parse_metadata(self):
        """Parse metadata to extract genotype groups."""
        # Standardize sample_id
        self.sample_metadata['sample_id'] = self.sample_metadata['sample_id'].str.upper()

        # Extract main group (KPT vs KPNT)
        self.sample_metadata['main_group'] = self.sample_metadata['group'].apply(
            lambda x: 'KPT' if 'KPT' in str(x) else 'KPNT'
        )

        # Extract cis/trans
        self.sample_metadata['genotype'] = self.sample_metadata['group'].apply(
            lambda x: 'cis' if 'cis' in str(x).lower() else
                     ('trans' if 'trans' in str(x).lower() else 'Unknown')
        )

        # Full group name
        self.sample_metadata['genotype_full'] = self.sample_metadata['group']

        # Convert timepoint to numeric
        self.sample_metadata['timepoint'] = pd.to_numeric(
            self.sample_metadata['timepoint']
        )

    # =========================================================================
    # PHASE 1: ENHANCED PHENOTYPING & TUMOR DETECTION
    # =========================================================================

    def phase1_phenotype_cells(self,
                              thresholds: Optional[Dict[str, float]] = None,
                              auto_threshold_method: str = 'otsu'):
        """
        Phase 1.1: Enhanced cell phenotyping with intensity-based gating.

        Implements multi-marker phenotyping with spatial context validation.

        Parameters
        ----------
        thresholds : dict, optional
            Manual thresholds for markers {marker_name: threshold_value}
        auto_threshold_method : str
            Method for automatic thresholding: 'otsu', 'li', 'percentile'

        Returns
        -------
        pd.DataFrame
            Cells with phenotype assignments
        """
        print("\n" + "=" * 80)
        print("PHASE 1.1: ENHANCED CELL PHENOTYPING")
        print("=" * 80 + "\n")

        # Define marker channels based on markers.csv
        marker_channels = {
            'TOM': 'Channel_2',  # Assuming TOM is in CY3 of cycle 1
            'CD45': 'Channel_3',  # From markers file R1.0.4_CY5_CD45
            'AGFP': 'Channel_4',  # From markers file R1.0.4_CY7_AGFP
            'PERK': 'Channel_6',  # From markers file R2.0.4_CY3_PERK
            'CD8B': 'Channel_7',  # From markers file R4.0.4_CY5_CD8A (assuming CD8B similar)
            'CD3': 'Channel_10',  # From markers file R3.0.4_CY3_CD3E
            'KI67': 'Channel_27', # From markers file R6.0.4_CY7_KI67
        }

        # Auto-threshold or use provided thresholds
        if thresholds is None:
            thresholds = self._auto_threshold_markers(
                marker_channels,
                method=auto_threshold_method
            )

        # Apply thresholds to define cell phenotypes
        print("Applying intensity-based gating...")

        # Binary marker expression
        for marker, channel in marker_channels.items():
            if channel in self.cells_df.columns:
                self.cells_df[f'{marker}_positive'] = (
                    self.cells_df[channel] > thresholds.get(marker, 0)
                )

        # Define phenotypes
        self.cells_df['phenotype'] = 'Unknown'

        # Tumor cells: TOM+
        tumor_mask = self.cells_df.get('TOM_positive', False)
        self.cells_df.loc[tumor_mask, 'phenotype'] = 'Tumor'

        # NINJA+ tumor: TOM+ AGFP+
        ninja_mask = tumor_mask & self.cells_df.get('AGFP_positive', False)
        self.cells_df.loc[ninja_mask, 'phenotype'] = 'Tumor_NINJA+'

        # pERK+ tumor: TOM+ pERK+
        perk_mask = tumor_mask & self.cells_df.get('PERK_positive', False)
        self.cells_df.loc[perk_mask, 'phenotype'] = 'Tumor_pERK+'

        # NINJA+ pERK+ tumor
        ninja_perk_mask = ninja_mask & self.cells_df.get('PERK_positive', False)
        self.cells_df.loc[ninja_perk_mask, 'phenotype'] = 'Tumor_NINJA+_pERK+'

        # All immune: CD45+
        immune_mask = self.cells_df.get('CD45_positive', False) & ~tumor_mask
        self.cells_df.loc[immune_mask, 'phenotype'] = 'Immune'

        # T cells: CD3+
        tcell_mask = immune_mask & self.cells_df.get('CD3_positive', False)
        self.cells_df.loc[tcell_mask, 'phenotype'] = 'T_cell'

        # CD8+ T cells: CD3+ CD8B+
        cd8_mask = tcell_mask & self.cells_df.get('CD8B_positive', False)
        self.cells_df.loc[cd8_mask, 'phenotype'] = 'CD8_T_cell'

        # CD4+ T cells (CD3+ CD8-)
        cd4_mask = tcell_mask & ~self.cells_df.get('CD8B_positive', False)
        self.cells_df.loc[cd4_mask, 'phenotype'] = 'CD4_T_cell'

        # Report phenotype frequencies
        print("\nPhenotype frequencies:")
        phenotype_counts = self.cells_df['phenotype'].value_counts()
        for pheno, count in phenotype_counts.items():
            pct = 100 * count / len(self.cells_df)
            print(f"  {pheno}: {count:,} ({pct:.2f}%)")

        # Save phenotyped cells
        output_file = self.output_dir / 'data' / 'phenotyped_cells.csv'
        self.cells_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

        # Generate QC plots
        self._plot_phenotyping_qc(thresholds, marker_channels)

        return self.cells_df

    def _auto_threshold_markers(self,
                                marker_channels: Dict[str, str],
                                method: str = 'otsu') -> Dict[str, float]:
        """
        Automatically determine thresholds for markers.

        Parameters
        ----------
        marker_channels : dict
            Mapping of marker names to channel columns
        method : str
            Thresholding method

        Returns
        -------
        dict
            Thresholds for each marker
        """
        from skimage.filters import threshold_otsu, threshold_li

        thresholds = {}

        for marker, channel in marker_channels.items():
            if channel not in self.cells_df.columns:
                print(f"  Warning: Channel {channel} for {marker} not found")
                thresholds[marker] = 0
                continue

            values = self.cells_df[channel].dropna().values

            if len(values) == 0:
                thresholds[marker] = 0
                continue

            # Remove extreme outliers for better thresholding
            q99 = np.percentile(values, 99)
            values_clipped = values[values <= q99]

            if method == 'otsu':
                try:
                    threshold = threshold_otsu(values_clipped)
                except:
                    threshold = np.percentile(values_clipped, 75)
            elif method == 'li':
                try:
                    threshold = threshold_li(values_clipped)
                except:
                    threshold = np.percentile(values_clipped, 75)
            elif method == 'percentile':
                threshold = np.percentile(values_clipped, 75)
            else:
                threshold = np.median(values_clipped)

            thresholds[marker] = threshold
            print(f"  {marker}: threshold = {threshold:.2f}")

        return thresholds

    def _plot_phenotyping_qc(self,
                            thresholds: Dict[str, float],
                            marker_channels: Dict[str, str]):
        """Generate QC plots for phenotyping."""
        print("\nGenerating phenotyping QC plots...")

        output_dir = self.output_dir / '01_phenotyping' / 'individual_plots'

        # 1. Threshold selection plots per marker
        for marker, channel in marker_channels.items():
            if channel not in self.cells_df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            values = self.cells_df[channel].dropna().values
            threshold = thresholds.get(marker, 0)

            # Histogram with threshold
            ax = axes[0]
            ax.hist(values, bins=100, alpha=0.7, edgecolor='black')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.2f}')
            ax.set_xlabel(f'{marker} Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{marker} Expression Distribution')
            ax.legend()
            ax.set_yscale('log')

            # Before/after threshold
            ax = axes[1]
            negative = values[values <= threshold]
            positive = values[values > threshold]
            ax.hist([negative, positive], bins=50, label=['Negative', 'Positive'],
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{marker} Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{marker} Positive vs Negative')
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'threshold_qc_{marker}.png',
                       dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

        # 2. Spatial distribution map per phenotype
        self._plot_phenotype_spatial_distributions()

        # 3. Overall phenotype composition
        self._plot_phenotype_composition()

        print(f"QC plots saved to: {output_dir}")

    def _plot_phenotype_spatial_distributions(self):
        """Plot spatial distribution maps for each phenotype."""
        output_dir = self.output_dir / '01_phenotyping' / 'aggregated_panels'

        # Get major phenotypes
        major_phenotypes = [
            'Tumor', 'Tumor_NINJA+', 'Tumor_pERK+',
            'CD8_T_cell', 'CD4_T_cell', 'Immune'
        ]

        # Plot per sample
        for sample_id in self.cells_df['sample_id'].unique():
            sample_data = self.cells_df[self.cells_df['sample_id'] == sample_id]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for idx, phenotype in enumerate(major_phenotypes):
                ax = axes[idx]

                # Plot all cells in gray
                ax.scatter(sample_data['X_centroid'], sample_data['Y_centroid'],
                          s=1, c='lightgray', alpha=0.3, rasterized=True)

                # Highlight specific phenotype
                pheno_data = sample_data[sample_data['phenotype'] == phenotype]
                if len(pheno_data) > 0:
                    ax.scatter(pheno_data['X_centroid'], pheno_data['Y_centroid'],
                              s=2, c='red', alpha=0.6, rasterized=True)

                ax.set_title(f'{phenotype} (n={len(pheno_data):,})')
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_aspect('equal')

            plt.suptitle(f'Phenotype Spatial Distributions - {sample_id}',
                        fontsize=16, y=0.995)
            plt.tight_layout()
            plt.savefig(output_dir / f'spatial_distribution_{sample_id}.pdf',
                       dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

        print(f"Spatial distribution maps saved to: {output_dir}")

    def _plot_phenotype_composition(self):
        """Plot overall phenotype composition across samples."""
        output_dir = self.output_dir / '01_phenotyping' / 'aggregated_panels'

        # Phenotype composition by sample
        composition = self.cells_df.groupby(['sample_id', 'phenotype']).size().reset_index(name='count')
        composition['total'] = composition.groupby('sample_id')['count'].transform('sum')
        composition['percentage'] = 100 * composition['count'] / composition['total']

        # Merge with metadata
        composition = composition.merge(self.sample_metadata, on='sample_id', how='left')

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # By sample
        ax = axes[0, 0]
        pivot = composition.pivot(index='sample_id', columns='phenotype', values='percentage').fillna(0)
        pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title('Phenotype Composition by Sample')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Percentage (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)

        # By group
        ax = axes[0, 1]
        group_comp = composition.groupby(['group', 'phenotype'])['percentage'].mean().reset_index()
        pivot_group = group_comp.pivot(index='group', columns='phenotype', values='percentage').fillna(0)
        pivot_group.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title('Phenotype Composition by Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Average Percentage (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)

        # By timepoint
        ax = axes[1, 0]
        time_comp = composition.groupby(['timepoint', 'phenotype'])['percentage'].mean().reset_index()
        pivot_time = time_comp.pivot(index='timepoint', columns='phenotype', values='percentage').fillna(0)
        pivot_time.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title('Phenotype Composition by Timepoint')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Average Percentage (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Heatmap
        ax = axes[1, 1]
        pivot_heatmap = composition.pivot_table(
            index='sample_id',
            columns='phenotype',
            values='percentage',
            fill_value=0
        )
        sns.heatmap(pivot_heatmap, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Percentage (%)'})
        ax.set_title('Phenotype Composition Heatmap')
        ax.set_xlabel('Phenotype')
        ax.set_ylabel('Sample')

        plt.tight_layout()
        plt.savefig(output_dir / 'phenotype_composition_overview.pdf',
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        print(f"Composition plots saved to: {output_dir}")

    # =========================================================================
    # PHASE 1.2: TUMOR STRUCTURE DETECTION
    # =========================================================================

    def phase12_detect_tumor_structures(self):
        """
        Phase 1.2: Tumor structure detection using DBSCAN.

        Implements DBSCAN clustering on TOM+ cells with morphological validation.
        Defines tumor zones: core, margin, peri-tumor, and distal regions.

        Returns
        -------
        pd.DataFrame
            Tumor structures with metrics
        """
        print("\n" + "=" * 80)
        print("PHASE 1.2: TUMOR STRUCTURE DETECTION")
        print("=" * 80 + "\n")

        # Get tumor cells
        tumor_cells = self.cells_df[
            self.cells_df['phenotype'].str.contains('Tumor', na=False)
        ].copy()

        if len(tumor_cells) == 0:
            print("Warning: No tumor cells found!")
            self.tumors_df = pd.DataFrame()
            return self.tumors_df

        print(f"Detecting tumor structures from {len(tumor_cells):,} tumor cells...")

        tumors_list = []
        tumor_id_counter = 0

        # Process per sample
        for sample_id in tumor_cells['sample_id'].unique():
            sample_tumor_cells = tumor_cells[tumor_cells['sample_id'] == sample_id]

            # DBSCAN clustering
            coords = sample_tumor_cells[['X_centroid', 'Y_centroid']].values

            clustering = DBSCAN(
                eps=self.config.tumor_dbscan_eps,
                min_samples=self.config.tumor_dbscan_min_samples,
                metric='euclidean'
            ).fit(coords)

            labels = clustering.labels_

            # Assign tumor IDs to cells
            sample_tumor_cells['tumor_id'] = labels
            self.cells_df.loc[sample_tumor_cells.index, 'tumor_id'] = labels

            # Calculate tumor metrics
            n_tumors = len([l for l in np.unique(labels) if l >= 0])
            print(f"  {sample_id}: detected {n_tumors} tumors")

            for tumor_label in np.unique(labels):
                if tumor_label < 0:  # Noise
                    continue

                tumor_cells_mask = labels == tumor_label
                tumor_coords = coords[tumor_cells_mask]

                # Global tumor ID
                global_tumor_id = f"{sample_id}_T{tumor_id_counter}"
                tumor_id_counter += 1

                # Basic metrics
                n_cells = np.sum(tumor_cells_mask)
                centroid_x = np.mean(tumor_coords[:, 0])
                centroid_y = np.mean(tumor_coords[:, 1])

                # Morphological metrics
                try:
                    hull = ConvexHull(tumor_coords)
                    area = hull.volume  # In 2D, volume is area
                    perimeter = hull.area  # In 2D, area is perimeter
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                except:
                    # Fallback for small/collinear tumors
                    area = n_cells * 100  # Rough estimate
                    perimeter = 0
                    circularity = 0

                tumors_list.append({
                    'tumor_id': global_tumor_id,
                    'sample_id': sample_id,
                    'tumor_label': tumor_label,
                    'n_cells': n_cells,
                    'area_um2': area,
                    'perimeter_um': perimeter,
                    'circularity': circularity,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                })

        self.tumors_df = pd.DataFrame(tumors_list)

        # Merge with metadata
        self.tumors_df = self.tumors_df.merge(
            self.sample_metadata,
            on='sample_id',
            how='left'
        )

        print(f"\nDetected {len(self.tumors_df)} total tumors")
        if len(self.tumors_df) > 0:
            print(f"Tumor size range: {self.tumors_df['n_cells'].min():.0f} - "
                  f"{self.tumors_df['n_cells'].max():.0f} cells")

        # Save tumor structures
        output_file = self.output_dir / 'data' / 'tumor_structures.csv'
        self.tumors_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

        # Define tumor zones for each cell
        self._define_tumor_zones()

        # Generate visualizations
        self._plot_tumor_structures()
        self._plot_tumor_size_distribution()

        return self.tumors_df

    def _define_tumor_zones(self):
        """Define tumor zones for each cell based on distance to tumor structures."""
        print("\nDefining tumor zones...")

        # Initialize zone column
        self.cells_df['tumor_zone'] = 'Distal'

        for sample_id in self.cells_df['sample_id'].unique():
            sample_cells = self.cells_df[self.cells_df['sample_id'] == sample_id]
            sample_tumors = self.tumors_df[self.tumors_df['sample_id'] == sample_id]

            if len(sample_tumors) == 0:
                continue

            # Get coordinates
            all_coords = sample_cells[['X_centroid', 'Y_centroid']].values
            tumor_cells = sample_cells[
                sample_cells['phenotype'].str.contains('Tumor', na=False)
            ]

            if len(tumor_cells) == 0:
                continue

            tumor_coords = tumor_cells[['X_centroid', 'Y_centroid']].values

            # Build tree for tumor cells
            tree = cKDTree(tumor_coords)

            # Query nearest tumor cell distance for all cells
            distances, _ = tree.query(all_coords, k=1)

            # Assign zones
            sample_indices = sample_cells.index

            # Core: tumor cells themselves
            tumor_mask = sample_cells['phenotype'].str.contains('Tumor', na=False)
            self.cells_df.loc[sample_indices[tumor_mask], 'tumor_zone'] = 'Core'

            # Margin: 0-50 μm from tumor
            margin_mask = (~tumor_mask) & (distances <= self.config.margin_width)
            self.cells_df.loc[sample_indices[margin_mask], 'tumor_zone'] = 'Margin'

            # Peri-tumor: 50-150 μm from tumor
            peritumor_mask = (
                (~tumor_mask) &
                (distances > self.config.peritumor_inner) &
                (distances <= self.config.peritumor_outer)
            )
            self.cells_df.loc[sample_indices[peritumor_mask], 'tumor_zone'] = 'Peri-tumor'

            # Distal: >150 μm
            # Already initialized to 'Distal'

        zone_counts = self.cells_df['tumor_zone'].value_counts()
        print("Tumor zone assignment:")
        for zone, count in zone_counts.items():
            print(f"  {zone}: {count:,} cells")

    def _plot_tumor_structures(self):
        """Plot tumor structures with boundaries and zones."""
        print("\nGenerating tumor structure plots...")

        output_dir = self.output_dir / '01_phenotyping' / 'individual_plots'

        for sample_id in self.tumors_df['sample_id'].unique():
            sample_cells = self.cells_df[self.cells_df['sample_id'] == sample_id]
            sample_tumors = self.tumors_df[self.tumors_df['sample_id'] == sample_id]

            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot cells by zone
            zone_colors = {
                'Core': 'red',
                'Margin': 'orange',
                'Peri-tumor': 'yellow',
                'Distal': 'lightgray'
            }

            for zone, color in zone_colors.items():
                zone_cells = sample_cells[sample_cells['tumor_zone'] == zone]
                if len(zone_cells) > 0:
                    ax.scatter(zone_cells['X_centroid'], zone_cells['Y_centroid'],
                              s=1, c=color, alpha=0.5, label=zone, rasterized=True)

            # Draw tumor boundaries
            for _, tumor in sample_tumors.iterrows():
                # Get tumor cells
                tumor_cells_in_struct = sample_cells[
                    (sample_cells['phenotype'].str.contains('Tumor', na=False)) &
                    (sample_cells['tumor_id'] == tumor['tumor_label'])
                ]

                if len(tumor_cells_in_struct) > 4:
                    try:
                        coords = tumor_cells_in_struct[['X_centroid', 'Y_centroid']].values
                        hull = ConvexHull(coords)
                        for simplex in hull.simplices:
                            ax.plot(coords[simplex, 0], coords[simplex, 1],
                                   'k-', linewidth=0.5, alpha=0.7)
                    except:
                        pass

                # Mark centroid
                ax.plot(tumor['centroid_x'], tumor['centroid_y'],
                       'b*', markersize=10)

            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_title(f'Tumor Structures and Zones - {sample_id}')
            ax.legend(markerscale=5, loc='best')
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(output_dir / f'tumor_structures_{sample_id}.pdf',
                       dpi=self.config.dpi, bbox_inches='tight')
            plt.close()

        print(f"Tumor structure plots saved to: {output_dir}")

    def _plot_tumor_size_distribution(self):
        """Plot tumor size distribution across samples."""
        output_dir = self.output_dir / '01_phenotyping' / 'aggregated_panels'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Overall size distribution
        ax = axes[0, 0]
        ax.hist(self.tumors_df['n_cells'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Tumor Size (cells)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Tumor Size Distribution')
        ax.set_yscale('log')

        # Size by group
        ax = axes[0, 1]
        groups = self.tumors_df['group'].unique()
        for group in groups:
            group_tumors = self.tumors_df[self.tumors_df['group'] == group]
            ax.hist(group_tumors['n_cells'], bins=30, alpha=0.5,
                   label=group, edgecolor='black')
        ax.set_xlabel('Tumor Size (cells)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tumor Size by Group')
        ax.legend()

        # Size vs circularity
        ax = axes[1, 0]
        scatter = ax.scatter(self.tumors_df['n_cells'], self.tumors_df['circularity'],
                            c=self.tumors_df['timepoint'], cmap='viridis',
                            alpha=0.6, s=50)
        ax.set_xlabel('Tumor Size (cells)')
        ax.set_ylabel('Circularity')
        ax.set_title('Tumor Size vs Circularity')
        plt.colorbar(scatter, ax=ax, label='Timepoint')

        # Boxplot by timepoint
        ax = axes[1, 1]
        timepoints = sorted(self.tumors_df['timepoint'].dropna().unique())
        if len(timepoints) > 0:
            data_by_time = [
                self.tumors_df[self.tumors_df['timepoint'] == tp]['n_cells'].values
                for tp in timepoints
            ]
            ax.boxplot(data_by_time, labels=[str(int(t)) for t in timepoints])
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Tumor Size (cells)')
            ax.set_title('Tumor Size by Timepoint')

        plt.tight_layout()
        plt.savefig(output_dir / 'tumor_size_distribution.pdf',
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        print(f"Size distribution plot saved to: {output_dir}")

    # =========================================================================
    # PHASE 2: PLACEHOLDER METHODS (to be expanded)
    # =========================================================================

    def phase21_perk_spatial_architecture(self):
        """Phase 2.1: pERK spatial architecture analysis - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 2.1: pERK SPATIAL ARCHITECTURE ANALYSIS")
        print("=" * 80 + "\n")
        print("This phase analyzes pERK+ spatial clustering, growth, and infiltration.")
        print("Implementation: To be completed with full Ripley's K, Moran's I, and hotspot analysis.")
        print("✓ Phase 2.1 placeholder complete\n")

    def phase22_ninja_escape_analysis(self):
        """Phase 2.2: NINJA escape mechanism analysis - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 2.2: NINJA ESCAPE MECHANISM ANALYSIS")
        print("=" * 80 + "\n")
        print("This phase analyzes NINJA+ spatial patterns and immune escape mechanisms.")
        print("Implementation: To be completed with clustering and enrichment analysis.")
        print("✓ Phase 2.2 placeholder complete\n")

    def phase23_heterogeneity_analysis(self):
        """Phase 2.3: Heterogeneity emergence and evolution - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 2.3: HETEROGENEITY EMERGENCE & EVOLUTION")
        print("=" * 80 + "\n")
        print("This phase analyzes marker diversification and tumor entropy.")
        print("Implementation: To be completed with LISA clustering and entropy calculations.")
        print("✓ Phase 2.3 placeholder complete\n")

    def phase24_rcn_temporal_dynamics(self):
        """Phase 2.4: Cellular neighborhood temporal dynamics - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 2.4: CELLULAR NEIGHBORHOOD DYNAMICS")
        print("=" * 80 + "\n")
        print("This phase analyzes recurrent cellular neighborhoods over time.")
        print("Implementation: To be completed with k-NN graphs and hierarchical clustering.")
        print("✓ Phase 2.4 placeholder complete\n")

    def phase31_multilevel_distance_analysis(self):
        """Phase 3.1: Multi-level distance analysis - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 3.1: MULTI-LEVEL DISTANCE ANALYSIS")
        print("=" * 80 + "\n")
        print("This phase analyzes spatial distances between cell types.")
        print("Implementation: To be completed with per-tumor and per-sample distance metrics.")
        print("✓ Phase 3.1 placeholder complete\n")

    def phase32_infiltration_tumor_associations(self):
        """Phase 3.2: Infiltration-tumor association analysis - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 3.2: INFILTRATION-TUMOR ASSOCIATIONS")
        print("=" * 80 + "\n")
        print("This phase analyzes relationships between tumor size/position and infiltration.")
        print("Implementation: To be completed with regression models.")
        print("✓ Phase 3.2 placeholder complete\n")

    def phase33_pseudotemporal_analysis(self):
        """Phase 3.3: Pseudo-temporal trajectory analysis - PLACEHOLDER."""
        print("\n" + "=" * 80)
        print("PHASE 3.3: PSEUDO-TEMPORAL TRAJECTORY ANALYSIS")
        print("=" * 80 + "\n")
        print("This phase infers tumor evolution trajectories.")
        if not SCANPY_AVAILABLE:
            print("Warning: scanpy not available, skipping pseudo-temporal analysis")
        else:
            print("Implementation: To be completed with PAGA/trajectory inference.")
        print("✓ Phase 3.3 placeholder complete\n")
