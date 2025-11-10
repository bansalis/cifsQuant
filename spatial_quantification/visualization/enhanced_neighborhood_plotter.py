"""
Enhanced Neighborhood Analysis Visualization
Comprehensive plots for regional infiltration and neighborhood composition
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings

from .plot_utils import (
    detect_plot_type,
    plot_with_stats,
    create_dual_plots,
    calculate_statistics
)


class EnhancedNeighborhoodPlotter:
    """
    Comprehensive visualization for enhanced neighborhood analysis.

    Generates:
    - Regional infiltration comparison (marker+ vs marker-)
    - Distance to marker regions over time
    - Neighborhood composition within regions
    - Per-cell neighborhood comparisons
    """

    def __init__(self, output_dir: Path, config: Dict):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : Path
            Output directory for plots
        config : dict
            Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Get plotting settings
        plotting_config = config.get('plotting', {})
        self.group_colors = plotting_config.get('group_colors', {
            'KPT': '#E41A1C', 'KPNT': '#377EB8'
        })
        self.timepoint_label = plotting_config.get('timepoint_label', 'Time (weeks)')

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    def plot_regional_infiltration(self, regional_infiltration_df: pd.DataFrame, marker: str):
        """
        Plot regional infiltration (marker+ vs marker-).

        Parameters
        ----------
        regional_infiltration_df : pd.DataFrame
            Regional infiltration data
        marker : str
            Marker name (pERK or NINJA)
        """
        if regional_infiltration_df is None or len(regional_infiltration_df) == 0:
            print(f"    ⚠ No data for {marker} regional infiltration")
            return

        # Detect immune populations from columns
        immune_cols = [col for col in regional_infiltration_df.columns
                      if col.endswith('_in_pos_region_percent')]
        if not immune_cols:
            print(f"    ⚠ No immune population columns found for {marker}")
            return

        immune_pops = [col.replace('_in_pos_region_percent', '') for col in immune_cols]

        # Create plots for each immune population
        for immune_pop in immune_pops:
            # Prepare data for marker+ region
            pos_col = f'{immune_pop}_in_pos_region_percent'
            neg_col = f'{immune_pop}_in_neg_region_percent'

            if pos_col not in regional_infiltration_df.columns or neg_col not in regional_infiltration_df.columns:
                continue

            # Create data for plotting (both marker+ and marker- in same dataframe)
            plot_data_list = []

            for idx, row in regional_infiltration_df.iterrows():
                # Marker+ region
                plot_data_list.append({
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', ''),
                    'region_type': f'{marker}+',
                    'percent_in_region': row[pos_col]
                })

                # Marker- region
                plot_data_list.append({
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', ''),
                    'region_type': f'{marker}-',
                    'percent_in_region': row[neg_col]
                })

            plot_data = pd.DataFrame(plot_data_list)

            # Create dual plots (with and without stats) for percent in region
            output_base = str(self.plots_dir / f'{marker}_{immune_pop}_percent_in_region')
            create_dual_plots(
                plot_data,
                value_col='percent_in_region',
                group_col='region_type',
                timepoint_col='timepoint',
                group_colors={f'{marker}+': self.group_colors.get('KPT', '#E41A1C'),
                            f'{marker}-': '#999999'},
                title_base=f'{immune_pop}: % in {marker}+ vs {marker}- Regions',
                ylabel='% Immune in Region',
                xlabel=self.timepoint_label,
                output_path_base=output_base
            )

            print(f"    ✓ Saved {marker} {immune_pop} infiltration plots (with and without stats)")

            # Also create distance plots if distance columns exist
            dist_pos_col = f'{immune_pop}_mean_dist_to_pos'
            dist_neg_col = f'{immune_pop}_mean_dist_to_neg'

            if dist_pos_col in regional_infiltration_df.columns and dist_neg_col in regional_infiltration_df.columns:
                dist_data_list = []

                for idx, row in regional_infiltration_df.iterrows():
                    if not pd.isna(row[dist_pos_col]):
                        dist_data_list.append({
                            'sample_id': row['sample_id'],
                            'timepoint': row.get('timepoint', 1),
                            'main_group': row.get('main_group', ''),
                            'region_type': f'{marker}+',
                            'distance': row[dist_pos_col]
                        })

                    if not pd.isna(row[dist_neg_col]):
                        dist_data_list.append({
                            'sample_id': row['sample_id'],
                            'timepoint': row.get('timepoint', 1),
                            'main_group': row.get('main_group', ''),
                            'region_type': f'{marker}-',
                            'distance': row[dist_neg_col]
                        })

                if dist_data_list:
                    dist_data = pd.DataFrame(dist_data_list)

                    output_base = str(self.plots_dir / f'{marker}_{immune_pop}_distance_to_region')
                    create_dual_plots(
                        dist_data,
                        value_col='distance',
                        group_col='region_type',
                        timepoint_col='timepoint',
                        group_colors={f'{marker}+': self.group_colors.get('KPT', '#E41A1C'),
                                    f'{marker}-': '#999999'},
                        title_base=f'{immune_pop}: Distance to {marker}+ vs {marker}- Regions',
                        ylabel='Mean Distance (μm)',
                        xlabel=self.timepoint_label,
                        output_path_base=output_base
                    )

                    print(f"    ✓ Saved {marker} {immune_pop} distance plots (with and without stats)")

    def plot_regional_neighborhoods(self, regional_neighborhoods_df: pd.DataFrame, marker: str):
        """
        Plot neighborhood composition in marker+ vs marker- regions.

        Parameters
        ----------
        regional_neighborhoods_df : pd.DataFrame
            Regional neighborhood composition data
        marker : str
            Marker name
        """
        if regional_neighborhoods_df is None or len(regional_neighborhoods_df) == 0:
            print(f"    ⚠ No data for {marker} regional neighborhoods")
            return

        # Get cell type columns
        cell_type_cols = [col for col in regional_neighborhoods_df.columns
                         if col.startswith('pos_region_') and col.endswith('_fraction')]

        if not cell_type_cols:
            print(f"    ⚠ No cell type columns found for {marker} regional neighborhoods")
            return

        # Extract cell type names
        cell_types = [col.replace('pos_region_', '').replace('_fraction', '')
                     for col in cell_type_cols]

        # Select key cell types for visualization
        key_types = ['CD8_T_cells', 'CD3_positive', 'CD45_positive', 'Tumor']
        available_key_types = [ct for ct in key_types if ct in cell_types]

        if not available_key_types:
            available_key_types = cell_types[:4]  # Take first 4

        # Create plots for each cell type
        for cell_type in available_key_types:
            col_pos = f'pos_region_{cell_type}_fraction'
            col_neg = f'neg_region_{cell_type}_fraction'

            if col_pos not in regional_neighborhoods_df.columns or col_neg not in regional_neighborhoods_df.columns:
                continue

            # Prepare data for plotting
            plot_data_list = []

            for idx, row in regional_neighborhoods_df.iterrows():
                # Marker+ region
                plot_data_list.append({
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', ''),
                    'region_type': f'{marker}+',
                    'fraction': row[col_pos] * 100  # Convert to percent
                })

                # Marker- region
                plot_data_list.append({
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', ''),
                    'region_type': f'{marker}-',
                    'fraction': row[col_neg] * 100  # Convert to percent
                })

            plot_data = pd.DataFrame(plot_data_list)

            # Create dual plots (with and without stats)
            output_base = str(self.plots_dir / f'{marker}_{cell_type}_neighborhood_composition')
            create_dual_plots(
                plot_data,
                value_col='fraction',
                group_col='region_type',
                timepoint_col='timepoint',
                group_colors={f'{marker}+': self.group_colors.get('KPT', '#E41A1C'),
                            f'{marker}-': '#999999'},
                title_base=f'{cell_type}: Neighborhood Composition in {marker}+ vs {marker}- Regions',
                ylabel='% of Neighborhood',
                xlabel=self.timepoint_label,
                output_path_base=output_base
            )

        print(f"    ✓ Saved {marker} regional neighborhood plots (with and without stats)")

    def plot_per_cell_neighborhoods(self, per_cell_df: pd.DataFrame, marker: str):
        """
        Plot per-cell neighborhood comparison (marker+ vs marker-).

        Parameters
        ----------
        per_cell_df : pd.DataFrame
            Per-cell neighborhood data (summary or individual)
        marker : str
            Marker name
        """
        if per_cell_df is None or len(per_cell_df) == 0:
            print(f"    ⚠ No data for {marker} per-cell neighborhoods")
            return

        # Get cell type columns
        neighbor_cols = [col for col in per_cell_df.columns
                        if col.startswith('neighbor_') and col.endswith('_fraction')]

        if not neighbor_cols:
            print(f"    ⚠ No neighbor columns found for {marker} per-cell neighborhoods")
            return

        cell_types = [col.replace('neighbor_', '').replace('_fraction', '')
                     for col in neighbor_cols]

        # Select key cell types
        key_types = ['CD8_T_cells', 'CD3_positive', 'CD45_positive', 'Tumor']
        available_key_types = [ct for ct in key_types if ct in cell_types]

        if not available_key_types:
            available_key_types = cell_types[:4]

        # Create plots for each cell type
        for cell_type in available_key_types:
            col = f'neighbor_{cell_type}_fraction'

            if col not in per_cell_df.columns:
                continue

            # Prepare data for plotting
            plot_data_list = []

            for idx, row in per_cell_df.iterrows():
                marker_status = row.get('cell_marker_status', 'unknown')

                plot_data_list.append({
                    'sample_id': row.get('sample_id', ''),
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', ''),
                    'marker_status': f'{marker}+' if marker_status == 'positive' else f'{marker}-',
                    'fraction': row[col] * 100  # Convert to percent
                })

            plot_data = pd.DataFrame(plot_data_list)

            if len(plot_data) == 0:
                continue

            # Create dual plots (with and without stats)
            output_base = str(self.plots_dir / f'{marker}_{cell_type}_per_cell_neighborhood')
            create_dual_plots(
                plot_data,
                value_col='fraction',
                group_col='marker_status',
                timepoint_col='timepoint',
                group_colors={f'{marker}+': self.group_colors.get('KPT', '#E41A1C'),
                            f'{marker}-': '#999999'},
                title_base=f'{cell_type}: Per-Cell Neighborhood ({marker}+ vs {marker}- cells)',
                ylabel='% of Local Neighborhood',
                xlabel=self.timepoint_label,
                output_path_base=output_base
            )

        print(f"    ✓ Saved {marker} per-cell neighborhood plots (with and without stats)")

    def generate_all_plots(self, results: Dict):
        """
        Generate all enhanced neighborhood visualization plots.

        Parameters
        ----------
        results : dict
            Results dictionary from EnhancedNeighborhoodAnalysis
        """
        print("  Generating enhanced neighborhood analysis plots...")

        # Get markers from config
        enh_config = self.config.get('enhanced_neighborhoods', {})
        marker_configs = enh_config.get('markers', [])

        # Extract marker names from config
        if marker_configs:
            markers = [m['name'] for m in marker_configs]
        else:
            # Default markers (fallback)
            markers = ['pERK', 'NINJA']

        print(f"    Processing {len(markers)} markers: {', '.join(markers)}")

        # Process each marker
        for marker in markers:
            # Regional infiltration
            key = f'{marker}_regional_infiltration'
            if key in results:
                self.plot_regional_infiltration(results[key], marker)
            else:
                print(f"    ⚠ {key} not found in results")

            # Regional neighborhoods
            key = f'{marker}_regional_neighborhoods'
            if key in results:
                self.plot_regional_neighborhoods(results[key], marker)
            else:
                print(f"    ⚠ {key} not found in results")

            # Per-cell neighborhoods - use summary data for plotting
            key_summary = f'{marker}_per_cell_neighborhoods_summary'
            key_individual = f'{marker}_per_cell_neighborhoods_individual'

            if key_summary in results:
                self.plot_per_cell_neighborhoods(results[key_summary], marker)
            elif key_individual in results:
                # If only individual data exists, aggregate it for plotting
                print(f"    Using individual cell data for {marker} (aggregating for plot)")
                self.plot_per_cell_neighborhoods(results[key_individual], marker)
            else:
                print(f"    ⚠ No per-cell neighborhood data found for {marker}")

        print(f"  ✓ All enhanced neighborhood plots saved to {self.plots_dir}/")
