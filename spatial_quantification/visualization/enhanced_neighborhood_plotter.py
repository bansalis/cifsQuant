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

    def plot_comprehensive_immune_composition(self, regional_infiltration_df: pd.DataFrame, marker: str):
        """
        Create comprehensive immune composition plots for marker+ vs marker- regions.

        Two versions:
        1. Simple: marker+ vs marker- only
        2. Detailed: marker+ vs marker- separated by group (KPT/KPNT)
        """
        if regional_infiltration_df is None or len(regional_infiltration_df) == 0:
            return

        # Detect immune populations
        immune_cols = [col for col in regional_infiltration_df.columns
                      if col.endswith('_in_pos_region_percent')]
        if not immune_cols:
            return

        immune_pops = [col.replace('_in_pos_region_percent', '') for col in immune_cols]

        # Version 1: Simple comparison (marker+ vs marker-)
        self._plot_immune_composition_simple(regional_infiltration_df, marker, immune_pops)

        # Version 2: Detailed comparison (marker+/- x KPT/KPNT)
        self._plot_immune_composition_detailed(regional_infiltration_df, marker, immune_pops)

        # Heatmap of immune composition
        self._plot_immune_composition_heatmap(regional_infiltration_df, marker, immune_pops)

    def _plot_immune_composition_simple(self, df: pd.DataFrame, marker: str, immune_pops: List[str]):
        """Side-by-side boxplots: marker+ vs marker- for all immune populations."""

        # Prepare data for all immune populations
        plot_data_list = []

        for immune_pop in immune_pops:
            pos_col = f'{immune_pop}_in_pos_region_percent'
            neg_col = f'{immune_pop}_in_neg_region_percent'

            if pos_col not in df.columns or neg_col not in df.columns:
                continue

            for idx, row in df.iterrows():
                # Marker+ region
                plot_data_list.append({
                    'immune_population': immune_pop,
                    'region_type': f'{marker}+',
                    'percent': row[pos_col],
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', '')
                })

                # Marker- region
                plot_data_list.append({
                    'immune_population': immune_pop,
                    'region_type': f'{marker}-',
                    'percent': row[neg_col],
                    'sample_id': row['sample_id'],
                    'timepoint': row.get('timepoint', 1),
                    'main_group': row.get('main_group', '')
                })

        if not plot_data_list:
            return

        plot_df = pd.DataFrame(plot_data_list)

        # Create subplots for each immune population
        n_pops = len(immune_pops)
        fig, axes = plt.subplots(1, n_pops, figsize=(5*n_pops, 6))

        if n_pops == 1:
            axes = [axes]

        for idx, immune_pop in enumerate(immune_pops):
            ax = axes[idx]
            pop_data = plot_df[plot_df['immune_population'] == immune_pop]

            # Create boxplot
            region_types = [f'{marker}+', f'{marker}-']
            positions = [0, 1]
            box_data = []
            colors = []

            for region_type in region_types:
                data = pop_data[pop_data['region_type'] == region_type]['percent'].dropna()
                box_data.append(data)
                if region_type == f'{marker}+':
                    colors.append(self.group_colors.get('KPT', '#E41A1C'))
                else:
                    colors.append('#999999')

            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='black'))

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add individual points
            for pos, region_type, color in zip(positions, region_types, colors):
                data = pop_data[pop_data['region_type'] == region_type]['percent'].dropna()
                x = np.random.normal(pos, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.4, s=30, color=color, zorder=5)

            # Add statistics
            from scipy import stats
            if len(box_data[0]) >= 2 and len(box_data[1]) >= 2:
                stat, pval = stats.mannwhitneyu(box_data[0], box_data[1], alternative='two-sided')
                if pval < 0.001:
                    sig = '***'
                elif pval < 0.01:
                    sig = '**'
                elif pval < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'

                y_max = max([d.max() for d in box_data])
                y_range = y_max - min([d.min() for d in box_data])
                y_bracket = y_max + y_range * 0.05
                h = y_range * 0.02

                ax.plot([0, 0, 1, 1], [y_bracket, y_bracket+h, y_bracket+h, y_bracket],
                       lw=1.5, c='black')
                ax.text(0.5, y_bracket+h, sig,
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_xticks(positions)
            ax.set_xticklabels(region_types, fontsize=10)
            ax.set_ylabel('% in Region', fontsize=11, fontweight='bold')
            ax.set_title(immune_pop.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{marker}: Immune Composition ({marker}+ vs {marker}- Regions)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_immune_composition_simple.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} immune composition (simple) plot")

    def _plot_immune_composition_detailed(self, df: pd.DataFrame, marker: str, immune_pops: List[str]):
        """Side-by-side boxplots: marker+/- x KPT/KPNT for all immune populations."""

        if 'main_group' not in df.columns:
            return

        groups = sorted(df['main_group'].unique())
        if len(groups) < 2:
            return

        # Prepare data
        plot_data_list = []

        for immune_pop in immune_pops:
            pos_col = f'{immune_pop}_in_pos_region_percent'
            neg_col = f'{immune_pop}_in_neg_region_percent'

            if pos_col not in df.columns or neg_col not in df.columns:
                continue

            for idx, row in df.iterrows():
                group = row.get('main_group', '')

                # Marker+ region
                plot_data_list.append({
                    'immune_population': immune_pop,
                    'category': f'{marker}+ {group}',
                    'percent': row[pos_col],
                    'sample_id': row['sample_id'],
                    'region_type': f'{marker}+',
                    'main_group': group
                })

                # Marker- region
                plot_data_list.append({
                    'immune_population': immune_pop,
                    'category': f'{marker}- {group}',
                    'percent': row[neg_col],
                    'sample_id': row['sample_id'],
                    'region_type': f'{marker}-',
                    'main_group': group
                })

        if not plot_data_list:
            return

        plot_df = pd.DataFrame(plot_data_list)

        # Create subplots
        n_pops = len(immune_pops)
        fig, axes = plt.subplots(1, n_pops, figsize=(6*n_pops, 6))

        if n_pops == 1:
            axes = [axes]

        for idx, immune_pop in enumerate(immune_pops):
            ax = axes[idx]
            pop_data = plot_df[plot_df['immune_population'] == immune_pop]

            # Categories: marker+/- x group
            categories = []
            for group in groups:
                categories.append(f'{marker}+ {group}')
            for group in groups:
                categories.append(f'{marker}- {group}')

            positions = list(range(len(categories)))
            box_data = []
            colors = []

            for cat in categories:
                data = pop_data[pop_data['category'] == cat]['percent'].dropna()
                box_data.append(data)

                # Color by group
                if 'KPT' in cat:
                    color = self.group_colors.get('KPT', '#E41A1C')
                elif 'KPNT' in cat:
                    color = self.group_colors.get('KPNT', '#377EB8')
                else:
                    color = '#999999'

                # Lighter for marker- regions
                if f'{marker}-' in cat:
                    colors.append(color + '80')  # Add alpha
                else:
                    colors.append(color)

            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='black'))

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add individual points
            for pos, cat, color in zip(positions, categories, colors):
                data = pop_data[pop_data['category'] == cat]['percent'].dropna()
                x = np.random.normal(pos, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.4, s=30, color=color, zorder=5)

            ax.set_xticks(positions)
            ax.set_xticklabels([cat.replace(f'{marker}+', '+').replace(f'{marker}-', '-')
                               for cat in categories], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('% in Region', fontsize=11, fontweight='bold')
            ax.set_title(immune_pop.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{marker}: Immune Composition by Group ({marker}+/- x KPT/KPNT)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_immune_composition_detailed.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} immune composition (detailed) plot")

    def _plot_immune_composition_heatmap(self, df: pd.DataFrame, marker: str, immune_pops: List[str]):
        """Heatmap of immune composition in marker+ vs marker- regions."""

        # Calculate mean percentages
        heatmap_data = []

        for immune_pop in immune_pops:
            pos_col = f'{immune_pop}_in_pos_region_percent'
            neg_col = f'{immune_pop}_in_neg_region_percent'

            if pos_col not in df.columns or neg_col not in df.columns:
                continue

            row_data = {
                'Immune Population': immune_pop.replace('_', ' '),
                f'{marker}+ Region': df[pos_col].mean(),
                f'{marker}- Region': df[neg_col].mean()
            }
            heatmap_data.append(row_data)

        if not heatmap_data:
            return

        heatmap_df = pd.DataFrame(heatmap_data).set_index('Immune Population')

        fig, ax = plt.subplots(figsize=(8, len(immune_pops)*0.8))
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': '% in Region'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        ax.set_title(f'{marker}: Immune Composition Heatmap',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('')

        plt.tight_layout()

        plot_path = self.plots_dir / f'{marker}_immune_composition_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved {marker} immune composition heatmap")

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
                # Add comprehensive immune composition plots
                self.plot_comprehensive_immune_composition(results[key], marker)
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
