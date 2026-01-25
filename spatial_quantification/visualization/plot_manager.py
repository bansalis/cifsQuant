"""
Plot Manager for Spatial Quantification
Orchestrate all plotting operations
"""

from pathlib import Path
from typing import Dict
from .individual_plots import IndividualPlots
from .composite_plots import CompositePlots
from ..stats.comparisons import GroupComparison
from ..stats.temporal import TemporalAnalysis


class PlotManager:
    """Manage all plotting operations."""

    def __init__(self, config: Dict, output_dir: Path):
        """
        Initialize plot manager.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        output_dir : Path
            Base output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)

        # Get grouping column from metadata config
        meta_config = config.get('metadata', {})
        self.group_col = meta_config.get('primary_grouping') or meta_config.get('group_column', 'group')

        # Create plotting instances
        self.individual_plots = IndividualPlots(config, output_dir)
        self.composite_plots = CompositePlots(config, output_dir)

        # Create statistical instances
        self.group_comparison = GroupComparison(config)
        self.temporal_analysis = TemporalAnalysis(config)

    def plot_population_dynamics(self, population_results: Dict):
        """
        Create all plots for population dynamics analysis.

        Parameters
        ----------
        population_results : dict
            Results from PopulationDynamics.run()
        """
        print("\nGenerating population dynamics plots...")

        comparisons = self.config['population_dynamics'].get('comparisons', [])

        for pop_name, pop_data in population_results.items():
            if 'fraction' in pop_name:
                # Fractional population
                value_col = 'fraction'
            else:
                # Regular population - plot count and density
                for value_col in ['count', 'density_per_mm2']:
                    if value_col not in pop_data.columns:
                        continue

                    # Plot for each comparison
                    for comp in comparisons:
                        config_groups = comp.get('groups', [])
                        group_col = self.group_col

                        # Auto-detect groups from the data
                        if group_col in pop_data.columns:
                            actual_groups = sorted(pop_data[group_col].dropna().unique().tolist())
                        else:
                            actual_groups = []

                        # Only use config groups if they exist in the data, otherwise use actual groups
                        if config_groups:
                            groups = [g for g in config_groups if g in actual_groups]
                            if not groups:
                                groups = actual_groups
                        else:
                            groups = actual_groups

                        if not groups:
                            continue

                        # Auto-detect timepoints from data
                        if 'timepoint' in pop_data.columns:
                            timepoints = sorted(pop_data['timepoint'].dropna().unique().tolist())
                        else:
                            timepoints = comp.get('timepoints', [])

                        # Only calculate between-group statistics if we have 2+ groups
                        stats = None
                        if len(groups) >= 2 and timepoints:
                            try:
                                stats = self.group_comparison.compare_at_each_timepoint(
                                    pop_data, value_col, group_col, groups, timepoints
                                )
                            except Exception as e:
                                print(f"  ⚠ Could not calculate stats for {pop_name}: {e}")

                        # Create meaningful output name
                        if len(groups) == 1:
                            output_name = f"{pop_name}_{value_col}_{groups[0]}_over_time"
                        else:
                            output_name = f"{pop_name}_{value_col}_{'_vs_'.join(groups)}"

                        self.individual_plots.plot_population_over_time(
                            pop_data,
                            population=pop_name,
                            value_col=value_col,
                            group_col=group_col,
                            groups=groups,
                            stats=stats,
                            output_name=output_name
                        )

                        # Save statistics if available
                        if stats is not None:
                            stats_path = self.output_dir / 'population_dynamics' / f'{output_name}_stats.csv'
                            stats_path.parent.mkdir(parents=True, exist_ok=True)
                            stats.to_csv(stats_path, index=False)

        print(f"  ✓ Generated population dynamics plots")

    def plot_distance_analysis(self, distance_results: Dict):
        """
        Create all plots for distance analysis.

        Parameters
        ----------
        distance_results : dict
            Results from DistanceAnalysis.run()
        """
        print("\nGenerating distance analysis plots...")

        # Create distance plotter for histogram visualizations
        from .distance_analysis_plotter import DistanceAnalysisPlotter
        distance_plotter = DistanceAnalysisPlotter(self.output_dir / 'distance_analysis', self.config)

        comparisons = self.config['distance_analysis'].get('comparisons', [])

        for pairing_name, pairing_data in distance_results.items():
            # Extract source and target from pairing_name
            parts = pairing_name.split('_to_')
            if len(parts) == 2:
                source, target = parts
            else:
                continue

            # Plot for each comparison
            for comp in comparisons:
                config_groups = comp.get('groups', [])
                group_col = self.group_col

                # Auto-detect groups from the data
                if group_col in pairing_data.columns:
                    actual_groups = sorted(pairing_data[group_col].dropna().unique().tolist())
                else:
                    actual_groups = []

                # Only use config groups if they exist in the data, otherwise use actual groups
                if config_groups:
                    groups = [g for g in config_groups if g in actual_groups]
                    if not groups:
                        groups = actual_groups  # Fall back to actual groups
                else:
                    groups = actual_groups

                if not groups:
                    print(f"  ⚠ No groups found for {pairing_name}, skipping plot")
                    continue

                # Time series plot - auto-detect timepoints
                if 'timepoint' in pairing_data.columns:
                    timepoints = sorted(pairing_data['timepoint'].dropna().unique().tolist())
                else:
                    timepoints = comp.get('timepoints', [])

                # Only calculate between-group statistics if we have 2+ groups
                stats = None
                if len(groups) >= 2 and timepoints:
                    try:
                        stats = self.group_comparison.compare_at_each_timepoint(
                            pairing_data, 'mean_distance', group_col, groups, timepoints
                        )
                    except Exception as e:
                        print(f"  ⚠ Could not calculate stats for {pairing_name}: {e}")

                # Create meaningful output name
                if len(groups) == 1:
                    output_name = f"{pairing_name}_{groups[0]}_over_time"
                else:
                    output_name = f"{pairing_name}_{'_vs_'.join(groups)}"

                self.individual_plots.plot_population_over_time(
                    pairing_data,
                    population=f'{source} to {target}',
                    value_col='mean_distance',
                    group_col=group_col,
                    groups=groups,
                    stats=stats,
                    output_name=output_name
                )

                # Distance distribution
                self.individual_plots.plot_distance_distribution(
                    pairing_data,
                    source=source,
                    target=target,
                    group_col=group_col,
                    groups=groups,
                    output_name=f'{pairing_name}_distribution'
                )

                # NEW: Binned distance histograms (show peak shifts)
                if self.config.get('spatial_visualization', {}).get('distance_histograms', {}).get('enabled', True):
                    distance_bins = self.config.get('spatial_visualization', {}).get('distance_histograms', {}).get('distance_bins', None)
                    distance_plotter.plot_distance_histograms_binned(
                        pairing_data,
                        source=source,
                        target=target,
                        group_col=group_col,
                        groups=groups,
                        distance_bins=distance_bins
                    )

                # Save statistics
                stats_path = self.output_dir / 'distance_analysis' / f'{output_name}_stats.csv'
                stats.to_csv(stats_path, index=False)

        print(f"  ✓ Generated distance analysis plots")

    def plot_infiltration_analysis(self, infiltration_results: Dict):
        """
        Create all plots for infiltration analysis.

        Parameters
        ----------
        infiltration_results : dict
            Results from InfiltrationAnalysis.run()
        """
        print("\nGenerating infiltration analysis plots...")

        # Plots are generated within InfiltrationAnalysisOptimized
        # Includes: 3-panel spatial plots (marker+/-, Gi* hotspots, DBSCAN clusters)
        print(f"  ✓ Infiltration plots generated by analysis module")
        print(f"    Spatial plots: {self.output_dir / 'infiltration_analysis' / 'spatial_plots'}")
        print(f"    Data saved to: {self.output_dir / 'infiltration_analysis'}")

    def plot_neighborhood_analysis(self, neighborhood_results: Dict):
        """
        Create all plots for neighborhood analysis.

        Parameters
        ----------
        neighborhood_results : dict
            Results from NeighborhoodAnalysis.run()
        """
        print("\nGenerating neighborhood analysis plots...")

        # Plots are generated within NeighborhoodAnalysisOptimized
        # Includes: composition heatmaps, temporal evolution, stacked areas, spatial maps
        print(f"  ✓ Neighborhood plots generated by analysis module")
        print(f"    Composition heatmaps, temporal evolution, stacked area charts")
        print(f"    Spatial maps: {self.output_dir / 'neighborhood_analysis' / 'spatial_plots'}")
        print(f"    Data saved to: {self.output_dir / 'neighborhoods'}")
