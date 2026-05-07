"""
Harrell-Davis Shift Plot Analysis
Compare nearest-neighbor distance distributions between stage groups using
the Harrell-Davis quantile estimator and bootstrap confidence intervals.

Reproduces Nirmal et al. 2022 Fig 3F style shift plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.stats import beta as beta_dist
import warnings


class ShiftPlotAnalysis:
    """
    Harrell-Davis shift plots for comparing nearest-neighbor distance
    distributions across melanoma stage groups.

    For each (source, target) cell-population pair and each stage-group
    comparison, the shift plot shows — at each percentile — how much the
    distance distribution has shifted between the two groups.  Bootstrap
    confidence intervals flag which shifts are statistically significant.

    Config section: config['shift_plot_analysis']
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        """
        Initialize shift-plot analysis.

        Parameters
        ----------
        adata : AnnData
            Annotated data.  Cell coordinates must be in adata.obs['x'] and
            adata.obs['y'].  Phenotype membership as boolean columns named
            'is_<PhenotypeName>' in adata.obs.
        config : dict
            Full pipeline configuration dictionary.
        output_dir : Path
            Root output directory.  Results are written to a subdirectory
            'shift_plot_analysis/'.
        """
        self.adata = adata
        self.full_config = config
        self.config = config.get('shift_plot_analysis', {})
        self.output_dir = Path(output_dir) / 'shift_plot_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Run the full shift-plot analysis.

        Returns
        -------
        dict
            Nested results keyed by pair_name → comparison_label → metrics.
            Each leaf dict contains:
                percentiles   : list of int
                hd_group1     : list of float  (HD quantile estimates for group 1)
                hd_group2     : list of float  (HD quantile estimates for group 2)
                difference    : list of float  (group2 − group1)
                significant   : list of bool
                ci_low        : list of float  (lower bound of 95 % bootstrap CI)
                ci_high       : list of float  (upper bound of 95 % bootstrap CI)
                n_group1      : int
                n_group2      : int
                group1_label  : str
                group2_label  : str
        """
        print("\n" + "=" * 80)
        print("SHIFT PLOT ANALYSIS  (Harrell-Davis quantile estimator)")
        print("=" * 80)

        if not self.config.get('enabled', True):
            print("  Shift-plot analysis is disabled in config — skipping.")
            return self.results

        pairs = self.config.get('pairs', [])
        comparisons = self.config.get('comparisons', [])
        stage_col = self.config.get('stage_column', 'stage_group')
        proximity_radius = self.config.get('proximity_radius', 1000)
        percentiles = self.config.get('percentiles', list(range(10, 100, 10)))
        n_bootstrap = self.config.get('n_bootstrap', 1000)
        alpha = self.config.get('alpha', 0.05)

        if stage_col not in self.adata.obs.columns:
            warnings.warn(
                f"Stage column '{stage_col}' not found in adata.obs. "
                "Shift-plot analysis cannot proceed."
            )
            return self.results

        if not pairs:
            warnings.warn("No cell-population pairs configured for shift-plot analysis.")
            return self.results

        if not comparisons:
            warnings.warn("No stage-group comparisons configured for shift-plot analysis.")
            return self.results

        print(f"\n  Cell-population pairs     : {len(pairs)}")
        print(f"  Stage-group comparisons   : {len(comparisons)}")
        print(f"  Percentiles               : {percentiles}")
        print(f"  Bootstrap samples         : {n_bootstrap}")
        print(f"  Proximity radius          : {proximity_radius} μm")
        print(f"  Significance threshold α  : {alpha}")

        for pair in pairs:
            source = pair['source']
            target = pair['target']
            pair_name = pair.get('name', f'{source}_to_{target}')

            print(f"\n  Pair: {pair_name}  ({source} → {target})")

            # Pre-compute per-sample distances so we don't repeat KDTree queries
            distances_by_stage = self._collect_distances_by_stage(
                source, target, stage_col, proximity_radius
            )

            if not distances_by_stage:
                warnings.warn(
                    f"  No distance data found for pair '{pair_name}' — skipping."
                )
                continue

            self.results[pair_name] = {}

            for comp in comparisons:
                g1_label = comp['group1']
                g2_label = comp['group2']
                comp_label = comp.get('label', f'{g1_label} vs {g2_label}')

                print(f"    Comparison: {comp_label}")

                g1_dists = distances_by_stage.get(g1_label, np.array([]))
                g2_dists = distances_by_stage.get(g2_label, np.array([]))

                if len(g1_dists) < 5 or len(g2_dists) < 5:
                    warnings.warn(
                        f"    Insufficient cells for '{comp_label}' "
                        f"(n1={len(g1_dists)}, n2={len(g2_dists)}) — skipping."
                    )
                    continue

                comp_result = self._compute_shift(
                    g1_dists, g2_dists,
                    percentiles, n_bootstrap, alpha,
                    g1_label, g2_label
                )
                self.results[pair_name][comp_label] = comp_result

                n_sig = sum(comp_result['significant'])
                print(
                    f"      n1={comp_result['n_group1']:,}  "
                    f"n2={comp_result['n_group2']:,}  "
                    f"significant percentiles: {n_sig}/{len(percentiles)}"
                )

        self._save_results()
        self._generate_plots()

        print("\n  Shift-plot analysis complete.")
        print(f"  Results saved to: {self.output_dir}/")
        print("=" * 80 + "\n")

        return self.results

    # ------------------------------------------------------------------
    # Distance collection
    # ------------------------------------------------------------------

    def _collect_distances_by_stage(
        self,
        source_pop: str,
        target_pop: str,
        stage_col: str,
        proximity_radius: float
    ) -> Dict[str, np.ndarray]:
        """
        Collect all nearest-neighbour distances from source_pop to target_pop,
        grouped by stage label.

        Returns a dict mapping stage_label → 1-D float array of distances.
        Distances larger than proximity_radius are excluded.
        """
        source_col = f'is_{source_pop}'
        target_col = f'is_{target_pop}'

        if source_col not in self.adata.obs.columns:
            warnings.warn(f"Source population '{source_pop}' not found in adata.obs.")
            return {}
        if target_col not in self.adata.obs.columns:
            warnings.warn(f"Target population '{target_pop}' not found in adata.obs.")
            return {}

        obs = self.adata.obs

        # Ensure spatial coordinates exist (stored in obsm['spatial'])
        if 'spatial' not in self.adata.obsm:
            warnings.warn(
                "adata.obsm does not contain 'spatial' coordinates. "
                "Cannot compute distances."
            )
            return {}

        all_coords = self.adata.obsm['spatial']  # shape (n_cells, 2)

        distances_by_stage: Dict[str, List[np.ndarray]] = {}

        for sample_id in obs['sample_id'].unique():
            sample_mask_bool = (obs['sample_id'] == sample_id).values
            sample_obs = obs[sample_mask_bool]
            sample_coords = all_coords[sample_mask_bool]  # (n_sample, 2)

            # Determine stage for this sample
            stage = sample_obs[stage_col].iloc[0]
            if pd.isna(stage):
                continue

            stage = str(stage)

            source_mask = sample_obs[source_col].fillna(False).astype(bool).values
            target_mask = sample_obs[target_col].fillna(False).astype(bool).values

            if source_mask.sum() == 0 or target_mask.sum() == 0:
                continue

            source_coords = sample_coords[source_mask]
            target_coords = sample_coords[target_mask]

            dists = self._nearest_neighbour_distances(
                source_coords, target_coords, proximity_radius
            )

            if len(dists) == 0:
                continue

            distances_by_stage.setdefault(stage, []).append(dists)

        # Concatenate per stage
        return {
            stage: np.concatenate(arrays)
            for stage, arrays in distances_by_stage.items()
        }

    def _nearest_neighbour_distances(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        proximity_radius: float
    ) -> np.ndarray:
        """
        Return 1-D array of nearest-neighbour distances from each source cell
        to the closest target cell.  Distances beyond proximity_radius are
        excluded.

        Parameters
        ----------
        source_coords : np.ndarray, shape (N, 2)
        target_coords : np.ndarray, shape (M, 2)
        proximity_radius : float
        """
        tree = cKDTree(target_coords)
        distances, _ = tree.query(source_coords, k=1, distance_upper_bound=proximity_radius)

        # cKDTree returns inf when no neighbour is within the upper bound
        valid = np.isfinite(distances)
        return distances[valid]

    # ------------------------------------------------------------------
    # Harrell-Davis estimator and bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def harrell_davis_quantile(data: np.ndarray, q: float) -> float:
        """
        Harrell-Davis estimator of the q-th quantile.

        HD(q) = Σ w_i · x_(i)

        where x_(i) are the order statistics and the weights w_i come from
        the incomplete Beta function:
            w_i = B(i/n; n·q, n·(1−q)) − B((i−1)/n; n·q, n·(1−q))

        Parameters
        ----------
        data : np.ndarray
            1-D array of observations.
        q : float
            Quantile in (0, 1).

        Returns
        -------
        float
            HD quantile estimate.
        """
        n = len(data)
        if n == 0:
            return np.nan

        sorted_data = np.sort(data)
        i_vals = np.arange(n)

        # Vectorised beta CDF difference for all i simultaneously
        a = n * q
        b = n * (1.0 - q)

        cdf_upper = beta_dist.cdf((i_vals + 1) / n, a, b)
        cdf_lower = beta_dist.cdf(i_vals / n, a, b)
        weights = cdf_upper - cdf_lower

        # Normalise to handle floating-point rounding
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.nan
        weights = weights / weight_sum

        return float(np.dot(weights, sorted_data))

    def _compute_hd_profile(
        self, data: np.ndarray, percentile_vals: List[int]
    ) -> np.ndarray:
        """
        Compute the Harrell-Davis estimate at each percentile.

        Parameters
        ----------
        data : np.ndarray
        percentile_vals : list of int  (e.g. [10, 20, ..., 90])

        Returns
        -------
        np.ndarray  shape (len(percentile_vals),)
        """
        q_vals = np.array(percentile_vals) / 100.0
        return np.array([
            self.harrell_davis_quantile(data, q) for q in q_vals
        ])

    def _compute_shift(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        percentiles: List[int],
        n_bootstrap: int,
        alpha: float,
        group1_label: str,
        group2_label: str
    ) -> Dict:
        """
        Compute the shift between two distance distributions at each percentile
        using the Harrell-Davis estimator and bootstrap CIs.

        The observed shift at percentile p is:  HD_p(data2) − HD_p(data1)

        Bootstrap CI: resample independently from data1 and data2, compute
        shift at each percentile, collect n_bootstrap replicates, derive the
        (alpha/2, 1−alpha/2) interval.

        Parameters
        ----------
        data1, data2 : np.ndarray
        percentiles : list of int
        n_bootstrap : int
        alpha : float
        group1_label, group2_label : str

        Returns
        -------
        dict  with keys: percentiles, hd_group1, hd_group2, difference,
                         significant, ci_low, ci_high, n_group1, n_group2,
                         group1_label, group2_label
        """
        n_pct = len(percentiles)

        # Observed HD profiles
        hd1 = self._compute_hd_profile(data1, percentiles)
        hd2 = self._compute_hd_profile(data2, percentiles)
        observed_diff = hd2 - hd1  # shape (n_pct,)

        # Bootstrap
        rng = np.random.default_rng(seed=42)
        boot_diffs = np.empty((n_bootstrap, n_pct))

        for b in range(n_bootstrap):
            resample1 = rng.choice(data1, size=len(data1), replace=True)
            resample2 = rng.choice(data2, size=len(data2), replace=True)
            boot_hd1 = self._compute_hd_profile(resample1, percentiles)
            boot_hd2 = self._compute_hd_profile(resample2, percentiles)
            boot_diffs[b] = boot_hd2 - boot_hd1

        # Percentile bootstrap CI
        ci_low = np.nanpercentile(boot_diffs, 100 * alpha / 2, axis=0)
        ci_high = np.nanpercentile(boot_diffs, 100 * (1 - alpha / 2), axis=0)

        # Significant if the CI does not include zero
        significant = (ci_low > 0) | (ci_high < 0)

        return {
            'percentiles': percentiles,
            'hd_group1': hd1.tolist(),
            'hd_group2': hd2.tolist(),
            'difference': observed_diff.tolist(),
            'significant': significant.tolist(),
            'ci_low': ci_low.tolist(),
            'ci_high': ci_high.tolist(),
            'n_group1': int(len(data1)),
            'n_group2': int(len(data2)),
            'group1_label': group1_label,
            'group2_label': group2_label,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_results(self):
        """Save shift-plot results to CSV files for reproducibility."""
        rows = []
        for pair_name, comparisons in self.results.items():
            for comp_label, data in comparisons.items():
                for i, pct in enumerate(data['percentiles']):
                    rows.append({
                        'pair': pair_name,
                        'comparison': comp_label,
                        'percentile': pct,
                        'hd_group1': data['hd_group1'][i],
                        'hd_group2': data['hd_group2'][i],
                        'difference': data['difference'][i],
                        'ci_low': data['ci_low'][i],
                        'ci_high': data['ci_high'][i],
                        'significant': data['significant'][i],
                        'n_group1': data['n_group1'],
                        'n_group2': data['n_group2'],
                        'group1_label': data['group1_label'],
                        'group2_label': data['group2_label'],
                    })

        if rows:
            df = pd.DataFrame(rows)
            out_path = self.output_dir / 'shift_plot_results.csv'
            df.to_csv(out_path, index=False)
            print(f"\n  Saved shift-plot results to: {out_path}")

    def _generate_plots(self):
        """Generate shift-plot visualisations (delegates to ShiftPlotPlotter)."""
        if not self.results:
            return

        if not self.config.get('generate_plots', True):
            return

        try:
            from ..visualization.shift_plot_plotter import ShiftPlotPlotter
            plotter = ShiftPlotPlotter(self.output_dir, self.full_config)
            plotter.generate_all_plots(self.results)
        except ImportError as exc:
            warnings.warn(
                f"ShiftPlotPlotter could not be imported — skipping plot generation. "
                f"({exc})"
            )
        except Exception as exc:
            warnings.warn(f"Plot generation failed: {exc}")
