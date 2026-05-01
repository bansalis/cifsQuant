"""
pERK MFI (Mean Fluorescence Intensity) Analysis

Analyzes pERK expression level (raw + normalized) from adata.layers and tests
whether pERK MFI correlates with spatial proximity to CD8 T cells.

Data sources:
    adata.layers['raw']        - raw intensity pre-normalization
    adata.layers['normalized'] - per-marker 99th-percentile scaled to 0-1
    adata.X                    - copy of normalized (thresholds applied here)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scipy.spatial import cKDTree
import warnings


class PerkMFIAnalysis:
    """
    Analyze pERK mean fluorescence intensity and its spatial relationship with T cells.

    Analyses:
    1. pERK MFI distribution in pERK+ vs pERK- (gated) cells per group
    2. pERK MFI range (IQR, CV) per sample to assess gating consistency
    3. pERK MFI vs distance to nearest CD8 T cell
    4. pERK MFI vs CD8 T cell density within a radius window
    5. Per-tumor median pERK MFI vs per-tumor CD8 count
    """

    def __init__(self, adata, config: Dict, output_dir: Path):
        self.adata = adata
        self.config = config
        self.mfi_config = config.get('perk_mfi_analysis', {})
        self.output_dir = Path(output_dir) / 'perk_mfi_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.marker_name = self.mfi_config.get('marker_name', 'PERK')
        self.tumor_phenotype = self.mfi_config.get('tumor_phenotype', 'Tumor')
        self.tcell_pops = self.mfi_config.get('tcell_populations', ['CD8_T_cells', 'T_cells'])
        self.window_radius = self.mfi_config.get('window_radius', 50)
        self.distance_bins = self.mfi_config.get('distance_bins', [0, 25, 50, 100, 200, 500])

        meta_config = config.get('metadata', {})
        self.group_col = (meta_config.get('primary_grouping') or
                          meta_config.get('group_column', 'group'))

        self.results = {}

    def run(self) -> Dict:
        """Run complete pERK MFI analysis."""
        print("\n" + "="*80)
        print("pERK MFI ANALYSIS")
        print("="*80)

        # Resolve pERK index in var_names
        perk_idx = self._get_marker_index()
        if perk_idx is None:
            print(f"  ⚠ Marker '{self.marker_name}' not found in adata.var_names; skipping")
            return {}

        # Extract MFI values for all cells
        print(f"\n1. Extracting MFI values (marker index={perk_idx})...")
        cell_df = self._build_cell_df(perk_idx)
        if cell_df is None or len(cell_df) == 0:
            print("  ⚠ No cells extracted; skipping")
            return {}
        print(f"    ✓ Extracted {len(cell_df)} cells")

        # 1. MFI distribution in gated+ vs gated- cells
        print("\n2. Comparing MFI distributions (gated+ vs gated-)...")
        mfi_dist = self._mfi_distribution_by_gate(cell_df)
        if mfi_dist is not None:
            self.results['mfi_distribution_by_gate'] = mfi_dist

        # 2. Per-sample MFI range (IQR, CV)
        print("\n3. Computing per-sample MFI statistics...")
        sample_stats = self._per_sample_mfi_stats(cell_df)
        if sample_stats is not None:
            self.results['per_sample_mfi_stats'] = sample_stats

        # 3 & 4. pERK MFI vs T cell proximity (distance + density)
        print("\n4. Computing T cell proximity metrics...")
        proximity_df = self._compute_tcell_proximity(cell_df)
        if proximity_df is not None:
            self.results['mfi_vs_tcell_proximity'] = proximity_df
            print(f"    ✓ Proximity metrics for {len(proximity_df)} tumor cells")

        # 5. Per-tumor summary: median pERK MFI vs CD8 count
        print("\n5. Computing per-tumor summary...")
        per_tumor = self._per_tumor_summary(cell_df)
        if per_tumor is not None:
            self.results['per_tumor_mfi_summary'] = per_tumor

        # Save results
        self._save_results()

        # Generate plots
        print("\n6. Generating plots...")
        try:
            from spatial_quantification.visualization.perk_mfi_plotter import PerkMFIPlotter
            plotter = PerkMFIPlotter(self.output_dir, self.config)
            plotter.generate_all_plots(self.results)
        except Exception as e:
            print(f"  ⚠ Could not generate pERK MFI plots: {e}")

        print(f"\n✓ pERK MFI analysis complete. Results: {self.output_dir}/")
        print("="*80 + "\n")
        return self.results

    def _get_marker_index(self) -> Optional[int]:
        """Get index of marker in adata.var_names."""
        var_names = list(self.adata.var_names)
        # Exact match first
        if self.marker_name in var_names:
            return var_names.index(self.marker_name)
        # Case-insensitive
        lower_names = [v.lower() for v in var_names]
        target_lower = self.marker_name.lower()
        if target_lower in lower_names:
            return lower_names.index(target_lower)
        return None

    def _build_cell_df(self, perk_idx: int) -> Optional[pd.DataFrame]:
        """
        Build a per-cell DataFrame with pERK MFI (raw + normalized),
        phenotype columns, spatial coordinates, and metadata.
        """
        obs = self.adata.obs.copy()

        # Extract raw MFI
        if 'raw' in self.adata.layers:
            raw_layer = self.adata.layers['raw']
            obs['perk_mfi_raw'] = np.asarray(raw_layer[:, perk_idx]).ravel()
        else:
            obs['perk_mfi_raw'] = np.nan

        # Extract normalized MFI
        if 'normalized' in self.adata.layers:
            norm_layer = self.adata.layers['normalized']
            obs['perk_mfi_normalized'] = np.asarray(norm_layer[:, perk_idx]).ravel()
        else:
            # Fall back to adata.X
            import scipy.sparse as sp
            X = self.adata.X
            if sp.issparse(X):
                X = X.toarray()
            obs['perk_mfi_normalized'] = np.asarray(X[:, perk_idx]).ravel()

        # Add spatial coordinates
        if 'spatial' in self.adata.obsm:
            obs['x'] = self.adata.obsm['spatial'][:, 0]
            obs['y'] = self.adata.obsm['spatial'][:, 1]
        elif 'X_centroid' in obs.columns and 'Y_centroid' in obs.columns:
            obs['x'] = obs['X_centroid']
            obs['y'] = obs['Y_centroid']
        else:
            obs['x'] = np.nan
            obs['y'] = np.nan

        # Restrict to tumor cells
        tumor_col = f'is_{self.tumor_phenotype}'
        if tumor_col in obs.columns:
            obs = obs[obs[tumor_col].astype(bool)].copy()

        if len(obs) == 0:
            return None

        return obs

    def _mfi_distribution_by_gate(self, cell_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Summarize pERK MFI distributions per group/sample for gated+ vs gated- cells."""
        perk_gate_col = 'is_pERK_positive_tumor'
        if perk_gate_col not in cell_df.columns:
            # Try alternative column names
            candidates = [c for c in cell_df.columns if 'pERK_positive' in c or 'PERK_positive' in c]
            if not candidates:
                print("    ⚠ pERK gating column not found")
                return None
            perk_gate_col = candidates[0]

        rows = []
        for sample in cell_df['sample_id'].unique():
            sdata = cell_df[cell_df['sample_id'] == sample]
            group = sdata[self.group_col].iloc[0] if self.group_col in sdata.columns else ''
            timepoint = sdata['timepoint'].iloc[0] if 'timepoint' in sdata.columns else np.nan

            for gate_status, gate_label in [(True, 'positive'), (False, 'negative')]:
                gate_mask = sdata[perk_gate_col].astype(bool) == gate_status
                gdata = sdata[gate_mask]
                if len(gdata) == 0:
                    continue

                for mfi_col in ['perk_mfi_raw', 'perk_mfi_normalized']:
                    vals = gdata[mfi_col].dropna().values
                    if len(vals) == 0:
                        continue
                    rows.append({
                        'sample_id': sample,
                        'group': group,
                        'timepoint': timepoint,
                        'gate_status': gate_label,
                        'mfi_type': mfi_col.replace('perk_mfi_', ''),
                        'n_cells': len(vals),
                        'mean': np.mean(vals),
                        'median': np.median(vals),
                        'std': np.std(vals),
                        'q25': np.percentile(vals, 25),
                        'q75': np.percentile(vals, 75),
                        'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
                        'cv': np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else np.nan,
                    })

        if rows:
            df = pd.DataFrame(rows)
            print(f"    ✓ MFI distributions for {df['sample_id'].nunique()} samples")
            return df
        return None

    def _per_sample_mfi_stats(self, cell_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Per-sample pERK MFI statistics for consistency assessment."""
        perk_col = 'is_pERK_positive_tumor'
        if perk_col not in cell_df.columns:
            return None

        rows = []
        for sample in cell_df['sample_id'].unique():
            sdata = cell_df[cell_df['sample_id'] == sample]
            pos_data = sdata[sdata[perk_col].astype(bool)]
            group = sdata[self.group_col].iloc[0] if self.group_col in sdata.columns else ''
            timepoint = sdata['timepoint'].iloc[0] if 'timepoint' in sdata.columns else np.nan

            row = {'sample_id': sample, 'group': group, 'timepoint': timepoint,
                   'n_perk_positive': int(sdata[perk_col].sum()),
                   'n_tumor_cells': len(sdata)}

            for mfi_type in ['raw', 'normalized']:
                col = f'perk_mfi_{mfi_type}'
                if col not in pos_data.columns:
                    continue
                vals = pos_data[col].dropna().values
                if len(vals) == 0:
                    continue
                iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
                cv = np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else np.nan
                row[f'{mfi_type}_median'] = np.median(vals)
                row[f'{mfi_type}_iqr'] = iqr
                row[f'{mfi_type}_cv'] = cv
                row[f'{mfi_type}_p95'] = np.percentile(vals, 95)

            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def _compute_tcell_proximity(self, cell_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        For each tumor cell, compute:
        - Distance to nearest T cell (for each configured T cell population)
        - T cell density within window_radius µm

        Uses vectorized numpy operations — no row-by-row iteration.
        """
        if cell_df['x'].isna().all():
            return None

        result_frames = []

        for sample in cell_df['sample_id'].unique():
            sdata = cell_df[cell_df['sample_id'] == sample].copy()
            tumor_coords = sdata[['x', 'y']].values

            if len(tumor_coords) == 0:
                continue

            group = sdata[self.group_col].iloc[0] if self.group_col in sdata.columns else ''
            main_group = sdata['main_group'].iloc[0] if 'main_group' in sdata.columns else group
            timepoint = sdata['timepoint'].iloc[0] if 'timepoint' in sdata.columns else np.nan

            # All sample cells for T cell lookup (not restricted to tumor)
            sample_obs_mask = self.adata.obs['sample_id'] == sample
            all_sample_obs = self.adata.obs[sample_obs_mask]
            if 'spatial' in self.adata.obsm:
                all_sample_coords = self.adata.obsm['spatial'][sample_obs_mask.values]
            else:
                all_sample_coords = all_sample_obs[['X_centroid', 'Y_centroid']].values

            # Build result frame with metadata columns
            frame = pd.DataFrame({
                'sample_id': sample,
                'group': group,
                'main_group': main_group,
                'timepoint': timepoint,
                'perk_mfi_raw': sdata['perk_mfi_raw'].values,
                'perk_mfi_normalized': sdata['perk_mfi_normalized'].values,
            })

            # Boolean gate column (safe access)
            gate_col = 'is_pERK_positive_tumor'
            if gate_col in sdata.columns:
                frame['is_perk_positive'] = sdata[gate_col].values.astype(float)
            else:
                frame['is_perk_positive'] = np.nan

            if 'tumor_region_id' in sdata.columns:
                frame['tumor_region_id'] = sdata['tumor_region_id'].values

            # Vectorized T cell proximity
            for pop in self.tcell_pops:
                col = f'is_{pop}'
                if col not in all_sample_obs.columns:
                    continue
                tcell_mask = all_sample_obs[col].values.astype(bool)
                if tcell_mask.sum() == 0:
                    frame[f'dist_to_nearest_{pop}'] = np.nan
                    frame[f'n_{pop}_within_{self.window_radius}um'] = 0
                    continue
                tcell_coords = all_sample_coords[tcell_mask]
                tree = cKDTree(tcell_coords)
                nn_dists, _ = tree.query(tumor_coords, k=1)
                counts_in_radius = np.array(
                    tree.query_ball_point(tumor_coords, r=self.window_radius, return_length=True)
                )
                frame[f'dist_to_nearest_{pop}'] = nn_dists
                frame[f'n_{pop}_within_{self.window_radius}um'] = counts_in_radius

            result_frames.append(frame)

        if result_frames:
            return pd.concat(result_frames, ignore_index=True)
        return None

    def _per_tumor_summary(self, cell_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Per-tumor median pERK MFI vs immune cell counts."""
        if 'tumor_region_id' not in cell_df.columns:
            return None

        perk_col = 'is_pERK_positive_tumor'
        rows = []

        for sample in cell_df['sample_id'].unique():
            sdata = cell_df[cell_df['sample_id'] == sample]
            group = sdata[self.group_col].iloc[0] if self.group_col in sdata.columns else ''
            timepoint = sdata['timepoint'].iloc[0] if 'timepoint' in sdata.columns else np.nan

            tumor_ids = sdata['tumor_region_id'].dropna().unique()
            tumor_ids = [t for t in tumor_ids if t != -1]

            for tid in tumor_ids:
                tdata = sdata[sdata['tumor_region_id'] == tid]
                if len(tdata) < 10:
                    continue

                row = {
                    'sample_id': sample,
                    'tumor_id': tid,
                    'group': group,
                    'timepoint': timepoint,
                    'n_tumor_cells': len(tdata),
                }

                if perk_col in tdata.columns:
                    pos_mask = tdata[perk_col].astype(bool)
                    pos_vals_raw = tdata.loc[pos_mask, 'perk_mfi_raw'].dropna()
                    pos_vals_norm = tdata.loc[pos_mask, 'perk_mfi_normalized'].dropna()
                    row['n_perk_positive'] = int(pos_mask.sum())
                    row['pct_perk_positive'] = pos_mask.sum() / len(tdata) * 100
                    row['median_perk_mfi_raw'] = np.median(pos_vals_raw) if len(pos_vals_raw) > 0 else np.nan
                    row['median_perk_mfi_normalized'] = np.median(pos_vals_norm) if len(pos_vals_norm) > 0 else np.nan

                # Count T cells in same tumor region
                sample_obs_mask = self.adata.obs['sample_id'] == sample
                all_obs = self.adata.obs[sample_obs_mask]
                tumor_region_mask = all_obs.get('tumor_region_id', pd.Series()) == tid

                for pop in self.tcell_pops:
                    col = f'is_{pop}'
                    if col in all_obs.columns:
                        row[f'n_{pop}_in_tumor'] = int(
                            (all_obs[col].values.astype(bool) & tumor_region_mask.values).sum()
                        )

                rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def _save_results(self):
        """Save all results to CSV."""
        for name, df in self.results.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                path = self.output_dir / f'{name}.csv'
                df.to_csv(path, index=False)
        print(f"  ✓ Saved {len(self.results)} pERK MFI datasets to {self.output_dir}/")
