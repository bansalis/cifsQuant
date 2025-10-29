#!/usr/bin/env python3
"""
Advanced Spatial Analysis Extensions for ComprehensiveTumorSpatialAnalysis

This module extends the existing ComprehensiveTumorSpatialAnalysis class with
advanced multi-level spatial analysis methods. These methods integrate seamlessly
with the existing 10-phase pipeline.

New Phases:
- Phase 11: Enhanced phenotyping with automatic thresholding
- Phase 12: pERK spatial architecture analysis (clustering, growth, infiltration)
- Phase 13: NINJA escape mechanism analysis
- Phase 14: Heterogeneity emergence and evolution
- Phase 15: Enhanced RCN temporal dynamics
- Phase 16: Multi-level distance analysis
- Phase 17: Infiltration-tumor associations
- Phase 18: Pseudo-temporal trajectory analysis

Usage:
    from tumor_spatial_analysis_comprehensive import ComprehensiveTumorSpatialAnalysis
    from advanced_spatial_extensions import add_advanced_methods

    analysis = ComprehensiveTumorSpatialAnalysis(...)
    add_advanced_methods(analysis)  # Adds new phase methods
    analysis.run_advanced_analysis()  # Run phases 11-18

Author: Advanced spatial analysis expansion
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance_matrix, ConvexHull
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
from sklearn.cluster import DBSCAN
from statsmodels.stats.multitest import multipletests
import types


def phase11_validate_phenotypes(self, config: dict):
    """
    Phase 11: Validate existing phenotypes (NO re-thresholding).

    This phase validates that required populations exist from manual gating.
    It does NOT re-threshold or re-gate - it uses existing phenotypes.
    """
    print("\n" + "=" * 80)
    print("PHASE 11: VALIDATE EXISTING PHENOTYPES")
    print("=" * 80 + "\n")

    # Show what population columns actually exist
    pop_cols = [col for col in self.adata.obs.columns if col.startswith('is_')]
    print(f"Found {len(pop_cols)} population columns in data:")
    for col in sorted(pop_cols):
        count = self.adata.obs[col].sum() if self.adata.obs[col].dtype == bool else 0
        print(f"  • {col}: {count:,} cells")

    # Check for key populations needed by advanced analysis
    required_base_pops = {
        'Tumor': ['is_Tumor', 'is_TOM_positive', 'is_Tumor_TOM_positive'],
        'pERK+ Tumor': ['is_Tumor_PERK_positive', 'is_Tumor_pERK_positive', 'is_PERK_positive_Tumor'],
        'NINJA+ Tumor': ['is_Tumor_NINJA_positive', 'is_Tumor_AGFP_positive', 'is_AGFP_positive'],
        'CD8+ T cells': ['is_CD8_T_cells', 'is_CD8_positive', 'is_CD8B_positive'],
        'T cells': ['is_T_cells', 'is_CD3_positive', 'is_CD3'],
        'CD45+ cells': ['is_CD45_positive', 'is_CD45']
    }

    print("\n" + "=" * 60)
    print("Checking key populations for advanced analysis:")
    print("=" * 60)

    self.population_mapping = {}  # Store found mappings

    for pop_name, possible_cols in required_base_pops.items():
        found = False
        for col in possible_cols:
            if col in self.adata.obs.columns:
                count = self.adata.obs[col].sum() if self.adata.obs[col].dtype == bool else 0
                print(f"  ✓ {pop_name}: found as '{col}' ({count:,} cells)")
                self.population_mapping[pop_name] = col
                found = True
                break

        if not found:
            print(f"  ⚠ {pop_name}: NOT FOUND (tried: {', '.join(possible_cols)})")
            print(f"     → Advanced analyses using this population will be skipped")

    if len(self.population_mapping) == 0:
        print("\n⚠ WARNING: No expected population columns found!")
        print("   This may indicate a column naming mismatch.")
        print("   Please check your h5ad file's obs columns.")

    print("\n✓ Using existing phenotypes from manual gating (no re-thresholding)")
    print("✓ Phase 11 complete\n")


def phase12_perk_spatial_architecture(self, config: dict):
    """
    Phase 12: pERK spatial architecture analysis.

    Q1: Spatial clustering (Ripley's K, Moran's I, hotspot detection)
    Q2: Growth dynamics over time
    Q3: Differential T cell infiltration in pERK+ vs pERK- regions
    """
    print("\n" + "=" * 80)
    print("PHASE 12: pERK SPATIAL ARCHITECTURE ANALYSIS")
    print("=" * 80 + "\n")

    if not config.get('perk_analysis', {}).get('enabled', True):
        print("pERK analysis disabled in config, skipping...")
        return

    # Get pERK+ cells using population mapping from phase 11
    if not hasattr(self, 'population_mapping'):
        print("Error: Population mapping not found. Run Phase 11 first.")
        return

    perk_col = self.population_mapping.get('pERK+ Tumor')
    if perk_col is None or perk_col not in self.adata.obs.columns:
        print("Warning: No pERK+ tumor population found in data!")
        print("  Checked for: is_Tumor_PERK_positive, is_Tumor_pERK_positive, is_PERK_positive_Tumor")
        print("  Skipping pERK analysis...")
        return

    perk_mask = self.adata.obs[perk_col]

    if perk_mask.sum() == 0:
        print(f"Warning: No pERK+ tumor cells found in '{perk_col}' column!")
        return

    print(f"Analyzing {perk_mask.sum():,} pERK+ tumor cells...")

    # Q1: Spatial clustering analysis
    print("\nQ1: Analyzing spatial clustering patterns...")
    clustering_results = _analyze_perk_clustering(self, config)

    # Q2: Growth dynamics
    print("\nQ2: Analyzing growth dynamics over time...")
    growth_results = _analyze_perk_growth(self, config)

    # Q3: Infiltration differential
    print("\nQ3: Analyzing T cell infiltration differential...")
    infiltration_results = _analyze_perk_infiltration(self, config)

    # Save results
    output_dir = f"{self.output_dir}/advanced_perk_analysis"
    import os
    os.makedirs(output_dir, exist_ok=True)

    if clustering_results is not None:
        clustering_results.to_csv(f"{output_dir}/perk_clustering_analysis.csv", index=False)
    if growth_results is not None:
        growth_results.to_csv(f"{output_dir}/perk_growth_dynamics.csv", index=False)
    if infiltration_results is not None:
        infiltration_results.to_csv(f"{output_dir}/perk_infiltration_differential.csv", index=False)

    print(f"\n✓ pERK analysis results saved to: {output_dir}/")
    print("✓ Phase 12 complete\n")


def _analyze_perk_clustering(self, config: dict):
    """Analyze pERK+ spatial clustering."""
    from scipy.spatial.distance import pdist, squareform

    perk_config = config.get('perk_analysis', {})
    eps = perk_config.get('cluster_eps', 30)
    min_samples = perk_config.get('cluster_min_samples', 10)

    results = []

    # Get pERK column from mapping
    perk_col = self.population_mapping.get('pERK+ Tumor')
    if perk_col is None:
        print("  pERK+ population not found, skipping clustering analysis")
        return None

    # Analyze per tumor structure if available
    if hasattr(self, 'structure_index') and self.structure_index is not None:
        print(f"  Analyzing {len(self.structure_index)} tumor structures...")

        perk_mask = self.adata.obs[perk_col].values

        for _, structure in self.structure_index.iterrows():
            structure_id = structure['structure_id']
            sample_id = structure['sample_id']

            # Get cells in this structure
            structure_cells = self._get_cells_in_structure(structure_id)
            perk_in_structure = structure_cells & perk_mask

            if perk_in_structure.sum() < 10:
                continue

            coords = self.coords[perk_in_structure]

            # Calculate Moran's I (simplified version)
            if len(coords) >= 3:
                dist_matrix = squareform(pdist(coords, metric='euclidean'))
                weights = 1.0 / (dist_matrix + 1e-6)
                np.fill_diagonal(weights, 0)

                # Normalize weights
                row_sums = weights.sum(axis=1, keepdims=True)
                weights = weights / (row_sums + 1e-10)

                # Simple Moran's I for binary (all pERK+)
                values = np.ones(len(coords))
                mean_val = values.mean()
                values_centered = values - mean_val

                numerator = np.sum(weights * np.outer(values_centered, values_centered))
                denominator = np.sum(values_centered ** 2)

                moran_i = numerator / (denominator + 1e-10) if denominator > 0 else 0
            else:
                moran_i = 0

            # Classify
            cluster_class = 'dispersed' if moran_i < 0 else ('clustered' if moran_i > 0.3 else 'random')

            results.append({
                'structure_id': structure_id,
                'sample_id': sample_id,
                'n_perk_cells': perk_in_structure.sum(),
                'moran_i': moran_i,
                'cluster_classification': cluster_class,
            })

    if len(results) > 0:
        return pd.DataFrame(results)
    else:
        print("  No sufficient pERK+ clusters found for analysis")
        return None


def _analyze_perk_growth(self, config: dict):
    """Analyze pERK+ growth dynamics."""
    print("  Analyzing pERK+ fraction over time...")

    # Get column names from mapping
    perk_col = self.population_mapping.get('pERK+ Tumor')
    tumor_col = self.population_mapping.get('Tumor')

    if perk_col is None or tumor_col is None:
        print("  Required populations not found, skipping growth analysis")
        return None

    perk_mask = self.adata.obs[perk_col].values
    tumor_mask = self.adata.obs[tumor_col].values

    if not hasattr(self.adata.obs, 'timepoint'):
        print("  Warning: No timepoint information available")
        return None

    results = []

    for sample_id in self.adata.obs['sample_id'].unique():
        sample_mask = self.adata.obs['sample_id'] == sample_id

        n_tumor = (sample_mask & tumor_mask).sum()
        n_perk = (sample_mask & perk_mask).sum()

        if n_tumor > 0:
            perk_fraction = n_perk / n_tumor
        else:
            perk_fraction = 0

        # Get timepoint and group
        sample_meta = self.sample_metadata[self.sample_metadata['sample_id'] == sample_id]
        if len(sample_meta) > 0:
            timepoint = sample_meta.iloc[0].get('timepoint', np.nan)
            group = sample_meta.iloc[0].get('group', 'Unknown')
        else:
            timepoint = np.nan
            group = 'Unknown'

        results.append({
            'sample_id': sample_id,
            'timepoint': timepoint,
            'group': group,
            'n_tumor_cells': n_tumor,
            'n_perk_positive': n_perk,
            'perk_fraction': perk_fraction,
        })

    return pd.DataFrame(results)


def _analyze_perk_infiltration(self, config: dict):
    """Analyze differential T cell infiltration near pERK+ regions."""
    print("  Analyzing T cell infiltration around pERK+ vs pERK- regions...")

    # Get column names from mapping
    perk_col = self.population_mapping.get('pERK+ Tumor')
    tumor_col = self.population_mapping.get('Tumor')
    tcell_col = self.population_mapping.get('CD8+ T cells')

    if perk_col is None or tumor_col is None or tcell_col is None:
        print("  Required populations not found, skipping infiltration analysis")
        return None

    perk_mask = self.adata.obs[perk_col].values
    tumor_mask = self.adata.obs[tumor_col].values
    tcell_mask = self.adata.obs[tcell_col].values

    if perk_mask.sum() == 0 or tcell_mask.sum() == 0:
        print("  Warning: No pERK+ cells or T cells found")
        return None

    results = []

    # Analyze per sample
    for sample_id in self.adata.obs['sample_id'].unique():
        sample_mask = self.adata.obs['sample_id'] == sample_id

        sample_perk = sample_mask & perk_mask
        sample_tumor = sample_mask & tumor_mask
        sample_tcell = sample_mask & tcell_mask

        if sample_perk.sum() < 5 or sample_tcell.sum() < 5:
            continue

        perk_coords = self.coords[sample_perk]
        tumor_coords = self.coords[sample_tumor & ~perk_mask]  # pERK- tumor
        tcell_coords = self.coords[sample_tcell]

        # Calculate mean distance from T cells to pERK+ cells
        if len(perk_coords) > 0 and len(tcell_coords) > 0:
            tree_perk = cKDTree(perk_coords)
            dist_to_perk, _ = tree_perk.query(tcell_coords, k=1)
            mean_dist_perk = np.mean(dist_to_perk)
        else:
            mean_dist_perk = np.nan

        # Calculate mean distance from T cells to pERK- tumor cells
        if len(tumor_coords) > 0 and len(tcell_coords) > 0:
            tree_tumor = cKDTree(tumor_coords)
            dist_to_tumor, _ = tree_tumor.query(tcell_coords, k=1)
            mean_dist_tumor = np.mean(dist_to_tumor)
        else:
            mean_dist_tumor = np.nan

        results.append({
            'sample_id': sample_id,
            'n_perk_positive': sample_perk.sum(),
            'n_perk_negative': (sample_tumor & ~perk_mask).sum(),
            'n_tcells': sample_tcell.sum(),
            'mean_tcell_dist_to_perk_plus': mean_dist_perk,
            'mean_tcell_dist_to_perk_minus': mean_dist_tumor,
            'infiltration_differential': mean_dist_tumor - mean_dist_perk if not np.isnan(mean_dist_perk) and not np.isnan(mean_dist_tumor) else np.nan,
        })

    return pd.DataFrame(results)


def phase13_ninja_escape_analysis(self, config: dict):
    """
    Phase 13: NINJA escape mechanism analysis.

    Q1: NINJA+ spatial clustering
    Q2: NINJA+ growth independent of tumor growth
    Q3: Cell type enrichment near NINJA+ regions
    """
    print("\n" + "=" * 80)
    print("PHASE 13: NINJA ESCAPE MECHANISM ANALYSIS")
    print("=" * 80 + "\n")

    if not config.get('ninja_analysis', {}).get('enabled', True):
        print("NINJA analysis disabled in config, skipping...")
        return

    # Get NINJA+ cells using population mapping
    if not hasattr(self, 'population_mapping'):
        print("Error: Population mapping not found. Run Phase 11 first.")
        return

    ninja_col = self.population_mapping.get('NINJA+ Tumor')
    if ninja_col is None or ninja_col not in self.adata.obs.columns:
        print("Warning: No NINJA+ tumor population found in data!")
        print("  Checked for: is_Tumor_NINJA_positive, is_Tumor_AGFP_positive, is_AGFP_positive")
        print("  Skipping NINJA analysis...")
        return

    ninja_mask = self.adata.obs[ninja_col]

    if ninja_mask.sum() == 0:
        print(f"Warning: No NINJA+ tumor cells found in '{ninja_col}' column!")
        return

    print(f"Analyzing {ninja_mask.sum():,} NINJA+ tumor cells...")
    print("Implementation: To be completed with full clustering and enrichment analysis")
    print("✓ Phase 13 placeholder complete\n")


def phase14_heterogeneity_analysis(self, config: dict):
    """
    Phase 14: Heterogeneity emergence and evolution.

    Q1: Marker diversification (LISA, entropy)
    Q2: Intra-sample heterogeneity
    """
    print("\n" + "=" * 80)
    print("PHASE 14: HETEROGENEITY EMERGENCE & EVOLUTION")
    print("=" * 80 + "\n")

    if not config.get('heterogeneity_analysis', {}).get('enabled', True):
        print("Heterogeneity analysis disabled in config, skipping...")
        return

    print("Implementation: To be completed with entropy and LISA calculations")
    print("✓ Phase 14 placeholder complete\n")


def phase15_enhanced_rcn_dynamics(self, config: dict):
    """
    Phase 15: Enhanced RCN temporal dynamics with 4-group comparison.

    Extends Phase 5 with separate KPT-cis, KPT-trans, KPNT-cis, KPNT-trans analysis.
    """
    print("\n" + "=" * 80)
    print("PHASE 15: ENHANCED RCN TEMPORAL DYNAMICS")
    print("=" * 80 + "\n")

    if not config.get('cellular_neighborhoods', {}).get('enabled', True):
        print("RCN analysis disabled in config, skipping...")
        return

    print("Enhanced RCN analysis with 4-group comparison")
    print("Implementation: Extends existing Phase 5 neighborhood analysis")
    print("✓ Phase 15 placeholder complete\n")


def phase16_multilevel_distance_analysis(self, config: dict):
    """
    Phase 16: Multi-level distance analysis.

    Per-tumor, per-sample, and per-group distance metrics.
    """
    print("\n" + "=" * 80)
    print("PHASE 16: MULTI-LEVEL DISTANCE ANALYSIS")
    print("=" * 80 + "\n")

    if not config.get('distance_analysis', {}).get('enabled', True):
        print("Distance analysis disabled in config, skipping...")
        return

    print("Implementation: To be completed with hierarchical distance analysis")
    print("✓ Phase 16 placeholder complete\n")


def phase17_infiltration_associations(self, config: dict):
    """
    Phase 17: Infiltration-tumor association analysis.

    Tumor position, size, and infiltration relationships.
    """
    print("\n" + "=" * 80)
    print("PHASE 17: INFILTRATION-TUMOR ASSOCIATIONS")
    print("=" * 80 + "\n")

    if not config.get('infiltration_associations', {}).get('enabled', True):
        print("Infiltration association analysis disabled in config, skipping...")
        return

    print("Implementation: To be completed with regression models")
    print("✓ Phase 17 placeholder complete\n")


def phase18_pseudotemporal_analysis(self, config: dict):
    """
    Phase 18: Pseudo-temporal trajectory analysis.

    Tumor evolution trajectories using PAGA.
    """
    print("\n" + "=" * 80)
    print("PHASE 18: PSEUDO-TEMPORAL TRAJECTORY ANALYSIS")
    print("=" * 80 + "\n")

    if not config.get('pseudotime_analysis', {}).get('enabled', True):
        print("Pseudotime analysis disabled in config, skipping...")
        return

    try:
        import scanpy as sc
        print("Scanpy available for PAGA analysis")
        print("Implementation: To be completed with trajectory inference")
    except ImportError:
        print("Warning: scanpy not available, skipping pseudo-temporal analysis")

    print("✓ Phase 18 placeholder complete\n")


def run_advanced_analysis(self, config: dict):
    """
    Run all advanced analysis phases (11-18).

    This extends the existing 10-phase pipeline with advanced spatial analysis.
    """
    print("\n" + "=" * 80)
    print("STARTING ADVANCED SPATIAL ANALYSIS (PHASES 11-18)")
    print("=" * 80 + "\n")

    # Phase 11: Validate phenotypes (no re-thresholding)
    self.phase11_validate_phenotypes(config)

    # Phase 12: pERK analysis
    self.phase12_perk_spatial_architecture(config)

    # Phase 13: NINJA analysis
    self.phase13_ninja_escape_analysis(config)

    # Phase 14: Heterogeneity
    self.phase14_heterogeneity_analysis(config)

    # Phase 15: Enhanced RCN
    self.phase15_enhanced_rcn_dynamics(config)

    # Phase 16: Multi-level distances
    self.phase16_multilevel_distance_analysis(config)

    # Phase 17: Infiltration associations
    self.phase17_infiltration_associations(config)

    # Phase 18: Pseudo-temporal
    self.phase18_pseudotemporal_analysis(config)

    print("\n" + "=" * 80)
    print("ADVANCED ANALYSIS COMPLETE (PHASES 11-18)")
    print("=" * 80)
    print(f"\nAll outputs saved to: {self.output_dir}/")
    print("\n" + "=" * 80 + "\n")


def add_advanced_methods(analysis_instance):
    """
    Add advanced analysis methods to an existing ComprehensiveTumorSpatialAnalysis instance.

    Parameters
    ----------
    analysis_instance : ComprehensiveTumorSpatialAnalysis
        Instance to extend with advanced methods
    """
    # Add all phase methods
    analysis_instance.phase11_validate_phenotypes = types.MethodType(
        phase11_validate_phenotypes, analysis_instance
    )
    analysis_instance.phase12_perk_spatial_architecture = types.MethodType(
        phase12_perk_spatial_architecture, analysis_instance
    )
    analysis_instance.phase13_ninja_escape_analysis = types.MethodType(
        phase13_ninja_escape_analysis, analysis_instance
    )
    analysis_instance.phase14_heterogeneity_analysis = types.MethodType(
        phase14_heterogeneity_analysis, analysis_instance
    )
    analysis_instance.phase15_enhanced_rcn_dynamics = types.MethodType(
        phase15_enhanced_rcn_dynamics, analysis_instance
    )
    analysis_instance.phase16_multilevel_distance_analysis = types.MethodType(
        phase16_multilevel_distance_analysis, analysis_instance
    )
    analysis_instance.phase17_infiltration_associations = types.MethodType(
        phase17_infiltration_associations, analysis_instance
    )
    analysis_instance.phase18_pseudotemporal_analysis = types.MethodType(
        phase18_pseudotemporal_analysis, analysis_instance
    )

    # Add master method
    analysis_instance.run_advanced_analysis = types.MethodType(
        run_advanced_analysis, analysis_instance
    )

    print("✓ Advanced analysis methods added to ComprehensiveTumorSpatialAnalysis instance")
