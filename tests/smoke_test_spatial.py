"""Smoke tests: Stage 3 spatial quantification."""
import sys
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPhenotypeBuilder:

    def test_builds_simple_positive_phenotype(self, minimal_adata):
        """PhenotypeBuilder should create is_T_cells column from positive: [CD3]."""
        from spatial_quantification.core.phenotype_builder import PhenotypeBuilder

        config = {
            'phenotypes': {
                'T_cells': {'positive': ['CD3']},
            }
        }
        builder = PhenotypeBuilder(minimal_adata, config)
        builder.build_all_phenotypes()

        assert 'is_T_cells' in minimal_adata.obs.columns
        # T_cells = is_CD3 (already in adata.obs from fixture)
        expected = minimal_adata.obs['is_CD3'].values
        actual = minimal_adata.obs['is_T_cells'].values
        np.testing.assert_array_equal(expected, actual)

    def test_builds_base_constrained_phenotype(self, minimal_adata):
        """Phenotype with base: must be a subset of the parent phenotype."""
        from spatial_quantification.core.phenotype_builder import PhenotypeBuilder

        config = {
            'phenotypes': {
                'T_cells': {'positive': ['CD3']},
                'CD8_T_cells': {'base': 'T_cells', 'positive': ['CD8']},
            }
        }
        builder = PhenotypeBuilder(minimal_adata, config)
        builder.build_all_phenotypes()

        assert 'is_CD8_T_cells' in minimal_adata.obs.columns
        # CD8_T_cells must be subset of T_cells
        cd8_mask = minimal_adata.obs['is_CD8_T_cells']
        t_mask = minimal_adata.obs['is_T_cells']
        assert (cd8_mask & ~t_mask).sum() == 0, "CD8_T_cells must be subset of T_cells"

    def test_skips_missing_marker_gracefully(self, minimal_adata):
        """Phenotype referencing a nonexistent marker should be skipped, not crash."""
        from spatial_quantification.core.phenotype_builder import PhenotypeBuilder

        config = {
            'phenotypes': {
                'Ghost_cells': {'positive': ['NONEXISTENT_MARKER']},
                'T_cells': {'positive': ['CD3']},
            }
        }
        builder = PhenotypeBuilder(minimal_adata, config)
        # Should not raise
        builder.build_all_phenotypes()
        # Valid phenotype should still be built
        assert 'is_T_cells' in minimal_adata.obs.columns


class TestMetadataManager:

    def test_merges_metadata_into_adata(self, minimal_adata, tmp_path):
        """MetadataManager should add group column to adata.obs."""
        from spatial_quantification.core.metadata_manager import MetadataManager

        meta_csv = tmp_path / 'sample_metadata.csv'
        meta_df = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'group': ['GroupA', 'GroupB'],
        })
        meta_df.to_csv(meta_csv, index=False)

        # MetadataManager requires timepoint_column in config['metadata']
        meta_df['timepoint'] = [0, 4]
        meta_df.to_csv(meta_csv, index=False)

        config = {
            'input': {'metadata': str(meta_csv)},
            'metadata': {
                'sample_column': 'sample_id',
                'group_column': 'group',
                'timepoint_column': 'timepoint',
            },
        }
        manager = MetadataManager(meta_df, config)
        manager.process()
        manager.merge_with_adata(minimal_adata)

        assert 'group' in minimal_adata.obs.columns
        assert set(minimal_adata.obs['group'].unique()) == {'GroupA', 'GroupB'}


class TestPopulationDynamicsInit:

    def test_instantiation_does_not_crash(self, minimal_adata, tmp_path):
        """PopulationDynamics should be instantiatable on synthetic data."""
        from spatial_quantification.analyses.population_dynamics import PopulationDynamics

        # Add required phenotype columns to adata
        minimal_adata.obs['is_T_cells'] = minimal_adata.obs['is_CD3']
        minimal_adata.obs['group'] = ['GroupA'] * 100 + ['GroupB'] * 100

        config = {
            'population_dynamics': {
                'enabled': True,
                'populations': ['T_cells'],
                'comparisons': [{'name': 'A_vs_B', 'groups': ['GroupA', 'GroupB']}],
                'metrics': ['count'],
                'statistics': {'test_method': 'mannwhitneyu', 'alpha': 0.05,
                               'fdr_correction': 'benjamini_hochberg'},
            },
            'metadata': {'sample_column': 'sample_id', 'group_column': 'group'},
        }
        output_dir = tmp_path / 'pop_dynamics'
        output_dir.mkdir()

        # Should not raise on init
        analysis = PopulationDynamics(minimal_adata, config, output_dir)
        assert analysis is not None


class TestDistanceAnalysisInit:

    def test_instantiation_does_not_crash(self, minimal_adata, tmp_path):
        """DistanceAnalysis should be instantiatable on synthetic data."""
        from spatial_quantification.analyses.distance_analysis import DistanceAnalysis

        minimal_adata.obs['is_T_cells'] = minimal_adata.obs['is_CD3']
        minimal_adata.obs['is_B_cells'] = minimal_adata.obs['is_B220']
        minimal_adata.obs['group'] = ['GroupA'] * 100 + ['GroupB'] * 100

        config = {
            'distance_analysis': {
                'enabled': True,
                'pairings': [{'source': 'T_cells', 'targets': ['B_cells']}],
                'metrics': ['mean_nearest_neighbor'],
                'levels': ['per_sample'],
                'comparisons': [{'name': 'A_vs_B', 'groups': ['GroupA', 'GroupB']}],
                'max_distance_plot': 500,
            },
            'metadata': {'sample_column': 'sample_id', 'group_column': 'group'},
            'statistics': {'fdr_correction': 'benjamini_hochberg', 'alpha': 0.05},
        }
        output_dir = tmp_path / 'distance'
        output_dir.mkdir()

        analysis = DistanceAnalysis(minimal_adata, config, output_dir)
        assert analysis is not None


class TestLoadConfig:

    def test_project_yaml_format(self, tmp_path):
        """load_config() extracts spatial subsection from project.yaml format."""
        import yaml
        from spatial_quantification.run_spatial_quantification import load_config

        project = {
            'markers': {'DAPI': 'DAPI'},
            'spatial': {
                'phenotypes': {'T_cells': {'positive': ['CD3']}},
                'input': {'gated_data': 'gated.h5ad', 'metadata': 'meta.csv'},
                'output': {'base_directory': 'results'},
                'statistics': {'test': 'mann_whitney'},
                'visualization': {'enabled': False},
            }
        }
        config_file = tmp_path / 'project.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(project, f)

        config = load_config(str(config_file))
        assert 'phenotypes' in config
        assert 'spatial' not in config  # subsection extracted, not nested

    def test_standalone_spatial_config(self, tmp_path):
        """load_config() accepts flat spatial_config.yaml without 'spatial:' wrapper."""
        import yaml
        from spatial_quantification.run_spatial_quantification import load_config

        standalone = {
            'phenotypes': {'T_cells': {'positive': ['CD3']}},
            'input': {'gated_data': 'gated.h5ad', 'metadata': 'meta.csv'},
            'output': {'base_directory': 'results'},
            'statistics': {'test': 'mann_whitney'},
            'visualization': {'enabled': False},
        }
        config_file = tmp_path / 'spatial_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(standalone, f)

        config = load_config(str(config_file))
        assert 'phenotypes' in config
