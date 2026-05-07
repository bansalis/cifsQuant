"""Shared fixtures for cifsQuant smoke tests."""
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

# Make cifsQuant importable from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def minimal_project_yaml():
    """Minimal valid project.yaml dict for config validation tests."""
    return {
        'markers': {
            'DAPI': 'DAPI',
            'Cy3_CD3': 'CD3',
            'Cy5_CD8': 'CD8',
            'FITC_B220': 'B220',
        },
        'marker_hierarchy': {
            'CD8': 'CD3',
        },
        'dapi_channel': 0,
        'nuc_diameter': 12,
        'cyto_diameter': 24,
        'gating': {
            'use_shared_gates': True,
            'normalization_method': 'percentile_99',
            'gates': {
                'DAPI': None,
                'CD3': None,
                'CD8': None,
                'B220': None,
            },
            'tile_correction': {
                'enabled': False,
                'markers': [],
            },
            'liberal_gating': {
                'enabled': False,
                'liberal_markers': [],
            },
        },
        'spatial': {
            'input': {
                'gated_data': 'manual_gating_output/gated_data.h5ad',
                'metadata': 'sample_metadata.csv',
            },
            'output': {
                'base_directory': 'spatial_quantification_results',
                'formats': ['png'],
                'dpi': 300,
            },
            'metadata': {
                'sample_column': 'sample_id',
                'group_column': 'group',
            },
            'tumor_definition': {
                'base_phenotype': 'T_cells',
                'required_positive': ['CD3'],
                'required_negative': [],
                'structure_detection': {
                    'method': 'DBSCAN',
                    'eps': 1000,
                    'min_samples': 500,
                },
            },
            'phenotypes': {
                'T_cells': {
                    'positive': ['CD3'],
                    'negative': [],
                },
                'CD8_T_cells': {
                    'base': 'T_cells',
                    'positive': ['CD8'],
                },
                'B_cells': {
                    'positive': ['B220'],
                },
            },
            'per_tumor_analysis': {'enabled': False},
            'population_dynamics': {'enabled': False},
            'distance_analysis': {'enabled': False},
            'statistics': {
                'test': 'mann_whitney',
                'fdr_method': 'benjamini-hochberg',
                'alpha': 0.05,
            },
            'visualization': {'enabled': False},
            'performance': {'n_jobs': 1},
            'advanced': {'random_seed': 42, 'verbose': False},
        },
    }


@pytest.fixture
def minimal_adata():
    """Small synthetic AnnData with spatial coords and phenotype columns."""
    import anndata

    n_cells = 200
    rng = np.random.default_rng(42)

    # Marker expression matrix
    markers = ['DAPI', 'CD3', 'CD8', 'B220']
    X = rng.uniform(0, 1, size=(n_cells, len(markers)))

    obs = pd.DataFrame({
        'sample_id': ['S1'] * 100 + ['S2'] * 100,
        'group':     ['GroupA'] * 100 + ['GroupB'] * 100,
        'X_centroid': rng.uniform(0, 2000, size=n_cells),
        'Y_centroid': rng.uniform(0, 2000, size=n_cells),
        # Simulated gate booleans
        'is_CD3':  (X[:, 1] > 0.5).astype(bool),
        'is_CD8':  (X[:, 2] > 0.6).astype(bool),
        'is_B220': (X[:, 3] > 0.5).astype(bool),
    })
    obs.index = [f'cell_{i}' for i in range(n_cells)]

    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=markers),
    )
    adata.obsm['spatial'] = np.column_stack([obs['X_centroid'], obs['Y_centroid']])

    return adata


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for analysis output."""
    return tmp_path / 'test_results'
