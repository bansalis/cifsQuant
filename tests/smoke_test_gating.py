"""Smoke tests: Stage 2 manual_gating.py logic."""
import sys
import os
import numpy as np
import pandas as pd
import pytest
import anndata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_minimal_adata(markers=None, n_cells=200, seed=42):
    """Synthetic AnnData with raw intensities (pre-normalization)."""
    if markers is None:
        markers = ['DAPI', 'CD3', 'CD8', 'B220']
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5000, size=(n_cells, len(markers))).astype(np.float32)
    obs = pd.DataFrame({
        'sample_id': [f'S{i % 2 + 1}' for i in range(n_cells)],
        'X_centroid': rng.uniform(0, 2000, n_cells),
        'Y_centroid': rng.uniform(0, 2000, n_cells),
    })
    obs.index = [f'cell_{i}' for i in range(n_cells)]
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=markers))
    return adata


class TestNormalization:

    def test_percentile99_normalization_scales_to_unit(self):
        """normalize_data() should scale intensities so most values land in [0, 1]."""
        from manual_gating import normalize_data

        adata = _make_minimal_adata()
        normalize_data(adata, method='percentile_99')

        normalized = adata.layers['normalized']
        assert normalized.max() <= 1.05, 'Expected max ≈ 1.0 after p99 clip'
        assert normalized.min() >= 0.0, 'Expected min >= 0 after clipping'

    def test_normalization_preserves_shape(self):
        """normalize_data() must not change adata dimensions."""
        from manual_gating import normalize_data

        adata = _make_minimal_adata()
        n_cells, n_vars = adata.shape
        normalize_data(adata, method='percentile_99')
        assert adata.layers['normalized'].shape == (n_cells, n_vars)

    def test_normalization_stores_raw_layer(self):
        """normalize_data() must write a 'raw' layer preserving original values."""
        from manual_gating import normalize_data

        adata = _make_minimal_adata()
        original_X = adata.X.copy()
        normalize_data(adata, method='percentile_99')
        assert np.allclose(adata.layers['raw'], original_X)

    def test_normalize_column_percentile99(self):
        """Column-level helper mirrors the production p99 scaling."""
        from manual_gating import normalize_column
        values = pd.Series(np.random.default_rng(1).uniform(0, 5000, 200))
        normalized = normalize_column(values, method='percentile_99')
        assert normalized.max() <= 1.01
        assert normalized.min() >= 0.0

    def test_normalize_column_zscore(self):
        """zscore normalization should produce ~N(0,1) values."""
        from manual_gating import normalize_column
        values = pd.Series(np.random.default_rng(2).normal(100, 20, 500))
        normalized = normalize_column(values, method='zscore')
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1


class TestGateApplication:

    def _make_gated_adata(self, threshold_dict=None, n_cells=100, seed=7):
        """Synthetic adata with layers['aligned'] for gate testing."""
        markers = ['CD3', 'CD8']
        rng = np.random.default_rng(seed)
        aligned = rng.uniform(0, 1, size=(n_cells, len(markers))).astype(np.float32)
        obs = pd.DataFrame({
            'sample_id': [f'S{i % 2 + 1}' for i in range(n_cells)],
        })
        obs.index = [f'cell_{i}' for i in range(n_cells)]
        adata = anndata.AnnData(
            X=aligned.copy(), obs=obs, var=pd.DataFrame(index=markers)
        )
        adata.layers['aligned'] = aligned
        return adata

    def test_gate_at_zero_threshold(self):
        """threshold=0 gates all strictly-positive values; zero-intensity cells
        stay negative (production gating in apply_gates() is strictly >)."""
        from manual_gating import apply_gate
        values = pd.Series([0.0, 0.1, 0.5])
        result = apply_gate(values, threshold=0.0)
        assert list(result) == [False, True, True]

    def test_apply_gates_creates_gated_layer(self):
        """apply_gates() should produce a binary 'gated' layer."""
        from manual_gating import apply_gates

        adata = self._make_gated_adata()
        gates = {'CD3': 0.5, 'CD8': 0.5}
        apply_gates(adata, gates)

        assert 'gated' in adata.layers
        unique_vals = np.unique(adata.layers['gated'])
        assert set(unique_vals).issubset({0.0, 1.0}), 'Gated layer must be binary'

    def test_high_threshold_gates_nothing(self):
        """Gate at 1.0 should yield no positive cells (all aligned values < 1.0)."""
        from manual_gating import apply_gates

        adata = self._make_gated_adata()
        gates = {'CD3': 1.0, 'CD8': 1.0}
        apply_gates(adata, gates)

        assert adata.layers['gated'].sum() == 0

    def test_low_threshold_gates_all(self):
        """Gate at 0.0 should yield all positive cells (all aligned > 0)."""
        from manual_gating import apply_gates

        adata = self._make_gated_adata()
        gates = {'CD3': 0.0, 'CD8': 0.0}
        apply_gates(adata, gates)

        assert adata.layers['gated'].sum() == adata.n_obs * adata.n_vars

    def test_gate_threshold_is_exclusive(self):
        """Cells at exactly the threshold should be gated negative (strictly greater than)."""
        from manual_gating import apply_gates

        markers = ['CD3']
        obs = pd.DataFrame({'sample_id': ['S1', 'S1', 'S1']})
        obs.index = ['c0', 'c1', 'c2']
        aligned = np.array([[0.3], [0.5], [0.7]], dtype=np.float32)
        adata = anndata.AnnData(X=aligned.copy(), obs=obs, var=pd.DataFrame(index=markers))
        adata.layers['aligned'] = aligned
        apply_gates(adata, {'CD3': 0.5})
        # 0.3 → negative, 0.5 → negative (not strictly greater), 0.7 → positive
        assert list(adata.layers['gated'][:, 0]) == [0.0, 0.0, 1.0]


class TestLoadProjectConfig:

    def test_load_project_config_overrides_markers(self, tmp_path):
        """load_project_config() should update module-level MARKERS dict."""
        import yaml
        import manual_gating as mg

        project = {
            'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3'},
            'marker_hierarchy': {},
            'gating': {
                'use_shared_gates': True,
                'normalization_method': 'percentile_99',
                'gates': {'DAPI': None, 'CD3': None},
                'tile_correction': {'enabled': False, 'markers': []},
                'liberal_gating': {'enabled': False, 'liberal_markers': []},
            },
        }
        config_file = tmp_path / 'project.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(project, f)

        mg.load_project_config(str(config_file))

        assert 'Cy3_CD3' in mg.MARKERS
        assert mg.MARKERS['Cy3_CD3'] == 'CD3'

    def test_load_project_config_overrides_gates(self, tmp_path):
        """load_project_config() should update GATES dict."""
        import yaml
        import manual_gating as mg

        project = {
            'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3'},
            'marker_hierarchy': {},
            'gating': {
                'use_shared_gates': True,
                'normalization_method': 'percentile_99',
                'gates': {'DAPI': None, 'CD3': 0.42},
                'tile_correction': {'enabled': False, 'markers': []},
                'liberal_gating': {'enabled': False, 'liberal_markers': []},
            },
        }
        config_file = tmp_path / 'project.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(project, f)

        mg.load_project_config(str(config_file))

        assert mg.GATES.get('CD3') == pytest.approx(0.42)
