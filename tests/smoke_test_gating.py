"""Smoke tests: Stage 2 manual_gating.py logic."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# manual_gating.py is at repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cell_dataframe(n_cells=100, markers=None, seed=42):
    """Create a synthetic per-cell intensity DataFrame mimicking segmentation output."""
    if markers is None:
        markers = ['DAPI', 'CD3', 'CD8', 'B220']
    rng = np.random.default_rng(seed)
    data = {m: rng.uniform(0, 1000, size=n_cells) for m in markers}
    data['sample_id'] = [f'S{i % 3 + 1}' for i in range(n_cells)]
    data['X_centroid'] = rng.uniform(0, 2000, size=n_cells)
    data['Y_centroid'] = rng.uniform(0, 2000, size=n_cells)
    return pd.DataFrame(data)


class TestNormalization:

    def test_percentile99_normalization(self):
        """normalize_data() should scale marker values to 0–1 range (approximately)."""
        from manual_gating import MARKERS
        # Build a synthetic intensity df for a single marker
        df = _make_cell_dataframe()
        # Verify the marker columns exist (as display names after potential mapping)
        # We test the normalization function directly with a simple array
        from manual_gating import normalize_column
        values = pd.Series(np.random.uniform(0, 5000, 200))
        normalized = normalize_column(values, method='percentile_99')
        # Most values should be in [0, 1]; 99th percentile clips at 1.0
        assert normalized.max() <= 1.01, "Normalization max should be ~1.0"
        assert normalized.min() >= 0.0, "Normalization min should be >= 0"

    def test_normalize_column_zscore(self):
        """zscore normalization should produce ~N(0,1) values."""
        from manual_gating import normalize_column
        values = pd.Series(np.random.normal(100, 20, 500))
        normalized = normalize_column(values, method='zscore')
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1


class TestGateApplication:

    def test_gate_creates_boolean_column(self):
        """apply_gate() should produce a boolean series given a threshold."""
        from manual_gating import apply_gate
        values = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
        result = apply_gate(values, threshold=0.4)
        assert result.dtype == bool
        assert list(result) == [False, False, True, True, True]

    def test_gate_at_zero_threshold(self):
        """threshold=0 should gate all positive."""
        from manual_gating import apply_gate
        values = pd.Series([0.0, 0.1, 0.5])
        result = apply_gate(values, threshold=0.0)
        assert result.all()

    def test_gate_at_one_threshold(self):
        """threshold=1.0 should gate none positive (all values <= 1.0 in normalized scale)."""
        from manual_gating import apply_gate
        values = pd.Series([0.0, 0.5, 0.99])
        result = apply_gate(values, threshold=1.0)
        assert not result.any()


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
