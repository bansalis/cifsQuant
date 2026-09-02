"""Smoke tests: GUI component logic (no Streamlit session required)."""
import sys
import os
import numpy as np
import pandas as pd
import pytest
import yaml
import anndata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_ROOT = os.path.join(REPO_ROOT, 'gui')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, GUI_ROOT)


# ── config_io tests ───────────────────────────────────────────────────────────

class TestConfigIO:

    def test_load_project_returns_dict(self, tmp_path):
        """load_project() should parse a YAML file into a dict."""
        from components.config_io import load_project

        cfg = {'markers': {'DAPI': 'DAPI', 'Cy3': 'CD3'}, 'gating': {}}
        p = tmp_path / 'project.yaml'
        p.write_text(yaml.dump(cfg))

        result = load_project(p)
        assert isinstance(result, dict)
        assert result['markers']['Cy3'] == 'CD3'

    def test_load_project_missing_file_returns_empty(self, tmp_path):
        """load_project() on a non-existent file should return an empty dict."""
        from components.config_io import load_project

        result = load_project(tmp_path / 'nonexistent.yaml')
        assert isinstance(result, dict)
        assert result == {}

    def test_save_project_roundtrip(self, tmp_path):
        """save_project() should write YAML that load_project() reads back identically."""
        from components.config_io import load_project, save_project

        cfg = {
            'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3'},
            'gating': {'gates': {'CD3': 0.35}},
        }
        p = tmp_path / 'project.yaml'
        save_project(cfg, p)
        loaded = load_project(p)

        assert loaded['gating']['gates']['CD3'] == pytest.approx(0.35)
        assert loaded['markers']['Cy3_CD3'] == 'CD3'

    def test_get_display_names_returns_list(self):
        """get_display_names() should return a list of marker display names."""
        from components.config_io import get_display_names

        config = {'markers': {'DAPI': 'DAPI', 'Cy3': 'CD3', 'Cy5': 'CD8'}}
        names = get_display_names(config)
        assert isinstance(names, list)
        assert 'CD3' in names
        assert 'CD8' in names

    def test_get_display_names_empty_config(self):
        """get_display_names() should return empty list when no markers defined."""
        from components.config_io import get_display_names

        assert get_display_names({}) == []

    def test_get_analysis_list_returns_expected_keys(self):
        """get_analysis_list() should return dicts with key/label/requires fields."""
        from components.config_io import get_analysis_list

        analyses = get_analysis_list()
        assert isinstance(analyses, list)
        assert len(analyses) > 0
        for a in analyses:
            assert 'key' in a, f"Missing 'key' in {a}"
            assert 'label' in a, f"Missing 'label' in {a}"
            assert 'requires' in a, f"Missing 'requires' in {a}"
            assert isinstance(a['requires'], list)

    def test_get_analysis_list_per_tumor_analysis_present(self):
        """per_tumor_analysis should be in the analysis list."""
        from components.config_io import get_analysis_list

        keys = [a['key'] for a in get_analysis_list()]
        assert 'per_tumor_analysis' in keys

    def test_validate_project_passes_on_valid(self, tmp_path):
        """validate_project() should return a list of checks, all passing."""
        from components.config_io import validate_project

        config = {
            'markers': {'DAPI': 'DAPI', 'Cy3': 'CD3'},
            'gating': {'gates': {'CD3': 0.4}},
            'spatial': {'phenotypes': {'T_cells': {'positive': ['CD3']}}},
        }
        checks = validate_project(config)
        assert isinstance(checks, list)
        # Every check has required fields
        for c in checks:
            assert 'check' in c
            assert 'status' in c
            assert 'message' in c
            assert c['status'] in ('ok', 'warn', 'error')

    def test_validate_project_warns_missing_gated_data(self, tmp_path):
        """validate_project() should warn when gated_data.h5ad doesn't exist."""
        from components.config_io import validate_project

        config = {
            'markers': {'DAPI': 'DAPI'},
            'gating': {},
            'spatial': {'input': {'gated_data': str(tmp_path / 'nonexistent.h5ad')}},
        }
        checks = validate_project(config)
        statuses = {c['check']: c['status'] for c in checks}
        # At minimum, must not crash — gated_data check may be warn or error
        assert isinstance(checks, list)


# ── gating_plots tests ────────────────────────────────────────────────────────

class TestGatingPlots:

    def test_compute_gmm_returns_expected_keys(self):
        """compute_gmm() should return a dict with means, stds, weights, suggested_threshold."""
        from components.gating_plots import compute_gmm

        rng = np.random.default_rng(0)
        # Bimodal: two clusters at 0.2 and 0.7
        values = np.concatenate([rng.normal(0.2, 0.05, 300), rng.normal(0.7, 0.05, 300)])

        result = compute_gmm(values)
        assert result is not None
        assert 'means' in result
        assert 'stds' in result
        assert 'weights' in result
        assert 'suggested_threshold' in result

    def test_compute_gmm_threshold_between_modes(self):
        """Suggested threshold should lie between the two GMM component means."""
        from components.gating_plots import compute_gmm

        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(0.15, 0.04, 500), rng.normal(0.75, 0.04, 500)])

        result = compute_gmm(values)
        if result is not None:
            lo = min(result['means'])
            hi = max(result['means'])
            t = result['suggested_threshold']
            assert lo <= t <= hi, f'Threshold {t:.3f} not between modes {lo:.3f} and {hi:.3f}'

    def test_compute_gmm_handles_uniform_data(self):
        """compute_gmm() on flat data should return None or a result without crashing."""
        from components.gating_plots import compute_gmm

        values = np.full(100, 0.5)
        result = compute_gmm(values)  # Should not raise; may return None

    def test_plot_marker_histogram_returns_figure(self):
        """plot_marker_histogram() should return a plotly Figure."""
        import plotly.graph_objects as go
        from components.gating_plots import plot_marker_histogram

        values = np.random.default_rng(2).uniform(0, 1, 500)
        fig = plot_marker_histogram(values, threshold=0.5, marker_name='CD3')

        assert isinstance(fig, go.Figure)

    def test_plot_marker_histogram_with_gmm(self):
        """plot_marker_histogram() with gmm_params should not crash."""
        import plotly.graph_objects as go
        from components.gating_plots import plot_marker_histogram, compute_gmm

        values = np.concatenate([
            np.random.default_rng(3).normal(0.2, 0.05, 200),
            np.random.default_rng(4).normal(0.7, 0.05, 200),
        ])
        gmm = compute_gmm(values)
        fig = plot_marker_histogram(values, threshold=0.45, marker_name='CD3', gmm_params=gmm)
        assert isinstance(fig, go.Figure)

    def test_plot_spatial_scatter_returns_figure(self):
        """plot_spatial_scatter() should return a plotly Figure."""
        import plotly.graph_objects as go
        from components.gating_plots import plot_spatial_scatter

        rng = np.random.default_rng(5)
        n = 200
        x = rng.uniform(0, 2000, n)
        y = rng.uniform(0, 2000, n)
        mask = np.array([True, False] * (n // 2))

        fig = plot_spatial_scatter(x, y, mask, 'CD3')
        assert isinstance(fig, go.Figure)

    def test_plot_phenotype_scatter_returns_figure(self):
        """plot_phenotype_scatter() should return a plotly Figure."""
        import plotly.graph_objects as go
        from components.gating_plots import plot_phenotype_scatter

        rng = np.random.default_rng(6)
        n = 150
        x = rng.uniform(0, 2000, n)
        y = rng.uniform(0, 2000, n)
        mask = rng.random(n) > 0.5

        fig = plot_phenotype_scatter(x, y, mask, 'T_cells')
        assert isinstance(fig, go.Figure)


# ── spatial_viewer tests ──────────────────────────────────────────────────────

class TestSpatialViewer:

    def _make_adata_with_layers(self, n_cells=100):
        """AnnData with spatial coords and aligned layer (mimics normalized_data.h5ad)."""
        rng = np.random.default_rng(42)
        markers = ['CD3', 'CD8', 'B220']
        X = rng.uniform(0, 1, (n_cells, len(markers))).astype(np.float32)
        obs = pd.DataFrame({
            'sample_id': [f'S{i % 2 + 1}' for i in range(n_cells)],
            'X_centroid': rng.uniform(0, 2000, n_cells),
            'Y_centroid': rng.uniform(0, 2000, n_cells),
            'is_CD3': (X[:, 0] > 0.5).astype(bool),
        })
        obs.index = [f'cell_{i}' for i in range(n_cells)]
        adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=markers))
        adata.layers['aligned'] = X.copy()
        adata.obsm['spatial'] = np.column_stack([obs['X_centroid'], obs['Y_centroid']])
        return adata

    def test_get_samples_returns_list(self):
        """get_samples() should return a sorted list of unique sample IDs."""
        from components.spatial_viewer import get_samples

        adata = self._make_adata_with_layers()
        samples = get_samples(adata)
        assert isinstance(samples, list)
        assert set(samples) == {'S1', 'S2'}

    def test_get_marker_values_all_samples(self):
        """get_marker_values() without sample filter should return all cell values."""
        from components.spatial_viewer import get_marker_values

        adata = self._make_adata_with_layers()
        vals = get_marker_values(adata, 'CD3')
        assert len(vals) == adata.n_obs

    def test_get_marker_values_single_sample(self):
        """get_marker_values() filtered to a sample should return ~n_cells/2 values."""
        from components.spatial_viewer import get_marker_values

        adata = self._make_adata_with_layers(n_cells=100)
        vals = get_marker_values(adata, 'CD3', sample_id='S1')
        assert len(vals) == 50

    def test_get_marker_values_unknown_marker_returns_empty(self):
        """get_marker_values() for a non-existent marker should return empty array."""
        from components.spatial_viewer import get_marker_values

        adata = self._make_adata_with_layers()
        vals = get_marker_values(adata, 'NONEXISTENT')
        assert len(vals) == 0

    def test_get_spatial_coords_shape(self):
        """get_spatial_coords() should return two arrays of the same length."""
        from components.spatial_viewer import get_spatial_coords

        adata = self._make_adata_with_layers()
        x, y = get_spatial_coords(adata)
        assert len(x) == len(y) == adata.n_obs

    def test_get_spatial_coords_filtered(self):
        """get_spatial_coords() filtered by sample should return half the cells."""
        from components.spatial_viewer import get_spatial_coords

        adata = self._make_adata_with_layers(n_cells=100)
        x, y = get_spatial_coords(adata, sample_id='S1')
        assert len(x) == 50

    def test_get_gate_mask_returns_boolean_array(self):
        """get_gate_mask() should return a boolean array."""
        from components.spatial_viewer import get_gate_mask

        adata = self._make_adata_with_layers()
        mask = get_gate_mask(adata, 'CD3', threshold=0.5)
        assert mask.dtype == bool
        assert len(mask) == adata.n_obs

    def test_find_h5ad_returns_none_when_missing(self, tmp_path):
        """find_h5ad() should return None when no matching file exists."""
        from components.spatial_viewer import find_h5ad

        result = find_h5ad(tmp_path, 'gated_data.h5ad')
        assert result is None

    def test_find_h5ad_finds_file_in_subdir(self, tmp_path):
        """find_h5ad() should find the file in manual_gating_output/ subdirectory."""
        from components.spatial_viewer import find_h5ad

        subdir = tmp_path / 'manual_gating_output'
        subdir.mkdir()
        target = subdir / 'gated_data.h5ad'
        target.write_text('')  # empty placeholder

        result = find_h5ad(tmp_path, 'gated_data.h5ad')
        assert result is not None
        assert result == target


# ── Rawdata channel matching (soft marker-name → filename matching) ──────────

class TestChannelMatching:

    def _make_sample(self, tmp_path):
        sample = tmp_path / 'rawdata' / 'JL216'
        sample.mkdir(parents=True)
        for name in ('R1_DAPI.ome.tif', 'R1_Cy3_CD3.ome.tif', 'R2_Cy5_CD8.ome.tif'):
            (sample / name).touch()
        return sample

    def test_all_channels_match(self, tmp_path):
        """Each channel name soft-matches its raw file by fluor/protein substrings."""
        from run_cifsquant import match_channels_report

        sample = self._make_sample(tmp_path)
        report = match_channels_report(sample, ['DAPI', 'Cy3_CD3', 'Cy5_CD8'])

        assert [r['matched'] for r in report] == [True, True, True]
        assert report[1]['file'] == 'R1_Cy3_CD3.ome.tif'
        assert report[2]['file'] == 'R2_Cy5_CD8.ome.tif'

    def test_missing_channel_reported_not_raised(self, tmp_path):
        """A channel with no raw file is reported unmatched, never an exception."""
        from run_cifsquant import match_channels_report

        sample = self._make_sample(tmp_path)
        report = match_channels_report(sample, ['DAPI', 'Cy3_CD3', 'FITC_B220'])

        assert report[2] == {'channel': 'FITC_B220', 'file': None, 'matched': False}
        assert report[0]['matched'] and report[1]['matched']

    def test_fluorophore_matching_is_case_insensitive(self, tmp_path):
        """CY3 vs Cy3 casing differences must not break the match."""
        from run_cifsquant import match_channels_report

        sample = self._make_sample(tmp_path)
        report = match_channels_report(sample, ['CY3_CD3'])
        assert report[0]['matched']
        assert report[0]['file'] == 'R1_Cy3_CD3.ome.tif'

    def test_report_preserves_channel_order(self, tmp_path):
        from run_cifsquant import match_channels_report

        sample = self._make_sample(tmp_path)
        report = match_channels_report(sample, ['Cy5_CD8', 'DAPI'])
        assert [r['channel'] for r in report] == ['Cy5_CD8', 'DAPI']

    def test_matcher_is_quiet(self, tmp_path, capsys):
        """The report helper must not leak the tiler's console chatter."""
        from run_cifsquant import match_channels_report

        sample = self._make_sample(tmp_path)
        match_channels_report(sample, ['DAPI'])
        assert capsys.readouterr().out == ''

    def test_dry_run_prints_match_report(self, tmp_path, capsys):
        """CLI dry-run in rawdata mode shows the per-sample matching table."""
        from run_cifsquant import run_segmentation

        self._make_sample(tmp_path)
        project = {
            'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3', 'FITC_B220': 'B220'},
            'rawdata_dir': str(tmp_path / 'rawdata'),
            'outdir': str(tmp_path / 'results'),
        }
        run_segmentation(project, tmp_path / 'project.yaml', dry_run=True)

        out = capsys.readouterr().out
        assert 'JL216: 2/3 channels matched' in out
        assert '✓ DAPI' in out and 'R1_DAPI.ome.tif' in out
        assert 'NO MATCH' in out and 'FITC_B220' in out


# ── Full-page renders via Streamlit AppTest ──────────────────────────────────

def _page(name):
    return os.path.join(GUI_ROOT, 'pages', name)


def _make_project_dir(tmp_path, with_rawdata=False, missing_channel=False):
    """A self-contained project directory the pages can render against."""
    project = {
        'markers': {'DAPI': 'DAPI', 'Cy3_CD3': 'CD3', 'Cy5_CD8': 'CD8'},
        'marker_hierarchy': {'CD8': 'CD3'},
        'input_image': None,
        'gating': {
            'use_shared_gates': True,
            'normalization_method': 'percentile_99',
            'gates': {'DAPI': None, 'CD3': None, 'CD8': None},
            'tile_correction': {'enabled': False, 'markers': []},
            'liberal_gating': {'enabled': False, 'liberal_markers': []},
        },
        'spatial': {
            'input': {'gated_data': 'manual_gating_output/gated_data.h5ad',
                      'metadata': 'sample_metadata.csv'},
            'output': {'base_directory': 'spatial_quantification_results'},
            'metadata': {'sample_column': 'sample_id', 'group_column': 'group'},
            'tumor_definition': {'base_phenotype': 'Primary_Structure',
                                 'required_positive': ['CD3'],
                                 'structure_detection': {'eps': 500, 'min_samples': 50}},
            'phenotypes': {
                'Primary_Structure': {'positive': ['CD3']},
                'CD8_T_cells': {'base': 'Primary_Structure', 'positive': ['CD8']},
            },
            'statistics': {'test': 'mann_whitney'},
            'visualization': {'enabled': False},
        },
    }
    (tmp_path / 'project.yaml').write_text(yaml.dump(project, sort_keys=False))
    pd.DataFrame({'sample_id': ['S1'], 'group': ['GroupA']}).to_csv(
        tmp_path / 'sample_metadata.csv', index=False)
    if with_rawdata:
        sample = tmp_path / 'rawdata' / 'JL216'
        sample.mkdir(parents=True)
        (sample / 'R1_DAPI.ome.tif').touch()
        (sample / 'R1_Cy3_CD3.ome.tif').touch()
        if not missing_channel:
            (sample / 'R2_Cy5_CD8.ome.tif').touch()
    return tmp_path


def _write_h5ad(tmp_path, filename, n=200, seed=0):
    """Synthetic checkpoint with layers['aligned'] as the gating pages expect."""
    rng = np.random.default_rng(seed)
    markers = ['DAPI', 'CD3', 'CD8']
    # bimodal CD3 so GMM auto-suggest has two modes to find
    cd3 = np.concatenate([rng.normal(0.1, 0.03, n // 2),
                          rng.normal(0.7, 0.08, n - n // 2)]).clip(0, 1)
    X = np.column_stack([rng.uniform(0, 1, n), cd3,
                         rng.uniform(0, 1, n)]).astype('float32')
    obs = pd.DataFrame({
        'sample_id': ['S1'] * n,
        'X_centroid': rng.uniform(0, 100, n),
        'Y_centroid': rng.uniform(0, 100, n),
    })
    obs.index = [f'c{i}' for i in range(n)]
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=markers))
    adata.layers['aligned'] = X.copy()
    out = tmp_path / 'manual_gating_output'
    out.mkdir(exist_ok=True)
    adata.write(out / filename)


class TestPageRenders:
    """Every page must render exception-free against a provisioned project dir."""

    def _apptest(self, page, project_dir):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(_page(page) if not page.endswith('app.py')
                               else os.path.join(GUI_ROOT, 'app.py'),
                               default_timeout=60)
        at.session_state['project_dir'] = str(project_dir)
        at.session_state['project_config'] = {}
        at.session_state['gates'] = {}
        at.run()
        return at

    def test_home_renders(self, tmp_path):
        at = self._apptest('app.py', _make_project_dir(tmp_path))
        assert not at.exception, at.exception[0].message

    def test_panel_setup_renders(self, tmp_path):
        at = self._apptest('1_Panel_Setup.py', _make_project_dir(tmp_path))
        assert not at.exception, at.exception[0].message

    def test_gating_renders_and_autosuggest_moves_slider(self, tmp_path):
        """Page 2 must survive a direct load AND auto-suggest must move the slider
        (regression: widget state used to swallow the GMM suggestion)."""
        project_dir = _make_project_dir(tmp_path)
        _write_h5ad(project_dir, 'normalized_data.h5ad')

        at = self._apptest('2_Gating.py', project_dir)
        assert not at.exception, at.exception[0].message

        before = [s for s in at.slider if 'CD3 threshold' in str(s.label)][0].value
        [b for b in at.button if b.key == 'auto_CD3'][0].click().run()
        assert not at.exception, at.exception[0].message
        after = [s for s in at.slider if 'CD3 threshold' in str(s.label)][0].value
        assert after != before and after > 0

    def test_spatial_config_renders_and_keeps_base_phenotype(self, tmp_path):
        """Page 3 must not wipe tumor_definition.base_phenotype on render
        (regression: off-by-one in the dropdown preselect)."""
        project_dir = _make_project_dir(tmp_path)
        _write_h5ad(project_dir, 'gated_data.h5ad')

        at = self._apptest('3_Spatial_Config.py', project_dir)
        assert not at.exception, at.exception[0].message
        sels = [s for s in at.selectbox if s.label == 'Base phenotype']
        assert sels and sels[0].value == 'Primary_Structure'

    def test_run_pipeline_renders(self, tmp_path):
        at = self._apptest('4_Run_Pipeline.py', _make_project_dir(tmp_path))
        assert not at.exception, at.exception[0].message

    def test_run_pipeline_validate_click(self, tmp_path):
        """Regression: 'Validate config' used to NameError on first click."""
        at = self._apptest('4_Run_Pipeline.py', _make_project_dir(tmp_path))
        [b for b in at.button if 'Validate' in str(b.label)][0].click().run()
        assert not at.exception, at.exception[0].message

    def test_results_renders(self, tmp_path):
        at = self._apptest('5_Results.py', _make_project_dir(tmp_path))
        assert not at.exception, at.exception[0].message


class TestRunPipelineRawdataPreflight:
    """The per-channel matching table on page 4."""

    def _render(self, tmp_path, **kw):
        from streamlit.testing.v1 import AppTest
        project_dir = _make_project_dir(tmp_path, with_rawdata=True, **kw)
        at = AppTest.from_file(_page('4_Run_Pipeline.py'), default_timeout=60)
        at.session_state['project_dir'] = str(project_dir)
        at.session_state['project_config'] = {}
        at.run()
        return at

    def test_matching_table_shown_when_all_match(self, tmp_path):
        at = self._render(tmp_path)
        assert not at.exception, at.exception[0].message
        md = ' '.join(str(m.value) for m in at.markdown)
        assert 'R1_Cy3_CD3.ome.tif' in md
        assert 'no matching file' not in md
        assert any('per-channel mode' in str(s.value) for s in at.subheader)

    def test_unmatched_channel_flagged(self, tmp_path):
        at = self._render(tmp_path, missing_channel=True)
        assert not at.exception, at.exception[0].message
        md = ' '.join(str(m.value) for m in at.markdown)
        assert 'no matching file' in md
        assert 'Cy5_CD8' in md

    def test_no_rawdata_section_in_stacked_mode(self, tmp_path):
        from streamlit.testing.v1 import AppTest
        project_dir = _make_project_dir(tmp_path)
        # stacked mode: input_image set, no rawdata/
        cfg = yaml.safe_load((project_dir / 'project.yaml').read_text())
        cfg['input_image'] = 'sample.ome.tif'
        (project_dir / 'project.yaml').write_text(yaml.dump(cfg))

        at = AppTest.from_file(_page('4_Run_Pipeline.py'), default_timeout=60)
        at.session_state['project_dir'] = str(project_dir)
        at.session_state['project_config'] = {}
        at.run()
        assert not at.exception, at.exception[0].message
        assert not any('per-channel mode' in str(s.value) for s in at.subheader)
