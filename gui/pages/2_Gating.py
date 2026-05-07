"""Page 2: Interactive per-marker gating with live histogram + spatial scatter."""
import streamlit as st
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'gui'))

from components.config_io import load_project, save_project, get_display_names
from components.gating_plots import (
    plot_marker_histogram, plot_spatial_scatter, compute_gmm
)
from components.spatial_viewer import (
    load_normalized, get_samples, get_marker_values, get_spatial_coords,
    get_gate_mask, find_h5ad
)

st.set_page_config(page_title='Gating · cifsQuant', layout='wide')
st.title('2 · Interactive Gating')
st.caption('Set per-marker thresholds. Adjust the slider and watch the histogram and spatial map update live.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'

if not yaml_path.exists():
    st.warning('No project.yaml found.')
    st.stop()

config = st.session_state.get('project_config') or load_project(yaml_path)
gating = config.setdefault('gating', {})
gates_config = gating.setdefault('gates', {})
display_names = get_display_names(config)

# ── Load normalized data ─────────────────────────────────────────────────────
norm_path = find_h5ad(project_dir, 'normalized_data.h5ad')
gated_path = find_h5ad(project_dir, 'gated_data.h5ad')

if norm_path is None:
    st.warning(
        '`normalized_data.h5ad` not found. Run Stage 2 gating first to generate the normalization checkpoint:\n\n'
        '```bash\npython run_cifsquant.py --project project.yaml --stages gating\n```\n\n'
        'The GUI will use the checkpoint to enable fast threshold tuning without re-normalizing.'
    )
    st.stop()

adata = load_normalized(str(norm_path))
samples = get_samples(adata)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Controls')
    sample_id = st.selectbox('Sample', ['all'] + samples)
    show_gmm = st.toggle('Show GMM fit', value=True, help='Overlay 2-component GMM on histogram')
    st.divider()
    st.markdown('**Gate summary**')
    for marker in display_names:
        if marker == 'DAPI':
            continue
        vals = get_marker_values(adata, marker)
        thresh = float(st.session_state.gates.get(marker, gates_config.get(marker) or 0.0))
        n_pos = int((vals >= thresh).sum())
        pct = 100 * n_pos / len(vals) if len(vals) > 0 else 0
        st.caption(f'{marker}: {pct:.1f}% +')

# ── Per-marker tabs ──────────────────────────────────────────────────────────
gateable = [m for m in display_names if m != 'DAPI']
if not gateable:
    st.info('No markers to gate. Define your panel in Page 1 first.')
    st.stop()

tabs = st.tabs(gateable)

for marker, tab in zip(gateable, tabs):
    with tab:
        vals = get_marker_values(adata, marker, sample_id if sample_id != 'all' else None)
        if len(vals) == 0:
            st.warning(f'No data for {marker} in sample {sample_id}')
            continue

        # Current threshold (session gates override config)
        default_thresh = float(st.session_state.gates.get(marker, gates_config.get(marker) or 0.0))
        if default_thresh is None:
            default_thresh = 0.0

        # GMM computation (cached per marker+sample)
        gmm_cache_key = f'gmm_{marker}_{sample_id}'
        if gmm_cache_key not in st.session_state:
            st.session_state[gmm_cache_key] = compute_gmm(vals)
        gmm = st.session_state[gmm_cache_key] if show_gmm else None

        col_hist, col_spatial = st.columns(2)

        with col_hist:
            thresh = st.slider(
                f'{marker} threshold',
                min_value=0.0, max_value=1.0,
                value=default_thresh,
                step=0.005,
                key=f'slider_{marker}',
                help='Drag to adjust gate. Green region = positive cells.',
            )
            st.session_state.gates[marker] = thresh

            col_auto, col_mode = st.columns(2)
            with col_auto:
                if st.button('Auto-suggest', key=f'auto_{marker}', help='Run GMM and suggest threshold'):
                    fresh_gmm = compute_gmm(vals)
                    if fresh_gmm:
                        st.session_state.gates[marker] = fresh_gmm['suggested_threshold']
                        st.session_state[gmm_cache_key] = fresh_gmm
                        st.rerun()
            with col_mode:
                liberal = st.toggle(
                    'Liberal', key=f'lib_{marker}',
                    value=marker in gating.get('liberal_gating', {}).get('liberal_markers', []),
                    help='More sensitive (lower) threshold for rare/dim populations',
                )

            fig_hist = plot_marker_histogram(vals, thresh, marker, gmm)
            st.plotly_chart(fig_hist, use_container_width=True, key=f'hist_{marker}')

        with col_spatial:
            x, y = get_spatial_coords(adata, sample_id if sample_id != 'all' else None)
            gate_mask = get_gate_mask(adata, marker, thresh, sample_id if sample_id != 'all' else None)
            fig_spatial = plot_spatial_scatter(x, y, gate_mask, marker)
            st.plotly_chart(fig_spatial, use_container_width=True, key=f'spatial_{marker}')

        # Hierarchy note
        hierarchy = config.get('marker_hierarchy', {})
        if marker in hierarchy and hierarchy[marker]:
            st.caption(f'Hierarchy: {marker}+ cells enforced as subset of {hierarchy[marker]}+')

# ── Summary table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader('Gate summary')

summary_rows = []
for marker in gateable:
    vals = get_marker_values(adata, marker)
    thresh = float(st.session_state.gates.get(marker, gates_config.get(marker) or 0.0))
    n_pos = int((vals >= thresh).sum())
    pct = 100 * n_pos / len(vals) if len(vals) > 0 else 0
    mode = 'manual' if st.session_state.gates.get(marker) is not None else 'auto'
    summary_rows.append({'Marker': marker, 'Threshold': round(thresh, 4),
                         'Mode': mode, '# Positive': n_pos, '% Positive': round(pct, 1)})

import pandas as pd
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ── Apply Gates ──────────────────────────────────────────────────────────────
st.divider()
col_apply, col_note = st.columns([1, 3])
with col_apply:
    apply = st.button('Apply Gates & Save gated_data.h5ad', type='primary',
                      help='Writes threshold values to project.yaml and runs apply_gates() to produce gated_data.h5ad')
with col_note:
    st.caption(
        'This applies the thresholds set above and writes `manual_gating_output/gated_data.h5ad`. '
        'You can also run gating from the CLI: `python run_cifsquant.py --project project.yaml --stages gating`'
    )

if apply:
    # Update gates in config
    for marker in gateable:
        thresh = st.session_state.gates.get(marker)
        if thresh is not None:
            gating['gates'][marker] = round(float(thresh), 4)

    # Update liberal gating list
    lg = gating.setdefault('liberal_gating', {'enabled': False, 'liberal_markers': []})
    lib_markers = [m for m in gateable if st.session_state.get(f'lib_{m}', False)]
    if lib_markers:
        lg['enabled'] = True
        lg['liberal_markers'] = lib_markers
    else:
        lg['enabled'] = False
        lg['liberal_markers'] = []

    save_project(config, yaml_path)
    st.session_state.project_config = config

    # Run gating via CLI (preserves all gating logic, tile correction, hierarchy enforcement)
    import subprocess, threading, queue

    out_dir = project_dir / 'manual_gating_output'
    out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(REPO_ROOT / 'manual_gating.py'),
        '--results_dir', str(project_dir / 'results'),
        '--project', str(yaml_path),
        '--skip_normalization',   # reuse normalized_data.h5ad checkpoint
    ]

    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    lines = []

    with st.spinner('Applying gates…'):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_dir)
            )
            if result.returncode == 0:
                st.success('Gates applied. `gated_data.h5ad` written.')
                # Clear cache so next load reads fresh file
                load_normalized.clear()
            else:
                st.error('Gating failed. Check the log below.')
                st.code(result.stderr[-3000:], language='text')
        except FileNotFoundError:
            st.warning(
                'Could not run manual_gating.py automatically. '
                'Run from CLI:\n```bash\npython run_cifsquant.py --project project.yaml --stages gating\n```'
            )
