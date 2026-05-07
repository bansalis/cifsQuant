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
from components.pipeline_runner import run_stage_inline

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

# ── Prerequisite checks ───────────────────────────────────────────────────────
if not display_names:
    st.error('**Step 1 incomplete.** No markers are defined in your project.yaml.')
    st.markdown('Go to **1 · Panel Setup**, define your imaging panel, and save before coming here.')
    st.page_link('pages/1_Panel_Setup.py', label='Go to Panel Setup')
    st.stop()

results_dir = project_dir / 'results'
has_seg_results = results_dir.exists() and any(results_dir.glob('*/final/combined_quantification.csv'))
norm_path = find_h5ad(project_dir, 'normalized_data.h5ad')
gated_path = find_h5ad(project_dir, 'gated_data.h5ad')

if not has_seg_results and norm_path is None:
    st.error('**Segmentation not run yet.** No results found.')
    st.markdown(
        'Go to **4 · Run Pipeline**, enable **Stage 1 · Segmentation**, and run it first. '
        'Once segmentation is complete, come back here.'
    )
    st.page_link('pages/4_Run_Pipeline.py', label='Go to Run Pipeline')
    st.stop()

if norm_path is None:
    st.info(
        'Segmentation results found but normalization checkpoint not yet generated. '
        'Go to **4 · Run Pipeline** and run **Stage 2 · Gating** to build the normalization checkpoint, '
        'which enables fast interactive threshold adjustment here.'
    )
    st.page_link('pages/4_Run_Pipeline.py', label='Go to Run Pipeline')
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

# ── Apply Gates / Re-run Gating ──────────────────────────────────────────────
st.divider()
col_apply, col_rerun, col_note = st.columns([1, 1, 2])
with col_apply:
    apply = st.button('Apply Gates & Save gated_data.h5ad', type='primary',
                      help='Saves current thresholds and re-gates using the normalization checkpoint (fast — no re-normalization)')
with col_rerun:
    if st.button('Re-run full gating stage', help='Re-normalizes from scratch and re-applies gates. Use if panel or samples changed.'):
        save_project(config, yaml_path)
        run_stage_inline('gating', project_dir, yaml_path)
        load_normalized.clear()
        st.rerun()
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
