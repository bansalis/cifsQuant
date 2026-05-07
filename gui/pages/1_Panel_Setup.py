"""Page 1: Panel definition, marker hierarchy, and segmentation parameters."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'gui'))

from components.config_io import (
    load_project, save_project, detect_channels_from_results, get_display_names
)

st.set_page_config(page_title='Panel Setup · cifsQuant', layout='wide')
st.title('1 · Panel Setup')
st.caption('Define your imaging panel, marker hierarchy, and segmentation parameters.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'

if not yaml_path.exists():
    st.warning('No project.yaml found. Go to the home page and create one from the template.')
    st.stop()

config = st.session_state.get('project_config') or load_project(yaml_path)

# ── Section 1: Panel (markers) ─────────────────────────────────────────────
st.header('Imaging panel')

col_info, col_detect = st.columns([3, 1])
with col_info:
    st.markdown('Map each **channel name** (as it appears in segmentation CSVs) to a **display name** used throughout the pipeline.')
with col_detect:
    if st.button('Auto-detect channels', help='Reads column headers from results/ CSVs'):
        detected = detect_channels_from_results(project_dir)
        if detected:
            existing_keys = set(config.get('markers', {}).keys())
            for ch in detected:
                if ch not in existing_keys:
                    config.setdefault('markers', {})[ch] = ch
            st.success(f'Detected {len(detected)} channels')
        else:
            st.info('No segmentation CSVs found in results/. Run Stage 1 first, or add markers manually.')

# Editable markers table
markers = config.get('markers', {})
marker_df = pd.DataFrame(
    [{'Channel name': k, 'Display name': v} for k, v in markers.items()]
)

edited = st.data_editor(
    marker_df,
    num_rows='dynamic',
    use_container_width=True,
    column_config={
        'Channel name': st.column_config.TextColumn('Channel name', help='Exact column name in segmentation CSV'),
        'Display name': st.column_config.TextColumn('Display name', help='Short name used in gating, phenotypes, and plots'),
    },
    key='marker_table',
)
config['markers'] = {row['Channel name']: row['Display name'] for _, row in edited.iterrows()
                     if row['Channel name'] and row['Display name']}

# ── Section 2: Marker hierarchy ─────────────────────────────────────────────
st.header('Marker hierarchy')
st.markdown(
    'Parent-child constraints: a child marker+ cell **must** also be parent marker+. '
    'Set parent to `— none —` to allow a marker on any cell type (e.g. KI67).'
)

display_names = get_display_names(config)
hierarchy = config.get('marker_hierarchy', {})

# Build editable hierarchy table
hier_rows = [{'Child marker': k, 'Parent marker': v if v else '— none —'}
             for k, v in hierarchy.items()]
hier_df = pd.DataFrame(hier_rows) if hier_rows else pd.DataFrame(columns=['Child marker', 'Parent marker'])

parent_options = ['— none —'] + display_names
edited_hier = st.data_editor(
    hier_df,
    num_rows='dynamic',
    use_container_width=True,
    column_config={
        'Child marker': st.column_config.SelectboxColumn('Child marker', options=display_names),
        'Parent marker': st.column_config.SelectboxColumn('Parent marker', options=parent_options),
    },
    key='hierarchy_table',
)
config['marker_hierarchy'] = {
    row['Child marker']: (None if row['Parent marker'] == '— none —' else row['Parent marker'])
    for _, row in edited_hier.iterrows()
    if row['Child marker']
}

# ── Section 3: Segmentation parameters ──────────────────────────────────────
st.header('Segmentation parameters')
st.markdown('These are passed to Nextflow as `-params-file project.yaml` when running Stage 1.')

col1, col2, col3 = st.columns(3)
with col1:
    config['dapi_channel'] = st.number_input(
        'DAPI channel index', min_value=0, value=int(config.get('dapi_channel', 0)),
        help='Zero-indexed position of the DAPI channel in the OME-TIFF'
    )
    config['nuc_diameter'] = st.number_input(
        'Nucleus diameter (px)', min_value=4, max_value=60, value=int(config.get('nuc_diameter', 12)),
        help='Expected nucleus diameter in pixels. Measure a representative nucleus in FIJI.'
    )
with col2:
    config['cyto_diameter'] = st.number_input(
        'Cell diameter (px)', min_value=4, max_value=120, value=int(config.get('cyto_diameter', 24)),
        help='Expected full cell diameter. Typically ~2× nucleus diameter.'
    )
    config['cellpose_model'] = st.selectbox(
        'Cellpose model', ['cyto2', 'cyto', 'nuclei', 'cyto3'],
        index=['cyto2', 'cyto', 'nuclei', 'cyto3'].index(config.get('cellpose_model', 'cyto2')),
    )
with col3:
    config['custom_channel_weights'] = st.text_input(
        'Custom channel weights', value=config.get('custom_channel_weights', ''),
        help='Weighted cytoplasm channels, e.g. 0:0.7,3:0.3. Leave blank for uniform.'
    )
    config['use_gpu'] = st.toggle('Use GPU', value=bool(config.get('use_gpu', True)))

# ── Section 4: Tile correction ──────────────────────────────────────────────
st.header('Tile artifact correction')

gating = config.setdefault('gating', {})
tc = gating.setdefault('tile_correction', {'enabled': False, 'markers': []})

tc['enabled'] = st.toggle('Enable tile correction', value=bool(tc.get('enabled', False)),
                           help='Corrects scanner tile-boundary intensity artifacts in dim/nuclear markers')
if tc['enabled']:
    tc['markers'] = st.multiselect(
        'Markers to correct', options=display_names,
        default=[m for m in tc.get('markers', []) if m in display_names],
        help='Select markers that show visible grid patterns or periodic intensity steps at tile boundaries',
    )

# ── Segmentation diagnostics ─────────────────────────────────────────────────
diag_dir = project_dir / 'tile_correction_diagnostics'
if diag_dir.exists():
    pngs = list(diag_dir.glob('*.png'))
    if pngs:
        st.header('Segmentation QC — Tile Correction Diagnostics')
        st.caption(f'{len(pngs)} diagnostic plots from last run')
        cols = st.columns(min(3, len(pngs)))
        for i, png in enumerate(sorted(pngs)[:9]):
            with cols[i % 3]:
                st.image(str(png), caption=png.stem, use_container_width=True)

# ── Save ────────────────────────────────────────────────────────────────────
st.divider()
if st.button('💾 Save to project.yaml', type='primary'):
    save_project(config, yaml_path)
    st.session_state.project_config = config
    st.success('Saved. Proceed to **2 · Interactive Gating**.')
