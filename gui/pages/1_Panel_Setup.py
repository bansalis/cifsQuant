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
from components.pipeline_runner import run_stage_inline

st.set_page_config(page_title='Panel Setup · cifsQuant', layout='wide')
st.title('1 · Panel Setup')
st.caption('Define your imaging panel, marker hierarchy, and segmentation parameters. Save before moving to the next step.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))
yaml_path = project_dir / 'project.yaml'

if not yaml_path.exists():
    st.warning('No project.yaml found. Go to the home page and create one from the template.')
    st.page_link('app.py', label='Go to Home', icon='🏠')
    st.stop()

config = st.session_state.get('project_config') or load_project(yaml_path)

# ── Section 1: Panel (markers) ─────────────────────────────────────────────
st.header('Imaging panel')
st.markdown('Map each **channel name** (as it appears in segmentation CSVs) to a **display name** used throughout the pipeline.')

col_info, col_detect = st.columns([3, 1])
with col_detect:
    if st.button('Auto-detect channels', help='Reads column headers from results/ CSVs'):
        detected = detect_channels_from_results(project_dir)
        if detected:
            existing_keys = set(config.get('markers', {}).keys())
            for ch in detected:
                if ch not in existing_keys:
                    config.setdefault('markers', {})[ch] = ch
            st.success(f'Detected {len(detected)} channels. Edit display names below, then click Apply.')
        else:
            st.info('No segmentation CSVs found in results/ yet. Add markers manually or run Stage 1 first.')

# Editable markers table — wrapped in a form so edits don't rerun on every keystroke
markers = config.get('markers', {})
marker_df = pd.DataFrame(
    [{'Channel name': k, 'Display name': v} for k, v in markers.items()]
)

with st.form('markers_form'):
    st.markdown('**Edit table, then click Apply.**')
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
    if st.form_submit_button('Apply panel changes', type='primary'):
        config['markers'] = {
            row['Channel name']: row['Display name']
            for _, row in edited.iterrows()
            if row['Channel name'] and row['Display name']
        }
        st.session_state.project_config = config
        st.success(f'{len(config["markers"])} markers applied.')

# ── Section 2: Marker hierarchy ─────────────────────────────────────────────
st.header('Marker hierarchy')
st.markdown(
    'Parent-child constraints: a child marker+ cell **must** also be parent marker+. '
    'Example: CD8 → parent CD3 means CD8+ cells are always a subset of CD3+ cells.'
)

display_names = get_display_names(config)
hierarchy = config.get('marker_hierarchy', {})

hier_rows = [{'Child marker': k, 'Parent marker': v if v else '— none —'}
             for k, v in hierarchy.items()]
hier_df = pd.DataFrame(hier_rows) if hier_rows else pd.DataFrame(columns=['Child marker', 'Parent marker'])

parent_options = ['— none —'] + display_names

with st.form('hierarchy_form'):
    st.markdown('**Edit table, then click Apply.**')
    edited_hier = st.data_editor(
        hier_df,
        num_rows='dynamic',
        use_container_width=True,
        column_config={
            'Child marker': st.column_config.SelectboxColumn('Child marker', options=display_names, required=True),
            'Parent marker': st.column_config.SelectboxColumn('Parent marker', options=parent_options, required=True),
        },
        key='hierarchy_table',
    )
    if st.form_submit_button('Apply hierarchy changes', type='primary'):
        config['marker_hierarchy'] = {
            row['Child marker']: (None if row['Parent marker'] == '— none —' else row['Parent marker'])
            for _, row in edited_hier.iterrows()
            if row['Child marker']
        }
        st.session_state.project_config = config
        st.success('Hierarchy applied.')

# ── Section 3: Segmentation parameters ──────────────────────────────────────
st.header('Segmentation parameters')
st.markdown('Passed to Nextflow when running Stage 1 segmentation.')

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
    config['cyto_model'] = st.selectbox(
        'Cellpose model (cytoplasm)', ['cyto2', 'cyto', 'nuclei', 'cyto3'],
        index=['cyto2', 'cyto', 'nuclei', 'cyto3'].index(config.get('cyto_model', 'cyto2')),
        help='Written to cyto_model, read by the segmentation stage.',
    )
with col3:
    config['custom_channel_weights'] = st.text_input(
        'Custom channel weights', value=config.get('custom_channel_weights', ''),
        help='Weighted cytoplasm channels, e.g. 0:0.7,3:0.3. Leave blank for uniform.'
    )
    st.caption('GPU is auto-detected by the segmentation stage (torch.cuda).')

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
        help='Select markers that show visible grid patterns at tile boundaries',
    )

# ── Segmentation diagnostics ─────────────────────────────────────────────────
# manual_gating.py writes diagnostics next to itself (repo root), so fall back
# there when the project directory has none
diag_dir = project_dir / 'tile_correction_diagnostics'
if not diag_dir.exists():
    diag_dir = REPO_ROOT / 'tile_correction_diagnostics'
if diag_dir.exists():
    pngs = list(diag_dir.glob('*.png'))
    if pngs:
        st.header('Tile Correction Diagnostics')
        st.caption(f'{len(pngs)} diagnostic plots from last run')
        cols = st.columns(min(3, len(pngs)))
        for i, png in enumerate(sorted(pngs)[:9]):
            with cols[i % 3]:
                st.image(str(png), caption=png.stem, use_container_width=True)

# ── Save ────────────────────────────────────────────────────────────────────
st.divider()
col_save, col_run = st.columns(2)
with col_save:
    if st.button('💾 Save to project.yaml', type='primary'):
        save_project(config, yaml_path)
        st.session_state.project_config = config
        st.success('Saved. Ready to run segmentation.')

with col_run:
    n_markers = len(config.get('markers', {}))
    st.markdown('**Run Stage 1 · Segmentation**')
    st.caption('Requires Docker + Nextflow ≥ 23.10 on your PATH. Save first.')
    rawdata_dir = project_dir / 'rawdata'
    n_images = len(list(rawdata_dir.glob('*.ome.tif'))) if rawdata_dir.exists() else 0
    if n_images:
        st.caption(f'{n_images} OME-TIFF images found in rawdata/')
    else:
        st.caption('No OME-TIFF files found in rawdata/')

    if st.button('▶ Run Segmentation', disabled=not (n_markers and yaml_path.exists()),
                 help='Runs Nextflow + Cellpose segmentation on images in rawdata/'):
        if not yaml_path.exists():
            st.error('Save project.yaml first.')
        else:
            run_stage_inline('segmentation', project_dir, yaml_path)
            st.info('Segmentation complete. Proceed to **2 · Gating**.')
