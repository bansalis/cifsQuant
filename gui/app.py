"""
cifsQuant GUI — entry point.

Launch: streamlit run gui/app.py

The GUI is an optional wrapper around the CLI pipeline.
All core pipeline functionality remains accessible via CLI:
  python run_cifsquant.py --project project.yaml [--stages ...] [--dry-run]
  python manual_gating.py --results_dir results --project project.yaml
  python spatial_quantification/run_spatial_quantification.py --config project.yaml
"""
import streamlit as st
from pathlib import Path
import sys
import os

# Make sure the repo root is importable (for run_cifsquant, manual_gating, etc.)
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(
    page_title='cifsQuant',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Session state: project directory + config ──────────────────────────────
if 'project_dir' not in st.session_state:
    st.session_state.project_dir = str(REPO_ROOT)
if 'project_config' not in st.session_state:
    st.session_state.project_config = {}
if 'gates' not in st.session_state:
    st.session_state.gates = {}

# ── Sidebar: project directory selector ───────────────────────────────────
with st.sidebar:
    st.title('🔬 cifsQuant')
    st.caption('Spatial CyCIF Analysis Pipeline')
    st.divider()

    project_dir_input = st.text_input(
        'Project directory',
        value=st.session_state.project_dir,
        help='Root directory containing project.yaml, results/, and manual_gating_output/',
    )
    if project_dir_input != st.session_state.project_dir:
        st.session_state.project_dir = project_dir_input
        st.session_state.project_config = {}
        st.session_state.gates = {}

    project_dir = Path(st.session_state.project_dir)
    yaml_path = project_dir / 'project.yaml'

    if yaml_path.exists():
        st.success(f'project.yaml found')
        if not st.session_state.project_config:
            from gui.components.config_io import load_project
            st.session_state.project_config = load_project(yaml_path)
    else:
        st.warning('No project.yaml in this directory')
        if st.button('Create from template'):
            import shutil
            template = REPO_ROOT / 'project.yaml'
            if template.exists():
                shutil.copy(template, yaml_path)
                from gui.components.config_io import load_project
                st.session_state.project_config = load_project(yaml_path)
                st.rerun()

    # ── Pipeline progress ──────────────────────────────────────────────────
    st.divider()
    _cfg = st.session_state.get('project_config', {})
    _results_dir = project_dir / 'results'
    _norm_h5ad   = project_dir / 'manual_gating_output' / 'normalized_data.h5ad'
    _gated_h5ad  = project_dir / 'manual_gating_output' / 'gated_data.h5ad'
    _spatial_dir = project_dir / 'spatial_quantification_results'

    _panel_ok   = bool(_cfg.get('markers'))
    _seg_ok     = _results_dir.exists() and any(_results_dir.glob('*/final/combined_quantification.csv'))
    _gating_ok  = _gated_h5ad.exists()
    _spatial_ok = _spatial_dir.exists() and any(_spatial_dir.iterdir())

    def _step(label, done, active):
        icon = '✅' if done else ('▶' if active else '○')
        st.markdown(f'{icon} {label}')

    st.caption('**Progress**')
    _step('Panel Setup', _panel_ok, not _panel_ok)
    _step('Segmentation', _seg_ok, _panel_ok and not _seg_ok)
    _step('Gating', _gating_ok, _seg_ok and not _gating_ok)
    _step('Spatial Config', _gating_ok, _gating_ok)
    _step('Spatial Analysis', _spatial_ok, _gating_ok and not _spatial_ok)

    st.divider()
    st.caption('**Steps**')
    st.page_link('pages/1_Panel_Setup.py',    label='1 · Panel Setup',        icon='🧬')
    st.page_link('pages/4_Run_Pipeline.py',   label='  Run segmentation',     icon='▶️')
    st.page_link('pages/2_Gating.py',         label='2 · Interactive Gating', icon='🎚️')
    st.page_link('pages/3_Spatial_Config.py', label='3 · Spatial Config',     icon='⚙️')
    st.page_link('pages/4_Run_Pipeline.py',   label='4 · Run Pipeline',       icon='▶️')
    st.page_link('pages/5_Results.py',        label='5 · Results Browser',    icon='📊')
    st.divider()
    st.caption('**CLI usage** (no GUI required):')
    st.code('python run_cifsquant.py \\\n  --project project.yaml', language='bash')

# ── Home page ────────────────────────────────────────────────────────────
st.title('cifsQuant — Spatial CyCIF Analysis')
st.markdown("""
End-to-end spatial quantification pipeline for cyclic immunofluorescence (CyCIF) imaging.

---

### How to use this GUI

Use the sidebar to navigate through the five steps:

| Step | Page | What you do |
|---|---|---|
| **1** | Panel Setup | Define your imaging panel and segmentation parameters |
| **2** | Interactive Gating | Set per-marker thresholds with live histogram feedback |
| **3** | Spatial Config | Define cell phenotypes and choose spatial analyses |
| **4** | Run Pipeline | Execute one or more stages and monitor progress |
| **5** | Results Browser | Browse plots and tables from completed analyses |

> **Note:** The GUI is optional. All functionality is available via the command line:
> ```bash
> python run_cifsquant.py --project project.yaml
> python run_cifsquant.py --project project.yaml --stages gating spatial
> python run_cifsquant.py --project project.yaml --dry-run
> ```

---
""")

# Status overview
col1, col2, col3, col4 = st.columns(4)

results_dir = project_dir / 'results'
norm_h5ad = project_dir / 'manual_gating_output' / 'normalized_data.h5ad'
gated_h5ad = project_dir / 'manual_gating_output' / 'gated_data.h5ad'
output_dir = project_dir / 'spatial_quantification_results'

with col1:
    n_samples = len(list(results_dir.glob('*/final/combined_quantification.csv'))) if results_dir.exists() else 0
    st.metric('Segmented samples', n_samples, help='CSV files found in results/')

with col2:
    st.metric('Normalization', '✓ done' if norm_h5ad.exists() else '—', help=str(norm_h5ad))

with col3:
    st.metric('Gating', '✓ done' if gated_h5ad.exists() else '—', help=str(gated_h5ad))

with col4:
    n_analyses = len(list(output_dir.glob('*/'))) if output_dir.exists() else 0
    st.metric('Completed analyses', n_analyses)
