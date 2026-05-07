"""Page 5: Results browser — PNG gallery and CSV viewer for completed analyses."""
import streamlit as st
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'gui'))

from components.results_browser import render_results_browser

st.set_page_config(page_title='Results · cifsQuant', layout='wide')
st.title('5 · Results Browser')
st.caption('Browse plots and tables from completed analyses.')

project_dir = Path(st.session_state.get('project_dir', REPO_ROOT))

# Allow override of output directory
default_output = str(project_dir / 'spatial_quantification_results')
output_dir_input = st.text_input(
    'Results directory',
    value=default_output,
    help='Path to the spatial_quantification_results/ directory',
)
output_dir = Path(output_dir_input)

st.divider()

render_results_browser(output_dir)
