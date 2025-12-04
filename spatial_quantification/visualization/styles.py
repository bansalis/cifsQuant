"""
Plot Styles for Spatial Quantification
Publication-quality and exploratory styles
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def setup_publication_style():
    """Set up publication-quality plot style - Prism-like clean appearance."""
    # Use seaborn with custom settings
    sns.set_style('ticks')

    # Matplotlib settings
    mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
    mpl.rcParams['ps.fonttype'] = 42

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 14  # Larger base font
    mpl.rcParams['axes.labelsize'] = 16  # Bigger axis labels
    mpl.rcParams['axes.titlesize'] = 16  # Bigger titles
    mpl.rcParams['xtick.labelsize'] = 14  # Bigger tick labels
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 13  # Bigger legend
    mpl.rcParams['figure.titlesize'] = 18

    # Line widths - bolder for publication
    mpl.rcParams['axes.linewidth'] = 2.0  # Bolder axes
    mpl.rcParams['grid.linewidth'] = 0.0  # No grid
    mpl.rcParams['lines.linewidth'] = 3.0  # Bolder lines
    mpl.rcParams['patch.linewidth'] = 2.5  # Bolder box edges

    # Tick parameters
    mpl.rcParams['xtick.major.width'] = 2.0  # Bolder ticks
    mpl.rcParams['ytick.major.width'] = 2.0
    mpl.rcParams['xtick.major.size'] = 6  # Longer ticks
    mpl.rcParams['ytick.major.size'] = 6

    # Remove top and right spines for clean look
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    # Grid - OFF for publication
    mpl.rcParams['axes.grid'] = False

    # Figure
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'


def setup_exploratory_style():
    """Set up exploratory plot style (with raw data visible)."""
    sns.set_style('whitegrid')

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3

    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['savefig.bbox'] = 'tight'


def get_colorblind_palette():
    """Get colorblind-friendly color palette."""
    # Colorblind-friendly palette (Wong 2011)
    colors = {
        'blue': '#0173B2',
        'orange': '#DE8F05',
        'green': '#029E73',
        'yellow': '#ECE133',
        'cyan': '#56B4E9',
        'red': '#CC3311',
        'purple': '#949494',
        'pink': '#FFAABB'
    }
    return colors


def get_group_colors():
    """Get standard colors for groups."""
    return {
        'KPT': '#E41A1C',    # Red
        'KPNT': '#377EB8',   # Blue
        'cis': '#4DAF4A',    # Green
        'trans': '#FF7F00'   # Orange
    }


def add_significance_stars(ax, x1, x2, y, p_value, height_offset=0):
    """
    Add significance stars to plot.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to add stars to
    x1, x2 : float
        X positions for comparison
    y : float
        Y position for bracket
    p_value : float
        P-value for significance
    height_offset : float
        Additional height offset
    """
    # Determine significance level
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'

    # Draw bracket
    bracket_height = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    y_pos = y + height_offset

    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bracket_height,
                                y_pos + bracket_height, y_pos],
            'k-', linewidth=1.5)

    # Add stars
    ax.text((x1 + x2) / 2, y_pos + bracket_height,
           stars, ha='center', va='bottom', fontsize=12)


def format_p_value(p_value):
    """Format p-value for display."""
    if p_value < 0.001:
        return 'p < 0.001'
    elif p_value < 0.01:
        return f'p = {p_value:.3f}'
    else:
        return f'p = {p_value:.2f}'
