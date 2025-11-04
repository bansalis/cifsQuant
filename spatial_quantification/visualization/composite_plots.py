"""
Composite Plots for Spatial Quantification
Generate multi-panel figures
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List
from .styles import setup_publication_style


class CompositePlots:
    """Generate composite multi-panel figures."""

    def __init__(self, config: Dict, output_dir: Path):
        """
        Initialize composite plots.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        output_dir : Path
            Output directory
        """
        self.config = config['visualization']
        self.output_dir = Path(output_dir)
        self.formats = config['output'].get('formats', ['pdf'])
        self.dpi = config['output'].get('dpi', 300)

        setup_publication_style()

    def create_population_overview(self, plot_functions: List,
                                   output_name: str = 'population_overview'):
        """
        Create multi-panel overview of population dynamics.

        Parameters
        ----------
        plot_functions : list
            List of plotting functions to call
        output_name : str
            Output filename
        """
        n_plots = len(plot_functions)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols

        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.3, wspace=0.3)

        for i, plot_func in enumerate(plot_functions):
            row = i // ncols
            col = i % ncols
            ax = fig.add_subplot(gs[row, col])

            # Call plotting function with axis
            plot_func(ax)

        plt.tight_layout()

        # Save
        for fmt in self.formats:
            output_path = self.output_dir / f'{output_name}.{fmt}'
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')

        plt.close(fig)
