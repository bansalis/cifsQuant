"""Plotly-based interactive plots for the gating page."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_marker_histogram(
    values: np.ndarray,
    threshold: float,
    marker_name: str,
    gmm_params: dict | None = None,
) -> go.Figure:
    """
    Histogram + KDE + optional GMM fit with threshold line.

    gmm_params: {'means': [m0, m1], 'stds': [s0, s1], 'weights': [w0, w1]}
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=80,
        name='All cells',
        marker_color='#94A3B8',
        opacity=0.6,
        histnorm='probability density',
    ))

    # KDE overlay
    from scipy.stats import gaussian_kde
    if len(values) > 10:
        kde_x = np.linspace(values.min(), values.max(), 300)
        kde_y = gaussian_kde(values)(kde_x)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='#1E40AF', width=2),
        ))

    # GMM fit
    if gmm_params:
        from scipy.stats import norm
        x = np.linspace(values.min(), values.max(), 300)
        _line_colors = ['#6EE7B7', '#FCA5A5']
        _fill_colors = ['rgba(110,231,183,0.15)', 'rgba(252,165,165,0.15)']
        for i, (mean, std, weight) in enumerate(zip(
            gmm_params['means'], gmm_params['stds'], gmm_params['weights']
        )):
            y = weight * norm.pdf(x, mean, std)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                name=f'GMM component {i+1}',
                line=dict(color=_line_colors[i % 2], width=1.5, dash='dot'),
                fill='tozeroy', fillcolor=_fill_colors[i % 2],
            ))

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='#DC2626',
        line_width=2,
        annotation_text=f'Gate: {threshold:.3f}',
        annotation_position='top right',
        annotation_font_color='#DC2626',
    )

    # Shaded regions
    ymax = 1.0  # relative — will scale to data
    fig.add_vrect(
        x0=values.min(), x1=threshold,
        fillcolor='#EF4444', opacity=0.05,
        layer='below', line_width=0,
    )
    fig.add_vrect(
        x0=threshold, x1=values.max(),
        fillcolor='#22C55E', opacity=0.08,
        layer='below', line_width=0,
    )

    n_pos = int((values >= threshold).sum())
    pct_pos = 100 * n_pos / len(values)

    fig.update_layout(
        title=dict(text=f'{marker_name}  —  {n_pos:,} positive ({pct_pos:.1f}%)', font_size=14),
        xaxis_title='Normalized intensity',
        yaxis_title='Density',
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation='h', y=1.08, x=0),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='#FFFFFF',
        showlegend=True,
    )
    return fig


def plot_spatial_scatter(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    gate_mask: np.ndarray,
    marker_name: str,
    max_points: int = 30_000,
) -> go.Figure:
    """
    Spatial scatter colored by gate status. Subsamples if n_cells > max_points.
    """
    n = len(x_coords)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        x_coords = x_coords[idx]
        y_coords = y_coords[idx]
        gate_mask = gate_mask[idx]

    pos_mask = gate_mask.astype(bool)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_coords[~pos_mask], y=y_coords[~pos_mask],
        mode='markers',
        marker=dict(size=2, color='#CBD5E1', opacity=0.4),
        name=f'{marker_name}−',
    ))
    fig.add_trace(go.Scatter(
        x=x_coords[pos_mask], y=y_coords[pos_mask],
        mode='markers',
        marker=dict(size=2.5, color='#16A34A', opacity=0.7),
        name=f'{marker_name}+',
    ))

    n_pos = int(pos_mask.sum())
    fig.update_layout(
        title=dict(text=f'Spatial — {n_pos:,} / {n:,} shown positive', font_size=13),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x'),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='#0F172A',
        paper_bgcolor='#FFFFFF',
        legend=dict(orientation='h', y=1.08, x=0),
    )
    return fig


def compute_gmm(values: np.ndarray) -> dict | None:
    """Fit 2-component GMM and return parameters + suggested threshold."""
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, random_state=42)
        gm.fit(values.reshape(-1, 1))
        means = gm.means_.flatten()
        stds = np.sqrt(gm.covariances_.flatten())
        weights = gm.weights_.flatten()
        order = np.argsort(means)
        means, stds, weights = means[order], stds[order], weights[order]

        # Suggested threshold: valley between peaks
        from scipy.signal import find_peaks
        x = np.linspace(values.min(), values.max(), 500)
        from scipy.stats import norm
        density = sum(w * norm.pdf(x, m, s) for w, m, s in zip(weights, means, stds))
        valleys, _ = find_peaks(-density)
        if len(valleys) > 0:
            threshold = float(x[valleys[0]])
        else:
            threshold = float(np.mean(means))

        return {
            'means': means.tolist(),
            'stds': stds.tolist(),
            'weights': weights.tolist(),
            'suggested_threshold': threshold,
        }
    except Exception:
        return None


def plot_phenotype_scatter(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    phenotype_mask: np.ndarray,
    phenotype_name: str,
    max_points: int = 40_000,
) -> go.Figure:
    """Spatial scatter for a phenotype (for the spatial config preview)."""
    n = len(x_coords)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        x_coords = x_coords[idx]
        y_coords = y_coords[idx]
        phenotype_mask = phenotype_mask[idx]

    pos = phenotype_mask.astype(bool)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_coords[~pos], y=y_coords[~pos],
        mode='markers',
        marker=dict(size=1.5, color='#1E293B', opacity=0.15),
        name='Other cells',
    ))
    fig.add_trace(go.Scatter(
        x=x_coords[pos], y=y_coords[pos],
        mode='markers',
        marker=dict(size=2.5, color='#2563EB', opacity=0.8),
        name=phenotype_name,
    ))

    fig.update_layout(
        title=dict(text=f'{phenotype_name}: {pos.sum():,} cells', font_size=13),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x'),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='#0F172A',
        paper_bgcolor='#FFFFFF',
        legend=dict(orientation='h', y=1.08),
    )
    return fig
