#!/usr/bin/env python3
"""
Simple SCIMAP Analysis - Gating and Phenotyping Only
Focus: Load data → Gate markers → Phenotype cells → Generate plots
"""

import pandas as pd
import numpy as np
import scimap as sm
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import gamma
from scipy.optimize import minimize
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_triangle

warnings.filterwarnings('ignore')

# CONFIGURATION - EDIT THIS SECTION ONLY
CONFIG = {
    'markers': {
        'Channel_2': 'TOM', 
        'Channel_3': 'CD45',
        'Channel_4': 'AGFP',
        'Channel_6': 'PERK',
        'Channel_7': 'CD8B',
        'Channel_8': 'KI67',
        'Channel_10': 'CD3'
    },
    'phenotypes': [
        ['T_cells', 'CD3', 'pos'],
        ['CD8_T_cells', 'CD3,CD8B', 'allpos'],
        ['Immune', 'CD45', 'pos'],
        ['Proliferating', 'KI67', 'pos'],
        ['Proliferating_T', 'CD3,KI67', 'allpos'],
        ['Stressed_T', 'CD3,PERK', 'allpos']
    ],
    'sample_metadata': {
        'GUEST29': {'age_weeks': 8, 'genotype': 'cis', 'treatment': 'Control'},
        'GUEST30': {'age_weeks': 8, 'genotype': 'trans', 'treatment': 'Control'}
    },
    'manual_gates': {
        # Add manual thresholds here after reviewing plots
        # 'CD3': 0.6,
        # 'CD45': 0.7,
    }
}

def load_data(results_dir):
    """Load and combine all sample data"""
    print("Loading data...")
    
    all_data = []
    for sample_dir in Path(results_dir).iterdir():
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        quant_file = sample_dir / "final" / "combined_quantification.csv"
        
        if not quant_file.exists():
            continue
            
        df = pd.read_csv(quant_file)
        df = df.rename(columns=CONFIG['markers'])
        df['sample_id'] = sample_name
        
        # Add metadata
        if sample_name in CONFIG['sample_metadata']:
            meta = CONFIG['sample_metadata'][sample_name]
            for key, value in meta.items():
                df[key] = value
                
        all_data.append(df)
        print(f"  {sample_name}: {len(df):,} cells")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined: {len(combined_df):,} cells from {len(all_data)} samples")
    
    return combined_df

def inflection_gate(data, window=0.05):
    """
    Find inflection point in cumulative distribution.
    Used in SCIMAP for robust gating.
    """
    clean_data = np.sort(data[data > 0])
    
    if len(clean_data) < 100:
        return np.percentile(data, 90)
    
    # Cumulative distribution
    cdf = np.arange(1, len(clean_data) + 1) / len(clean_data)
    
    # Second derivative (inflection)
    grad = np.gradient(np.gradient(cdf))
    
    # Find maximum curvature above 50th percentile
    midpoint = len(clean_data) // 2
    inflection_idx = midpoint + np.argmax(np.abs(grad[midpoint:]))
    
    gate = clean_data[inflection_idx]
    
    # Verify biological plausibility
    pos_pct = (data > gate).mean()
    if pos_pct > 0.7:  # Too permissive
        gate = np.percentile(clean_data, 80)
    elif pos_pct < 0.02:  # Too stringent
        gate = np.percentile(clean_data, 70)
    
    return gate

def diagnose_raw_data(adata, output_dir):
    """Diagnose if transformations are causing issues"""
    
    print("\n" + "="*60)
    print("RAW DATA DIAGNOSIS")
    print("="*60)
    
    diag_dir = Path(output_dir) / "transformation_diagnosis"
    diag_dir.mkdir(exist_ok=True, parents=True)
    
    for marker in adata.var.index:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{marker} - Transformation Impact Analysis', fontsize=16)
        
        for i, sample in enumerate(adata.obs['sample_id'].unique()):
            sample_mask = adata.obs['sample_id'] == sample
            marker_idx = adata.var.index.get_loc(marker)
            
            # STEP 1: Raw data
            raw_vals = adata.X[sample_mask, marker_idx].copy()
            
            # STEP 2: After arcsinh
            cofactor = 5.0
            arcsinh_vals = np.arcsinh(raw_vals / cofactor)
            
            # STEP 3: After Combat (simulate)
            # Combat centers data - simulate what happens
            combat_vals = arcsinh_vals - arcsinh_vals.mean()
            combat_vals = combat_vals / arcsinh_vals.std()
            
            # Plot raw
            axes[0, i].hist(raw_vals[raw_vals > 0], bins=100, alpha=0.7)
            axes[0, i].set_title(f'{sample} - RAW')
            axes[0, i].set_ylabel('Frequency')
            
            # Plot arcsinh
            axes[1, i].hist(arcsinh_vals, bins=100, alpha=0.7, color='orange')
            axes[1, i].set_title(f'{sample} - After Arcsinh')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].axvline(0, color='red', linestyle='--')
            
            # Statistics
            print(f"\n{sample} - {marker}:")
            print(f"  Raw: min={raw_vals.min():.1f}, max={raw_vals.max():.1f}, "
                  f"zeros={np.sum(raw_vals==0)}/{len(raw_vals)}")
            print(f"  Arcsinh: min={arcsinh_vals.min():.3f}, max={arcsinh_vals.max():.3f}, "
                  f"mean={arcsinh_vals.mean():.3f}")
            print(f"  Combat(sim): min={combat_vals.min():.3f}, max={combat_vals.max():.3f}, "
                  f"mean={combat_vals.mean():.3f}")
            
            # Check if distributions overlap
            if i == 1:  # Second sample
                prev_sample = list(adata.obs['sample_id'].unique())[0]
                prev_mask = adata.obs['sample_id'] == prev_sample
                prev_raw = adata.X[prev_mask, marker_idx].copy()
                prev_arcsinh = np.arcsinh(prev_raw / cofactor)
                
                # KS test for distribution similarity
                from scipy.stats import ks_2samp
                ks_raw = ks_2samp(raw_vals[raw_vals > 0], prev_raw[prev_raw > 0])
                ks_arcsinh = ks_2samp(arcsinh_vals, prev_arcsinh)
                
                print(f"  KS test p-value (raw): {ks_raw.pvalue:.4f}")
                print(f"  KS test p-value (arcsinh): {ks_arcsinh.pvalue:.4f}")
                if ks_arcsinh.pvalue < 0.05 and ks_raw.pvalue >= 0.05:
                    print(f"  ⚠️ WARNING: Arcsinh CREATED batch effect!")
        
        plt.tight_layout()
        plt.savefig(diag_dir / f'{marker}_transformation_diagnosis.png', dpi=150)
        plt.close()
    
    print(f"\n✅ Diagnostics saved to: {diag_dir}")

def quantile_gate(data, bg_quantile=0.70, positive_fold=3.0, min_positive_fraction=0.02):
    """Quantile-based gating (Goltsev CODEX method)"""
    clean_data = data[data > 0]
    
    if len(clean_data) < 100:
        return np.percentile(data, 90)
    
    background = np.percentile(clean_data, bg_quantile * 100)
    cv = np.std(clean_data) / np.mean(clean_data)
    
    # Adaptive fold-change
    fold = positive_fold * (1.5 if cv > 0.8 else 1.2 if cv > 0.5 else 1.0)
    gate = background * fold
    
    # Verify fraction
    pos_frac = (clean_data > gate).mean()
    if pos_frac < min_positive_fraction:
        gate = np.percentile(clean_data, (1 - min_positive_fraction) * 100)
    elif pos_frac > 0.5:
        gate = np.percentile(clean_data, 75)
    
    return gate

def create_gating_plots(adata, output_dir):
    """Create comprehensive gating plots for manual threshold selection"""
    print("Creating gating plots...")
    
    plots_dir = Path(output_dir) / "gating_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Overview plot
    n_markers = len(adata.var.index)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    gating_summary = {}
    
    for i, marker in enumerate(adata.var.index):
        if i >= len(axes):
            break
            
        intensities = adata[:, marker].X.flatten()
        
        # Calculate threshold options
        thresholds = {
            'P85': np.percentile(intensities, 85),
            'P90': np.percentile(intensities, 90),
            'P95': np.percentile(intensities, 95),
            'Current': 0.5
        }
        
        # Store for summary
        gating_summary[marker] = {
            'current_positive_pct': (intensities > 0.5).mean() * 100,
            'thresholds': thresholds
        }
        
        # Overview plot
        axes[i].hist(intensities, bins=50, alpha=0.7, color='skyblue')
        axes[i].axvline(0.5, color='red', linestyle='--', linewidth=2)
        
        pos_pct = (intensities > 0.5).mean() * 100
        axes[i].set_title(f'{marker}\n{pos_pct:.1f}% positive')
        axes[i].set_xlabel('Intensity')
        
        # Individual detailed plot
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle(f'{marker} Gating Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution with threshold options
        axes2[0,0].hist(intensities, bins=100, alpha=0.7, density=True, color='skyblue')
        
        colors = ['red', 'orange', 'purple', 'brown']
        for j, (name, thresh) in enumerate(thresholds.items()):
            axes2[0,0].axvline(thresh, color=colors[j], linestyle='--', linewidth=2, 
                              label=f'{name}: {thresh:.3f}')
        
        axes2[0,0].set_title('Threshold Options')
        axes2[0,0].set_xlabel('Rescaled Intensity')
        axes2[0,0].set_ylabel('Density')
        axes2[0,0].legend()
        axes2[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Positive percentages at different thresholds
        threshold_range = np.linspace(0.1, 0.95, 50)
        pos_percentages = [(intensities > t).mean() * 100 for t in threshold_range]
        
        axes2[0,1].plot(threshold_range, pos_percentages, linewidth=2, color='green')
        axes2[0,1].axvline(0.5, color='red', linestyle='--', linewidth=2, 
                          label=f'Current: {pos_pct:.1f}%')
        axes2[0,1].set_xlabel('Threshold')
        axes2[0,1].set_ylabel('% Positive Cells')
        axes2[0,1].set_title('Positive % vs Threshold')
        axes2[0,1].legend()
        axes2[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Current classification
        pos_mask = intensities > 0.5
        
        # Sample for visualization if too many points
        if len(intensities) > 10000:
            sample_idx = np.random.choice(len(intensities), 10000, replace=False)
            sample_intensities = intensities[sample_idx]
            sample_pos_mask = pos_mask[sample_idx]
        else:
            sample_intensities = intensities
            sample_pos_mask = pos_mask
        
        np.random.seed(42)
        y_pos = np.random.normal(1, 0.05, sum(sample_pos_mask))
        y_neg = np.random.normal(0, 0.05, sum(~sample_pos_mask))
        
        axes2[1,0].scatter(sample_intensities[sample_pos_mask], y_pos, s=1, alpha=0.6, 
                          color='red', label=f'Positive: {sum(pos_mask):,}')
        axes2[1,0].scatter(sample_intensities[~sample_pos_mask], y_neg, s=1, alpha=0.6, 
                          color='blue', label=f'Negative: {sum(~pos_mask):,}')
        axes2[1,0].axvline(0.5, color='black', linestyle='--', linewidth=2)
        axes2[1,0].set_ylim(-0.3, 1.3)
        axes2[1,0].set_xlabel('Rescaled Intensity')
        axes2[1,0].set_title('Current Classification')
        axes2[1,0].legend()
        axes2[1,0].set_yticks([0, 1])
        axes2[1,0].set_yticklabels(['Negative', 'Positive'])
        
        # Plot 4: Instructions
        axes2[1,1].axis('off')
        
        instructions = f"""
TO SET MANUAL THRESHOLD:

1. Look at the plots above
2. Choose appropriate threshold
3. Edit scimap_pipeline.py CONFIG:

'manual_gates': {{
    '{marker}': 0.XX,  # Your threshold
}}

Current options:
  P85: {thresholds['P85']:.3f} ({(intensities > thresholds['P85']).mean()*100:.1f}%)
  P90: {thresholds['P90']:.3f} ({(intensities > thresholds['P90']).mean()*100:.1f}%)
  P95: {thresholds['P95']:.3f} ({(intensities > thresholds['P95']).mean()*100:.1f}%)

Then re-run the script.
        """
        
        axes2[1,1].text(0.05, 0.95, instructions, transform=axes2[1,1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{marker}_detailed_gating.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Save overview
    plt.suptitle('All Markers Gating Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'all_markers_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save gating summary
    with open(plots_dir / 'gating_summary.json', 'w') as f:
        json.dump(gating_summary, f, indent=2)
    
    print(f"✅ Gating plots saved to: {plots_dir}")
    return gating_summary

def run_phenotyping(adata, phenotypes):
    """Run cell phenotyping"""
    print("Running cell phenotyping...")
    
    # Convert to SCIMAP format
    max_markers = max(len(p[1].split(',')) for p in phenotypes)
    marker_cols = [f'marker{i+1}' for i in range(max_markers)]
    
    phenotype_data = []
    for phenotype, markers, gate in phenotypes:
        marker_list = markers.split(',')
        row = [phenotype] + marker_list + [''] * (max_markers - len(marker_list)) + [gate]
        phenotype_data.append(row)
        
    phenotype_workflow = pd.DataFrame(phenotype_data, columns=['phenotype'] + marker_cols + ['gate'])
    
    # Run phenotyping
    adata = sm.tl.phenotype_cells(adata, phenotype=phenotype_workflow, gate=0.5)
    
    # Print results
    phenotype_counts = adata.obs['phenotype'].value_counts()
    print(f"\n✅ Identified {len(phenotype_counts)} phenotypes:")
    for phenotype, count in phenotype_counts.items():
        print(f"  {phenotype}: {count:,} ({count/len(adata)*100:.1f}%)")
    
    return adata

def create_phenotype_plots(adata, output_dir):
    """Create phenotyping summary plots"""
    print("Creating phenotype plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Phenotype counts
    phenotype_counts = adata.obs['phenotype'].value_counts()
    axes[0,0].bar(range(len(phenotype_counts)), phenotype_counts.values)
    axes[0,0].set_xticks(range(len(phenotype_counts)))
    axes[0,0].set_xticklabels(phenotype_counts.index, rotation=45, ha='right')
    axes[0,0].set_title('Cell Phenotype Counts', fontweight='bold')
    axes[0,0].set_ylabel('Count')
    
    # Add count labels
    for i, v in enumerate(phenotype_counts.values):
        axes[0,0].text(i, v + max(phenotype_counts.values)*0.01, f'{v:,}', 
                       ha='center', va='bottom')
    
    # 2. Sample comparison if multiple samples
    if 'sample_id' in adata.obs.columns and adata.obs['sample_id'].nunique() > 1:
        sample_pheno = pd.crosstab(adata.obs['sample_id'], adata.obs['phenotype'], normalize='index')
        sample_pheno.plot(kind='bar', stacked=True, ax=axes[0,1], colormap='tab20')
        axes[0,1].set_title('Phenotype Proportions by Sample', fontweight='bold')
        axes[0,1].set_ylabel('Proportion')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1))
    
    # 3. Marker expression by phenotype
    phenotype_marker_mean = pd.DataFrame()
    for phenotype in adata.obs['phenotype'].unique():
        if phenotype != 'Unknown':
            cells = adata[adata.obs['phenotype'] == phenotype]
            if len(cells) > 0:
                mean_expression = pd.DataFrame(cells.X, columns=adata.var.index).mean()
                phenotype_marker_mean[phenotype] = mean_expression
    
    if not phenotype_marker_mean.empty:
        sns.heatmap(phenotype_marker_mean.T, annot=True, cmap='RdBu_r', 
                   center=0.5, ax=axes[1,0], fmt='.2f')
        axes[1,0].set_title('Mean Marker Expression by Phenotype', fontweight='bold')
    
    # 4. Spatial distribution if coordinates available
    if 'X_centroid' in adata.obs.columns and 'Y_centroid' in adata.obs.columns:
        # Subsample for visualization
        if len(adata) > 20000:
            idx = np.random.choice(len(adata), 20000, replace=False)
            plot_data = adata[idx]
        else:
            plot_data = adata
            
        phenotypes = plot_data.obs['phenotype'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(phenotypes)))
        
        for i, phenotype in enumerate(phenotypes):
            if phenotype != 'Unknown':
                mask = plot_data.obs['phenotype'] == phenotype
                if mask.sum() > 0:
                    axes[1,1].scatter(plot_data.obs.loc[mask, 'X_centroid'],
                                     plot_data.obs.loc[mask, 'Y_centroid'],
                                     c=[colors[i]], s=0.5, alpha=0.7, 
                                     label=phenotype)
        
        axes[1,1].set_xlabel('X coordinate')
        axes[1,1].set_ylabel('Y coordinate')
        axes[1,1].set_title('Spatial Distribution', fontweight='bold')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), markerscale=5)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'phenotype_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Phenotype plots saved")

def rosin_threshold(data):
    """Rosin unimodal thresholding from flow cytometry"""
    positive_vals = data[data > 0]
    if len(positive_vals) < 100:
        return np.percentile(data, 90)
    
    counts, bins = np.histogram(positive_vals, bins=256)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    peak_idx = np.argmax(counts)
    last_idx = len(counts) - 1
    while last_idx > peak_idx and counts[last_idx] < counts[peak_idx] * 0.01:
        last_idx -= 1
    
    if last_idx <= peak_idx:
        return np.percentile(positive_vals, 85)
    
    x1, y1 = peak_idx, counts[peak_idx]
    x2, y2 = last_idx, counts[last_idx]
    
    distances = []
    for i in range(peak_idx, last_idx):
        d = abs((y2-y1)*i - (x2-x1)*counts[i] + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances.append(d)
    
    if distances:
        threshold_idx = peak_idx + np.argmax(distances)
        return bin_centers[threshold_idx]
    
    return np.percentile(positive_vals, 85)

def fit_higher_order_mixtures(data):
    """Try 2,3,4 component Gaussian and Gamma mixtures, return conservative gate"""
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gamma
    from scipy.optimize import minimize
    
    positive_vals = data[data > 0]
    if len(positive_vals) < 100:
        return None
    
    gates = []
    
    # Try Gaussian mixtures (2,3,4 components)
    for n_comp in [2, 3, 4]:
        try:
            gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
            gmm.fit(positive_vals.reshape(-1, 1))
            
            # Sort components by mean
            means = gmm.means_.flatten()
            sorted_idx = np.argsort(means)
            
            # Gate between last two components (conservative)
            if n_comp >= 2:
                gate = (means[sorted_idx[-2]] + means[sorted_idx[-1]]) / 2
                gates.append(gate)
        except:
            pass
    
    # Try Gamma mixtures (2,3 components)
    for n_comp in [2, 3]:
        try:
            # Initialize with quantiles
            quantiles = np.percentile(positive_vals, np.linspace(10, 90, n_comp))
            
            def gamma_mixture_nll(params):
                weights = params[:n_comp-1]
                weights = np.append(weights, 1 - np.sum(weights))
                shapes = params[n_comp-1:2*n_comp-1]
                scales = params[2*n_comp-1:]
                
                if np.any(weights < 0) or np.any(weights > 1) or np.any(shapes <= 0) or np.any(scales <= 0):
                    return 1e10
                
                pdf = np.zeros_like(positive_vals)
                for w, a, b in zip(weights, shapes, scales):
                    pdf += w * gamma.pdf(positive_vals, a, scale=b)
                
                pdf = np.maximum(pdf, 1e-10)
                return -np.sum(np.log(pdf))
            
            init_params = [1/n_comp] * (n_comp-1) + [2.0] * n_comp + quantiles.tolist()
            bounds = [(0.01, 0.99)] * (n_comp-1) + [(0.1, 50)] * n_comp + [(0.1, 50000)] * n_comp
            
            result = minimize(gamma_mixture_nll, init_params, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                shapes = result.x[n_comp-1:2*n_comp-1]
                scales = result.x[2*n_comp-1:]
                means = shapes * scales
                
                # Gate between last two components
                if n_comp >= 2:
                    sorted_idx = np.argsort(means)
                    gate = (means[sorted_idx[-2]] + means[sorted_idx[-1]]) / 2
                    gates.append(gate)
        except:
            pass
    
    if gates:
        # Return most conservative (highest) gate
        return np.percentile(gates, 75)  # 75th percentile of all gates
    
    return None

def consensus_gate(data, marker_name, sample_name):
    """
    Consensus of standard methods + higher-order mixtures.
    Conservative approach for populations at both extremes.
    """
    from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_triangle
    
    positive_vals = data[data > 0]
    
    if len(positive_vals) < 100:
        return np.percentile(data, 90)
    
    methods = {}
    
    # Standard methods
    try:
        methods['otsu'] = threshold_otsu(positive_vals)
    except:
        pass
    
    try:
        methods['li'] = threshold_li(positive_vals)
    except:
        pass
    
    try:
        methods['yen'] = threshold_yen(positive_vals)
    except:
        pass
    
    try:
        methods['triangle'] = threshold_triangle(positive_vals)
    except:
        pass
    
    try:
        methods['rosin'] = rosin_threshold(data)
    except:
        pass
    
    # Higher-order mixtures (conservative)
    try:
        mixture_gate = fit_higher_order_mixtures(data)
        if mixture_gate is not None:
            methods['mixture'] = mixture_gate
    except:
        pass
    
    if not methods:
        return np.percentile(positive_vals, 85)
    
    # Print all methods
    print(f"{sample_name} {marker_name}:")
    for method, thresh in sorted(methods.items(), key=lambda x: x[1]):
        pct = (data > thresh).mean() * 100
        print(f"  {method}: {thresh:.0f} ({pct:.1f}%)")
    
    # Conservative consensus: use 75th percentile of all gates
    # This ensures we don't over-gate in either direction
    gate = np.percentile(list(methods.values()), 75)
    
    pct_final = (data > gate).mean() * 100
    print(f"  → consensus (P75): {gate:.0f} ({pct_final:.1f}%)")
    
    return gate

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple SCIMAP Analysis')
    parser.add_argument('--results_dir', required=True, help='Path to results directory')
    parser.add_argument('--output_dir', default='scimap_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE SCIMAP ANALYSIS - GATING & PHENOTYPING")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load data
    combined_df = load_data(args.results_dir)
    
    # 2. Convert to SCIMAP AnnData format manually
    print("\nConverting to SCIMAP format...")
    
    # Separate markers from metadata
    marker_cols = list(CONFIG['markers'].values())
    marker_data = combined_df[marker_cols].values
    
    # Create AnnData object
    import anndata
    adata = anndata.AnnData(X=marker_data)
    adata.var.index = marker_cols
    adata.obs = combined_df.drop(columns=marker_cols).reset_index(drop=True)
    
    print(f"AnnData: {adata.n_obs:,} cells × {adata.n_vars} markers")
    diagnose_raw_data(adata, output_dir)
    # Check for empty channels
    print("Checking data integrity...")
    for sample in adata.obs['sample_id'].unique():
        sample_data = adata[adata.obs['sample_id'] == sample]
        for marker in adata.var.index:
            marker_data = sample_data.X[:, adata.var.index.get_loc(marker)]
            if marker_data.max() == 0:
                print(f"Warning: {sample} {marker} has all zero values")
                # Set minimum non-zero value
                idx = adata.obs['sample_id'] == sample
                marker_idx = adata.var.index.get_loc(marker)
                adata.X[idx, marker_idx] = np.maximum(marker_data, 0.001)


    # Check for data corruption
    print("Validating data integrity...")
    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        sample_data = adata[sample_mask]
        print(f"{sample}: {sample_data.n_obs} cells")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(sample_data.X)) or np.any(np.isinf(sample_data.X)):
            print(f"  WARNING: {sample} contains NaN/inf values")
            adata.X[sample_mask] = np.nan_to_num(adata.X[sample_mask], nan=0, posinf=0, neginf=0)
        
        # Check marker ranges
        for i, marker in enumerate(adata.var.index):
            marker_vals = sample_data.X[:, i]
            if marker_vals.max() == marker_vals.min():
                print(f"  WARNING: {sample} {marker} has constant values")
                # Add tiny noise to break ties
                adata.X[sample_mask, i] += np.random.normal(0, 1e-6, sum(sample_mask))
    print("\nMarker variance summary:")
    print("\nChecking marker variation per sample before rescaling...")
    for sample in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample
        for i, marker in enumerate(adata.var.index):
            vals = adata.X[sample_mask, i]
            std = np.std(vals)
            if std == 0:
                print(f"  WARNING: {sample} {marker} has zero variation (std=0)")

    for sample in adata.obs['sample_id'].unique():
        print(f"\n{sample} marker ranges:")
        sample_mask = adata.obs['sample_id'] == sample
        for i, marker in enumerate(adata.var.index):
            vals = adata.X[sample_mask, i]
            print(f"  {marker}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")

    
    print("\n" + "="*60)
    print("CONSENSUS GATING (Standard + Higher-Order Mixtures)")
    print("="*60)

    adata_gated = adata.copy()
    raw_gates = {}

    for sample in adata.obs['sample_id'].unique():
        raw_gates[sample] = {}
        sample_mask = adata.obs['sample_id'] == sample
        
        for marker in adata.var.index:
            marker_idx = adata.var.index.get_loc(marker)
            raw_vals = adata.X[sample_mask, marker_idx].copy()
            
            gate = consensus_gate(raw_vals, marker, sample)
            raw_gates[sample][marker] = gate
            
            adata_gated.X[sample_mask, marker_idx] = (raw_vals > gate).astype(float)

    adata = adata_gated
    CONFIG['consensus_gates_used'] = raw_gates
    CONFIG['manual_gates'] = {m: 0.5 for m in adata.var.index}

    print("\n✅ Consensus gating complete - data is now binary")

    print("\nGMM gates applied per sample:")
    for sample in raw_gates.keys():
        print(f"\n{sample}:")
        for marker, gate in raw_gates[sample].items():
            print(f"  {marker}: {gate:.3f}")

    # 4. Create gating plots (data is already gated)
    gating_summary = create_gating_plots(adata, output_dir)

    # 5. Data already gated, just verify
    print("\nVerifying gating (data already binary)...")
    for marker in adata.var.index:
        pos_pct = adata[:, marker].X.mean() * 100
        pos_count = int(adata[:, marker].X.sum())
        print(f"  {marker}: {pos_count:,} positive ({pos_pct:.1f}%)")
    
    # 6. Run phenotyping
    adata = run_phenotyping(adata, CONFIG['phenotypes'])
    
    # 7. Create phenotype plots
    create_phenotype_plots(adata, output_dir)
    
    # 8. Save results
    print("\nSaving results...")
    adata.write(output_dir / 'scimap_gated_phenotyped.h5ad')
    adata.obs.to_csv(output_dir / 'phenotyped_cells.csv', index=False)
    
    # Summary
    summary = {
        'total_cells': len(adata),
        'markers': list(adata.var.index),
        'phenotype_counts': adata.obs['phenotype'].value_counts().to_dict(),
        'manual_gates_used': CONFIG.get('manual_gates', {}),
        'samples': adata.obs['sample_id'].nunique() if 'sample_id' in adata.obs.columns else 1
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📊 Cells analyzed: {len(adata):,}")
    print(f"🏷️ Phenotypes: {len(adata.obs['phenotype'].unique())}")
    print(f"📁 Output: {output_dir}")
    print("\nFiles created:")
    print("- gating_plots/ (threshold visualization)")
    print("- phenotype_summary.png")  
    print("- scimap_gated_phenotyped.h5ad")
    print("- phenotyped_cells.csv")
    print("- analysis_summary.json")
    
    if not CONFIG.get('manual_gates'):
        print(f"\n💡 Review plots in {output_dir}/gating_plots/")
        print("💡 Set manual gates in CONFIG if needed, then re-run")

if __name__ == '__main__':
    main()