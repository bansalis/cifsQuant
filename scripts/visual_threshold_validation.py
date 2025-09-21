#!/usr/bin/env python3
"""
Visual Threshold Validation
Creates plots showing example cells near thresholds for manual validation
"""

import os
import sys
import yaml
import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

def load_configs(config_path, phenotype_config_path):
    """Load both experiment and phenotyping configs"""
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    with open(phenotype_config_path, 'r') as f:
        pheno_config = yaml.safe_load(f)
    return exp_config, pheno_config

def load_data_and_thresholds(phenotyped_h5ad_path, thresholds_csv_path):
    """Load phenotyped data and thresholds"""
    print(f"Loading phenotyped data: {phenotyped_h5ad_path}")
    adata = ad.read_h5ad(phenotyped_h5ad_path)
    
    print(f"Loading thresholds: {thresholds_csv_path}")
    thresholds_df = pd.read_csv(thresholds_csv_path)
    thresholds = dict(zip(thresholds_df['channel'], thresholds_df['threshold']))
    
    return adata, thresholds

def find_raw_image_path(slide_id, rawdata_path):
    """Find the raw OME-TIFF file for a given slide"""
    rawdata_dir = Path(rawdata_path)
    
    # Look for files matching the slide ID
    possible_files = list(rawdata_dir.glob(f"{slide_id}*.ome.tif")) + \
                    list(rawdata_dir.glob(f"{slide_id}*.tif"))
    
    if possible_files:
        return str(possible_files[0])
    else:
        return None

def load_raw_image_region(raw_image_path, center_x, center_y, crop_size=200, pyramid_level=1):
    """
    Load a small region around a cell from the raw OME-TIFF
    
    Parameters:
    - center_x, center_y: cell coordinates 
    - crop_size: size of crop in pixels
    - pyramid_level: which pyramid level to use (1 for smaller images)
    """
    try:
        with tifffile.TiffFile(raw_image_path) as tif:
            # Use pyramid level 1 for faster loading
            if len(tif.series) > pyramid_level:
                series = tif.series[pyramid_level]
                # Scale coordinates based on pyramid level
                scale_factor = 2 ** pyramid_level
                center_x = int(center_x / scale_factor)
                center_y = int(center_y / scale_factor)
                crop_size = int(crop_size / scale_factor)
            else:
                series = tif.series[0]
            
            # Get image shape
            if len(series.pages) > 0:
                first_page = series.pages[0]
                height, width = first_page.shape
                n_channels = len(series.pages)
            else:
                return None
            
            # Calculate crop bounds
            half_crop = crop_size // 2
            x_start = max(0, center_x - half_crop)
            x_end = min(width, center_x + half_crop)
            y_start = max(0, center_y - half_crop) 
            y_end = min(height, center_y + half_crop)
            
            # Load all channels for this region
            crop_data = np.zeros((n_channels, y_end - y_start, x_end - x_start))
            
            for ch_idx in range(min(n_channels, 10)):  # Limit to 10 channels
                try:
                    page = series.pages[ch_idx]
                    channel_data = page.asarray()
                    crop_data[ch_idx] = channel_data[y_start:y_end, x_start:x_end]
                except:
                    continue
                    
            return crop_data
            
    except Exception as e:
        print(f"Error loading image region: {e}")
        return None

def get_cells_near_threshold(adata, channel, threshold, n_examples=20, window=0.1):
    """
    Get cells near the threshold for visual inspection
    
    Parameters:
    - window: fraction of threshold range to consider "near" (0.1 = ±10%)
    """
    values = adata[:, channel].X.flatten()
    
    # Define "near threshold" window
    lower_bound = threshold * (1 - window)
    upper_bound = threshold * (1 + window)
    
    # Find cells in the window
    near_threshold_mask = (values >= lower_bound) & (values <= upper_bound)
    near_indices = np.where(near_threshold_mask)[0]
    
    if len(near_indices) == 0:
        return None, None, None
    
    # Sample examples
    n_sample = min(n_examples, len(near_indices))
    sampled_indices = np.random.choice(near_indices, n_sample, replace=False)
    
    # Get positive and negative examples
    pos_mask = values[sampled_indices] > threshold
    neg_mask = values[sampled_indices] <= threshold
    
    pos_indices = sampled_indices[pos_mask]
    neg_indices = sampled_indices[neg_mask]
    
    return pos_indices, neg_indices, values[sampled_indices]

def create_cell_image_crops(adata, channel, threshold, pos_indices, neg_indices, 
                           exp_config, output_dir, max_examples=6):
    """Create image crops showing actual cells near threshold"""
    
    if pos_indices is None and neg_indices is None:
        return 0
    
    rawdata_path = exp_config.get('rawdata_path', 'rawdata')
    
    # Combine indices and labels
    all_indices = []
    all_labels = []
    
    if pos_indices is not None and len(pos_indices) > 0:
        n_pos = min(max_examples//2, len(pos_indices))
        all_indices.extend(pos_indices[:n_pos])
        all_labels.extend(['Positive'] * n_pos)
    
    if neg_indices is not None and len(neg_indices) > 0:
        n_neg = min(max_examples//2, len(neg_indices))
        all_indices.extend(neg_indices[:n_neg])
        all_labels.extend(['Negative'] * n_neg)
    
    if len(all_indices) == 0:
        return 0
    
    # Create figure
    n_examples = len(all_indices)
    n_cols = min(3, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Raw Image Crops - {channel} (Threshold: {threshold:.1f})', fontsize=14)
    
    crops_created = 0
    
    for idx, (cell_idx, label) in enumerate(zip(all_indices, all_labels)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get cell coordinates and slide info
        cell_data = adata[cell_idx].obs.iloc[0]
        slide_id = cell_data['slide_id']
        x_coord = cell_data['X_centroid']
        y_coord = cell_data['Y_centroid']
        intensity = float(adata[cell_idx, channel].X.flatten()[0])  # Properly extract scalar
        
        # Find raw image path
        raw_image_path = find_raw_image_path(slide_id, rawdata_path)
        
        if raw_image_path is None:
            ax.text(0.5, 0.5, f'Raw image not found\nfor {slide_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label}: {intensity:.1f}')
            continue
        
        # Load image crop
        crop_data = load_raw_image_region(raw_image_path, x_coord, y_coord, 
                                        crop_size=150, pyramid_level=1)
        
        if crop_data is None:
            ax.text(0.5, 0.5, f'Failed to load\nimage region', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label}: {intensity:.1f}')
            continue
        
        # Find the channel index in the raw data
        channel_mapping = exp_config['channels']
        channel_idx = None
        for marker, ch_name in channel_mapping.items():
            if ch_name == channel:
                # Try to map to channel index (assuming Channel_X format)
                try:
                    channel_idx = int(ch_name.split('_')[1]) - 1  # Channel_1 -> index 0
                    break
                except:
                    continue
        
        if channel_idx is None or channel_idx >= crop_data.shape[0]:
            channel_idx = 0  # Default to first channel
        
        # Display the relevant channel
        crop_channel = crop_data[channel_idx]
        
        # Normalize for display
        if crop_channel.max() > crop_channel.min():
            crop_normalized = (crop_channel - crop_channel.min()) / (crop_channel.max() - crop_channel.min())
        else:
            crop_normalized = crop_channel
        
        # Show image
        im = ax.imshow(crop_normalized, cmap='viridis', interpolation='nearest')
        
        # Mark cell center
        center_y_crop, center_x_crop = crop_channel.shape[0]//2, crop_channel.shape[1]//2
        ax.plot(center_x_crop, center_y_crop, 'r+', markersize=10, markeredgewidth=2)
        
        # Color border based on classification
        border_color = 'blue' if label == 'Positive' else 'orange'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        
        ax.set_title(f'{label}: {intensity:.1f}\nSlide: {slide_id}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        crops_created += 1
    
    # Remove empty subplots
    for idx in range(n_examples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.remove()
    
    plt.tight_layout()
    
    # Save plot
    crop_dir = Path(output_dir) / "raw_image_crops"
    crop_dir.mkdir(exist_ok=True)
    plt.savefig(crop_dir / f'{channel}_threshold_crops.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if crops_created > 0:
        print(f"  Created {crops_created} image crops for {channel}")
    
    return crops_created

def create_threshold_validation_plot(adata, channel, threshold, pos_indices, neg_indices, 
                                   sampled_values, output_path):
    """Create visualization showing threshold and example cells"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Overall distribution with threshold
    ax1 = axes[0, 0]
    all_values = adata[:, channel].X.flatten()
    all_values = all_values[all_values > 0]  # Remove zeros
    
    ax1.hist(all_values, bins=100, alpha=0.7, density=True, color='lightblue', 
             label=f'All cells (n={len(all_values):,})')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.1f}')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{channel} - Overall Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Zoomed view around threshold
    ax2 = axes[0, 1]
    # Focus on region around threshold
    zoom_range = threshold * 2  # Show 2x threshold range
    zoom_mask = (all_values >= 0) & (all_values <= zoom_range)
    zoom_values = all_values[zoom_mask]
    
    if len(zoom_values) > 0:
        ax2.hist(zoom_values, bins=50, alpha=0.7, density=True, color='lightgreen')
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold: {threshold:.1f}')
        
        # Show sampled cells
        if len(sampled_values) > 0:
            for val in sampled_values:
                color = 'blue' if val > threshold else 'orange'
                ax2.axvline(val, color=color, alpha=0.3, linewidth=1)
        
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{channel} - Threshold Region')
    ax2.legend()
    
    # 3. Example positive cells
    ax3 = axes[1, 0]
    if len(pos_indices) > 0:
        pos_values = adata[pos_indices, channel].X.flatten()
        ax3.scatter(range(len(pos_values)), pos_values, 
                   color='blue', alpha=0.7, s=50)
        ax3.axhline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.1f}')
        ax3.set_xlabel('Cell Index')
        ax3.set_ylabel('Intensity')
        ax3.set_title(f'Positive Examples (n={len(pos_indices)})')
        ax3.legend()
        
        # Add value labels
        for i, val in enumerate(pos_values[:10]):  # Label first 10
            ax3.annotate(f'{val:.0f}', (i, val), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No positive examples\nnear threshold', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Positive Examples (n=0)')
    
    # 4. Example negative cells
    ax4 = axes[1, 1]
    if len(neg_indices) > 0:
        neg_values = adata[neg_indices, channel].X.flatten()
        ax4.scatter(range(len(neg_values)), neg_values, 
                   color='orange', alpha=0.7, s=50)
        ax4.axhline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.1f}')
        ax4.set_xlabel('Cell Index')
        ax4.set_ylabel('Intensity')
        ax4.set_title(f'Negative Examples (n={len(neg_indices)})')
        ax4.legend()
        
        # Add value labels
        for i, val in enumerate(neg_values[:10]):  # Label first 10
            ax4.annotate(f'{val:.0f}', (i, val), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No negative examples\nnear threshold', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Negative Examples (n=0)')
    
    plt.suptitle(f'Threshold Validation: {channel}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_phenotype_validation_plots(adata, exp_config, pheno_config, thresholds, output_dir):
    """Create validation plots for each phenotype"""
    print("Creating phenotype validation plots...")
    
    plot_dir = Path(output_dir) / "phenotype_validation"
    plot_dir.mkdir(exist_ok=True)
    
    channel_mapping = exp_config['channels']
    
    for pheno_name, rule in tqdm(pheno_config['phenotype_rules'].items(), 
                                desc="Creating validation plots"):
        
        # Get all markers involved in this phenotype
        all_markers = rule['positive'] + rule['negative']
        
        # Create subplot for this phenotype
        n_markers = len(all_markers)
        if n_markers == 0:
            continue
            
        fig, axes = plt.subplots(n_markers, 2, figsize=(12, 4*n_markers))
        if n_markers == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Phenotype Validation: {pheno_name}\n{rule["description"]}', 
                    fontsize=14)
        
        for idx, marker in enumerate(all_markers):
            if marker not in channel_mapping:
                continue
                
            channel = channel_mapping[marker]
            if channel not in thresholds:
                continue
                
            threshold = thresholds[channel]
            is_positive = marker in rule['positive']
            
            # Get example cells
            pos_indices, neg_indices, sampled_values = get_cells_near_threshold(
                adata, channel, threshold, n_examples=30, window=0.15
            )
            
            # Plot distribution
            ax_dist = axes[idx, 0]
            values = adata[:, channel].X.flatten()
            values = values[values > 0]
            
            ax_dist.hist(values, bins=100, alpha=0.7, density=True, color='lightblue')
            ax_dist.axvline(threshold, color='red', linestyle='--', linewidth=2,
                           label=f'Threshold: {threshold:.1f}')
            
            # Highlight expected region for this phenotype
            if is_positive:
                ax_dist.axvspan(threshold, values.max(), alpha=0.2, color='green',
                               label='Expected Positive')
            else:
                ax_dist.axvspan(0, threshold, alpha=0.2, color='orange',
                               label='Expected Negative')
            
            ax_dist.set_xlabel('Intensity')
            ax_dist.set_ylabel('Density')
            ax_dist.set_title(f'{marker} ({channel})')
            ax_dist.legend()
            ax_dist.set_yscale('log')
            
            # Plot example cells
            ax_examples = axes[idx, 1]
            
            if pos_indices is not None and len(pos_indices) > 0:
                pos_vals = adata[pos_indices, channel].X.flatten()
                ax_examples.scatter(range(len(pos_vals)), pos_vals, 
                                  color='blue', alpha=0.7, s=30, label='Positive')
            
            if neg_indices is not None and len(neg_indices) > 0:
                neg_vals = adata[neg_indices, channel].X.flatten() 
                offset = len(pos_indices) if pos_indices is not None else 0
                ax_examples.scatter(range(offset, offset + len(neg_vals)), neg_vals,
                                  color='orange', alpha=0.7, s=30, label='Negative')
            
            ax_examples.axhline(threshold, color='red', linestyle='--', linewidth=2)
            ax_examples.set_xlabel('Example Cell Index')
            ax_examples.set_ylabel('Intensity')
            ax_examples.set_title(f'Example Cells Near Threshold')
            ax_examples.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'{pheno_name}_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Phenotype validation plots saved to: {plot_dir}")

def create_summary_statistics(adata, pheno_config, output_dir):
    """Create summary statistics table"""
    
    stats_data = []
    
    for pheno_name, rule in pheno_config['phenotype_rules'].items():
        if pheno_name in adata.obs.columns:
            n_positive = adata.obs[pheno_name].sum()
            percentage = (n_positive / adata.n_obs) * 100
            
            # Per-slide breakdown
            slide_stats = adata.obs.groupby('slide_id')[pheno_name].agg(['sum', 'count'])
            slide_percentages = (slide_stats['sum'] / slide_stats['count'] * 100).round(2)
            
            stats_data.append({
                'phenotype': pheno_name,
                'description': rule['description'],
                'total_positive': n_positive,
                'total_percentage': round(percentage, 2),
                'positive_markers': ', '.join(rule['positive']),
                'negative_markers': ', '.join(rule['negative']),
                'slide_breakdown': slide_percentages.to_dict()
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Save summary
    summary_path = Path(output_dir) / "phenotype_validation" / "validation_summary.csv"
    stats_df.to_csv(summary_path, index=False)
    
    print(f"Validation summary saved to: {summary_path}")
    return stats_df

def create_comprehensive_channel_validation(adata, exp_config, thresholds, output_dir):
    """Create comprehensive validation for each channel including raw image crops"""
    print("Creating comprehensive channel validation with image crops...")
    
    plot_dir = Path(output_dir) / "threshold_validation"
    plot_dir.mkdir(exist_ok=True)
    
    total_crops = 0
    
    for channel, threshold in tqdm(thresholds.items(), desc="Processing channels"):
        if channel not in adata.var_names:
            continue
        
        # Get example cells near threshold
        pos_indices, neg_indices, sampled_values = get_cells_near_threshold(
            adata, channel, threshold, n_examples=20, window=0.15
        )
        
        if pos_indices is None and neg_indices is None:
            print(f"  Skipping {channel} - no cells near threshold")
            continue
        
        # Create statistical validation plot
        plot_path = plot_dir / f'{channel}_validation.png'
        create_threshold_validation_plot(adata, channel, threshold, pos_indices, 
                                       neg_indices, sampled_values, plot_path)
        
        # Create raw image crops
        crops_created = create_cell_image_crops(adata, channel, threshold, pos_indices, 
                                              neg_indices, exp_config, output_dir, 
                                              max_examples=6)
        total_crops += crops_created if crops_created else 0
    
    print(f"Created validation plots and {total_crops} total image crops")
    return total_crops

def main():
    parser = argparse.ArgumentParser(description='Visual threshold validation with raw image crops')
    parser.add_argument('--config', default='/app/spatial_analysis/configs/experiment_config.yaml')
    parser.add_argument('--phenotype-config', default='/app/spatial_analysis/configs/phenotyping_config.yaml')
    parser.add_argument('--phenotyped-data', required=True, help='Path to phenotyped h5ad file')
    parser.add_argument('--thresholds', required=True, help='Path to thresholds CSV file')
    parser.add_argument('--output', default='/app/spatial_analysis/outputs')
    
    args = parser.parse_args()
    
    # Load configs
    exp_config, pheno_config = load_configs(args.config, args.phenotype_config)
    print(f"Creating validation plots for: {exp_config['experiment_name']}")
    
    # Load data
    adata, thresholds = load_data_and_thresholds(args.phenotyped_data, args.thresholds)
    
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} channels")
    print(f"Available phenotypes: {[col for col in adata.obs.columns if col in pheno_config['phenotype_rules']]}")
    
    # Stage 1: Create comprehensive channel validation with image crops
    print("\n=== Stage 1: Channel Threshold Validation with Raw Image Crops ===")
    total_crops = create_comprehensive_channel_validation(adata, exp_config, thresholds, args.output)
    
    # Stage 2: Create phenotype validation plots (existing functionality)
    print("\n=== Stage 2: Phenotype Rule Validation ===")
    create_phenotype_validation_plots(adata, exp_config, pheno_config, thresholds, args.output)
    
    # Create summary statistics
    stats_df = create_summary_statistics(adata, pheno_config, args.output)
    
    print(f"\n✅ Visual validation complete!")
    print(f"📸 Created {total_crops} raw image crops showing cells near thresholds")
    print(f"📊 Created phenotype validation plots")
    print(f"📋 Generated summary statistics")
    
    print("\nPhenotype Summary:")
    for _, row in stats_df.iterrows():
        print(f"  {row['phenotype']}: {row['total_positive']} cells ({row['total_percentage']:.2f}%)")
    
    print(f"\n📁 Check these directories:")
    print(f"  - threshold_validation/: Statistical plots")
    print(f"  - raw_image_crops/: Actual cell images near thresholds")
    print(f"  - phenotype_validation/: Phenotype rule validation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())