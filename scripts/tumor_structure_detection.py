#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Use only packages available in phenotype Docker
try:
    from skimage import morphology, measure, segmentation
    from sklearn.cluster import DBSCAN
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False
    print("Using basic clustering without scikit-image/sklearn")

def detect_tumor_structures(phenotype_csv_path, output_dir, min_tumor_size=50, 
                          dilation_radius=30, dbscan_eps=50, min_samples=10):
    """
    Detect tumor structures from phenotyped cells using spatial clustering 
    and morphological operations.
    
    Parameters:
    -----------
    phenotype_csv_path : str
        Path to phenotyped_cells.csv from phenotype_analysis.py
    output_dir : str  
        Directory to save results
    min_tumor_size : int
        Minimum number of cells to constitute a tumor
    dilation_radius : int
        Radius for morphological dilation (pixels)
    dbscan_eps : float
        DBSCAN clustering epsilon parameter
    min_samples : int
        Minimum samples for DBSCAN core points
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load phenotyped data
    df = pd.read_csv(phenotype_csv_path)
    print(f"Loaded {len(df)} phenotyped cells")
    print(f"Available columns: {list(df.columns)}")
    
    # Use consensus_phenotype column
    tumor_cells = df[df['consensus_phenotype'].str.contains('Tumor|tumor|PanCK|Epithelial', na=False, case=False)].copy()
    
    if len(tumor_cells) == 0:
        # Try hierarchical_phenotype if consensus doesn't have tumor cells
        tumor_cells = df[df['hierarchical_phenotype'].str.contains('Tumor|tumor|PanCK|Epithelial', na=False, case=False)].copy()
    
    if len(tumor_cells) == 0:
        print("No tumor cells found in consensus_phenotype or hierarchical_phenotype columns.")
        print(f"Available phenotypes: {df['consensus_phenotype'].unique()}")
        return

    print(f"Found {len(tumor_cells)} tumor cells ({len(tumor_cells)/len(df)*100:.1f}%)")
    
    if len(tumor_cells) < min_tumor_size:
        print("Insufficient tumor cells detected")
        return
    
    # Get image dimensions with padding
    max_x, max_y = int(df['X_centroid'].max()) + 100, int(df['Y_centroid'].max()) + 100
    print(f"Image dimensions: {max_y} x {max_x}")
    
    # Downsample for faster processing if image is very large
    downsample_factor = 1
    if max_x > 20000 or max_y > 20000:
        downsample_factor = 4
        print(f"Large image detected, downsampling by factor {downsample_factor}")
    
    # Create tumor cell density map
    tumor_coords = tumor_cells[['X_centroid', 'Y_centroid']].values
    if downsample_factor > 1:
        tumor_coords = tumor_coords / downsample_factor
    
    print("Running DBSCAN clustering...")
    # Use DBSCAN to identify tumor cell clusters
    if HAS_ADVANCED:
        clustering = DBSCAN(eps=dbscan_eps/downsample_factor, min_samples=min_samples)
        tumor_cells['cluster'] = clustering.fit_predict(tumor_coords)
        
        # Filter out noise (-1) and small clusters
        valid_clusters = []
        for cluster_id in np.unique(tumor_cells['cluster']):
            if cluster_id == -1:  # noise
                continue
            cluster_size = np.sum(tumor_cells['cluster'] == cluster_id)
            if cluster_size >= min_tumor_size:
                valid_clusters.append(cluster_id)
        
        print(f"Found {len(valid_clusters)} tumor clusters")
    else:
        # Simple fallback - treat all tumor cells as one cluster
        tumor_cells['cluster'] = 1
        valid_clusters = [1]
        print("Using simple clustering (all tumor cells as one cluster)")
    
    # Create binary tumor mask using morphological operations
    mask_height = (max_y // downsample_factor) + 100
    mask_width = (max_x // downsample_factor) + 100
    tumor_mask = np.zeros((mask_height, mask_width), dtype=bool)
    
    print("Creating tumor mask...")
    # Mark tumor cell locations
    for _, cell in tumor_cells.iterrows():
        if cell['cluster'] in valid_clusters:
            x = int(cell['X_centroid'] / downsample_factor)
            y = int(cell['Y_centroid'] / downsample_factor)
            if 0 <= y < mask_height and 0 <= x < mask_width:
                tumor_mask[y, x] = True
    
    print("Applying morphological operations...")
    # Apply morphological operations to create tumor regions (use smaller radius for downsampled)
    structure_radius = max(1, dilation_radius // downsample_factor)
    if HAS_ADVANCED:
        structure = morphology.disk(structure_radius)
        tumor_regions = morphology.binary_dilation(tumor_mask, structure)
        tumor_regions = morphology.binary_closing(tumor_regions, morphology.disk(structure_radius//2))
        tumor_regions = morphology.remove_small_objects(tumor_regions, min_size=(min_tumor_size*10)//downsample_factor)
        tumor_labels = measure.label(tumor_regions)
    else:
        # Simple fallback without scikit-image
        tumor_labels = np.where(tumor_mask, 1, 0)
        print("Using basic clustering (scikit-image not available)")
    
    print(f"Found {np.max(tumor_labels)} tumor regions")
    
    # Assign tumor regions to cells (optimized)
    print("Assigning cells to tumor regions...")
    df['tumor_region'] = 0
    df['distance_to_tumor'] = np.inf
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    for i in range(0, len(df), chunk_size):
        end_idx = min(i + chunk_size, len(df))
        chunk = df.iloc[i:end_idx]
        
        for idx, row in chunk.iterrows():
            x = int(row['X_centroid'] / downsample_factor)
            y = int(row['Y_centroid'] / downsample_factor)
            
            if 0 <= y < tumor_labels.shape[0] and 0 <= x < tumor_labels.shape[1]:
                region_id = tumor_labels[y, x]
                df.at[idx, 'tumor_region'] = region_id
                
                # Skip distance calculation for cells in tumor regions and large images
                if region_id == 0 and downsample_factor == 1 and len(df) < 100000:
                    # Only calculate distance for smaller datasets
                    tumor_boundary = tumor_regions.astype(int) if HAS_ADVANCED else tumor_mask.astype(int)
                    if np.any(tumor_boundary):
                        distance_map = ndimage.distance_transform_edt(~tumor_boundary)
                        df.at[idx, 'distance_to_tumor'] = distance_map[y, x] * downsample_factor
        
        if (i // chunk_size + 1) % 10 == 0:
            print(f"Processed {end_idx}/{len(df)} cells...")

    # Calculate tumor metrics
    print("Calculating tumor statistics...")
    tumor_stats = []
    for region_id in np.unique(tumor_labels)[1:]:  # skip 0 (background)
        region_cells = df[df['tumor_region'] == region_id]
        region_tumor_cells = region_cells[region_cells['consensus_phenotype'].str.contains('Tumor|tumor|PanCK|Epithelial', na=False, case=False)]
        region_immune_cells = region_cells[region_cells['consensus_phenotype'].str.contains('T cell|B cell|NK|Macrophage|Dendritic|CD', na=False, case=False)]

        # Calculate region properties
        if HAS_ADVANCED:
            region_mask = (tumor_labels == region_id).astype(np.uint8)
            region_props = measure.regionprops(region_mask)[0]
            
            stats = {
                'tumor_id': region_id,
                'total_cells': len(region_cells),
                'tumor_cells': len(region_tumor_cells),
                'immune_cells': len(region_immune_cells),
                'infiltration_rate': len(region_immune_cells) / len(region_cells) if len(region_cells) > 0 else 0,
                'area_pixels': region_props.area * (downsample_factor ** 2),
                'centroid_x': region_props.centroid[1] * downsample_factor,
                'centroid_y': region_props.centroid[0] * downsample_factor,
                'major_axis': region_props.major_axis_length * downsample_factor,
                'minor_axis': region_props.minor_axis_length * downsample_factor,
                'eccentricity': region_props.eccentricity,
                'solidity': region_props.solidity
            }
        else:
            # Fallback without skimage
            stats = {
                'tumor_id': region_id,
                'total_cells': len(region_cells),
                'tumor_cells': len(region_tumor_cells),
                'immune_cells': len(region_immune_cells),
                'infiltration_rate': len(region_immune_cells) / len(region_cells) if len(region_cells) > 0 else 0,
                'area_pixels': len(region_cells) * 100,  # rough estimate
                'centroid_x': region_cells['X_centroid'].mean(),
                'centroid_y': region_cells['Y_centroid'].mean(),
                'major_axis': 0,
                'minor_axis': 0,
                'eccentricity': 0,
                'solidity': 0
            }
        tumor_stats.append(stats)
    
    tumor_stats_df = pd.DataFrame(tumor_stats)
    
    # Save results
    df.to_csv(output_path / 'cells_with_tumor_regions.csv', index=False)
    tumor_stats_df.to_csv(output_path / 'tumor_statistics.csv', index=False)
    
    # Save tumor mask as TIFF
    try:
        import tifffile
        # Scale back to original resolution if downsampled
        if downsample_factor > 1:
            from scipy import ndimage
            original_shape = (max_y, max_x)
            tumor_mask_full = ndimage.zoom(tumor_labels.astype(np.float32), downsample_factor, order=0)
            # Crop to exact original size
            tumor_mask_full = tumor_mask_full[:original_shape[0], :original_shape[1]]
        else:
            tumor_mask_full = tumor_labels
        
        tifffile.imwrite(str(output_path / 'tumor_sections_mask.tiff'), 
                        tumor_mask_full.astype(np.uint16))
        print(f"Saved tumor mask: tumor_sections_mask.tiff")
    except ImportError:
        print("tifffile not available, skipping mask export")
        np.save(str(output_path / 'tumor_sections_mask.npy'), tumor_labels)
    
    # Create visualizations
    create_tumor_visualizations(df, tumor_labels, tumor_stats_df, output_path)
    
    print(f"Results saved to {output_path}")
    return df, tumor_stats_df

def create_tumor_visualizations(df, tumor_labels, tumor_stats, output_path):
    """Create comprehensive tumor structure visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Cell type spatial distribution
    tumor_cells = df[df['consensus_phenotype'].str.contains('Tumor|tumor|PanCK|Epithelial', na=False, case=False)]
    immune_cells = df[df['consensus_phenotype'].str.contains('T cell|B cell|NK|Macrophage|Dendritic|CD', na=False, case=False)]
    
    axes[0, 0].scatter(immune_cells['X_centroid'], immune_cells['Y_centroid'], 
                      s=1, alpha=0.6, c='blue', label='Immune cells')
    axes[0, 0].scatter(tumor_cells['X_centroid'], tumor_cells['Y_centroid'], 
                      s=1, alpha=0.8, c='red', label='Tumor cells')
    axes[0, 0].set_title('Cell Type Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    
    # 2. Tumor regions overlay
    axes[0, 1].imshow(tumor_labels, cmap='tab20', alpha=0.7)
    axes[0, 1].scatter(df['X_centroid'], df['Y_centroid'], s=0.5, alpha=0.3, c='black')
    axes[0, 1].set_title('Identified Tumor Regions')
    
    # 3. Distance to tumor heatmap
    infiltrating_cells = df[df['tumor_region'] > 0]
    if len(infiltrating_cells) > 0:
        axes[0, 2].scatter(infiltrating_cells['X_centroid'], infiltrating_cells['Y_centroid'],
                          c=infiltrating_cells['distance_to_tumor'], s=1, cmap='viridis_r')
        axes[0, 2].set_title('Distance to Tumor (Infiltrating Cells)')
    else:
        axes[0, 2].text(0.5, 0.5, 'No infiltrating cells', ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # 4. Tumor size distribution
    if len(tumor_stats) > 0:
        axes[1, 0].hist(tumor_stats['total_cells'], bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Total Cells per Tumor')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Tumor Size Distribution')
        
        # 5. Infiltration rate vs tumor size
        axes[1, 1].scatter(tumor_stats['total_cells'], tumor_stats['infiltration_rate'], 
                          s=50, alpha=0.7, c='purple')
        axes[1, 1].set_xlabel('Tumor Size (cells)')
        axes[1, 1].set_ylabel('Immune Infiltration Rate')
        axes[1, 1].set_title('Infiltration vs Tumor Size')
        
        # 6. Tumor shape analysis
        axes[1, 2].scatter(tumor_stats['eccentricity'], tumor_stats['solidity'],
                          s=tumor_stats['total_cells']/10, alpha=0.7, c='green')
        axes[1, 2].set_xlabel('Eccentricity')
        axes[1, 2].set_ylabel('Solidity')
        axes[1, 2].set_title('Tumor Shape Analysis')
    else:
        for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
            ax.text(0.5, 0.5, 'No tumor data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path / 'tumor_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if len(tumor_stats) > 0:
        stats_summary = {
            'Total Tumors': len(tumor_stats),
            'Median Tumor Size': tumor_stats['total_cells'].median(),
            'Mean Infiltration Rate': tumor_stats['infiltration_rate'].mean(),
            'Large Tumors (>200 cells)': len(tumor_stats[tumor_stats['total_cells'] > 200])
        }
        
        bars = ax.bar(stats_summary.keys(), stats_summary.values(), color=['red', 'orange', 'blue', 'green'])
        ax.set_title('Tumor Analysis Summary', fontsize=16)
        ax.set_ylabel('Count/Rate', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_summary.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_summary.values())*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No tumor statistics available', ha='center', va='center', transform=ax.transAxes)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'tumor_summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Detect tumor structures from phenotyped cells')
    parser.add_argument('--input', required=True, help='Path to results/ directory or specific phenotyped_cells.csv')
    parser.add_argument('--output', required=True, help='Output base directory (usually results/)')
    parser.add_argument('--min-tumor-size', type=int, default=50,
                        help='Minimum cells per tumor')
    parser.add_argument('--dilation-radius', type=int, default=30,
                        help='Morphological dilation radius')
    parser.add_argument('--dbscan-eps', type=float, default=50,
                        help='DBSCAN clustering epsilon')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input is results directory (batch mode) or single file
    if input_path.is_dir():
        # Look for phenotyped_cells.csv in results structure: results/[SLIDE]/phenotype_analysis/phenotyped_cells.csv
        # OR in phenotype_analysis structure: phenotype_analysis/[SLIDE]/phenotyped_cells.csv
        
        # Try results directory structure first
        phenotype_files = list(input_path.glob("*/final/combined_quantification.csv"))
        
        if not phenotype_files:
            # Try phenotype_analysis structure 
            phenotype_files = list(input_path.glob("**/phenotyped_cells.csv"))
        
        if not phenotype_files:
            print(f"No quantification or phenotype files found in {input_path}")
            print("Expected structure:")
            print("  results/[SLIDE]/final/combined_quantification.csv")
            print("  OR phenotype_analysis/[SLIDE]/phenotyped_cells.csv")
            return
        
        for csv_file in phenotype_files:
            # Determine slide name and appropriate file
            if csv_file.name == "combined_quantification.csv":
                # Using results structure - need to run phenotyping first
                slide_name = csv_file.parent.parent.name
                print(f"\nFound slide {slide_name} - running phenotyping first...")
                
                # Run phenotyping on this slide's data
                phenotype_output = csv_file.parent.parent / "phenotype_analysis"
                phenotype_output.mkdir(exist_ok=True)
                
                # Simple phenotyping for tumor detection
                df = pd.read_csv(csv_file)
                
                # Basic phenotyping using Channel_2 for tumors
                df['consensus_phenotype'] = 'Unknown'
                if 'Channel_2' in df.columns:
                    tumor_threshold = df['Channel_2'].quantile(0.7)
                    df.loc[df['Channel_2'] > tumor_threshold, 'consensus_phenotype'] = 'Tumor'
                
                # Save quick phenotyped data
                phenotyped_file = phenotype_output / "phenotyped_cells.csv"
                
                # Add sample_id if missing
                if 'sample_id' not in df.columns:
                    df['sample_id'] = slide_name
                
                df[['sample_id', 'X_centroid', 'Y_centroid', 'consensus_phenotype']].to_csv(phenotyped_file, index=False)
                
                # Now process tumor detection
                output_dir = csv_file.parent.parent / "tumor_analysis"
                
            else:
                # Using phenotype_analysis structure
                slide_name = csv_file.parent.name
                if slide_name in ["phenotype_analysis", "data"]:
                    slide_name = "combined_data"
                phenotyped_file = csv_file
                output_dir = Path(args.output) / slide_name / "tumor_analysis"
            
            print(f"\nProcessing tumor detection for {slide_name}...")
            
            detect_tumor_structures(
                str(phenotyped_file),
                str(output_dir),
                min_tumor_size=args.min_tumor_size,
                dilation_radius=args.dilation_radius,
                dbscan_eps=args.dbscan_eps
            )
    else:
        # Single file processing
        detect_tumor_structures(
            str(input_path),
            args.output,
            min_tumor_size=args.min_tumor_size,
            dilation_radius=args.dilation_radius,
            dbscan_eps=args.dbscan_eps
        )

if __name__ == '__main__':
    main()