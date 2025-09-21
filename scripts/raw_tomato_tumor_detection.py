#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

try:
    from skimage import morphology, measure, filters
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def detect_tumors_from_raw_tomato(results_dir, output_dir, tomato_threshold=0.7, 
                                 min_region_size=1000, closing_size=10):
    """
    Direct tumor detection from raw tomato channel image.
    
    Uses the actual tomato channel TIFF, not cell interpolation.
    
    Parameters:
    -----------
    results_dir : str
        Path to results/SLIDE/ directory containing tiles/
    tomato_threshold : float
        Intensity threshold (quantile) for tomato signal
    min_region_size : int
        Minimum pixels for tumor region
    closing_size : int
        Morphological closing size
    """
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find tomato channel tiles
    tiles_dir = results_path / "tiles"
    if not tiles_dir.exists():
        print(f"No tiles directory found in {results_path}")
        return
    
    # Look for tile files - should contain Channel_2 (tomato)
    tile_files = list(tiles_dir.glob("tile_*.tif*"))
    if not tile_files:
        print(f"No tile files found in {tiles_dir}")
        return
    
    print(f"Found {len(tile_files)} tile files")
    
    # Load and stitch tomato channel from tiles
    tomato_image = load_tomato_channel_from_tiles(tile_files)
    
    if tomato_image is None:
        print("Could not load tomato channel")
        return
    
    print(f"Loaded tomato image: {tomato_image.shape}")
    
    # Threshold tomato channel
    tomato_cutoff = np.percentile(tomato_image[tomato_image > 0], tomato_threshold * 100)
    print(f"Tomato intensity cutoff: {tomato_cutoff:.1f} ({tomato_threshold*100}th percentile)")
    
    # Create binary tumor mask
    tumor_binary = tomato_image > tomato_cutoff
    
    print(f"Tomato+ pixels: {np.sum(tumor_binary):,} ({np.sum(tumor_binary)/tumor_binary.size*100:.1f}%)")
    
    # Apply morphological operations
    if HAS_SKIMAGE:
        tumor_regions = process_raw_tomato_regions(tumor_binary, min_region_size, closing_size)
    else:
        tumor_regions = np.where(tumor_binary, 1, 0)
    
    print(f"Final tumor regions: {np.max(tumor_regions)}")
    
    # Load cell data and assign to regions
    cell_file = results_path / "final" / "combined_quantification.csv"
    if cell_file.exists():
        df = pd.read_csv(cell_file)
        df_with_regions = assign_cells_to_raw_regions(df, tumor_regions)
        
        # Calculate statistics
        region_stats = calculate_raw_region_stats(df_with_regions, tumor_regions, tomato_image)
        
        # Save results
        save_raw_tomato_results(df_with_regions, region_stats, tumor_regions, 
                               tomato_image, output_path)
    else:
        print(f"No cell data found at {cell_file}")
        # Save just the regions
        save_raw_tomato_results(None, None, tumor_regions, tomato_image, output_path)
    
    print(f"Raw tomato tumor detection completed: {output_path}")

def load_tomato_channel_from_tiles(tile_files):
    """Load and stitch tomato channel (Channel_2) from tile files"""
    
    print("Loading tomato channel from tiles...")
    
    # Parse tile positions and load Channel_2
    tiles_data = []
    
    for tile_file in tile_files:
        try:
            # Extract position from filename: tile_y000000_x003584.tif
            filename = tile_file.name
            y_pos = int(filename.split('_y')[1].split('_')[0])
            x_pos = int(filename.split('_x')[1].split('.')[0])
            
            # Load TIFF file
            with tifffile.TiffFile(tile_file) as tif:
                tile_data = tif.asarray()
                
                # Handle different tile formats
                if len(tile_data.shape) == 3:
                    # Multi-channel tile - get Channel_2 (index 1, 0-based)
                    if tile_data.shape[0] > 1:
                        tomato_channel = tile_data[1]  # Channel_2
                    else:
                        print(f"Warning: {filename} only has 1 channel")
                        continue
                elif len(tile_data.shape) == 2:
                    # Single channel tile - assume it's the right one
                    tomato_channel = tile_data
                else:
                    print(f"Warning: Unexpected tile shape {tile_data.shape} in {filename}")
                    continue
                
                tiles_data.append({
                    'y_pos': y_pos,
                    'x_pos': x_pos,
                    'data': tomato_channel,
                    'filename': filename
                })
                
        except Exception as e:
            print(f"Error loading {tile_file}: {e}")
            continue
    
    if not tiles_data:
        print("No valid tiles loaded")
        return None
    
    print(f"Loaded {len(tiles_data)} tiles with tomato channel")
    
    # Determine full image dimensions
    max_y = max(tile['y_pos'] + tile['data'].shape[0] for tile in tiles_data)
    max_x = max(tile['x_pos'] + tile['data'].shape[1] for tile in tiles_data)
    
    print(f"Reconstructing image: {max_y} x {max_x}")
    
    # Create full tomato image
    tomato_image = np.zeros((max_y, max_x), dtype=np.float32)
    
    for tile in tiles_data:
        y_start = tile['y_pos']
        x_start = tile['x_pos']
        tile_data = tile['data']
        
        y_end = y_start + tile_data.shape[0]
        x_end = x_start + tile_data.shape[1]
        
        # Place tile in full image
        tomato_image[y_start:y_end, x_start:x_end] = tile_data
    
    print(f"Tomato channel range: {tomato_image.min():.1f} - {tomato_image.max():.1f}")
    
    return tomato_image

def process_raw_tomato_regions(tumor_binary, min_region_size, closing_size):
    """Process binary tomato mask into clean tumor regions"""
    
    print("Processing raw tomato regions...")
    
    # Remove small noise
    cleaned = morphology.remove_small_objects(tumor_binary, min_size=100)
    
    # Close gaps
    closed = morphology.binary_closing(cleaned, morphology.disk(closing_size))
    
    # Fill holes
    filled = morphology.remove_small_holes(closed, area_threshold=500)
    
    # Remove small regions
    final_regions = morphology.remove_small_objects(filled, min_size=min_region_size)
    
    # Label connected components
    labeled_regions = measure.label(final_regions)
    
    print(f"Morphological processing: {np.sum(tumor_binary)} → {np.sum(final_regions)} pixels")
    
    return labeled_regions

def assign_cells_to_raw_regions(df, tumor_regions):
    """Assign cells to tumor regions detected from raw tomato"""
    
    print("Assigning cells to raw tomato regions...")
    
    df = df.copy()
    df['tomato_region'] = 0
    df['in_tomato_tumor'] = False
    
    for i, row in df.iterrows():
        x = int(row['X_centroid'])
        y = int(row['Y_centroid'])
        
        if (0 <= x < tumor_regions.shape[1] and 
            0 <= y < tumor_regions.shape[0]):
            
            region_id = tumor_regions[y, x]
            df.at[i, 'tomato_region'] = region_id
            df.at[i, 'in_tomato_tumor'] = region_id > 0
    
    cells_in_tumors = np.sum(df['in_tomato_tumor'])
    print(f"Assigned {cells_in_tumors}/{len(df)} cells to tumor regions")
    
    return df

def calculate_raw_region_stats(df, tumor_regions, tomato_image):
    """Calculate statistics for raw tomato regions"""
    
    stats = []
    
    for region_id in np.unique(tumor_regions)[1:]:  # Skip 0
        region_mask = tumor_regions == region_id
        region_area = np.sum(region_mask)
        
        # Raw tomato intensities in this region
        region_tomato = tomato_image[region_mask]
        
        # Cells in this region
        region_cells = df[df['tomato_region'] == region_id]
        
        # Calculate region properties
        if HAS_SKIMAGE:
            props = measure.regionprops(region_mask.astype(int))[0]
            centroid = props.centroid
            major_axis = props.major_axis_length
            minor_axis = props.minor_axis_length
            eccentricity = props.eccentricity
            solidity = props.solidity
        else:
            y_coords, x_coords = np.where(region_mask)
            centroid = (np.mean(y_coords), np.mean(x_coords))
            major_axis = minor_axis = np.sqrt(region_area)
            eccentricity = solidity = 0.5
        
        stat = {
            'region_id': region_id,
            'area_pixels': region_area,
            'total_cells': len(region_cells),
            'raw_tomato_mean': np.mean(region_tomato),
            'raw_tomato_max': np.max(region_tomato),
            'raw_tomato_std': np.std(region_tomato),
            'centroid_x': centroid[1],
            'centroid_y': centroid[0],
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'eccentricity': eccentricity,
            'solidity': solidity,
        }
        
        # Cell-based statistics if available
        if len(region_cells) > 0:
            stat.update({
                'cell_density': len(region_cells) / region_area,
                'mean_cell_tomato': region_cells['Channel_2'].mean(),
                'max_cell_tomato': region_cells['Channel_2'].max(),
            })
            
            if 'Area' in region_cells.columns:
                stat['mean_cell_area'] = region_cells['Area'].mean()
        
        stats.append(stat)
    
    return pd.DataFrame(stats)

def save_raw_tomato_results(df, stats, regions, tomato_image, output_path):
    """Save results from raw tomato analysis"""
    
    # Save data files
    if df is not None:
        df.to_csv(output_path / 'cells_with_raw_tomato_regions.csv', index=False)
    
    if stats is not None:
        stats.to_csv(output_path / 'raw_tomato_region_statistics.csv', index=False)
    
    # Save images
    tifffile.imwrite(str(output_path / 'raw_tomato_channel.tiff'), 
                    tomato_image.astype(np.float32))
    tifffile.imwrite(str(output_path / 'raw_tomato_tumor_regions.tiff'), 
                    regions.astype(np.uint16))
    
    # Create visualizations
    create_raw_tomato_plots(df, stats, regions, tomato_image, output_path)

def create_raw_tomato_plots(df, stats, regions, tomato_image, output_path):
    """Create visualization plots for raw tomato analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Raw tomato channel
    im1 = axes[0, 0].imshow(tomato_image, cmap='Reds', alpha=0.9)
    axes[0, 0].set_title('Raw Tomato Channel')
    plt.colorbar(im1, ax=axes[0, 0], label='Raw Intensity')
    
    # 2. Detected tumor regions
    im2 = axes[0, 1].imshow(regions, cmap='tab20', alpha=0.8)
    axes[0, 1].set_title(f'Tumor Regions from Raw Tomato ({np.max(regions)})')
    plt.colorbar(im2, ax=axes[0, 1], label='Region ID')
    
    # 3. Overlay
    axes[0, 2].imshow(tomato_image, cmap='Reds', alpha=0.7)
    axes[0, 2].contour(regions, levels=range(1, np.max(regions)+1), colors='white', linewidths=2)
    axes[0, 2].set_title('Tomato Channel + Region Boundaries')
    
    # 4. Intensity histogram
    tomato_flat = tomato_image[tomato_image > 0]
    axes[1, 0].hist(tomato_flat, bins=100, alpha=0.7, color='red')
    cutoff = np.percentile(tomato_flat, 70)
    axes[1, 0].axvline(cutoff, color='black', linestyle='--', 
                      label=f'70th percentile: {cutoff:.1f}')
    axes[1, 0].set_xlabel('Raw Tomato Intensity')
    axes[1, 0].set_ylabel('Pixel Count')
    axes[1, 0].set_title('Raw Tomato Intensity Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # 5 & 6. Statistics if available
    if stats is not None and len(stats) > 0:
        # Region size vs intensity
        axes[1, 1].scatter(stats['area_pixels'], stats['raw_tomato_mean'],
                          s=100, alpha=0.7, c='orange')
        axes[1, 1].set_xlabel('Region Area (pixels)')
        axes[1, 1].set_ylabel('Mean Raw Tomato Intensity')
        axes[1, 1].set_title('Region Size vs Raw Intensity')
        
        # Summary
        axes[1, 2].bar(['Total Regions', 'Large Regions', 'High Intensity'], 
                      [len(stats),
                       len(stats[stats['area_pixels'] > 2000]),
                       len(stats[stats['raw_tomato_mean'] > cutoff])],
                      color=['red', 'orange', 'darkred'])
        axes[1, 2].set_title('Raw Tomato Region Summary')
        
        for i, height in enumerate([len(stats),
                                   len(stats[stats['area_pixels'] > 2000]),
                                   len(stats[stats['raw_tomato_mean'] > cutoff])]):
            axes[1, 2].text(i, height + 0.1, str(height), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'raw_tomato_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    with open(output_path / 'raw_tomato_report.txt', 'w') as f:
        f.write(f"Raw Tomato Tumor Detection Report\n")
        f.write(f"==================================\n\n")
        f.write(f"Raw tomato image shape: {tomato_image.shape}\n")
        f.write(f"Tomato intensity range: {tomato_image.min():.1f} - {tomato_image.max():.1f}\n")
        f.write(f"Detected tumor regions: {np.max(regions)}\n")
        f.write(f"Total tumor pixels: {np.sum(regions > 0):,}\n")
        
        if stats is not None and len(stats) > 0:
            f.write(f"\nRegion Statistics:\n")
            f.write(f"- Average region size: {stats['area_pixels'].mean():.0f} pixels\n")
            f.write(f"- Largest region: {stats['area_pixels'].max():,} pixels\n")
            f.write(f"- Average raw intensity: {stats['raw_tomato_mean'].mean():.1f}\n")
            f.write(f"- Total cells in tumors: {stats['total_cells'].sum():,}\n")

def main():
    parser = argparse.ArgumentParser(description='Raw tomato channel tumor detection')
    parser.add_argument('--input', required=True, help='Path to results/ directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--tomato-threshold', type=float, default=0.7,
                        help='Intensity percentile threshold (default: 0.7)')
    parser.add_argument('--min-region-size', type=int, default=1000,
                        help='Minimum region size in pixels (default: 1000)')
    parser.add_argument('--closing-size', type=int, default=10,
                        help='Morphological closing size (default: 10)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Process all slides that have tiles directories
        slide_dirs = [d for d in input_path.iterdir() if d.is_dir() and (d / "tiles").exists()]
        
        if not slide_dirs:
            print(f"No slide directories with tiles/ found in {input_path}")
            return
        
        for slide_dir in slide_dirs:
            slide_name = slide_dir.name
            output_dir = slide_dir / "raw_tomato_analysis"
            
            print(f"\n=== Processing {slide_name} ===")
            
            detect_tumors_from_raw_tomato(
                str(slide_dir),
                str(output_dir),
                tomato_threshold=args.tomato_threshold,
                min_region_size=args.min_region_size,
                closing_size=args.closing_size
            )
    else:
        detect_tumors_from_raw_tomato(
            str(input_path),
            args.output,
            tomato_threshold=args.tomato_threshold,
            min_region_size=args.min_region_size,
            closing_size=args.closing_size
        )

if __name__ == '__main__':
    main()