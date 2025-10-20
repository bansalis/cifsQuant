#!/usr/bin/env python3
# diagnose_segmentation.py - Visual QC for segmentation quality

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

def create_diagnostic_plots(sample_dir, n_tiles=6):
    """Create side-by-side plots of DAPI, nuclei mask, cell mask"""
    
    sample_dir = Path(sample_dir).absolute()
    tiles_dir = sample_dir / "tiles"
    nuclei_dir = sample_dir / "nuclei_masks"
    cell_dir = sample_dir / "cell_masks"
    tile_info_file = tiles_dir / "tile_info.json"
    
    # Check if required directories and files exist
    if not all([tiles_dir.exists(), nuclei_dir.exists(), cell_dir.exists(), tile_info_file.exists()]):
        print(f"Skipping {sample_dir.name}: Missing required directories/files")
        return False
    
    try:
        with open(tile_info_file) as f:
            tile_info = json.load(f)
    except Exception as e:
        print(f"Skipping {sample_dir.name}: Error reading tile_info.json - {e}")
        return False
    
    # Filter tiles with cells
    tiles_with_cells = []
    for info in tile_info:
        tile_base = Path(info['filename']).stem
        # Remove _corrected suffix if present
        tile_base = tile_base.replace('_corrected', '')
        
        nuclei_mask = (nuclei_dir / f"{tile_base}_corrected_nuclei_mask.tif").absolute()
        if not nuclei_mask.exists():
            nuclei_mask = (nuclei_dir / f"{tile_base}_nuclei_mask.tif").absolute()
        
        if nuclei_mask.exists():
            try:
                mask = tifffile.imread(str(nuclei_mask))
                n_cells = len(np.unique(mask[mask > 0]))
                if n_cells > 20:
                    tiles_with_cells.append((tile_base, n_cells, info, nuclei_mask))
            except Exception as e:
                print(f"  Error reading {nuclei_mask.name}: {e}")
                continue
    
    if not tiles_with_cells:
        print(f"Skipping {sample_dir.name}: No tiles with >20 cells")
        return False
    
    # Sort and select tiles
    tiles_with_cells.sort(key=lambda x: x[1], reverse=True)
    selected = [tiles_with_cells[0]]
    if len(tiles_with_cells) > 1:
        selected.append(tiles_with_cells[len(tiles_with_cells)//2])
        selected.append(tiles_with_cells[-1])
    
    selected = selected[:n_tiles]
    
    # Create plots
    fig, axes = plt.subplots(len(selected), 4, figsize=(20, 5*len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (tile_base, n_cells, info, nuclei_mask) in enumerate(selected):
        tile_file = (tiles_dir / f"{tile_base}.tif").absolute()
        cell_file = (cell_dir / f"{tile_base}_corrected_cell.tif").absolute()
        if not cell_file.exists():
            cell_file = (cell_dir / f"{tile_base}_cell.tif").absolute()
        
        if not all([tile_file.exists(), nuclei_mask.exists(), cell_file.exists()]):
            print(f"  Missing files for {tile_base}")
            continue
        
        try:
            tile = tifffile.imread(str(tile_file))
            dapi = tile[0]
            nuclei_mask_img = tifffile.imread(str(nuclei_mask))
            cell_mask = tifffile.imread(str(cell_file))
            
            axes[idx, 0].imshow(dapi, cmap='gray', vmin=np.percentile(dapi, 1), 
                               vmax=np.percentile(dapi, 99))
            axes[idx, 0].set_title(f'DAPI - {tile_base}\nCells: {n_cells}')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(nuclei_mask_img, cmap='tab20', interpolation='nearest')
            axes[idx, 1].set_title(f'Nuclei Mask\n{len(np.unique(nuclei_mask_img))-1} nuclei')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(cell_mask, cmap='tab20', interpolation='nearest')
            axes[idx, 2].set_title(f'Cell Mask\n{len(np.unique(cell_mask))-1} cells')
            axes[idx, 2].axis('off')
            
            overlay = np.zeros((*dapi.shape, 3))
            overlay[..., 0] = dapi / dapi.max()
            overlay[..., 1] = (cell_mask > 0) * 0.5
            overlay[..., 2] = (nuclei_mask_img > 0) * 0.5
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title(f'Overlay\nPos: ({info["x_start"]}, {info["y_start"]})')
            axes[idx, 3].axis('off')
        except Exception as e:
            print(f"  Error processing tile {tile_base}: {e}")
            continue
    
    plt.tight_layout()
    output_file = sample_dir / "segmentation_diagnostics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    check_tile_boundaries(sample_dir, tile_info)
    return True

def check_tile_boundaries(sample_dir, tile_info):
    """Check for intensity discontinuities at tile boundaries"""
    
    sample_dir = Path(sample_dir).absolute()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    tiles_sorted = sorted(tile_info, key=lambda x: (x['y_start'], x['x_start']))
    
    plotted = 0
    for idx in range(len(tiles_sorted)-1):
        if plotted >= 3:
            break
            
        tile1 = tiles_sorted[idx]
        tile2 = tiles_sorted[idx+1]
        
        if abs(tile1['y_start'] - tile2['y_start']) < 100:
            tile1_path = (sample_dir / "tiles" / tile1['filename']).absolute()
            tile2_path = (sample_dir / "tiles" / tile2['filename']).absolute()
            
            if not tile1_path.exists() or not tile2_path.exists():
                continue
            
            try:
                img1 = tifffile.imread(str(tile1_path))[0]
                img2 = tifffile.imread(str(tile2_path))[0]
                
                boundary1 = img1[:, -100:]
                boundary2 = img2[:, :100]
                
                axes[0, plotted].imshow(boundary1, cmap='gray')
                axes[0, plotted].set_title(f'Tile {idx} right edge')
                axes[0, plotted].axis('off')
                
                axes[1, plotted].imshow(boundary2, cmap='gray')
                axes[1, plotted].set_title(f'Tile {idx+1} left edge')
                axes[1, plotted].axis('off')
                
                plotted += 1
            except:
                continue
    
    if plotted > 0:
        plt.suptitle('Tile Boundary Intensity Check', fontsize=16)
        plt.tight_layout()
        output_file = sample_dir / "boundary_check.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1]).absolute()
    else:
        results_dir = Path("results")
        if not results_dir.exists():
            print("Error: results/ directory not found in current directory")
            print("Usage: python diagnose_segmentation.py [results_directory]")
            sys.exit(1)
        results_dir = results_dir.absolute()
    
    if not results_dir.exists():
        print("Error: results/ directory not found")
        sys.exit(1)
    
    sample_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    for sample_dir in sample_dirs:
        print(f"\nProcessing {sample_dir.name}...")
        create_diagnostic_plots(sample_dir)
    
    print("\nDone!")