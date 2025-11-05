#!/usr/bin/env python3
"""Optimized tiling from per-channel TIFFs with batching and parallel extraction"""

import tifffile
import numpy as np
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
import shutil

def extract_tile_from_channel(args):
    """Extract one tile from channel data"""
    channel_data, coord = args
    return channel_data[coord['y']:coord['y_end'], coord['x']:coord['x_end']]

def prescreen_dapi(dapi_data, coords, contrast_threshold=2.0):
    """Filter tile coords based on DAPI signal"""
    valid = []
    for coord in coords:
        y, x = coord['y'], coord['x']
        ye, xe = coord['y_end'], coord['x_end']
        tile_dapi = dapi_data[y:ye, x:xe]
        
        # Check 1: Not empty
        if tile_dapi.max() == 0:
            continue
        
        # Check 2: Has contrast (nuclei vs background)
        p95, p5 = np.percentile(tile_dapi, [95, 5])
        dynamic_range = p95 - p5
        if p95 > 100 and dynamic_range > 50 and p95 / (p5 + 1) > 1.5:
            valid.append(coord)
    
    return valid

def main():
    parser = argparse.ArgumentParser(description='Optimized tiling from per-channel TIFFs')
    parser.add_argument('--sample_dir', required=True, help='Directory with per-channel .ome.tif files')
    parser.add_argument('--markers_csv', required=True, help='Markers CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for tiles')
    parser.add_argument('--tile_size', type=int, default=4096, help='Tile size in pixels')
    parser.add_argument('--overlap', type=int, default=512, help='Overlap between tiles')
    parser.add_argument('--dapi_channel', type=int, default=3, help='DAPI channel index for prescreening')
    parser.add_argument('--max_workers', type=int, default=8, help='Max parallel workers')
    parser.add_argument('--batch_size', type=int, default=20, help='Tiles per batch (controls memory)')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get channel files
    channel_files = sorted(Path(args.sample_dir).glob('*.ome.tif'))
    if not channel_files:
        print(f"❌ No .ome.tif files found in {args.sample_dir}")
        sys.exit(1)
    
    print(f"FAST MODE: Using {len(channel_files)} per-channel TIFF files")
    
    # Get dimensions from first file
    with tifffile.TiffFile(channel_files[0]) as tif:
        height, width = tif.pages[0].shape
    
    print(f"Image: {len(channel_files)} channels, {height}x{width} pixels")
    
    # Generate all possible tile coordinates
    step = args.tile_size - args.overlap
    all_coords = []
    for y in range(0, height - args.overlap, step):
        for x in range(0, width - args.overlap, step):
            all_coords.append({
                'y': y,
                'x': x,
                'y_end': min(y + args.tile_size, height),
                'x_end': min(x + args.tile_size, width),
                'name': f"tile_y{y:06d}_x{x:06d}.tif"
            })
    
    print(f"Total possible tiles: {len(all_coords)}")
    
    # Check for existing tiles (RESUME)
    existing = {f.name for f in Path(args.output_dir).glob('tile_*.tif')}
    coords_to_process = [c for c in all_coords if c['name'] not in existing]
    
    if not coords_to_process:
        print(f"✓ All {len(existing)} tiles exist, skipping...")
        
        # Ensure metadata files exist
        if not (Path(args.output_dir) / 'tile_info.json').exists():
            tile_info = [{
                'filename': c['name'],
                'y_start': c['y'],
                'x_start': c['x'],
                'y_end': c['y_end'],
                'x_end': c['x_end'],
                'height': c['y_end'] - c['y'],
                'width': c['x_end'] - c['x'],
                'channels': len(channel_files)
            } for c in all_coords if c['name'] in existing]
            
            with open(Path(args.output_dir) / 'tile_info.json', 'w') as f:
                json.dump(tile_info, f, indent=2)
        
        if not (Path(args.output_dir) / 'tile_list.txt').exists():
            with open(Path(args.output_dir) / 'tile_list.txt', 'w') as f:
                for name in sorted(existing):
                    f.write(name + '\n')
        
        if not (Path(args.output_dir) / 'markers_tiled.csv').exists():
            shutil.copy(args.markers_csv, Path(args.output_dir) / 'markers_tiled.csv')
        
        sys.exit(0)
    
    print(f"RESUME MODE: Found {len(existing)} existing tiles, skipping them...")
    print(f"Resuming: {len(coords_to_process)} tiles remaining to process")
    
    # Prescreen with DAPI to avoid processing empty tiles
    print(f"Prescreening {len(coords_to_process)} tile locations from DAPI channel...")
    dapi_data = tifffile.imread(channel_files[args.dapi_channel])
    valid_coords = prescreen_dapi(dapi_data, coords_to_process)
    del dapi_data
    
    empty_count = len(coords_to_process) - len(valid_coords)
    print(f"Prescreening complete:")
    print(f"  ✓ {len(valid_coords)} tiles with good signal")
    print(f"  ✗ {empty_count} empty tiles (no DAPI)")
    print(f"  → Processing {len(valid_coords)}/{len(coords_to_process)} tiles ({100*len(valid_coords)/len(coords_to_process):.1f}%)")
    
    if not valid_coords:
        print("⚠ No tiles passed prescreening criteria!")
        print("   All tiles were either empty or had insufficient signal.")
        sys.exit(0)
    
    # Process in batches to control memory
    coord_batches = [valid_coords[i:i+args.batch_size] 
                     for i in range(0, len(valid_coords), args.batch_size)]
    
    print(f"\nProcessing {len(valid_coords)} tiles in {len(coord_batches)} batches of {args.batch_size}")
    print(f"Estimated memory per batch: ~{args.batch_size * len(channel_files) * 4096 * 4096 * 2 / 1e9:.1f} GB")
    
    from concurrent.futures import ThreadPoolExecutor

    def extract_and_append_channel(args):
        """Thread-safe: extract tile from channel and append to file"""
        channel_data, coord, output_path, ch_idx, n_channels = args
        
        tile_extract = channel_data[coord['y']:coord['y_end'], coord['x']:coord['x_end']]
        
        # Read existing partial tile, update this channel, write back
        if ch_idx == 0:
            # First channel - create new file
            tile = np.zeros((n_channels, tile_extract.shape[0], tile_extract.shape[1]), dtype=np.uint16)
            tile[0] = tile_extract
        else:
            # Read existing, update one channel
            tile = tifffile.imread(output_path)
            tile[ch_idx] = tile_extract
        
        tifffile.imwrite(output_path, tile, photometric='minisblack', compression='lzw')
        return True

    # Process in batches
    BATCH_SIZE = 10  # Smaller batches = less memory
    coord_batches = [valid_coords[i:i+BATCH_SIZE] for i in range(0, len(valid_coords), BATCH_SIZE)]

    print(f"\nProcessing {len(valid_coords)} tiles in {len(coord_batches)} batches of {BATCH_SIZE}")

    for batch_idx, coord_batch in enumerate(coord_batches):
        print(f"\nBatch {batch_idx+1}/{len(coord_batches)}: {len(coord_batch)} tiles")
        
        # Process each channel
        for ch_idx, ch_file in enumerate(channel_files):
            print(f"  Ch {ch_idx+1}/{len(channel_files)}...", end='', flush=True)
            
            # Load channel ONCE
            channel_data = tifffile.imread(ch_file)
            
            # ThreadPoolExecutor - shares memory, no copying
            with ThreadPoolExecutor(max_workers=4) as executor:
                tasks = [(channel_data, coord, 
                        str(Path(args.output_dir) / coord['name']),
                        ch_idx, len(channel_files)) 
                        for coord in coord_batch]
                list(executor.map(extract_and_append_channel, tasks))
            
            del channel_data
            print(" ✓", flush=True)
        
        print(f"  Batch {batch_idx+1} complete!")

    # Save metadata for ALL tiles (existing + newly created)
    all_existing = existing | {c['name'] for c in valid_coords}
    tile_info = [{
        'filename': c['name'],
        'y_start': c['y'],
        'x_start': c['x'],
        'y_end': c['y_end'],
        'x_end': c['x_end'],
        'height': c['y_end'] - c['y'],
        'width': c['x_end'] - c['x'],
        'channels': len(channel_files)
    } for c in all_coords if c['name'] in all_existing]
    
    with open(Path(args.output_dir) / 'tile_info.json', 'w') as f:
        json.dump(tile_info, f, indent=2)
    
    with open(Path(args.output_dir) / 'tile_list.txt', 'w') as f:
        for info in tile_info:
            f.write(info['filename'] + '\n')
    
    # Copy markers CSV
    shutil.copy(args.markers_csv, Path(args.output_dir) / 'markers_tiled.csv')
    
    print(f"\n{'='*60}")
    print(f"✓ TILING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total tiles created: {len(tile_info)}")
    print(f"  - Previously existed: {len(existing)}")
    print(f"  - Newly created: {len(valid_coords)}")
    print(f"  - Skipped (empty): {empty_count}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()