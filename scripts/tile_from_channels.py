#!/usr/bin/env python3
"""
Helper script to tile from per-channel TIFF files using markers.csv
This runs BEFORE Nextflow to create tiles compatible with the existing pipeline.
"""

import tifffile
import numpy as np
import json
import pandas as pd
import glob
import sys
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import argparse


def match_marker_to_file(marker_name, available_files):
    """Match a marker name from markers.csv to an actual file with flexible fuzzy matching."""
    # Extract cycle/round (e.g., "2.0.4" from "R2.0.4_CY3_PERK")
    parts = marker_name.replace('R', '').split('_')
    cycle_round = parts[0]  # e.g., "2.0.4"

    # Build search terms from marker name
    search_terms = []
    search_terms.append(cycle_round)  # Must have cycle/round

    # Add channel and marker parts
    for part in parts[1:]:
        if part and len(part) > 1:  # Skip single chars
            search_terms.append(part)

    best_match = None
    best_score = 0

    for filepath in available_files:
        filename = Path(filepath).name.upper()
        marker_upper = marker_name.upper()

        score = 0

        # MUST match cycle/round exactly (critical!)
        if cycle_round not in Path(filepath).name:
            continue

        # Score based on how many terms match (case-insensitive)
        for term in search_terms:
            if term.upper() in filename:
                score += 2

        # Bonus: Check if any word in marker is in filename
        # This handles variations like PERK vs pERK
        marker_words = [w for w in marker_name.split('_') if len(w) > 2]
        for word in marker_words:
            # Check if word appears in filename (case-insensitive, partial match OK)
            if word.upper() in filename or word.lower() in filename.lower():
                score += 1
            # Also check without first char (e.g., "ERK" in "pERK")
            if len(word) > 3 and word[1:].upper() in filename:
                score += 1

        # Strong bonus if marker components appear in order
        if all(term.upper() in filename for term in search_terms):
            score += 10

        if score > best_score:
            best_score = score
            best_match = filepath

    if best_match and best_score >= 3:  # Require minimum score
        return best_match

    # Fallback: very lenient regex search
    # Build pattern from marker parts
    pattern_parts = []
    for part in parts:
        if part and len(part) > 1:
            pattern_parts.append(part)

    pattern = '.*'.join(pattern_parts)
    for filepath in available_files:
        if re.search(pattern, str(filepath), re.IGNORECASE):
            # Verify cycle/round matches
            if cycle_round in Path(filepath).name:
                return filepath

    return None


def prescreen_tiles_from_dapi(dapi_file, coords, tile_size):
    """
    FAST prescreening: Open DAPI once, scan all tile locations, return good coordinates.
    This is MUCH faster than opening DAPI file separately for each tile.
    """
    print(f"Prescreening {len(coords)} tile locations from DAPI channel...")

    # Open DAPI file ONCE
    with tifffile.TiffFile(dapi_file) as tif:
        dapi_full = tif.pages[0].asarray()

    good_coords = []
    empty_count = 0
    low_signal_count = 0

    for y, x, y_end, x_end, channel_files, dapi_ch in coords:
        # Extract tile region
        dapi_tile = dapi_full[y:y_end, x:x_end]

        # Quick checks
        if dapi_tile.max() == 0:
            empty_count += 1
            continue

        # Check for nuclei signal - use faster approximate percentiles
        p95 = np.percentile(dapi_tile, 95)
        p5 = np.percentile(dapi_tile, 5)
        dynamic_range = p95 - p5

        # Require: contrast AND minimum intensity
        if p95 < 100 or dynamic_range < 50 or p95 / (p5 + 1) < 1.5:
            low_signal_count += 1
            continue

        # This tile passes screening
        good_coords.append((y, x, y_end, x_end, channel_files, dapi_ch))

    del dapi_full
    gc.collect()

    print(f"Prescreening complete:")
    print(f"  ✓ {len(good_coords)} tiles with good signal")
    print(f"  ✗ {empty_count} empty tiles (no DAPI)")
    print(f"  ✗ {low_signal_count} low-signal tiles (weak contrast)")
    print(f"  → Processing {len(good_coords)}/{len(coords)} tiles ({100*len(good_coords)/len(coords):.1f}%)")

    return good_coords


def extract_tile(args):
    """Extract tile from per-channel files."""
    y, x, y_end, x_end, channel_files, dapi_ch = args

    try:
        n_channels = len(channel_files)

        # NO prescreening here - that's done once upfront!
        # Read all channels directly
        tile = np.zeros((n_channels, y_end - y, x_end - x), dtype=np.uint16)

        for c in range(n_channels):
            with tifffile.TiffFile(channel_files[c]) as tif:
                tile[c] = tif.pages[0].asarray()[y:y_end, x:x_end]

        filename = f"tile_y{y:06d}_x{x:06d}.tif"

        # Write UNCOMPRESSED for MAXIMUM SPEED - compression is the bottleneck!
        # Uncompressed tiles are 10-20x faster to write on slow WSL mounts
        tifffile.imwrite(filename, tile,
                        photometric='minisblack',
                        metadata={'axes': 'CYX'},
                        bigtiff=True)

        del tile
        gc.collect()

        return {
            'filename': filename,
            'y_start': y, 'x_start': x,
            'y_end': y_end, 'x_end': x_end,
            'height': y_end - y, 'width': x_end - x,
            'channels': n_channels,
            'has_data': True
        }
    except Exception as e:
        print(f"Error tile y{y}_x{x}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Tile per-channel TIFFs using markers.csv')
    parser.add_argument('--sample_dir', required=True, help='Directory with per-channel TIFFs')
    parser.add_argument('--markers_csv', required=True, help='markers.csv file')
    parser.add_argument('--output_dir', required=True, help='Output directory for tiles')
    parser.add_argument('--tile_size', type=int, default=8192, help='Tile size')
    parser.add_argument('--overlap', type=int, default=1024, help='Tile overlap')
    parser.add_argument('--dapi_channel', type=int, default=0, help='DAPI channel index')
    parser.add_argument('--max_workers', type=int, default=3, help='Number of parallel workers (default: 3 for memory safety)')
    parser.add_argument('--use_fast_temp', action='store_true', help='Write to fast Linux FS first, then bulk move (5x faster on WSL!)')

    args = parser.parse_args()

    # Read markers.csv
    markers_df = pd.read_csv(args.markers_csv)
    marker_names = markers_df['marker_name'].tolist()

    print(f"Reading markers.csv: {len(marker_names)} channels defined")

    # Find available files
    available_files = glob.glob(f"{args.sample_dir}/*.ome.tif")
    print(f"Found {len(available_files)} .ome.tif files in {args.sample_dir}")

    # Match markers to files and convert to ABSOLUTE paths
    channel_files = []
    for marker in marker_names:
        matched_file = match_marker_to_file(marker, available_files)
        if matched_file:
            # Convert to absolute path BEFORE changing directories
            abs_path = os.path.abspath(matched_file)
            channel_files.append(abs_path)
            print(f"  ✓ {marker} → {Path(matched_file).name}")
        else:
            print(f"  ✗ WARNING: No file found for marker '{marker}'")
            sys.exit(1)

    if len(channel_files) < len(marker_names) * 0.8:
        print(f"ERROR: Only matched {len(channel_files)}/{len(marker_names)} markers")
        sys.exit(1)

    print(f"FAST MODE: Using {len(channel_files)} per-channel TIFF files")

    # Get dimensions from first channel
    with tifffile.TiffFile(channel_files[0]) as tif:
        height = tif.pages[0].shape[0]
        width = tif.pages[0].shape[1]

    n_channels = len(channel_files)
    print(f"Image: {n_channels} channels, {height}x{width} pixels")

    # Generate tile coordinates
    step = args.tile_size - args.overlap
    coords = []
    for y in range(0, height - args.overlap, step):
        for x in range(0, width - args.overlap, step):
            y_end = min(y + args.tile_size, height)
            x_end = min(x + args.tile_size, width)
            coords.append((y, x, y_end, x_end, channel_files, args.dapi_channel))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    final_output_dir = os.path.abspath(args.output_dir)

    # TWO-STAGE WRITE STRATEGY for WSL performance
    if args.use_fast_temp:
        import tempfile
        import shutil
        # Use fast Linux filesystem for initial writes (5x faster than /mnt!)
        temp_dir = tempfile.mkdtemp(prefix='tiles_', dir='/tmp')
        work_dir = temp_dir
        print(f"⚡ FAST MODE: Writing to Linux FS first ({temp_dir})")
        print(f"   Will bulk-move to {final_output_dir} when done (5x faster!)")
    else:
        work_dir = final_output_dir

    # RESUME FUNCTIONALITY: Check for existing tiles
    existing_tiles = set()
    if os.path.exists(final_output_dir):
        for f in os.listdir(final_output_dir):
            if f.startswith('tile_') and f.endswith('.tif'):
                existing_tiles.add(f)

    if existing_tiles:
        print(f"RESUME MODE: Found {len(existing_tiles)} existing tiles, skipping them...")
        coords_to_process = []
        for y, x, y_end, x_end, ch_files, dapi in coords:
            tile_name = f"tile_y{y:06d}_x{x:06d}.tif"
            if tile_name not in existing_tiles:
                coords_to_process.append((y, x, y_end, x_end, ch_files, dapi))
        coords = coords_to_process
        print(f"Resuming: {len(coords)} tiles remaining to process")
    else:
        print(f"Processing {len(coords)} tiles from scratch...")

    # FAST PRESCREENING: Scan DAPI once to filter good tiles
    # This is MUCH faster than opening DAPI separately for each tile!
    if len(coords) > 0:
        dapi_file = channel_files[args.dapi_channel]
        coords = prescreen_tiles_from_dapi(dapi_file, coords, args.tile_size)

        if len(coords) == 0:
            print("⚠ No tiles passed prescreening criteria!")
            print("   All tiles were either empty or had insufficient signal.")
            sys.exit(1)

    os.chdir(work_dir)

    # Process tiles
    tile_info = []
    batch_size = 8
    tiles_completed = len(existing_tiles)  # Count of already-done tiles

    for batch_start in range(0, len(coords), batch_size):
        batch = coords[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(extract_tile, c): c for c in batch}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    tile_info.append(result)

        total_done = tiles_completed + len(tile_info)
        print(f"Progress: {total_done} tiles completed ({len(tile_info)} new, {tiles_completed} existing)", flush=True)
        gc.collect()

    total_tiles = tiles_completed + len(tile_info)
    print(f"\nTILING SUMMARY: {total_tiles} total tiles ({len(tile_info)} newly created, {tiles_completed} already existed)")

    # Save outputs
    with open('tile_info.json', 'w') as f:
        json.dump(tile_info, f, indent=2)

    with open('tile_list.txt', 'w') as f:
        for info in tile_info:
            f.write(info['filename'] + "\n")

    # Copy markers CSV
    import shutil
    shutil.copy(args.markers_csv, 'markers_tiled.csv')

    print(f"Tiling completed! Created {len(tile_info)} tiles")

    # BULK MOVE from temp to final destination if using fast temp
    if args.use_fast_temp:
        print(f"\n⚡ Bulk-moving {total_tiles} tiles to final destination...")
        print(f"   From: {work_dir}")
        print(f"   To:   {final_output_dir}")

        import time
        start_time = time.time()

        # Move all tile files
        for filename in os.listdir(work_dir):
            if filename.endswith('.tif') or filename.endswith('.json') or filename.endswith('.txt') or filename.endswith('.csv'):
                src = os.path.join(work_dir, filename)
                dst = os.path.join(final_output_dir, filename)
                shutil.move(src, dst)

        # Clean up temp dir
        os.rmdir(work_dir)

        elapsed = time.time() - start_time
        print(f"   ✓ Bulk move completed in {elapsed:.1f}s")
        print(f"   Final tiles location: {final_output_dir}")


if __name__ == '__main__':
    main()
