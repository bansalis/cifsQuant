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


def extract_tile(args):
    """Extract tile from per-channel files."""
    y, x, y_end, x_end, channel_files, dapi_ch = args

    try:
        n_channels = len(channel_files)

        # Read all channels directly - NO pre-checks for maximum speed
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
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers (default: 8 for speed)')

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

    print(f"Processing {len(coords)} tiles...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    # Process tiles
    tile_info = []
    batch_size = 8

    for batch_start in range(0, len(coords), batch_size):
        batch = coords[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(extract_tile, c): c for c in batch}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    tile_info.append(result)

        print(f"Progress: {len(tile_info)}/{len(coords)}", flush=True)
        gc.collect()

    print(f"\nTILING SUMMARY: {len(coords)} total tiles, {len(tile_info)} with data")

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


if __name__ == '__main__':
    main()
