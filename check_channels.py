#!/usr/bin/env python3
"""
Diagnostic script to check channel counts in images and markers file.
This helps debug channel mismatch errors.
"""

import sys
import os
import pandas as pd
import tifffile
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Check channel counts in images and markers')
    parser.add_argument('--image', help='Path to image file (TIFF)', required=False)
    parser.add_argument('--markers', help='Path to markers CSV', default='markers.csv')
    parser.add_argument('--tiles-dir', help='Path to tiles directory', required=False)
    args = parser.parse_args()

    print("=" * 60)
    print("Channel Count Diagnostic Tool")
    print("=" * 60)

    # Check markers file
    if os.path.exists(args.markers):
        print(f"\n📄 Markers file: {args.markers}")
        df = pd.read_csv(args.markers)
        print(f"   Columns: {list(df.columns)}")

        if 'marker_name' in df.columns:
            marker_names = df['marker_name'].tolist()
        elif len(df.columns) == 1:
            marker_names = df.iloc[:, 0].tolist()
        else:
            print(f"   ⚠️  WARNING: Unexpected format!")
            marker_names = []

        print(f"   Marker count: {len(marker_names)}")
        print(f"   Markers: {', '.join(marker_names[:5])}{'...' if len(marker_names) > 5 else ''}")
    else:
        print(f"\n❌ Markers file not found: {args.markers}")
        marker_names = []

    # Check single image
    if args.image:
        if os.path.exists(args.image):
            print(f"\n🖼️  Image: {args.image}")
            img = tifffile.imread(args.image)
            print(f"   Shape: {img.shape}")
            print(f"   Dtype: {img.dtype}")

            if img.ndim == 3:
                n_channels = img.shape[0]
                print(f"   Channel count: {n_channels}")

                if marker_names:
                    if len(marker_names) == n_channels:
                        print(f"   ✅ MATCH: Markers ({len(marker_names)}) == Channels ({n_channels})")
                    else:
                        print(f"   ❌ MISMATCH: Markers ({len(marker_names)}) != Channels ({n_channels})")
                        print(f"   Difference: {abs(len(marker_names) - n_channels)} channels")
            else:
                print(f"   ⚠️  WARNING: Image is not 3D (expected CxHxW)")
        else:
            print(f"\n❌ Image file not found: {args.image}")

    # Check tiles directory
    if args.tiles_dir:
        tiles_path = Path(args.tiles_dir)
        if tiles_path.exists():
            print(f"\n📁 Tiles directory: {args.tiles_dir}")
            tile_files = list(tiles_path.glob('tile_*.tif'))
            print(f"   Found {len(tile_files)} tiles")

            if tile_files:
                # Check first tile
                first_tile = tile_files[0]
                print(f"\n   Checking first tile: {first_tile.name}")
                img = tifffile.imread(first_tile)
                print(f"   Shape: {img.shape}")

                if img.ndim == 3:
                    n_channels = img.shape[0]
                    print(f"   Channel count: {n_channels}")

                    if marker_names:
                        if len(marker_names) == n_channels:
                            print(f"   ✅ MATCH: Markers ({len(marker_names)}) == Channels ({n_channels})")
                        else:
                            print(f"   ❌ MISMATCH: Markers ({len(marker_names)}) != Channels ({n_channels})")
                            print(f"   Difference: {abs(len(marker_names) - n_channels)} channels")
        else:
            print(f"\n❌ Tiles directory not found: {args.tiles_dir}")

    print("\n" + "=" * 60)
    print("\n💡 Solutions:")
    print("   1. If mismatch: Update markers.csv to match image channel count")
    print("   2. Delete TILE_LARGE_IMAGE cache: rm -rf work/")
    print("   3. Re-run pipeline without -resume to regenerate markers_tiled.csv")
    print("=" * 60)

if __name__ == '__main__':
    main()
