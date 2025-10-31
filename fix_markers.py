#!/usr/bin/env python3
"""
Script to automatically fix markers.csv to match image channel counts.
"""

import sys
import os
import pandas as pd
import tifffile
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Fix markers.csv to match image channel count')
    parser.add_argument('--image', help='Path to image file (TIFF)', required=True)
    parser.add_argument('--markers', help='Path to input markers CSV', default='markers.csv')
    parser.add_argument('--output', help='Path to output markers CSV', default='markers_fixed.csv')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.markers):
        print(f"❌ Error: Markers file not found: {args.markers}")
        sys.exit(1)

    print("=" * 60)
    print("Markers File Fixer")
    print("=" * 60)

    # Read image to get channel count
    print(f"\n📖 Reading image: {args.image}")
    img = tifffile.imread(args.image)
    print(f"   Shape: {img.shape}")

    if img.ndim != 3:
        print(f"❌ Error: Image must be 3D (CxHxW), got {img.ndim}D")
        sys.exit(1)

    n_channels = img.shape[0]
    print(f"   ✅ Image has {n_channels} channels")

    # Read markers file
    print(f"\n📖 Reading markers: {args.markers}")
    df = pd.read_csv(args.markers)
    print(f"   Columns: {list(df.columns)}")

    # Extract marker names
    if 'marker_name' in df.columns:
        marker_names = df['marker_name'].tolist()
    elif len(df.columns) == 1:
        marker_names = df.iloc[:, 0].tolist()
    else:
        print(f"❌ Error: Unexpected markers file format!")
        print(f"   Expected 'marker_name' column or single column")
        sys.exit(1)

    print(f"   Original marker count: {len(marker_names)}")

    # Adjust marker count
    if len(marker_names) == n_channels:
        print(f"\n✅ Already matches! No changes needed.")
        marker_names_fixed = marker_names
    elif len(marker_names) > n_channels:
        print(f"\n⚠️  Too many markers ({len(marker_names)}), truncating to {n_channels}")
        marker_names_fixed = marker_names[:n_channels]
        print(f"   Removed: {', '.join(marker_names[n_channels:])}")
    else:
        print(f"\n⚠️  Too few markers ({len(marker_names)}), padding to {n_channels}")
        marker_names_fixed = marker_names.copy()
        for i in range(len(marker_names), n_channels):
            new_name = f"Channel_{i+1}"
            marker_names_fixed.append(new_name)
            print(f"   Added: {new_name}")

    # Write fixed markers file
    print(f"\n💾 Writing fixed markers: {args.output}")
    output_df = pd.DataFrame({'marker_name': marker_names_fixed})
    output_df.to_csv(args.output, index=False)
    print(f"   ✅ Wrote {len(marker_names_fixed)} markers")

    print("\n" + "=" * 60)
    print("✨ Done! Next steps:")
    print(f"   1. Review the fixed file: cat {args.output}")
    print(f"   2. If correct, replace original: mv {args.output} {args.markers}")
    print(f"   3. Delete cache: rm -rf work/")
    print(f"   4. Re-run pipeline: nextflow run mcmicro-tiled.nf ...")
    print("=" * 60)

if __name__ == '__main__':
    main()
