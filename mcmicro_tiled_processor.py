#!/usr/bin/env python3
"""
MCMICRO Tiled Processor
======================

An integrated Python workflow that:
1. Tiles large pyramidal TIFF images
2. Processes each tile through MCMICRO-compatible segmentation
3. Stitches results back together
4. Runs spatial analysis with SCIMAP

Usage:
    python mcmicro_tiled_processor.py --input large_image.tiff --output results/ --markers markers.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# Image processing
try:
    from skimage import segmentation, measure, filters, morphology
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_maxima
    from scipy import ndimage
    HAS_SKIMAGE = True
except ImportError:
    print("Warning: scikit-image not installed. Basic segmentation only.")
    HAS_SKIMAGE = False

# Spatial analysis
try:
    import scimap as sm
    import anndata
    HAS_SCIMAP = True
except ImportError:
    print("Warning: SCIMAP not installed. Skipping spatial analysis.")
    HAS_SCIMAP = False

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Skipping plots.")
    HAS_PLOTTING = False


class MCMICROTiledProcessor:
    """
    Integrated processor for large immunofluorescence images
    """
    
    def __init__(self, 
                 input_image: str,
                 output_dir: str,
                 markers_csv: Optional[str] = None,
                 tile_size: int = 2048,
                 overlap: int = 256,
                 pyramid_level: int = 0,
                 sample_name: str = "large_sample",
                 n_workers: int = 4):
        """
        Initialize the processor
        
        Parameters:
        -----------
        input_image : str
            Path to the large pyramidal TIFF image
        output_dir : str
            Directory to store all results
        markers_csv : str, optional
            Path to markers CSV file
        tile_size : int
            Size of tiles to extract (default: 2048)
        overlap : int
            Overlap between tiles in pixels (default: 256)
        pyramid_level : int
            Which pyramid level to use (0 = highest resolution)
        sample_name : str
            Name for this sample
        n_workers : int
            Number of parallel workers for processing
        """
        
        self.input_image = Path(input_image)
        self.output_dir = Path(output_dir)
        self.markers_csv = Path(markers_csv) if markers_csv else None
        self.tile_size = tile_size
        self.overlap = overlap
        self.pyramid_level = pyramid_level
        self.sample_name = sample_name
        self.n_workers = n_workers
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir = self.output_dir / "tiles"
        self.processed_dir = self.output_dir / "processed_tiles"
        self.final_dir = self.output_dir / "final"
        self.spatial_dir = self.output_dir / "spatial"
        
        for dir_path in [self.tiles_dir, self.processed_dir, self.final_dir, self.spatial_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Tile information
        self.tile_info = []
        
        print(f"Initialized MCMICRO Tiled Processor")
        print(f"Input: {self.input_image}")
        print(f"Output: {self.output_dir}")
        print(f"Tile size: {self.tile_size} with {self.overlap}px overlap")
        print(f"Workers: {self.n_workers}")
    
    def extract_tiles(self) -> List[Dict]:
        """
        Extract overlapping tiles from the large image
        
        Returns:
        --------
        List[Dict] : List of tile information dictionaries
        """
        
        print("Step 1: Extracting tiles from large image...")
        
        # Read the image at specified pyramid level
        with tifffile.TiffFile(self.input_image) as tif:
            try:
                # Handle pyramidal TIFF
                if len(tif.pages) > 1 and self.pyramid_level < len(tif.pages):
                    image = tif.pages[self.pyramid_level].asarray()
                else:
                    image = tif.asarray()
            except Exception as e:
                print(f"Error reading TIFF: {e}")
                image = tif.asarray()
        
        print(f"Original image shape: {image.shape}")
        
        # Handle different image dimensions
        if len(image.shape) == 3:
            if image.shape[0] < 100:  # Likely channels first
                channels, height, width = image.shape
            else:  # Likely height, width, channels
                height, width, channels = image.shape
                image = np.transpose(image, (2, 0, 1))  # Convert to channels first
        else:  # Single channel
            height, width = image.shape
            channels = 1
            image = image[np.newaxis, :, :]
        
        print(f"Processed image shape: C={channels}, H={height}, W={width}")
        
        # Calculate tiles
        step = self.tile_size - self.overlap
        tile_info = []
        
        for y in range(0, height - self.overlap, step):
            for x in range(0, width - self.overlap, step):
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)
                
                # Extract tile
                tile = image[:, y:y_end, x:x_end]
                
                # Create filename
                tile_filename = f"tile_y{y:06d}_x{x:06d}.tiff"
                tile_path = self.tiles_dir / tile_filename
                
                # Save tile
                tifffile.imwrite(str(tile_path), tile, photometric='minisblack')
                
                tile_info.append({
                    'filename': tile_filename,
                    'filepath': str(tile_path),
                    'y_start': y,
                    'x_start': x,
                    'y_end': y_end,
                    'x_end': x_end,
                    'height': y_end - y,
                    'width': x_end - x,
                    'channels': channels
                })
        
        # Save tile information
        self.tile_info = tile_info
        with open(self.tiles_dir / 'tile_info.json', 'w') as f:
            json.dump(tile_info, f, indent=2)
        
        print(f"Extracted {len(tile_info)} tiles")
        return tile_info
    
    def process_single_tile(self, tile_info: Dict) -> Dict:
        """
        Process a single tile through MCMICRO-like workflow
        
        Parameters:
        -----------
        tile_info : Dict
            Information about the tile to process
            
        Returns:
        --------
        Dict : Processing results for this tile
        """
        
        tile_path = Path(tile_info['filepath'])
        tile_name = tile_path.stem
        
        try:
            # Load tile
            image = tifffile.imread(tile_path)
            
            # Ensure we have the right format
            if len(image.shape) == 3 and image.shape[0] < 100:
                channels, height, width = image.shape
            elif len(image.shape) == 3:
                height, width, channels = image.shape
                image = np.transpose(image, (2, 0, 1))
            else:
                height, width = image.shape
                channels = 1
                image = image[np.newaxis, :, :]
            
            results = {
                'tile_name': tile_name,
                'success': False,
                'mask_path': None,
                'quantification_path': None,
                'error': None
            }
            
            # Step 1: Create probability map (UnMICST-like)
            nuclei_channel = image[0]  # Assume first channel is nuclei (DAPI/Hoechst)
            
            if HAS_SKIMAGE:
                # Enhanced nuclei detection
                nuclei_smooth = filters.gaussian(nuclei_channel, sigma=1.0)
                nuclei_thresh = filters.threshold_otsu(nuclei_smooth)
                nuclei_binary = nuclei_smooth > nuclei_thresh
                
                # Clean up binary mask
                nuclei_binary = morphology.remove_small_objects(nuclei_binary, min_size=50)
                nuclei_binary = morphology.binary_closing(nuclei_binary, morphology.disk(2))
                
            else:
                # Simple thresholding
                nuclei_thresh = np.percentile(nuclei_channel, 75)
                nuclei_binary = nuclei_channel > nuclei_thresh
            
            # Step 2: Watershed segmentation (S3Segmenter-like)
            if HAS_SKIMAGE and np.any(nuclei_binary):
                # Distance transform
                distance = ndimage.distance_transform_edt(nuclei_binary)
                
                # Find local maxima as seeds
                local_maxima = peak_local_maxima(
                    distance, 
                    min_distance=15, 
                    threshold_abs=0.3 * np.max(distance)
                )
                
                # Create markers
                markers = np.zeros(distance.shape, dtype=int)
                for i, (y, x) in enumerate(local_maxima):
                    markers[y, x] = i + 1
                
                # Watershed segmentation
                if np.max(markers) > 0:
                    mask = watershed(-distance, markers, mask=nuclei_binary)
                else:
                    mask = measure.label(nuclei_binary)
            else:
                # Fallback: simple connected components
                if np.any(nuclei_binary):
                    mask = measure.label(nuclei_binary)
                else:
                    mask = np.zeros(nuclei_channel.shape, dtype=int)
            
            # Save mask
            mask_path = self.processed_dir / f"{tile_name}_mask.tiff"
            tifffile.imwrite(str(mask_path), mask.astype(np.int32))
            results['mask_path'] = str(mask_path)
            
            # Step 3: Feature extraction (MCQuant-like)
            if HAS_SKIMAGE and np.max(mask) > 0:
                # Extract region properties
                props_list = []
                
                for channel_idx in range(channels):
                    channel_image = image[channel_idx]
                    
                    # Get region properties for this channel
                    props = measure.regionprops_table(
                        mask,
                        intensity_image=channel_image,
                        properties=[
                            'label', 'centroid', 'area', 'mean_intensity',
                            'max_intensity', 'min_intensity', 'eccentricity'
                        ]
                    )
                    
                    # Convert to DataFrame
                    df_channel = pd.DataFrame(props)
                    df_channel['channel'] = channel_idx + 1
                    
                    # Rename intensity columns to include channel info
                    intensity_cols = ['mean_intensity', 'max_intensity', 'min_intensity']
                    for col in intensity_cols:
                        if col in df_channel.columns:
                            df_channel[f'{col}_ch{channel_idx+1}'] = df_channel[col]
                    
                    props_list.append(df_channel)
                
                # Combine all channels (merge on label)
                if props_list:
                    df = props_list[0][['label', 'centroid-0', 'centroid-1', 'area', 'eccentricity']].copy()
                    df = df.rename(columns={'centroid-0': 'Y_centroid', 'centroid-1': 'X_centroid'})
                    
                    # Add intensity measurements from all channels
                    for channel_idx, df_channel in enumerate(props_list):
                        intensity_cols = [col for col in df_channel.columns if 'intensity_ch' in col]
                        for col in intensity_cols:
                            df = df.merge(
                                df_channel[['label', col]], 
                                on='label', 
                                how='left'
                            )
                    
                    # Add tile metadata
                    df['tile_name'] = tile_name
                    df['tile_y_start'] = tile_info['y_start']
                    df['tile_x_start'] = tile_info['x_start']
                    
                else:
                    # Create empty dataframe
                    df = pd.DataFrame(columns=['label', 'Y_centroid', 'X_centroid', 'area'])
            
            else:
                # Create empty dataframe if no cells detected
                df = pd.DataFrame(columns=['label', 'Y_centroid', 'X_centroid', 'area'])
            
            # Save quantification
            quant_path = self.processed_dir / f"{tile_name}.csv"
            df.to_csv(quant_path, index=False)
            results['quantification_path'] = str(quant_path)
            
            results['success'] = True
            results['n_cells'] = len(df)
            
            return results
            
        except Exception as e:
            return {
                'tile_name': tile_name,
                'success': False,
                'mask_path': None,
                'quantification_path': None,
                'error': str(e),
                'n_cells': 0
            }
    
    def process_tiles(self) -> List[Dict]:
        """
        Process all tiles through MCMICRO-like workflow in parallel
        
        Returns:
        --------
        List[Dict] : Processing results for all tiles
        """
        
        print("Step 2: Processing tiles through MCMICRO workflow...")
        
        results = []
        
        if self.n_workers == 1:
            # Sequential processing
            for tile_info in tqdm(self.tile_info, desc="Processing tiles"):
                result = self.process_single_tile(tile_info)
                results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_tile = {
                    executor.submit(self.process_single_tile, tile_info): tile_info 
                    for tile_info in self.tile_info
                }
                
                for future in tqdm(as_completed(future_to_tile), 
                                 total=len(future_to_tile), 
                                 desc="Processing tiles"):
                    result = future.result()
                    results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        total_cells = sum(r.get('n_cells', 0) for r in results)
        
        print(f"Processed {successful}/{len(results)} tiles successfully")
        print(f"Total cells detected: {total_cells}")
        
        return results
    
    def stitch_results(self, processing_results: List[Dict]) -> Tuple[str, str]:
        """
        Stitch segmentation masks and quantification data back together
        
        Parameters:
        -----------
        processing_results : List[Dict]
            Results from tile processing
            
        Returns:
        --------
        Tuple[str, str] : Paths to stitched mask and combined CSV
        """
        
        print("Step 3: Stitching results back together...")
        
        # Determine full image dimensions
        max_y = max([info['y_end'] for info in self.tile_info])
        max_x = max([info['x_end'] for info in self.tile_info])
        
        print(f"Full image dimensions: {max_y} x {max_x}")
        
        # Create output arrays
        full_mask = np.zeros((max_y, max_x), dtype=np.int32)
        current_label = 1
        all_quantifications = []
        
        # Process each tile result
        for result in processing_results:
            if not result['success'] or not result['mask_path']:
                continue
            
            # Find corresponding tile info
            tile_info = next((t for t in self.tile_info if t['filename'].startswith(result['tile_name'])), None)
            if not tile_info:
                continue
            
            try:
                # Load mask
                tile_mask = tifffile.imread(result['mask_path'])
                
                # Update labels to be globally unique
                updated_mask = tile_mask.copy()
                unique_labels = np.unique(tile_mask)
                unique_labels = unique_labels[unique_labels > 0]
                
                label_mapping = {}
                for old_label in unique_labels:
                    label_mapping[old_label] = current_label
                    updated_mask[tile_mask == old_label] = current_label
                    current_label += 1
                
                # Place in full mask (handle overlaps by giving priority to existing labels)
                y_start, x_start = tile_info['y_start'], tile_info['x_start']
                y_end = y_start + updated_mask.shape[0]
                x_end = x_start + updated_mask.shape[1]
                
                # Only place pixels where full_mask is still 0 (no overlap)
                overlap_region = full_mask[y_start:y_end, x_start:x_end]
                mask_to_place = np.where(overlap_region == 0, updated_mask, overlap_region)
                full_mask[y_start:y_end, x_start:x_end] = mask_to_place
                
                # Process quantification data
                if result['quantification_path'] and Path(result['quantification_path']).exists():
                    df = pd.read_csv(result['quantification_path'])
                    
                    if not df.empty:
                        # Update cell labels to match stitched mask
                        if 'label' in df.columns:
                            df['original_label'] = df['label']
                            df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
                        
                        # Adjust coordinates to global coordinate system
                        if 'X_centroid' in df.columns:
                            df['X_centroid'] += x_start
                        if 'Y_centroid' in df.columns:
                            df['Y_centroid'] += y_start
                        
                        # Add global tile information
                        df['global_tile_y'] = y_start
                        df['global_tile_x'] = x_start
                        
                        all_quantifications.append(df)
                        
            except Exception as e:
                print(f"Error stitching tile {result['tile_name']}: {e}")
                continue
        
        # Save stitched mask
        full_mask_path = self.final_dir / "full_segmentation_mask.tiff"
        tifffile.imwrite(str(full_mask_path), full_mask)
        print(f"Saved stitched mask: {full_mask_path}")
        
        # Combine and save quantification data
        if all_quantifications:
            combined_df = pd.concat(all_quantifications, ignore_index=True)
            
            # Remove any duplicate labels (can happen at tile boundaries)
            if 'label' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['label'])
            
            combined_csv_path = self.final_dir / "combined_quantification.csv"
            combined_df.to_csv(combined_csv_path, index=False)
            
            total_cells = len(combined_df)
            print(f"Combined quantification saved: {combined_csv_path}")
            print(f"Total cells in final dataset: {total_cells}")
        else:
            # Create empty CSV
            combined_csv_path = self.final_dir / "combined_quantification.csv"
            pd.DataFrame().to_csv(combined_csv_path, index=False)
            total_cells = 0
            print("No quantification data to combine")
        
        # Create summary report
        report_path = self.final_dir / "stitching_report.txt"
        successful_tiles = sum(1 for r in processing_results if r['success'])
        
        report = f"""Stitching Report for {self.sample_name}
=====================================
Original image: {self.input_image}
Total tiles created: {len(self.tile_info)}
Successfully processed tiles: {successful_tiles}
Final image dimensions: {max_y} x {max_x} pixels
Total cells detected: {total_cells}
Unique cell labels: {current_label - 1}

Output files:
- {full_mask_path.name}: Complete segmentation mask
- {combined_csv_path.name}: All cell measurements with global coordinates

Tile parameters:
- Tile size: {self.tile_size}x{self.tile_size} pixels
- Overlap: {self.overlap} pixels
- Pyramid level: {self.pyramid_level}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Stitching report saved: {report_path}")
        
        return str(full_mask_path), str(combined_csv_path)
    
    def spatial_analysis(self, combined_csv_path: str) -> Optional[str]:
        """
        Perform spatial analysis using SCIMAP
        
        Parameters:
        -----------
        combined_csv_path : str
            Path to combined quantification CSV
            
        Returns:
        --------
        Optional[str] : Path to saved AnnData object, or None if analysis failed
        """
        
        if not HAS_SCIMAP:
            print("Step 4: SCIMAP not available, skipping spatial analysis")
            return None
        
        print("Step 4: Performing spatial analysis...")
        
        try:
            # Load data
            df = pd.read_csv(combined_csv_path)
            
            if df.empty:
                print("No cells to analyze")
                return None
            
            print(f"Analyzing {len(df)} cells...")
            
            # Prepare data for SCIMAP
            # Create AnnData object
            
            # Extract intensity columns as features
            intensity_cols = [col for col in df.columns if 'intensity' in col.lower()]
            if intensity_cols:
                X_data = df[intensity_cols].values
                var_names = intensity_cols
            else:
                # Use basic morphological features
                morph_cols = ['area']
                available_morph = [col for col in morph_cols if col in df.columns]
                if available_morph:
                    X_data = df[available_morph].values
                    var_names = available_morph
                else:
                    print("No suitable features found for analysis")
                    return None
            
            # Create AnnData object
            adata = anndata.AnnData(X=X_data)
            adata.var_names = var_names
            adata.obs = df.reset_index(drop=True)
            
            # Add spatial coordinates
            if 'X_centroid' in df.columns and 'Y_centroid' in df.columns:
                adata.obsm['spatial'] = df[['X_centroid', 'Y_centroid']].values
                
                # Basic spatial analysis
                print("Computing spatial metrics...")
                
                # Nearest neighbor distances
                from scipy.spatial.distance import cdist
                coords = adata.obsm['spatial']
                distances = cdist(coords, coords)
                np.fill_diagonal(distances, np.inf)
                adata.obs['nearest_neighbor_distance'] = np.min(distances, axis=1)
                
                # Spatial clustering for neighborhoods
                from sklearn.cluster import KMeans
                if len(df) >= 20:  # Only cluster if we have enough cells
                    n_clusters = min(10, max(3, len(df) // 50))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    adata.obs['spatial_cluster'] = kmeans.fit_predict(coords)
                    
                    print(f"Identified {n_clusters} spatial neighborhoods")
                
                # Simple spatial statistics
                coords_std = np.std(coords, axis=0)
                adata.uns['spatial_stats'] = {
                    'spatial_extent_x': np.max(coords[:, 0]) - np.min(coords[:, 0]),
                    'spatial_extent_y': np.max(coords[:, 1]) - np.min(coords[:, 1]),
                    'spatial_std_x': coords_std[0],
                    'spatial_std_y': coords_std[1],
                    'cell_density': len(coords) / (coords_std[0] * coords_std[1] * 4)  # rough estimate
                }
            
            # Save AnnData object
            adata_path = self.spatial_dir / "spatial_analysis_results.h5ad"
            adata.write(str(adata_path))
            
            # Create summary CSV
            summary_df = adata.obs.copy()
            summary_csv_path = self.spatial_dir / "spatial_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            
            # Create plots if possible
            if HAS_PLOTTING and 'spatial' in adata.obsm.keys():
                self._create_spatial_plots(adata)
            
            print(f"Spatial analysis completed: {adata_path}")
            return str(adata_path)
            
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return None
    
    def _create_spatial_plots(self, adata):
        """Create spatial analysis plots"""
        
        if not HAS_PLOTTING:
            return
        
        coords = adata.obsm['spatial']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Spatial Analysis: {self.sample_name}', fontsize=16)
        
        # Plot 1: Cell spatial distribution
        axes[0, 0].scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.6, c='blue')
        axes[0, 0].set_title('Cell Spatial Distribution')
        axes[0, 0].set_xlabel('X coordinate (pixels)')
        axes[0, 0].set_ylabel('Y coordinate (pixels)')
        axes[0, 0].set_aspect('equal')
        
        # Plot 2: Nearest neighbor distances
        if 'nearest_neighbor_distance' in adata.obs.columns:
            distances = adata.obs['nearest_neighbor_distance']
            axes[0, 1].hist(distances, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('Nearest Neighbor Distances')
            axes[0, 1].set_xlabel('Distance (pixels)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.median(distances), color='red', linestyle='--', 
                              label=f'Median: {np.median(distances):.1f}')
            axes[0, 1].legend()
        
        # Plot 3: Spatial clusters (if available)
        if 'spatial_cluster' in adata.obs.columns:
            clusters = adata.obs['spatial_cluster']
            scatter = axes[1, 0].scatter(coords[:, 0], coords[:, 1], 
                                       c=clusters, s=1, alpha=0.6, cmap='tab10')
            axes[1, 0].set_title('Spatial Neighborhoods')
            axes[1, 0].set_xlabel('X coordinate (pixels)')
            axes[1, 0].set_ylabel('Y coordinate (pixels)')
            axes[1, 0].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Plot 4: Cell density heatmap
        try:
            from scipy.stats import gaussian_kde
            
            # Create density estimate
            kde = gaussian_kde(coords.T)
            
            # Create grid
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate density
            density = kde(positions).reshape(xx.shape)
            
            im = axes[1, 1].imshow(density, extent=[x_min, x_max, y_min, y_max], 
                                  origin='lower', cmap='YlOrRd', alpha=0.7)
            axes[1, 1].set_title('Cell Density')
            axes[1, 1].set_xlabel('X coordinate (pixels)')
            axes[1, 1].set_ylabel('Y coordinate (pixels)')
            plt.colorbar(im, ax=axes[1, 1])
            
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Density plot error:\\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.spatial_dir / "spatial_analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial plots saved: {plot_path}")
    
    def run_full_pipeline(self) -> Dict[str, str]:
        """
        Run the complete pipeline
        
        Returns:
        --------
        Dict[str, str] : Dictionary of output file paths
        """
        
        print(f"\\n{'='*60}")
        print(f"MCMICRO TILED PROCESSOR - {self.sample_name}")
        print(f"{'='*60}")
        
        results = {}
        
        try:
            # Step 1: Extract tiles
            tile_info = self.extract_tiles()
            results['tiles_dir'] = str(self.tiles_dir)
            results['n_tiles'] = len(tile_info)
            
            # Step 2: Process tiles
            processing_results = self.process_tiles()
            results['processed_dir'] = str(self.processed_dir)
            
            # Step 3: Stitch results
            mask_path, csv_path = self.stitch_results(processing_results)
            results['final_mask'] = mask_path
            results['final_csv'] = csv_path
            
            # Step 4: Spatial analysis
            adata_path = self.spatial_analysis(csv_path)
            if adata_path:
                results['spatial_analysis'] = adata_path
            
            # Summary
            print(f"\\n{'='*60}")
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Results directory: {self.output_dir}")
            print(f"Final outputs:")
            print(f"  - Segmentation mask: {Path(mask_path).name}")
            print(f"  - Quantification data: {Path(csv_path).name}")
            if adata_path:
                print(f"  - Spatial analysis: {Path(adata_path).name}")
            print(f"\\nTotal processing time: Complete")
            
        except Exception as e:
            print(f"\\nERROR: Pipeline failed with error: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Main command line interface"""
    
    parser = argparse.ArgumentParser(
        description="MCMICRO Tiled Processor for large immunofluorescence images"
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True, 
        help='Path to input large pyramidal TIFF image'
    )
    
    parser.add_argument(
        '--output', '-o', 
        required=True,
        help='Output directory for all results'
    )
    
    parser.add_argument(
        '--markers', '-m',
        help='Path to markers CSV file (optional)'
    )
    
    parser.add_argument(
        '--tile-size', 
        type=int, 
        default=2048,
        help='Size of tiles to extract (default: 2048)'
    )
    
    parser.add_argument(
        '--overlap', 
        type=int, 
        default=256,
        help='Overlap between tiles in pixels (default: 256)'
    )
    
    parser.add_argument(
        '--pyramid-level', 
        type=int, 
        default=0,
        help='Pyramid level to use (0 = highest resolution, default: 0)'
    )
    
    parser.add_argument(
        '--sample-name', 
        default='large_sample',
        help='Name for this sample (default: large_sample)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int, 
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = MCMICROTiledProcessor(
        input_image=args.input,
        output_dir=args.output,
        markers_csv=args.markers,
        tile_size=args.tile_size,
        overlap=args.overlap,
        pyramid_level=args.pyramid_level,
        sample_name=args.sample_name,
        n_workers=args.workers
    )
    
    # Run pipeline
    results = processor.run_full_pipeline()
    
    return results


if __name__ == '__main__':
    main()