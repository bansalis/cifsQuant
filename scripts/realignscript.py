#!/usr/bin/env python3
"""
Multiplexed IF Round-based Alignment and Stacking Tool

This script aligns multiple rounds of multiplexed immunofluorescence data
by first registering DAPI channels across rounds, then applying the same
shifts to all other channels within each round.

File naming convention expected: "INDIVIDUAL_ROUND_R000_STAIN-_SUFFIX.ome.tif"
Example: "GUEST29_1.0.4_R000_DAPI-_FINAL_F.ome.tif"

Requirements:
- tifffile
- numpy
- scikit-image
- opencv-python

Install with: pip install tifffile numpy scikit-image opencv-python
"""

# CONFIGURATION PARAMETERS - MODIFY THESE INSTEAD OF USING COMMAND LINE ARGUMENTS
CONFIG = {
    'input_dir': r'E:\CellDive_Imaging\batch22_timecourse\raw_images',  # Directory containing .ome.tif files
    'output_dir': r'E:\CellDive_Imaging\batch22_timecourse\rawdata',    # Directory to save output files
    'individuals': ['GUEST44'],  # List of individual IDs to process, or ['*'] for all
    'reference_round': "2.0.4",#None,            # Round to use as reference (e.g., "1.0.4"), None for auto
    'downsample_factor': 8,             # Downsample factor for registration
    'file_pattern': '*.ome.tif',        # File pattern to match
    'export_individual_channels': True, # Export shifted individual channel images
    'create_pyramidal_tiff': True,      # Create pyramidal TIFF output
    'create_uncompressed_stack': False,  # Create uncompressed full-resolution stack
    'dry_run': False,                   # Parse files and show organization without processing
    'use_gpu': True,                    # Enable GPU acceleration with CuPy
    'gpu_memory_fraction': 0.8          # Fraction of GPU memory to use (0.1-0.9)
}

import os
import re
import numpy as np
import tifffile
from skimage import registration, transform, filters, exposure
from skimage.feature import match_template
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import psutil
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set CUDA environment variables for Windows
def setup_cuda_environment():
    """Set up CUDA environment variables if not already set."""
    import os
    import glob
    
    if os.name == 'nt':  # Windows
        if 'CUDA_PATH' not in os.environ:
            # Common CUDA installation paths on Windows
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.*",
                r"C:\cuda\v12.*",
                r"C:\cuda\v11.*"
            ]
            
            cuda_path = None
            for pattern in cuda_paths:
                matches = glob.glob(pattern)
                if matches:
                    cuda_path = max(matches)  # Get highest version
                    break
            
            if cuda_path and os.path.exists(cuda_path):
                os.environ['CUDA_PATH'] = cuda_path
                os.environ['CUDA_HOME'] = cuda_path
                
                # Add CUDA bin to PATH if not present
                cuda_bin = os.path.join(cuda_path, 'bin')
                if cuda_bin not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
                
                logger.info(f"Set CUDA_PATH to: {cuda_path}")
                return True
            else:
                logger.warning("CUDA installation not found in common locations")
                return False
        else:
            logger.info(f"CUDA_PATH already set: {os.environ['CUDA_PATH']}")
            return True
    return True

# Setup CUDA environment before importing CuPy
setup_cuda_environment()

# GPU Setup and Testing
def check_gpu_availability():
    """
    Check if GPU acceleration with CuPy is available and working.
    Returns True if GPU is available and functional, False otherwise.
    """
    try:
        import cupy as cp
        logger.info("CuPy found, testing GPU availability...")
        
        if not cp.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return False
        
        # Test GPU memory and basic operations
        device = cp.cuda.Device()
        meminfo = device.mem_info
        total_memory_gb = meminfo[1] / 1024**3
        free_memory_gb = meminfo[0] / 1024**3
        
        logger.info(f"GPU memory: {total_memory_gb:.1f} GB total, {free_memory_gb:.1f} GB free")
        
        if free_memory_gb < 2.0:
            logger.warning("Insufficient GPU memory (<2GB free), falling back to CPU")
            return False
        
        # Test basic GPU operations
        test_array = cp.random.random((1000, 1000), dtype=cp.float32)
        test_fft = cp.fft.fft2(test_array)
        test_result = cp.mean(test_fft)
        
        logger.info("✅ GPU acceleration available and functional")
        return True
        
    except ImportError:
        logger.warning("CuPy not installed, falling back to CPU")
        return False
    except Exception as e:
        logger.warning(f"GPU test failed: {e}")
        # Check if it's a CUDA_PATH issue
        if "CUDA_PATH" in str(e) or "CUDA_HOME" in str(e):
            logger.info("Trying to continue without CUDA_PATH environment variable...")
            try:
                # Try importing without CUDA_PATH
                import cupy as cp
                if cp.cuda.is_available():
                    logger.info("✅ GPU available despite CUDA_PATH warning")
                    return True
            except:
                pass
        
        logger.warning("Falling back to CPU")
        return False

# Initialize GPU availability
GPU_AVAILABLE = check_gpu_availability() if CONFIG['use_gpu'] else False

if CONFIG['use_gpu'] and GPU_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    logger.info("Using GPU acceleration with CuPy")
else:
    logger.info("Using CPU processing")

def get_completed_individuals(output_dir: str) -> set:
    """
    Get list of individuals that have already been processed.
    Looks for pyramidal TIFF files as completion marker.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    
    completed = set()
    for file in output_path.glob("*_aligned_stack.tif"):
        # Extract individual ID from filename
        individual_id = file.stem.replace("_aligned_stack", "")
        completed.add(individual_id)
    
    return completed

def estimate_memory_required(data_shape: tuple, dtype: np.dtype, num_levels: int = 5) -> float:
    """
    Estimate memory required for pyramid generation in GB.
    Accounts for multiple pyramid levels and temporary arrays.
    """
    bytes_per_element = np.dtype(dtype).itemsize
    base_size_gb = (np.prod(data_shape) * bytes_per_element) / (1024**3)
    
    # Estimate total pyramid memory (sum of all levels + working memory)
    total_pyramid_gb = base_size_gb * 1.5  # ~1.33 for pyramid + 0.17 buffer
    
    return total_pyramid_gb

def check_memory_availability(required_gb: float) -> tuple:
    """
    Check if sufficient RAM and GPU memory is available.
    Returns (ram_ok, gpu_ok, ram_free_gb, gpu_free_gb)
    """
    # Check RAM
    ram = psutil.virtual_memory()
    ram_free_gb = ram.available / (1024**3)
    ram_ok = ram_free_gb >= required_gb
    
    # Check GPU memory if available
    gpu_free_gb = 0
    gpu_ok = False
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            meminfo = device.mem_info
            gpu_free_gb = meminfo[0] / (1024**3)
            gpu_ok = gpu_free_gb >= required_gb
        except:
            pass
    
    return ram_ok, gpu_ok, ram_free_gb, gpu_free_gb

class RoundBasedAligner:
    def __init__(self, individual_id: str, reference_round: Optional[str] = None, downsample_factor: int = 8):
        """
        Initialize the round-based aligner for a specific individual.
        
        Args:
            individual_id: ID of the individual (e.g., "GUEST29")
            reference_round: Round identifier to use as reference (e.g., "1.0.4"). 
                            If None, uses the first round alphabetically.
            downsample_factor: Factor to downsample images for faster registration
        """
        self.individual_id = individual_id
        self.reference_round = reference_round
        self.downsample_factor = downsample_factor
        self.round_shifts = {}
        self.physical_size_metadata = None
        
    def parse_filename(self, filepath: str) -> Tuple[str, str, str, str]:
        """
        Parse filename to extract individual, round and stain information.
        Stain name is between "R000_" and first occurrence of "-", "__", or "_AF".
        
        Returns:
            tuple: (individual_id, round_id, stain, full_filename)
        """
        filename = Path(filepath).name
        
        # Extract individual ID (first part before underscore)
        parts = filename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Filename doesn't match expected pattern: {filename}")
        
        individual_id = parts[0]
        
        # Extract round ID (X.Y.Z pattern)
        round_match = re.search(r'(\d+\.\d+\.\d+)', filename)
        if not round_match:
            raise ValueError(f"Could not find round pattern (X.Y.Z) in filename: {filename}")
        
        round_id = round_match.group(1)
        
        # Extract stain name - between "R000_" and first occurrence of "-", "__", or "_AF"
        # The stain name can contain underscores, so we need a more specific pattern
        stain_pattern = r'R000_(.+?)(?:-|__|_AF)'
        stain_match = re.search(stain_pattern, filename)
        
        if stain_match:
            stain = stain_match.group(1).upper()
        else:
            raise ValueError(f"Could not find stain pattern 'R000_STAIN' followed by '-', '__', or '_AF' in filename: {filename}")
        
        return individual_id, round_id, stain, filename

    def organize_files_by_round(self, file_list: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Organize files by round and stain for this individual.
        
        Returns:
            dict: {round_id: {stain: filepath, ...}, ...}
        """
        rounds_data = defaultdict(dict)
        
        for filepath in file_list:
            try:
                individual_id, round_id, stain, filename = self.parse_filename(filepath)
                
                # Only process files for this individual
                if individual_id == self.individual_id:
                    rounds_data[round_id][stain] = filepath
                    logger.debug(f"Parsed: {filename} -> Individual: {individual_id}, Round: {round_id}, Stain: {stain}")
            except ValueError as e:
                logger.warning(f"Skipping file due to parsing error: {e}")
                continue
        
        return dict(rounds_data)

    def load_channel(self, filepath: str) -> np.ndarray:
        """Load a single channel from OME-TIFF file."""
        logger.info(f"Loading {Path(filepath).name}")
        with tifffile.TiffFile(filepath) as tif:
            # Get the largest resolution (level 0)
            if hasattr(tif, 'series') and len(tif.series) > 0:
                return tif.series[0].asarray()
            else:
                return tif.asarray()

    def extract_physical_size_metadata(self, filepath: str) -> Dict[str, str]:
        """
        Extract physical size metadata from an OME-TIFF file.
        Returns dict with PhysicalSizeX, PhysicalSizeY, and their units.
        """
        def normalize_unit(unit_str: str) -> str:
            """Convert Unicode units to ASCII-safe equivalents."""
            if not unit_str:
                return 'um'
            # Replace µ (micro symbol) with 'u' for ASCII compatibility
            unit_str = unit_str.replace('µ', 'u').replace('μ', 'u')
            # Ensure it's ASCII
            try:
                unit_str.encode('ascii')
                return unit_str
            except UnicodeEncodeError:
                return 'um'  # Fallback to micrometers
        
        try:
            with tifffile.TiffFile(filepath) as tif:
                # Try to get OME-XML metadata
                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    
                    # Find the Pixels element
                    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//ome:Pixels', namespaces)
                    
                    if pixels is not None:
                        physical_size_x = pixels.get('PhysicalSizeX', '1.0')
                        physical_size_y = pixels.get('PhysicalSizeY', '1.0')
                        physical_size_x_unit = normalize_unit(pixels.get('PhysicalSizeXUnit', 'um'))
                        physical_size_y_unit = normalize_unit(pixels.get('PhysicalSizeYUnit', 'um'))
                        
                        return {
                            'PhysicalSizeX': physical_size_x,
                            'PhysicalSizeY': physical_size_y,
                            'PhysicalSizeXUnit': physical_size_x_unit,
                            'PhysicalSizeYUnit': physical_size_y_unit
                        }
                
                # Fallback: try to get from TIFF tags
                if tif.pages:
                    page = tif.pages[0]
                    if hasattr(page, 'tags'):
                        # Look for resolution tags
                        x_resolution = getattr(page.tags.get('XResolution'), 'value', None)
                        y_resolution = getattr(page.tags.get('YResolution'), 'value', None)
                        resolution_unit = getattr(page.tags.get('ResolutionUnit'), 'value', None)
                        
                        if x_resolution and y_resolution:
                            # Convert resolution to physical size (assuming pixels per unit)
                            if isinstance(x_resolution, (tuple, list)) and len(x_resolution) == 2:
                                physical_size_x = str(x_resolution[1] / x_resolution[0])
                            else:
                                physical_size_x = str(1.0 / x_resolution) if x_resolution else '1.0'
                            
                            if isinstance(y_resolution, (tuple, list)) and len(y_resolution) == 2:
                                physical_size_y = str(y_resolution[1] / y_resolution[0])
                            else:
                                physical_size_y = str(1.0 / y_resolution) if y_resolution else '1.0'
                            
                            unit = 'um'
                            if resolution_unit == 2:  # Inches
                                unit = 'inch'
                            elif resolution_unit == 3:  # Centimeters
                                unit = 'cm'
                            
                            return {
                                'PhysicalSizeX': physical_size_x,
                                'PhysicalSizeY': physical_size_y,
                                'PhysicalSizeXUnit': unit,
                                'PhysicalSizeYUnit': unit
                            }
        
        except Exception as e:
            logger.warning(f"Could not extract physical size metadata from {filepath}: {e}")
        
        # Default fallback with ASCII-safe units
        return {
            'PhysicalSizeX': '1.0',
            'PhysicalSizeY': '1.0',
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeYUnit': 'um'
        }

    def preprocess_for_registration(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for registration by downsampling and enhancing contrast.
        Uses GPU acceleration if available.
        """
        if GPU_AVAILABLE:
            return self._preprocess_gpu(image)
        else:
            return self._preprocess_cpu(image)

    def _preprocess_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated preprocessing using CuPy."""
        try:
            # Transfer to GPU
            img_gpu = cp.asarray(image, dtype=cp.float32)
            
            # Downsample for faster processing
            if self.downsample_factor > 1:
                scale = 1.0 / self.downsample_factor
                img_gpu = cp_ndimage.zoom(img_gpu, scale, order=1)
            
            # Enhance contrast for better feature detection
            img_min, img_max = cp.min(img_gpu), cp.max(img_gpu)
            if img_max > img_min:
                img_gpu = (img_gpu - img_min) / (img_max - img_min)
            
            # Apply gentle gaussian filter to reduce noise
            img_gpu = cp_ndimage.gaussian_filter(img_gpu, sigma=1.0)
            
            # Transfer back to CPU
            return img_gpu.get()
        
        except Exception as e:
            logger.warning(f"GPU preprocessing failed: {e}, falling back to CPU")
            return self._preprocess_cpu(image)

    def _preprocess_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU-based preprocessing (original implementation)."""
        # Downsample for faster processing
        if self.downsample_factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to float and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Enhance contrast for better feature detection
        image = exposure.rescale_intensity(image, out_range=(0, 1))
        
        # Apply gentle gaussian filter to reduce noise
        image = filters.gaussian(image, sigma=1.0)
        
        return image

    def compute_phase_correlation_shift(self, ref_img: np.ndarray, 
                                        moving_img: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute translation using phase correlation.
        Uses GPU acceleration if available.
        Returns (shift_y, shift_x, confidence) in pixels and confidence score.
        """
        if GPU_AVAILABLE:
            return self._phase_correlation_gpu(ref_img, moving_img)
        else:
            return self._phase_correlation_cpu(ref_img, moving_img)

    def _phase_correlation_gpu(self, ref_img: np.ndarray, moving_img: np.ndarray) -> Tuple[float, float, float]:
        """GPU-accelerated phase correlation using CuPy."""
        try:
            # Ensure images are the same size
            if ref_img.shape != moving_img.shape:
                logger.warning("Images have different shapes, resizing...")
                moving_img = cv2.resize(moving_img, (ref_img.shape[1], ref_img.shape[0]))
            
            # Transfer to GPU
            ref_gpu = cp.asarray(ref_img, dtype=cp.float32)
            moving_gpu = cp.asarray(moving_img, dtype=cp.float32)
            
            # FFT-based phase correlation
            f_ref = cp.fft.fft2(ref_gpu)
            f_mov = cp.fft.fft2(moving_gpu)
            
            # Cross-correlation in frequency domain
            cross_corr = f_ref * cp.conj(f_mov)
            cross_corr_magnitude = cp.abs(cross_corr)
            
            # Avoid division by zero
            cross_corr_magnitude = cp.where(cross_corr_magnitude == 0, 1e-10, cross_corr_magnitude)
            cross_corr_norm = cross_corr / cross_corr_magnitude
            
            # Inverse FFT to get correlation
            correlation = cp.fft.ifft2(cross_corr_norm).real
            
            # Find peak location
            peak_idx = cp.unravel_index(cp.argmax(correlation), correlation.shape)
            shift_y, shift_x = int(peak_idx[0].get()), int(peak_idx[1].get())
            
            # Convert to center-origin coordinates
            if shift_y > correlation.shape[0] // 2:
                shift_y -= correlation.shape[0]
            if shift_x > correlation.shape[1] // 2:
                shift_x -= correlation.shape[1]
            
            # Calculate confidence as peak value
            confidence = float(cp.max(correlation).get())
            
            return float(shift_y), float(shift_x), confidence
            
        except Exception as e:
            logger.warning(f"GPU phase correlation failed: {e}, falling back to CPU")
            return self._phase_correlation_cpu(ref_img, moving_img)

    def _phase_correlation_cpu(self, ref_img: np.ndarray, moving_img: np.ndarray) -> Tuple[float, float, float]:
        """CPU-based phase correlation using OpenCV."""
        # Ensure images are the same size
        if ref_img.shape != moving_img.shape:
            logger.warning("Images have different shapes, resizing...")
            moving_img = cv2.resize(moving_img, (ref_img.shape[1], ref_img.shape[0]))
        
        # Use OpenCV's phase correlation
        shift, confidence = cv2.phaseCorrelate(ref_img, moving_img)
        return shift[1], shift[0], confidence  # Return as (y, x, confidence)

    def compute_feature_based_shift(self, ref_img: np.ndarray, 
                                    moving_img: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute translation using ORB feature matching as fallback.
        Returns (shift_y, shift_x, confidence) in pixels and match confidence.
        """
        # Convert to uint8 for ORB
        ref_uint8 = (ref_img * 255).astype(np.uint8)
        moving_uint8 = (moving_img * 255).astype(np.uint8)
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(ref_uint8, None)
        kp2, des2 = orb.detectAndCompute(moving_uint8, None)
        
        if des1 is None or des2 is None:
            logger.warning("No features detected, returning zero shift")
            return 0.0, 0.0, 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            logger.warning("Insufficient matches for reliable registration")
            return 0.0, 0.0, 0.0
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points (use top 50% matches)
        good_matches = matches[:max(10, len(matches)//2)]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute mean translation
        shifts = pts2 - pts1
        mean_shift = np.median(shifts, axis=0)[0]
        
        # Calculate confidence based on consistency of matches
        # Standard deviation of shifts - lower is better
        shift_consistency = np.std(shifts.reshape(-1, 2), axis=0)
        consistency_score = 1.0 / (1.0 + np.mean(shift_consistency))
        
        # Match quality based on average distance
        avg_match_distance = np.mean([m.distance for m in good_matches])
        match_quality = max(0, 1.0 - avg_match_distance / 100.0)  # Normalize to 0-1
        
        # Combined confidence
        confidence = (consistency_score + match_quality) / 2.0
        
        return mean_shift[1], mean_shift[0], confidence  # Return as (y, x, confidence)

    def register_dapi_rounds(self, rounds_data: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[float, float, float]]:
        """
        Register DAPI channels across rounds to determine round-to-round shifts.
        
        Returns:
            dict: {round_id: (shift_y, shift_x, confidence), ...}
        """
        logger.info(f"Starting DAPI-based round registration for {self.individual_id}...")
        
        # Find rounds that have DAPI
        dapi_rounds = {}
        for round_id, stains in rounds_data.items():
            if 'DAPI' in stains:
                dapi_rounds[round_id] = stains['DAPI']
            else:
                logger.warning(f"No DAPI found in round {round_id}. Available stains: {list(stains.keys())}")
        
        if len(dapi_rounds) < 2:
            logger.error("Need at least 2 rounds with DAPI for registration")
            return {}
        
        # Determine reference round
        if self.reference_round is None or self.reference_round not in dapi_rounds:
            self.reference_round = sorted(dapi_rounds.keys())[0]
            logger.info(f"Using round {self.reference_round} as reference")
        
        # Load and preprocess reference DAPI
        ref_dapi_path = dapi_rounds[self.reference_round]
        ref_dapi = self.load_channel(ref_dapi_path)
        ref_processed = self.preprocess_for_registration(ref_dapi)
        
        # Extract physical size metadata from reference DAPI
        if self.physical_size_metadata is None:
            self.physical_size_metadata = self.extract_physical_size_metadata(ref_dapi_path)
            logger.info(f"Extracted physical size metadata: {self.physical_size_metadata}")
        
        round_shifts = {self.reference_round: (0.0, 0.0, 1.0)}  # Reference has perfect confidence
        
        for round_id, dapi_path in dapi_rounds.items():
            if round_id == self.reference_round:
                continue
                
            logger.info(f"Registering round {round_id} DAPI to reference round {self.reference_round}")
            
            # Load and preprocess moving DAPI
            moving_dapi = self.load_channel(dapi_path)
            moving_processed = self.preprocess_for_registration(moving_dapi)
            
            try:
                # Try phase correlation first
                shift_y, shift_x, confidence = self.compute_phase_correlation_shift(ref_processed, moving_processed)
                logger.info(f"Phase correlation shift for round {round_id}: ({shift_y:.2f}, {shift_x:.2f}), confidence: {confidence:.3f}")
                
                # Scale back to full resolution
                shift_y *= self.downsample_factor
                shift_x *= self.downsample_factor
                
            except Exception as e:
                logger.warning(f"Phase correlation failed for round {round_id}: {e}")
                logger.info("Falling back to feature-based registration...")
                
                try:
                    shift_y, shift_x, confidence = self.compute_feature_based_shift(ref_processed, moving_processed)
                    # Scale back to full resolution
                    shift_y *= self.downsample_factor
                    shift_x *= self.downsample_factor
                    
                except Exception as e2:
                    logger.warning(f"Feature-based registration also failed: {e2}")
                    shift_y, shift_x, confidence = 0.0, 0.0, 0.0
            
            round_shifts[round_id] = (shift_y, shift_x, confidence)
            logger.info(f"Final shift for round {round_id}: ({shift_y:.2f}, {shift_x:.2f}), confidence: {confidence:.3f}")
        
        self.round_shifts = round_shifts
        return round_shifts

    def export_individual_channels(self, rounds_data: Dict[str, Dict[str, str]], 
                                    round_shifts: Dict[str, Tuple[float, float, float]], 
                                    output_dir: str):
        """
        Export each individual channel as a shifted TIFF file.
        """
        logger.info(f"Exporting individual shifted channels for {self.individual_id}...")
        
        # Create output directory for individual channels
        channels_dir = Path(output_dir) / f"{self.individual_id}_shifted_channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate global bounding box
        all_shifts = [(shift[0], shift[1]) for shift in round_shifts.values()]
        max_shift_y = max(abs(shift[0]) for shift in all_shifts)
        max_shift_x = max(abs(shift[1]) for shift in all_shifts)
        
        # Get reference dimensions
        first_file = next(iter(next(iter(rounds_data.values())).values()))
        ref_image = self.load_channel(first_file)
        height, width = ref_image.shape[:2]
        
        output_height = int(height + 2 * max_shift_y)
        output_width = int(width + 2 * max_shift_x)
        
        for round_id in sorted(rounds_data.keys()):
            round_shift_data = round_shifts.get(round_id, (0.0, 0.0, 0.0))
            shift_y, shift_x = round_shift_data[0], round_shift_data[1]
            stains = rounds_data[round_id]
            
            for stain in sorted(stains.keys()):
                filepath = stains[stain]
                logger.info(f"Exporting R{round_id}_{stain}")
                
                # Load channel
                channel = self.load_channel(filepath)
                
                # Create shifted image
                shifted_image = np.zeros((output_height, output_width), dtype=channel.dtype)
                
                # Calculate placement
                start_y = int(max_shift_y - shift_y)
                start_x = int(max_shift_x - shift_x)
                end_y = start_y + channel.shape[0]
                end_x = start_x + channel.shape[1]
                
                # Place in shifted array
                shifted_image[start_y:end_y, start_x:end_x] = channel
                
                # Save as TIFF
                output_filename = f"{self.individual_id}_R{round_id}_{stain}_shifted.tif"
                output_path = channels_dir / output_filename
                
                tifffile.imwrite(str(output_path), shifted_image, 
                                compression='lzw', 
                                photometric='minisblack')
        
        logger.info(f"Individual channels exported to: {channels_dir}")

    def create_pyramid_with_subifd(self, aligned_stack: np.ndarray, 
                                    channel_names: List[str], 
                                    output_path: str):
        """
        Create pyramidal TIFF using proper SubIFD structure for Bio-Formats compatibility.
        Uses GPU acceleration for pyramid generation if available.
        """
        logger.info(f"Creating Bio-Formats compatible pyramidal TIFF: {output_path}")
        
        if GPU_AVAILABLE:
            pyramid_levels = self._generate_pyramid_gpu(aligned_stack)
        else:
            pyramid_levels = self._generate_pyramid_cpu(aligned_stack)
        
        # Create proper OME-XML metadata with channel names
        ome_xml = self._create_ome_xml(channel_names, aligned_stack.shape[2], 
                                        aligned_stack.shape[1], aligned_stack.shape[0], 
                                        aligned_stack.dtype)
        
        # Write using SubIFDs for Bio-Formats compatibility
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            # Calculate valid tile size (must be multiple of 16)
            tile_width = min(512, aligned_stack.shape[2])
            tile_height = min(512, aligned_stack.shape[1])
            
            # Ensure tile dimensions are multiples of 16
            tile_width = (tile_width // 16) * 16
            tile_height = (tile_height // 16) * 16
            
            # Minimum tile size of 16
            tile_width = max(16, tile_width)
            tile_height = max(16, tile_height)
            
            options = {
                'compression': 'lzw',
                'tile': (tile_width, tile_height),
                'photometric': 'minisblack'
            }
            
            # Write full resolution with SubIFDs
            if len(pyramid_levels) > 1:
                # Write with SubIFDs pointing to downsampled levels
                tif.write(pyramid_levels[0], 
                        subifds=len(pyramid_levels)-1,
                        description=ome_xml,
                        **options)
                
                # Write each sub-resolution level
                for level_data in pyramid_levels[1:]:
                    # Calculate tile size for this level
                    level_tile_width = min(512, level_data.shape[2])
                    level_tile_height = min(512, level_data.shape[1])
                    
                    # Ensure tile dimensions are multiples of 16
                    level_tile_width = (level_tile_width // 16) * 16
                    level_tile_height = (level_tile_height // 16) * 16
                    
                    # Minimum tile size of 16
                    level_tile_width = max(16, level_tile_width)
                    level_tile_height = max(16, level_tile_height)
                    
                    tif.write(level_data, 
                            compression='lzw',
                            tile=(level_tile_width, level_tile_height),
                            photometric='minisblack')
            else:
                # Single level
                tif.write(pyramid_levels[0],
                        description=ome_xml,
                        **options)
        
        logger.info(f"Bio-Formats pyramidal TIFF with {len(pyramid_levels)} levels saved!")

    def _generate_pyramid_gpu(self, aligned_stack: np.ndarray) -> List[np.ndarray]:
        """Generate pyramid levels using GPU acceleration with memory safeguards."""
        try:
            # Estimate memory requirements
            estimated_memory_gb = estimate_memory_required(aligned_stack.shape, aligned_stack.dtype)
            ram_ok, gpu_ok, ram_free_gb, gpu_free_gb = check_memory_availability(estimated_memory_gb)
            
            logger.info(f"Pyramid memory estimate: {estimated_memory_gb:.1f} GB")
            logger.info(f"Available: RAM {ram_free_gb:.1f} GB, GPU {gpu_free_gb:.1f} GB")
            
            if not gpu_ok:
                logger.warning(f"Insufficient GPU memory for pyramid generation, falling back to CPU")
                return self._generate_pyramid_cpu(aligned_stack)
            
            pyramid_levels = [aligned_stack]
            current_data_gpu = cp.asarray(aligned_stack)
            
            level_count = 1
            while min(current_data_gpu.shape[1:]) > 512:
                # Check memory before each level
                try:
                    device = cp.cuda.Device()
                    meminfo = device.mem_info
                    free_memory_gb = meminfo[0] / (1024**3)
                    
                    if free_memory_gb < 2.0:  # Safety threshold
                        logger.warning(f"Low GPU memory ({free_memory_gb:.1f} GB), stopping pyramid generation")
                        break
                    
                    new_height = current_data_gpu.shape[1] // 2
                    new_width = current_data_gpu.shape[2] // 2
                    
                    # GPU-accelerated downsampling
                    downsampled_gpu = cp_ndimage.zoom(current_data_gpu, (1, 0.5, 0.5), order=1)
                    
                    # Transfer back to CPU for storage
                    downsampled = downsampled_gpu.get()
                    pyramid_levels.append(downsampled)
                    
                    # Clear GPU memory
                    del current_data_gpu
                    current_data_gpu = downsampled_gpu
                    cp.cuda.runtime.deviceSynchronize()
                    
                    level_count += 1
                    logger.info(f"Generated pyramid level {level_count} with size {new_height}x{new_width} (GPU)")
                    
                except cp.cuda.memory.OutOfMemoryError as e:
                    logger.warning(f"GPU out of memory at level {level_count}: {e}")
                    logger.info("Clearing GPU memory and falling back to CPU for remaining levels")
                    
                    # Clear GPU memory
                    try:
                        del current_data_gpu
                        del downsampled_gpu
                    except:
                        pass
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                    # Continue with CPU for remaining levels
                    current_data = pyramid_levels[-1]
                    while min(current_data.shape[1:]) > 512:
                        new_height = current_data.shape[1] // 2
                        new_width = current_data.shape[2] // 2
                        
                        downsampled = np.zeros((current_data.shape[0], new_height, new_width), 
                                            dtype=current_data.dtype)
                        
                        for c in range(current_data.shape[0]):
                            downsampled[c] = cv2.resize(current_data[c], 
                                                        (new_width, new_height),
                                                        interpolation=cv2.INTER_AREA)
                        
                        pyramid_levels.append(downsampled)
                        current_data = downsampled
                        level_count += 1
                        logger.info(f"Generated pyramid level {level_count} with size {new_height}x{new_width} (CPU fallback)")
                    break
            
            # Clean up GPU memory
            try:
                del current_data_gpu
            except:
                pass
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            return pyramid_levels
            
        except Exception as e:
            logger.warning(f"GPU pyramid generation failed: {e}, falling back to CPU")
            # Clean up GPU memory on failure
            try:
                cp.cuda.runtime.deviceSynchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            gc.collect()
            return self._generate_pyramid_cpu(aligned_stack)

    def _generate_pyramid_cpu(self, aligned_stack: np.ndarray) -> List[np.ndarray]:
        """Generate pyramid levels using CPU with memory monitoring."""
        # Estimate memory requirements
        estimated_memory_gb = estimate_memory_required(aligned_stack.shape, aligned_stack.dtype)
        ram_ok, _, ram_free_gb, _ = check_memory_availability(estimated_memory_gb)
        
        logger.info(f"CPU pyramid memory estimate: {estimated_memory_gb:.1f} GB, available RAM: {ram_free_gb:.1f} GB")
        
        if not ram_ok and ram_free_gb < estimated_memory_gb * 0.5:
            logger.warning("Very low memory for pyramid generation, creating minimal pyramid")
            return self._generate_minimal_pyramid(aligned_stack)
        
        pyramid_levels = [aligned_stack]
        current_data = aligned_stack
        
        level_count = 1
        while min(current_data.shape[1:]) > 512:
            # Check available memory before each level
            ram = psutil.virtual_memory()
            available_memory_gb = ram.available / (1024**3)
            
            if available_memory_gb < 4.0:  # Safety threshold
                logger.warning(f"Low memory ({available_memory_gb:.1f} GB), stopping pyramid generation")
                break
            
            try:
                new_height = current_data.shape[1] // 2
                new_width = current_data.shape[2] // 2
                
                downsampled = np.zeros((current_data.shape[0], new_height, new_width), 
                                    dtype=current_data.dtype)
                
                for c in range(current_data.shape[0]):
                    downsampled[c] = cv2.resize(current_data[c], 
                                                (new_width, new_height),
                                                interpolation=cv2.INTER_AREA)
                
                pyramid_levels.append(downsampled)
                current_data = downsampled
                level_count += 1
                logger.info(f"Generated pyramid level {level_count} with size {new_height}x{new_width} (CPU)")
                
                # Force garbage collection to free memory
                gc.collect()
                
            except MemoryError as e:
                logger.warning(f"CPU out of memory at level {level_count}: {e}")
                logger.info("Creating minimal pyramid to save disk space")
                break
            except Exception as e:
                logger.warning(f"Error creating pyramid level {level_count}: {e}")
                break
        
        return pyramid_levels

    def _generate_minimal_pyramid(self, aligned_stack: np.ndarray) -> List[np.ndarray]:
        """Generate a minimal pyramid when memory is very limited."""
        logger.info("Creating minimal pyramid due to memory constraints")
        
        pyramid_levels = [aligned_stack]
        current_data = aligned_stack
        
        # Create only 1-2 levels with aggressive downsampling
        for level in range(1, 3):  # Max 2 additional levels
            try:
                # More aggressive downsampling (4x instead of 2x)
                scale_factor = 4 if level == 1 else 2
                new_height = current_data.shape[1] // scale_factor
                new_width = current_data.shape[2] // scale_factor
                
                if min(new_height, new_width) < 256:
                    break
                
                downsampled = np.zeros((current_data.shape[0], new_height, new_width), 
                                    dtype=current_data.dtype)
                
                for c in range(current_data.shape[0]):
                    downsampled[c] = cv2.resize(current_data[c], 
                                                (new_width, new_height),
                                                interpolation=cv2.INTER_AREA)
                
                pyramid_levels.append(downsampled)
                current_data = downsampled
                logger.info(f"Generated minimal pyramid level {level+1} with size {new_height}x{new_width}")
                
                # Force cleanup
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to create minimal pyramid level {level}: {e}")
                break
        
        return pyramid_levels

    def apply_shifts_and_stack(self, rounds_data: Dict[str, Dict[str, str]], 
                            round_shifts: Dict[str, Tuple[float, float, float]], 
                            output_path: str,
                            create_pyramidal: bool = True,
                            create_uncompressed: bool = True):
        """
        Apply computed round shifts to all channels and stack into TIFF files.
        Uses tiling approach if insufficient memory for full stack.
        """
        logger.info(f"Applying shifts and stacking all channels for {self.individual_id}...")
        
        # Collect all files with their shifts
        all_files = []
        all_shifts = []
        channel_names = []
        
        for round_id in sorted(rounds_data.keys()):
            round_shift_data = round_shifts.get(round_id, (0.0, 0.0, 0.0))
            shift_y, shift_x = round_shift_data[0], round_shift_data[1]
            stains = rounds_data[round_id]
            
            for stain in sorted(stains.keys()):
                filepath = stains[stain]
                all_files.append(filepath)
                all_shifts.append((shift_y, shift_x))
                channel_names.append(f"R{round_id}_{stain}")
        
        if not all_files:
            logger.error("No files to process")
            return
        
        logger.info(f"Processing {len(all_files)} channels total")
        
        # Load first channel to get dimensions
        first_channel = self.load_channel(all_files[0])
        height, width = first_channel.shape[:2]
        num_channels = len(all_files)
        dtype = first_channel.dtype
        
        # Calculate the output dimensions accounting for shifts
        max_shift_y = max(abs(shift[0]) for shift in all_shifts)
        max_shift_x = max(abs(shift[1]) for shift in all_shifts)
        
        # Create output array dimensions
        output_height = int(height + 2 * max_shift_y)
        output_width = int(width + 2 * max_shift_x)
        
        logger.info(f"Output dimensions: {output_height} x {output_width} x {num_channels}")
        
        # Estimate memory required for full stack
        bytes_per_element = np.dtype(dtype).itemsize
        stack_memory_gb = (num_channels * output_height * output_width * bytes_per_element) / (1024**3)
        
        # Check available memory
        ram = psutil.virtual_memory()
        available_memory_gb = ram.available / (1024**3)
        
        logger.info(f"Estimated memory for full stack: {stack_memory_gb:.2f} GB")
        logger.info(f"Available memory: {available_memory_gb:.2f} GB")
        
        # Decide whether to use full memory or tiling approach
        use_tiling = stack_memory_gb > available_memory_gb * 0.9  # Use 70% threshold for safety
        
        if use_tiling:
            logger.warning("Insufficient memory for full stack, using tiling approach")
            self._apply_shifts_tiled(all_files, all_shifts, channel_names, output_path,
                                    output_height, output_width, max_shift_y, max_shift_x,
                                    create_pyramidal, create_uncompressed)
        else:
            # Original approach with full memory
            logger.info("Using full memory approach")
            
            # Process channels
            aligned_stack = np.zeros((num_channels, output_height, output_width), dtype=dtype)
            
            for i, (filepath, (shift_y, shift_x)) in enumerate(zip(all_files, all_shifts)):
                logger.info(f"Processing channel {i+1}/{num_channels}: {channel_names[i]}")
                
                # Load channel
                channel = self.load_channel(filepath)
                
                # Calculate placement in output array
                start_y = int(max_shift_y - shift_y)
                start_x = int(max_shift_x - shift_x)
                end_y = start_y + channel.shape[0]
                end_x = start_x + channel.shape[1]
                
                # Place in output array
                aligned_stack[i, start_y:end_y, start_x:end_x] = channel
            
            # Save outputs using original methods
            self._save_outputs(aligned_stack, channel_names, output_path, 
                                create_pyramidal, create_uncompressed)
        
        # Print summary
        logger.info(f"\n=== REGISTRATION SUMMARY FOR {self.individual_id} ===")
        for round_id, shift_data in round_shifts.items():
            shift_y, shift_x, confidence = shift_data
            logger.info(f"Round {round_id}: shift = ({shift_y:.2f}, {shift_x:.2f}) pixels, confidence = {confidence:.3f}")

    def _apply_shifts_tiled(self, all_files: List[str], all_shifts: List[Tuple[float, float]], 
                            channel_names: List[str], output_path: str,
                            output_height: int, output_width: int, 
                            max_shift_y: float, max_shift_x: float,
                            create_pyramidal: bool, create_uncompressed: bool):
        """
        Apply shifts using a tiled approach to save memory.
        Writes directly to disk without loading full stack into memory.
        """
        output_path_obj = Path(output_path)
        num_channels = len(all_files)
        
        # Get dtype from first file
        with tifffile.TiffFile(all_files[0]) as tif:
            if hasattr(tif, 'series') and len(tif.series) > 0:
                dtype = tif.series[0].dtype
            else:
                dtype = tif.asarray().dtype
        
        # Determine tile size based on available memory
        ram = psutil.virtual_memory()
        available_memory_gb = ram.available / (1024**3)
        
        # Use tiles that fit comfortably in memory (aim for ~1GB per tile across all channels)
        bytes_per_element = np.dtype(dtype).itemsize
        tile_memory_gb = 1.0
        tile_size = int(np.sqrt((tile_memory_gb * 1024**3) / (num_channels * bytes_per_element)))
        tile_size = min(tile_size, 4096)  # Cap at 4096 for practical reasons
        tile_size = max(tile_size, 512)   # Minimum 512 for efficiency
        
        logger.info(f"Using tile size: {tile_size}x{tile_size}")
        
        if create_uncompressed:
            # Create uncompressed version using tiling
            uncompressed_path = output_path_obj.parent / (output_path_obj.stem + "_uncompressed.tif")
            logger.info(f"Creating uncompressed tiled stack: {uncompressed_path}")
            
            with tifffile.TiffWriter(str(uncompressed_path), bigtiff=True) as tif:
                # Write header with OME-XML metadata
                ome_xml = self._create_ome_xml(channel_names, output_width, output_height, 
                                                num_channels, dtype)
                
                # Process each channel one at a time
                for i, (filepath, (shift_y, shift_x)) in enumerate(zip(all_files, all_shifts)):
                    logger.info(f"Writing channel {i+1}/{num_channels}: {channel_names[i]}")
                    
                    # Create empty channel
                    channel_data = np.zeros((output_height, output_width), dtype=dtype)
                    
                    # Load and place the channel data
                    channel = self.load_channel(filepath)
                    start_y = int(max_shift_y - shift_y)
                    start_x = int(max_shift_x - shift_x)
                    end_y = start_y + channel.shape[0]
                    end_x = start_x + channel.shape[1]
                    channel_data[start_y:end_y, start_x:end_x] = channel
                    
                    # Write channel
                    if i == 0:
                        tif.write(channel_data, description=ome_xml, photometric='minisblack')
                    else:
                        tif.write(channel_data, photometric='minisblack')
                    
                    # Free memory
                    del channel
                    del channel_data
                    gc.collect()
        
        if create_pyramidal:
            # Create pyramidal version using tiled approach
            logger.info(f"Creating pyramidal tiled stack: {output_path}")
            self._create_pyramidal_tiled(all_files, all_shifts, channel_names, output_path,
                                        output_height, output_width, max_shift_y, max_shift_x,
                                        tile_size, dtype)

    def _create_pyramidal_tiled(self, all_files: List[str], all_shifts: List[Tuple[float, float]], 
                                channel_names: List[str], output_path: str,
                                output_height: int, output_width: int,
                                max_shift_y: float, max_shift_x: float,
                                tile_size: int, dtype: np.dtype):
        """
        Create pyramidal TIFF using tiled approach to conserve memory.
        """
        logger.info("Creating pyramidal TIFF with tiled approach...")
        
        # Calculate pyramid levels needed
        pyramid_levels = []
        current_height, current_width = output_height, output_width
        level = 0
        
        while min(current_height, current_width) > 256:
            pyramid_levels.append((current_height, current_width))
            current_height //= 2
            current_width //= 2
            level += 1
        
        logger.info(f"Creating {len(pyramid_levels)} pyramid levels")
        
        # Create OME-XML metadata
        ome_xml = self._create_ome_xml(channel_names, output_width, output_height, 
                                        len(channel_names), dtype)
        
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            # Process each pyramid level
            for level_idx, (level_height, level_width) in enumerate(pyramid_levels):
                scale_factor = 2 ** level_idx
                
                logger.info(f"Processing pyramid level {level_idx + 1}/{len(pyramid_levels)} "
                            f"({level_width}x{level_height})")
                
                # Calculate valid tile size (must be multiple of 16)
                tile_width = min(512, level_width)
                tile_height = min(512, level_height)
                
                # Ensure tile dimensions are multiples of 16
                tile_width = (tile_width // 16) * 16
                tile_height = (tile_height // 16) * 16
                
                # Minimum tile size of 16
                tile_width = max(16, tile_width)
                tile_height = max(16, tile_height)
                
                # For level 0, we need to write with subifds
                if level_idx == 0:
                    options = {
                        'compression': 'lzw',
                        'tile': (tile_width, tile_height),
                        'photometric': 'minisblack',
                        'subifds': len(pyramid_levels) - 1 if len(pyramid_levels) > 1 else 0,
                        'description': ome_xml
                    }
                else:
                    options = {
                        'compression': 'lzw',
                        'tile': (tile_width, tile_height),
                        'photometric': 'minisblack'
                    }
                
                # Process each channel for this level
                for channel_idx, (filepath, (shift_y, shift_x)) in enumerate(zip(all_files, all_shifts)):
                    logger.info(f"  Channel {channel_idx + 1}/{len(all_files)}: {channel_names[channel_idx]}")
                    
                    if level_idx == 0:
                        # Full resolution - load and shift
                        channel = self.load_channel(filepath)
                        channel_data = np.zeros((level_height, level_width), dtype=dtype)
                        
                        start_y = int(max_shift_y - shift_y)
                        start_x = int(max_shift_x - shift_x)
                        end_y = start_y + channel.shape[0]
                        end_x = start_x + channel.shape[1]
                        channel_data[start_y:end_y, start_x:end_x] = channel
                    else:
                        # Downsampled level - create from original data
                        channel = self.load_channel(filepath)
                        
                        # Downsample the channel first
                        channel_downsampled = cv2.resize(channel, 
                                                        (channel.shape[1] // scale_factor,
                                                        channel.shape[0] // scale_factor),
                                                        interpolation=cv2.INTER_AREA)
                        
                        # Create output array for this level
                        channel_data = np.zeros((level_height, level_width), dtype=dtype)
                        
                        # Apply shifts at downsampled scale
                        start_y = int((max_shift_y - shift_y) / scale_factor)
                        start_x = int((max_shift_x - shift_x) / scale_factor)
                        end_y = start_y + channel_downsampled.shape[0]
                        end_x = start_x + channel_downsampled.shape[1]
                        
                        # Ensure we don't exceed bounds
                        end_y = min(end_y, level_height)
                        end_x = min(end_x, level_width)
                        
                        channel_data[start_y:end_y, start_x:end_x] = channel_downsampled[:end_y-start_y, :end_x-start_x]
                        
                        del channel_downsampled
                    
                    # Write the channel data
                    if channel_idx == 0:
                        tif.write(channel_data, **options)
                    else:
                        # Recalculate tile size for non-first channels in case dimensions differ
                        current_tile_width = min(512, channel_data.shape[1])
                        current_tile_height = min(512, channel_data.shape[0])
                        
                        # Ensure tile dimensions are multiples of 16
                        current_tile_width = (current_tile_width // 16) * 16
                        current_tile_height = (current_tile_height // 16) * 16
                        
                        # Minimum tile size of 16
                        current_tile_width = max(16, current_tile_width)
                        current_tile_height = max(16, current_tile_height)
                        
                        tif.write(channel_data, photometric='minisblack', 
                                compression='lzw', 
                                tile=(current_tile_width, current_tile_height))
                    
                    # Clear memory
                    del channel
                    del channel_data
                    gc.collect()
        
        logger.info("Pyramidal TIFF creation complete!")

    def _create_ome_xml(self, channel_names: List[str], width: int, height: int, 
                        num_channels: int, dtype: np.dtype) -> str:
        """
        Create OME-XML metadata string.
        """
        from xml.etree.ElementTree import Element, SubElement, tostring
        
        ome = Element('OME')
        ome.set('xmlns', 'http://www.openmicroscopy.org/Schemas/OME/2016-06')
        ome.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        ome.set('xsi:schemaLocation', 'http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd')
        
        image = SubElement(ome, 'Image')
        image.set('ID', 'Image:0')
        image.set('Name', f'{self.individual_id}_aligned_stack')
        
        pixels = SubElement(image, 'Pixels')
        pixels.set('ID', 'Pixels:0')
        pixels.set('DimensionOrder', 'XYCZT')
        
        # Convert numpy dtype to OME-TIFF Type string
        dtype_str = str(dtype)
        dtype_map = {
            'uint8': 'uint8',
            'uint16': 'uint16',
            'int16': 'int16',
            'uint32': 'uint32',
            'int32': 'int32',
            'float32': 'float',
            'float64': 'double'
        }
        ome_type = dtype_map.get(dtype_str, 'uint16')
        
        pixels.set('Type', ome_type)
        pixels.set('SizeX', str(width))
        pixels.set('SizeY', str(height))
        pixels.set('SizeC', str(num_channels))
        pixels.set('SizeZ', '1')
        pixels.set('SizeT', '1')
        pixels.set('PhysicalSizeX', self.physical_size_metadata['PhysicalSizeX'])
        pixels.set('PhysicalSizeY', self.physical_size_metadata['PhysicalSizeY'])
        pixels.set('PhysicalSizeXUnit', self.physical_size_metadata['PhysicalSizeXUnit'])
        pixels.set('PhysicalSizeYUnit', self.physical_size_metadata['PhysicalSizeYUnit'])
        
        # Add channel information
        for i, channel_name in enumerate(channel_names):
            channel = SubElement(pixels, 'Channel')
            channel.set('ID', f'Channel:0:{i}')
            channel.set('Name', channel_name)
            channel.set('SamplesPerPixel', '1')
        
        # Add TiffData element
        tiff_data = SubElement(pixels, 'TiffData')
        tiff_data.set('IFD', '0')
        tiff_data.set('PlaneCount', str(num_channels))
        
        return tostring(ome, encoding='unicode')

    def _save_outputs(self, aligned_stack: np.ndarray, channel_names: List[str], 
                    output_path: str, create_pyramidal: bool, create_uncompressed: bool):
        """
        Save outputs using the original methods (when memory is sufficient).
        """
        output_path_obj = Path(output_path)
        
        if create_uncompressed:
            # Save uncompressed full-resolution stack
            uncompressed_path = output_path_obj.parent / (output_path_obj.stem + "_uncompressed.tif")
            logger.info(f"Saving uncompressed stack: {uncompressed_path}")
            
            ome_xml = self._create_ome_xml(channel_names, aligned_stack.shape[2], 
                                            aligned_stack.shape[1], aligned_stack.shape[0], 
                                            aligned_stack.dtype)
            
            with tifffile.TiffWriter(str(uncompressed_path), bigtiff=True) as tif:
                tif.write(aligned_stack,
                        description=ome_xml,
                        photometric='minisblack')
            logger.info("Uncompressed stack saved!")
        
        if create_pyramidal:
            # Save pyramidal TIFF with proper SubIFD structure
            self.create_pyramid_with_subifd(aligned_stack, channel_names, str(output_path))

def find_individuals(input_dir: str, file_pattern: str) -> List[str]:
   """
   Find all unique individual IDs in the input directory.
   """
   input_path = Path(input_dir)
   all_files = list(input_path.glob(file_pattern))
   
   individuals = set()
   for filepath in all_files:
       filename = filepath.name
       # Extract individual ID (first part before underscore)
       parts = filename.split('_')
       if len(parts) >= 1:
           individuals.add(parts[0])
   
   return sorted(list(individuals))

def main():
   """
   Main function to process all individuals.
   """
   # Validate configuration
   if not Path(CONFIG['input_dir']).exists():
       logger.error(f"Input directory does not exist: {CONFIG['input_dir']}")
       return
   
   # Create output directory
   output_dir = Path(CONFIG['output_dir'])
   output_dir.mkdir(parents=True, exist_ok=True)
   
   # Find individuals to process
   if CONFIG['individuals'] == ['*']:
       all_individuals = find_individuals(CONFIG['input_dir'], CONFIG['file_pattern'])
       logger.info(f"Auto-detected individuals: {all_individuals}")
   else:
       all_individuals = CONFIG['individuals']
   
   if not all_individuals:
       logger.error("No individuals found to process")
       return
   
   # Check for completed individuals and resume processing
   completed_individuals = get_completed_individuals(str(output_dir))
   individuals_to_process = [ind for ind in all_individuals if ind not in completed_individuals]
   
   if completed_individuals:
       logger.info(f"Found {len(completed_individuals)} completed individuals: {sorted(completed_individuals)}")
   
   if not individuals_to_process:
       logger.info("All individuals already processed!")
       return
   
   logger.info(f"Processing {len(individuals_to_process)} remaining individuals: {sorted(individuals_to_process)}")
   
   # Find all files
   input_path = Path(CONFIG['input_dir'])
   all_files = sorted(list(input_path.glob(CONFIG['file_pattern'])))
   
   if not all_files:
       logger.error(f"No files found matching pattern {CONFIG['file_pattern']} in {CONFIG['input_dir']}")
       return
   
   logger.info(f"Found {len(all_files)} files total")
   file_paths = [str(f) for f in all_files]
   
   # Process each individual
   for individual_id in individuals_to_process:
       logger.info(f"\n{'='*60}")
       logger.info(f"PROCESSING INDIVIDUAL: {individual_id}")
       logger.info(f"{'='*60}")
       
       # Initialize aligner for this individual
       aligner = RoundBasedAligner(
           individual_id=individual_id,
           reference_round=CONFIG['reference_round'],
           downsample_factor=CONFIG['downsample_factor']
       )
       
       # Organize files by round for this individual
       rounds_data = aligner.organize_files_by_round(file_paths)
       
       if not rounds_data:
           logger.warning(f"No files found for individual {individual_id}")
           continue
       
       # Display organization
       logger.info(f"\n=== FILE ORGANIZATION FOR {individual_id} ===")
       for round_id in sorted(rounds_data.keys()):
           stains = rounds_data[round_id]
           logger.info(f"Round {round_id}:")
           for stain, filepath in sorted(stains.items()):
               logger.info(f"  {stain}: {Path(filepath).name}")
       
       if CONFIG['dry_run']:
           logger.info(f"Dry run complete for {individual_id} - no processing performed")
           continue
       
       # Check for DAPI in each round
       missing_dapi = []
       for round_id, stains in rounds_data.items():
           if 'DAPI' not in stains:
               missing_dapi.append(round_id)
       
       if missing_dapi:
           logger.error(f"Individual {individual_id}: rounds missing DAPI: {missing_dapi}")
           logger.error("Cannot proceed without DAPI in all rounds")
           continue
       
       # Register rounds based on DAPI
       round_shifts = aligner.register_dapi_rounds(rounds_data)
       
       if not round_shifts:
           logger.error(f"Registration failed for individual {individual_id}")
           continue
       
       # Export individual channels if requested
       if CONFIG['export_individual_channels']:
           aligner.export_individual_channels(rounds_data, round_shifts, str(output_dir))
       
       # Create stacked outputs
       output_filename = f"{individual_id}_aligned_stack.ome.tif"
       output_path = output_dir / output_filename
       
       aligner.apply_shifts_and_stack(
           rounds_data, 
           round_shifts, 
           str(output_path),
           create_pyramidal=CONFIG['create_pyramidal_tiff'],
           create_uncompressed=CONFIG['create_uncompressed_stack']
       )
       
       logger.info(f"Completed processing for individual {individual_id}")
   
   if CONFIG['dry_run']:
       logger.info(f"\n{'='*60}")
       logger.info("DRY RUN COMPLETE - No files were processed")
       logger.info(f"{'='*60}")
       return
   
   logger.info(f"\n{'='*60}")
   logger.info("ALL PROCESSING COMPLETE!")
   logger.info(f"Output directory: {output_dir}")
   logger.info(f"{'='*60}")

if __name__ == "__main__":
   main()
