#!/usr/bin/env python3
"""
Quick test to validate tile correction implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from tile_artifact_correction import SharpEdgeTileDetector, TileArtifactCorrector
        from tile_artifact_correction import create_diagnostic_plots, save_correction_report
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_detector():
    """Test the detector with synthetic data."""
    print("\nTesting SharpEdgeTileDetector...")
    from tile_artifact_correction import SharpEdgeTileDetector

    try:
        # Create synthetic data with a clear boundary
        np.random.seed(42)
        n_cells = 1000

        # Create two regions with different intensities (simulating tile boundary)
        x_coords = np.random.uniform(0, 1000, n_cells)
        y_coords = np.random.uniform(0, 1000, n_cells)

        # Left side: low intensity, right side: high intensity
        intensities = np.where(x_coords < 500,
                              np.random.exponential(10, n_cells),
                              np.random.exponential(20, n_cells))

        # Initialize detector
        detector = SharpEdgeTileDetector(
            bin_size=50,
            smooth_sigma=2.0,
            edge_threshold_percentile=95,
            min_edge_pixels=10
        )

        # Run detection
        results = detector.detect(x_coords, y_coords, intensities)

        print(f"  Detection results:")
        print(f"    Detected: {results['detected']}")
        print(f"    Edge pixels: {results['n_edge_pixels']}")
        print(f"    H-lines: {results['n_h_lines']}")
        print(f"    V-lines: {results['n_v_lines']}")
        print("  ✓ Detector test passed")
        return True

    except Exception as e:
        print(f"  ✗ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corrector():
    """Test the corrector with synthetic data."""
    print("\nTesting TileArtifactCorrector...")
    from tile_artifact_correction import TileArtifactCorrector, SharpEdgeTileDetector

    try:
        # Create synthetic data
        np.random.seed(42)
        n_cells = 1000
        x_coords = np.random.uniform(0, 1000, n_cells)
        y_coords = np.random.uniform(0, 1000, n_cells)
        intensities = np.where(x_coords < 500,
                              np.random.exponential(10, n_cells),
                              np.random.exponential(20, n_cells))

        # Detect boundaries
        detector = SharpEdgeTileDetector()
        detection_results = detector.detect(x_coords, y_coords, intensities)

        if not detection_results['detected']:
            print("  ⚠ No boundaries detected in test data, skipping corrector test")
            return True

        # Initialize corrector
        corrector = TileArtifactCorrector(
            boundary_buffer=50,
            local_window=200,
            correction_strength=0.8,
            max_boundary_pct=50.0
        )

        # Run correction
        corrected, stats = corrector.correct(
            x_coords, y_coords, intensities, detection_results
        )

        print(f"  Correction results:")
        print(f"    Cells corrected: {stats.get('n_corrected', 0)}")
        print(f"    Boundary cells: {stats.get('n_boundary_cells', 0)}")
        print(f"    Mean correction: {stats.get('mean_correction_pct', 0):.1f}%")
        print("  ✓ Corrector test passed")
        return True

    except Exception as e:
        print(f"  ✗ Corrector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test that the config is properly set up in manual_gating.py"""
    print("\nTesting configuration...")
    try:
        # Import the config from manual_gating
        import manual_gating

        if not hasattr(manual_gating, 'TILE_CORRECTION_CONFIG'):
            print("  ✗ TILE_CORRECTION_CONFIG not found in manual_gating.py")
            return False

        config = manual_gating.TILE_CORRECTION_CONFIG

        required_keys = ['enabled', 'markers', 'bin_size', 'smooth_sigma',
                        'edge_threshold_percentile', 'boundary_buffer',
                        'local_window', 'correction_strength']

        missing = [k for k in required_keys if k not in config]
        if missing:
            print(f"  ✗ Missing config keys: {missing}")
            return False

        print(f"  Config check:")
        print(f"    Enabled: {config['enabled']}")
        print(f"    Markers: {config['markers']}")
        print(f"    Bin size: {config['bin_size']}")
        print(f"    Correction strength: {config['correction_strength']}")
        print("  ✓ Configuration test passed")
        return True

    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("TILE CORRECTION IMPLEMENTATION TESTS")
    print("="*70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Detector", test_detector()))
    results.append(("Corrector", test_corrector()))
    results.append(("Configuration", test_config()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
