"""
Integration tests for napari-flowreg motion correction.
These are slower tests that exercise the full pipeline.
Mark with @pytest.mark.slow to run only in CI nightly.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os


@pytest.mark.slow
def test_end_to_end_motion_correction_small():
    """Test complete motion correction on a small dataset."""
    # Check if pyflowreg is available
    try:
        # Set up environment before import
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        os.environ["NUMBA_CPU_NAME"] = "generic"
        
        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
        from pyflowreg.motion_correction.compensate_arr import compensate_arr
    except ImportError:
        pytest.skip("pyflowreg not installed")
    
    # Create small test data with motion
    np.random.seed(42)
    frames = 16  # Small number of frames
    size = 64    # Small frame size
    
    # Generate video with synthetic motion
    video = np.zeros((frames, size, size), dtype=np.float32)
    
    # Create a pattern
    for i in range(size//4, 3*size//4):
        for j in range(size//4, 3*size//4):
            video[0, i, j] = 1.0
    
    # Add motion (shift pattern)
    for t in range(1, frames):
        shift_x = int(2 * np.sin(t * 0.5))
        shift_y = int(2 * np.cos(t * 0.5))
        video[t] = np.roll(np.roll(video[0], shift_x, axis=1), shift_y, axis=0)
        # Add noise
        video[t] += np.random.randn(size, size) * 0.1
    
    # Create reference (average of first few frames)
    reference = np.mean(video[:4], axis=0)
    
    # Set up options for fast processing
    options = OFOptions(
        quality_setting=QualitySetting.FAST,
        alpha=1.5,
        sigma=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        levels=3,      # Fewer levels for speed
        iterations=10,  # Fewer iterations for speed
        eta=0.5,
        save_w=True,
        verbose=False
    )
    
    # Run motion correction
    registered, flow = compensate_arr(video, reference, options)
    
    # Validate output
    assert registered is not None
    assert registered.shape == video.shape
    assert flow is not None
    assert flow.shape == (frames, size, size, 2)
    
    # Check that motion was reduced
    # Original video should have more variance between frames than corrected
    original_diff = np.mean(np.abs(np.diff(video, axis=0)))
    corrected_diff = np.mean(np.abs(np.diff(registered, axis=0)))
    
    # Corrected should have less frame-to-frame difference
    assert corrected_diff < original_diff * 0.9  # At least 10% improvement


@pytest.mark.slow
def test_multiprocessing_path():
    """Test that multiprocessing path works correctly with SVML disabled."""
    try:
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        os.environ["NUMBA_CPU_NAME"] = "generic"
        
        from pyflowreg.motion_correction.parallelization.multiprocessing import (
            MultiprocessingExecutor, _init_shared, _process_frame_worker
        )
    except ImportError:
        pytest.skip("pyflowreg not installed")
    
    # Create small test batch
    batch = np.random.rand(4, 32, 32, 1).astype(np.float32)
    batch_proc = batch.copy()
    reference_raw = np.mean(batch, axis=0)
    reference_proc = reference_raw.copy()
    w_init = np.zeros((32, 32, 2), dtype=np.float32)
    
    # Create executor
    executor = MultiprocessingExecutor(n_workers=2)
    
    # Mock functions that would be imported in worker
    def mock_get_displacement(ref, frame, uv=None, **kwargs):
        return np.random.randn(32, 32, 2).astype(np.float32)
    
    def mock_imregister(frame, u, v, ref, **kwargs):
        return frame  # Return unchanged for testing
    
    # Run with minimal configuration
    try:
        registered, flows = executor.process_batch(
            batch=batch,
            batch_proc=batch_proc,
            reference_raw=reference_raw,
            reference_proc=reference_proc,
            w_init=w_init,
            get_displacement_func=mock_get_displacement,
            imregister_func=mock_imregister,
            interpolation_method='cubic',
            flow_params={'alpha': (1.5, 1.5), 'levels': 3}
        )
        
        # Validate output
        assert registered.shape == batch.shape
        assert flows.shape == (4, 32, 32, 2)
        
    finally:
        executor.cleanup()


@pytest.mark.slow
def test_widget_with_real_processing(make_napari_viewer_proxy):
    """Test widget with actual processing (small data)."""
    try:
        # Must set before any imports
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        os.environ["NUMBA_CPU_NAME"] = "generic"
        
        from napari_flowreg.flowreg_widget import FlowRegWidget
        # Delay pyflowreg import
    except ImportError:
        pytest.skip("napari_flowreg not available")
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Create small test video
    np.random.seed(42)
    video = np.random.rand(8, 32, 32).astype(np.float32)
    
    # Add to viewer
    viewer.add_image(video, name="test_video")
    widget._update_layer_lists()
    widget.input_combo.setCurrentText("test_video")
    
    # Configure for fast processing
    widget.quality_combo.setCurrentText("fast")
    widget.ref_method_combo.setCurrentText("Average all frames")
    
    # Get options
    options_dict = widget._create_options_dict()
    reference = widget._get_reference_frames()
    
    # Try actual processing with the worker
    try:
        # Import here to ensure SVML is disabled
        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
        from pyflowreg.motion_correction.compensate_arr import compensate_arr
        
        # Map quality
        quality_map = {
            "fast": QualitySetting.FAST,
            "balanced": QualitySetting.BALANCED,
            "quality": QualitySetting.QUALITY,
            "custom": QualitySetting.CUSTOM
        }
        
        options_dict["quality_setting"] = quality_map[options_dict["quality_setting"]]
        options = OFOptions(**options_dict)
        
        # Run processing
        registered, flow = compensate_arr(video, reference, options)
        
        # Validate
        assert registered.shape == video.shape
        assert flow.shape == (8, 32, 32, 2)
        
    except ImportError:
        pytest.skip("pyflowreg not available for real processing test")


@pytest.mark.slow 
def test_memory_efficiency():
    """Test that memory usage is reasonable for larger datasets."""
    try:
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        
        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
        from pyflowreg.motion_correction.compensate_arr import compensate_arr
    except ImportError:
        pytest.skip("pyflowreg not installed")
    
    # Test with medium-sized data
    video = np.random.rand(32, 128, 128).astype(np.float32)
    reference = np.mean(video[:4], axis=0)
    
    # Use fast settings
    options = OFOptions(
        quality_setting=QualitySetting.FAST,
        alpha=1.5,
        levels=4,
        iterations=20,
        eta=0.5,
        save_w=False,  # Don't save flow to reduce memory
        verbose=False
    )
    
    # Track memory before
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run processing
    registered, flow = compensate_arr(video, reference, options)
    
    # Check memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_increase = mem_after - mem_before
    
    # Should not use excessive memory (< 500 MB increase for this size)
    assert mem_increase < 500, f"Memory increased by {mem_increase} MB"
    
    # Validate output
    assert registered.shape == video.shape


@pytest.mark.slow
def test_error_handling_in_worker():
    """Test that errors in worker are handled gracefully."""
    import sys
    import os
    
    # Skip on macOS CI due to Qt abort issues
    if sys.platform == "darwin" and os.environ.get("CI"):
        pytest.skip("Skipping on macOS CI due to Qt initialization issues")
    
    from napari_flowreg.flowreg_widget import FlowRegWidget
    from unittest.mock import MagicMock
    import napari_flowreg.flowreg_widget as widgets_module
    
    # Create a mock viewer
    viewer = MagicMock()
    
    # Mock PYFLOWREG_AVAILABLE to allow processing to start
    original_available = widgets_module.PYFLOWREG_AVAILABLE
    widgets_module.PYFLOWREG_AVAILABLE = True
    
    try:
        widget = FlowRegWidget(viewer)
        
        # Create bad options that will cause error
        bad_options = {
            "quality_setting": "invalid_quality",  # This should cause error
            "alpha": -1.0,  # Invalid negative alpha
        }
        
        # Small valid data
        video = np.random.rand(4, 16, 16).astype(np.float32)
        reference = video[0]
        
        # Instead of running the actual worker, test that the error handling 
        # logic works correctly by directly calling the function
        with pytest.raises(ValueError) as exc_info:
            # Call the actual function (not the decorated version)
            # This simulates what happens inside the worker
            widget._run_motion_correction.__wrapped__(
                widget, video, reference, bad_options
            )
        
        # Check that the appropriate error was raised
        assert "Invalid quality setting" in str(exc_info.value)
        
    finally:
        # Restore original value
        widgets_module.PYFLOWREG_AVAILABLE = original_available