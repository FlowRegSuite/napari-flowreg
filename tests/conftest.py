"""
Pytest configuration and fixtures for napari-flowreg tests.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np

# Ensure hermetic test environment - disable SVML and set cache dir
@pytest.fixture(scope="session", autouse=True)
def hermetic_environment():
    """Set up hermetic test environment to prevent cache/env bleeding."""
    # Disable SVML for all platforms to ensure consistency
    os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    os.environ["NUMBA_CPU_NAME"] = "generic"
    
    # Use temporary cache directory for Numba
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(tmpdir) / "numba-cache")
        
        # For POSIX systems, use spawn to mirror Windows semantics
        if sys.platform != "win32":
            import multiprocessing as mp
            original_method = mp.get_start_method(allow_none=True)
            try:
                mp.set_start_method("spawn", force=True)
                yield
            finally:
                if original_method:
                    mp.set_start_method(original_method, force=True)
        else:
            yield


@pytest.fixture
def make_napari_viewer_proxy():
    """Create a headless napari viewer for testing."""
    try:
        from napari import Viewer
        from napari.utils._proxies import PublicOnlyProxy
        import sys
        
        viewer = Viewer(show=False)
        viewer_proxy = PublicOnlyProxy(viewer)
        yield viewer_proxy
        
        # Safer cleanup for Windows OpenGL context issues
        try:
            viewer.layers.clear()
            viewer.close()
        except (RuntimeError, AttributeError) as e:
            if "OpenGL" in str(e) or "glBindFramebuffer" in str(e):
                # Known issue on Windows with OpenGL context during teardown
                pass
            else:
                raise
    except ImportError:
        pytest.skip("napari not installed")


@pytest.fixture
def make_napari_viewer():
    """Create a napari viewer for testing (fallback for older napari)."""
    try:
        from napari import Viewer
        import sys
        
        viewer = Viewer(show=False)
        yield viewer
        
        # Safer cleanup for Windows OpenGL context issues
        try:
            viewer.layers.clear()
            viewer.close()
        except (RuntimeError, AttributeError) as e:
            if "OpenGL" in str(e) or "glBindFramebuffer" in str(e):
                # Known issue on Windows with OpenGL context during teardown
                pass
            else:
                raise
    except ImportError:
        pytest.skip("napari not installed")


@pytest.fixture
def sample_video_2d():
    """Create a small 2D video array for testing."""
    np.random.seed(42)
    # 10 frames, 32x32 pixels
    video = np.random.rand(10, 32, 32).astype(np.float32)
    # Add some motion pattern
    for t in range(10):
        shift = t * 0.5
        video[t] = np.roll(video[t], int(shift), axis=1)
    return video


@pytest.fixture
def sample_video_3d():
    """Create a small 3D video array for testing."""
    np.random.seed(42)
    # 10 frames, 16x32x32 pixels (z, y, x)
    video = np.random.rand(10, 16, 32, 32).astype(np.float32)
    return video


@pytest.fixture
def sample_video_multichannel():
    """Create a small multi-channel video array for testing."""
    np.random.seed(42)
    # 10 frames, 32x32 pixels, 2 channels
    video = np.random.rand(10, 32, 32, 2).astype(np.float32)
    return video


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pyflowreg_options():
    """Create a mock options dict for testing without pyflowreg."""
    return {
        "quality_setting": "fast",
        "alpha": 1.5,
        "sigma": [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],
        "levels": 3,
        "iterations": 10,
        "eta": 0.5,
        "save_w": True,
        "output_typename": "double",
        "verbose": False
    }