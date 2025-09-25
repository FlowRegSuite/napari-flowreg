"""
Test reader/writer functionality for napari-flowreg.
"""

import pytest
import numpy as np
from pathlib import Path
import h5py
import tifffile


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_reader_accepts_supported_files(temp_directory):
    """Test that reader accepts supported file types."""
    from napari_flowreg.io import napari_get_reader

    # Test accepted extensions
    supported = ['.h5', '.hdf5', '.tiff', '.tif', '.mat']
    for ext in supported:
        test_file = temp_directory / f"test{ext}"
        test_file.touch()  # Create empty file

        reader = napari_get_reader(str(test_file))
        # Reader should return a function or None
        # We accept None here since the file is empty
        assert reader is not None or ext == '.mat'  # MAT might need special handling


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_reader_rejects_unsupported_files(temp_directory):
    """Test that reader rejects unsupported file types."""
    from napari_flowreg.io import napari_get_reader
    
    # Test rejected extensions
    unsupported = ['.txt', '.csv', '.json', '.xyz']
    for ext in unsupported:
        test_file = temp_directory / f"test{ext}"
        test_file.touch()
        
        reader = napari_get_reader(str(test_file))
        assert reader is None


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_hdf5_round_trip(temp_directory, sample_video_2d):
    """Test HDF5 save/load round trip."""
    from napari_flowreg.io import write_compensated_data, napari_get_reader
    
    # Save data
    output_path = temp_directory / "test_output.h5"
    meta = {
        'name': 'test_data',
        'metadata': {'fps': 30.0, 'source': 'test'}
    }
    
    written = write_compensated_data(str(output_path), sample_video_2d, meta)
    assert len(written) == 1
    assert Path(written[0]).exists()
    
    # Load data back
    reader = napari_get_reader(str(output_path))
    assert reader is not None
    
    layer_data = reader(str(output_path))
    assert len(layer_data) > 0
    
    # Check data integrity
    loaded_data = layer_data[0][0]  # First element is the data
    assert loaded_data.shape == sample_video_2d.shape
    assert loaded_data.dtype == sample_video_2d.dtype
    np.testing.assert_array_almost_equal(loaded_data, sample_video_2d)


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_tiff_round_trip(temp_directory, sample_video_2d):
    """Test TIFF save/load round trip."""
    from napari_flowreg.io import write_compensated_data, napari_get_reader
    
    # Save data
    output_path = temp_directory / "test_output.tiff"
    meta = {'name': 'test_data'}
    
    written = write_compensated_data(str(output_path), sample_video_2d, meta)
    assert len(written) == 1
    assert Path(written[0]).exists()
    
    # Load data back
    reader = napari_get_reader(str(output_path))
    assert reader is not None
    
    layer_data = reader(str(output_path))
    assert len(layer_data) > 0
    
    # Check data integrity
    loaded_data = layer_data[0][0]
    assert loaded_data.shape == sample_video_2d.shape
    np.testing.assert_array_almost_equal(loaded_data, sample_video_2d)


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_multichannel_data_preservation(temp_directory, sample_video_multichannel):
    """Test that multi-channel data is preserved correctly."""
    from napari_flowreg.io import write_compensated_data, napari_get_reader
    
    # Save multi-channel data
    output_path = temp_directory / "test_multichannel.h5"
    meta = {'name': 'multichannel_data'}
    
    written = write_compensated_data(str(output_path), sample_video_multichannel, meta)
    assert len(written) == 1
    
    # Load back
    reader = napari_get_reader(str(output_path))
    layer_data = reader(str(output_path))
    
    loaded_data = layer_data[0][0]
    assert loaded_data.shape == sample_video_multichannel.shape
    assert loaded_data.shape[-1] == 2  # Two channels preserved


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_metadata_preservation(temp_directory, sample_video_2d):
    """Test that metadata is preserved during save/load."""
    from napari_flowreg.io import write_compensated_data, napari_get_reader
    
    # Create metadata
    output_path = temp_directory / "test_metadata.h5"
    meta = {
        'name': 'test_with_metadata',
        'metadata': {
            'fps': 30.0,
            'pixel_size': 0.325,
            'channel_names': ['GCaMP', 'tdTomato'],
            'processing': {
                'motion_corrected': True,
                'algorithm': 'flowreg',
                'parameters': {'alpha': 1.5, 'quality': 'balanced'}
            }
        }
    }
    
    # Save with metadata
    write_compensated_data(str(output_path), sample_video_2d, meta)
    
    # Load and check metadata
    reader = napari_get_reader(str(output_path))
    layer_data = reader(str(output_path))
    
    # Layer data format: (data, metadata, layer_type)
    if len(layer_data[0]) > 1:
        loaded_meta = layer_data[0][1]
        # Check some metadata was preserved
        assert loaded_meta is not None
        assert 'name' in loaded_meta or 'metadata' in loaded_meta


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_large_file_handling(temp_directory):
    """Test handling of large files (mock)."""
    from napari_flowreg.io import napari_get_reader
    
    # Create a mock large HDF5 file
    large_file = temp_directory / "large_file.h5"
    
    # Create file with chunked dataset
    with h5py.File(large_file, 'w') as f:
        # 100 frames, 512x512 - chunked for efficiency
        f.create_dataset('data', shape=(100, 512, 512), 
                        dtype=np.float32, chunks=(1, 512, 512))
    
    # Reader should handle it without loading all to memory
    reader = napari_get_reader(str(large_file))
    
    # Just check reader accepts it
    if reader is not None:
        # Reader should work with large files
        layer_data = reader(str(large_file))
        assert layer_data is not None


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_flow_field_saving(temp_directory):
    """Test saving flow fields alongside corrected data."""
    from napari_flowreg.io import write_flow_fields
    
    # Create mock flow field
    flow = np.random.randn(10, 32, 32, 2).astype(np.float32)
    
    output_path = temp_directory / "flow_fields.h5"
    meta = {'name': 'flow_fields'}
    
    # Save flow fields
    written = write_flow_fields(str(output_path), flow, meta)
    assert Path(written[0]).exists()
    
    # Check file contents
    with h5py.File(output_path, 'r') as f:
        assert 'flow' in f or 'data' in f
        saved_flow = f.get('flow', f.get('data'))
        assert saved_flow.shape == flow.shape


@pytest.mark.xfail(reason="IO module not yet implemented - pending implementation")
def test_reader_with_corrupt_file(temp_directory):
    """Test reader handles corrupt files gracefully."""
    from napari_flowreg.io import napari_get_reader
    
    # Create corrupt file
    corrupt_file = temp_directory / "corrupt.h5"
    corrupt_file.write_bytes(b"not a valid HDF5 file")
    
    # Reader should handle gracefully
    reader = napari_get_reader(str(corrupt_file))
    
    if reader is not None:
        # Should handle error gracefully
        try:
            layer_data = reader(str(corrupt_file))
            # Either returns None or empty list
            assert layer_data is None or len(layer_data) == 0
        except Exception:
            # Or raises a clear error
            pass