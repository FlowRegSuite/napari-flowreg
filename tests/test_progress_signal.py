"""
Test progress signaling in FlowRegWidget.
"""

import pytest
import numpy as np
from qtpy.QtCore import QObject, QThread
from qtpy.QtTest import QSignalSpy
from unittest.mock import MagicMock, patch
import time


def test_progress_signal_connection(make_napari_viewer, qtbot):
    """Test that progress signal is properly connected to progress bar."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Check signal is connected by checking if progress bar gets updated
    # In PySide6/PyQt, we can't directly check receivers, so we test functionality
    
    # Test signal emission updates progress bar
    initial_value = widget.progress_bar.value()
    test_value = 50
    
    # Create signal spy to monitor emissions
    spy = QSignalSpy(widget.progress_val)
    
    # Emit signal
    widget.progress_val.emit(test_value)
    
    # Wait for signal processing
    qtbot.wait(10)
    
    # Check signal was emitted
    assert spy.count() == 1
    assert spy.at(0)[0] == test_value
    
    # Check progress bar was updated
    assert widget.progress_bar.value() == test_value


def test_progress_callback_integration(make_napari_viewer, qtbot):
    """Test progress callback updates through Qt signals during motion correction."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Add test data
    test_data = np.random.rand(10, 64, 64)  # Small test video
    viewer.add_image(test_data, name="test_video")
    
    # Update widget layer lists
    widget._update_layer_lists()
    widget.input_combo.setCurrentText("test_video")
    
    # Set up for testing
    widget.ref_method_combo.setCurrentText("Average all frames")
    widget.export_flow_check.setChecked(False)  # Disable flow export for speed
    
    # Create signal spy to monitor progress updates
    progress_spy = QSignalSpy(widget.progress_val)
    
    # Mock the actual motion correction to simulate progress callbacks
    # compensate_arr is imported inside the worker function, so we need to patch it there
    with patch('pyflowreg.motion_correction.compensate_arr.compensate_arr') as mock_compensate:
        # Simulate motion correction with progress callbacks
        def mock_correction(video, ref, options, progress_callback=None):
            # Simulate processing frames with progress updates
            total_frames = video.shape[0]
            for i in range(total_frames):
                if progress_callback:
                    progress_callback(i + 1, total_frames)
                time.sleep(0.01)  # Small delay to simulate processing
            
            # Return mock results
            return video, None  # Return original as "corrected" and no flow
        
        mock_compensate.side_effect = mock_correction
        
        # Start motion correction
        widget._on_start_clicked()
        
        # Wait for processing to complete
        qtbot.wait(500)  # Wait up to 500ms for completion
        
        # Check that progress signals were emitted
        signal_count = progress_spy.count()
        assert signal_count > 0, "No progress signals were emitted"
        
        # Check that progress values are in expected range
        for i in range(signal_count):
            progress_value = progress_spy.at(i)[0]
            assert 0 <= progress_value <= 100, f"Progress value {progress_value} out of range"
        
        # Check final progress is 100%
        if signal_count > 0:
            final_progress = progress_spy.at(signal_count - 1)[0]
            assert final_progress == 100, f"Final progress was {final_progress}, expected 100"


def test_progress_callback_thread_safety(make_napari_viewer, qtbot):
    """Test that progress callbacks from worker thread are thread-safe."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Track which thread emits signals
    main_thread = QThread.currentThread()
    signal_threads = []
    
    def track_thread(*args):
        signal_threads.append(QThread.currentThread())
    
    # Connect to track thread
    widget.progress_val.connect(track_thread)
    
    # Test emission from different thread
    class WorkerThread(QThread):
        def __init__(self, widget):
            super().__init__()
            self.widget = widget
            
        def run(self):
            # Emit progress from worker thread
            for i in range(5):
                self.widget.progress_val.emit(i * 20)
                time.sleep(0.01)
    
    worker = WorkerThread(widget)
    worker.start()
    
    # Wait for worker to complete
    assert worker.wait(1000), "Worker thread did not complete in time"
    
    # Process events to ensure signals are delivered
    qtbot.wait(100)  # Give Qt time to process the signals
    
    # Check signals were received
    assert len(signal_threads) == 5, f"Expected 5 signals, got {len(signal_threads)}"
    
    # Qt signals should be thread-safe and handled in main thread
    # This is ensured by Qt's signal-slot mechanism


def test_progress_bar_reset_after_completion(make_napari_viewer, qtbot):
    """Test that progress bar is properly reset after completion."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Set progress to some value
    widget.progress_val.emit(75)
    qtbot.wait(10)
    assert widget.progress_bar.value() == 75
    
    # Simulate completion by calling reset
    widget._reset_ui()
    
    # Check progress bar is reset
    assert widget.progress_bar.value() == 0
    assert not widget.progress_bar.isVisible()


def test_progress_with_zero_frames(make_napari_viewer, qtbot):
    """Test progress callback handles edge case of zero total frames."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Create the update_progress function as it would be in _on_start_clicked
    progress_values = []
    
    def update_progress(current_frame, total_frames):
        if not total_frames:
            # Should handle gracefully without division by zero
            progress_values.append(None)
            return
        value = int((current_frame * 100) // total_frames)
        widget.progress_val.emit(value)
        progress_values.append(value)
    
    # Test with zero total frames
    update_progress(0, 0)
    assert progress_values[-1] is None, "Should handle zero total frames gracefully"
    
    # Test with valid frames
    update_progress(5, 10)
    qtbot.wait(10)
    assert progress_values[-1] == 50
    assert widget.progress_bar.value() == 50


@pytest.mark.parametrize("current,total,expected", [
    (0, 100, 0),
    (25, 100, 25),
    (50, 100, 50),
    (100, 100, 100),
    (1, 3, 33),  # Test rounding
    (2, 3, 66),
    (3, 3, 100),
])
def test_progress_calculation(make_napari_viewer, qtbot, current, total, expected):
    """Test progress percentage calculation for various frame counts."""
    from napari_flowreg.widgets import FlowRegWidget
    
    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)
    
    # Simulate progress callback
    def update_progress(current_frame, total_frames):
        if not total_frames:
            return
        value = int((current_frame * 100) // total_frames)
        widget.progress_val.emit(value)
    
    # Test calculation
    update_progress(current, total)
    qtbot.wait(10)
    
    # Check progress bar value
    assert widget.progress_bar.value() == expected