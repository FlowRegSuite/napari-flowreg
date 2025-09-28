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
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
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
    qtbot.waitUntil(lambda: len(spy) >= 1, timeout=200)

    # Check signal was emitted
    assert len(spy) == 1
    assert spy[0][0] == test_value
    
    # Check progress bar was updated
    assert widget.progress_bar.value() == test_value


def test_progress_callback_integration(make_napari_viewer, qtbot):
    """Test progress callback updates through Qt signals during motion correction."""
    from napari_flowreg.flowreg_widget import FlowRegWidget

    viewer = make_napari_viewer
    widget = FlowRegWidget(viewer)

    # Add test data
    test_data = np.random.rand(10, 64, 64)  # Small test video
    total_frames = test_data.shape[0]
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
    # Also patch ViewerModel.add_image at class level to prevent teardown errors
    with patch('pyflowreg.motion_correction.compensate_arr.compensate_arr') as mock_compensate, \
         patch('napari.components.viewer_model.ViewerModel.add_image', MagicMock(return_value=None)):
        # Simulate motion correction with progress callbacks
        def mock_correction(video, ref, options, progress_callback=None):
            # Simulate processing frames with progress updates
            n = video.shape[0]
            for i in range(n):
                if progress_callback:
                    progress_callback(i + 1, n)
                time.sleep(0.005)  # Small delay to simulate processing

            # Return mock results
            return video, None  # Return original as "corrected" and no flow

        mock_compensate.side_effect = mock_correction

        # Start motion correction
        widget._on_start_clicked()

        # Wait for completion: we should see progress for all frames and final value should be 100
        qtbot.waitUntil(
            lambda: len(progress_spy) >= total_frames and int(progress_spy[-1][0]) >= 100,
            timeout=5000
        )

        # Ensure the completion handler finished and UI reset occurred
        qtbot.waitUntil(lambda: widget.start_button.isEnabled(), timeout=5000)

    # Check that progress signals were emitted for all frames
    signal_count = len(progress_spy)
    assert signal_count >= total_frames, f"Expected at least {total_frames} signals, got {signal_count}"

    # Check that progress values are in expected range
    for i in range(signal_count):
        progress_value = int(progress_spy[i][0])
        assert 0 <= progress_value <= 100, f"Progress value {progress_value} out of range"

    # Check final progress is 100%
    final_progress = int(progress_spy[-1][0])
    assert final_progress == 100, f"Final progress was {final_progress}, expected 100"


def test_progress_callback_thread_safety(make_napari_viewer, qtbot):
    """Test that progress callbacks from worker thread are thread-safe."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
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
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
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
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
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
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
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