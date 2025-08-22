"""
Test FlowReg widget functionality and wiring.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from qtpy.QtCore import Qt


def test_widget_creation(make_napari_viewer_proxy):
    """Test that widget can be created and added to viewer."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    
    # Create widget
    widget = FlowRegWidget(viewer)
    assert widget is not None
    
    # Add to viewer
    viewer.window.add_dock_widget(widget, name="FlowReg")
    
    # Check widget is in the viewer
    assert "FlowReg" in viewer.window._dock_widgets


def test_widget_ui_elements(make_napari_viewer_proxy):
    """Test that all expected UI elements are present."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Check main UI elements exist
    assert hasattr(widget, 'input_combo')
    assert hasattr(widget, 'quality_combo')
    assert hasattr(widget, 'start_button')
    assert hasattr(widget, 'cancel_button')
    assert hasattr(widget, 'progress_bar')
    assert hasattr(widget, 'log_text')
    
    # Check parameter controls
    assert hasattr(widget, 'smooth_x')
    assert hasattr(widget, 'smooth_y')
    assert hasattr(widget, 'sigma_xy')
    assert hasattr(widget, 'sigma_t')
    assert hasattr(widget, 'levels_spin')
    assert hasattr(widget, 'iterations_spin')
    assert hasattr(widget, 'eta_spin')
    
    # Check reference controls
    assert hasattr(widget, 'ref_method_combo')
    assert hasattr(widget, 'ref_start_spin')
    assert hasattr(widget, 'ref_end_spin')


def test_layer_list_updates(make_napari_viewer_proxy, sample_video_2d):
    """Test that layer lists update when layers are added."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Initially no layers
    assert widget.input_combo.count() == 0
    
    # Add a layer
    viewer.add_image(sample_video_2d, name="test_video")
    
    # Update lists
    widget._update_layer_lists()
    
    # Check layer appears in combo boxes
    assert widget.input_combo.count() == 1
    assert widget.input_combo.itemText(0) == "test_video"
    assert widget.ref_layer_combo.count() == 1


def test_quality_setting_enables_custom_params(make_napari_viewer_proxy):
    """Test that custom parameters are enabled when quality is set to custom."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Initially custom params should be disabled
    assert not widget.levels_spin.isEnabled()
    assert not widget.iterations_spin.isEnabled()
    assert not widget.eta_spin.isEnabled()
    
    # Set quality to custom
    widget.quality_combo.setCurrentText("custom")
    widget._on_quality_changed("custom")
    
    # Now they should be enabled
    assert widget.levels_spin.isEnabled()
    assert widget.iterations_spin.isEnabled()
    assert widget.eta_spin.isEnabled()
    
    # Set back to balanced
    widget.quality_combo.setCurrentText("balanced")
    widget._on_quality_changed("balanced")
    
    # Should be disabled again
    assert not widget.levels_spin.isEnabled()
    assert not widget.iterations_spin.isEnabled()
    assert not widget.eta_spin.isEnabled()


def test_reference_method_enables_controls(make_napari_viewer_proxy):
    """Test that reference controls are enabled based on method."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Frame range method
    widget.ref_method_combo.setCurrentText("Frame range")
    widget._on_ref_method_changed("Frame range")
    assert widget.ref_start_spin.isEnabled()
    assert widget.ref_end_spin.isEnabled()
    assert not widget.ref_layer_combo.isEnabled()
    
    # External layer method
    widget.ref_method_combo.setCurrentText("External layer")
    widget._on_ref_method_changed("External layer")
    assert not widget.ref_start_spin.isEnabled()
    assert not widget.ref_end_spin.isEnabled()
    assert widget.ref_layer_combo.isEnabled()
    
    # Average all frames
    widget.ref_method_combo.setCurrentText("Average all frames")
    widget._on_ref_method_changed("Average all frames")
    assert not widget.ref_start_spin.isEnabled()
    assert not widget.ref_end_spin.isEnabled()
    assert not widget.ref_layer_combo.isEnabled()


def test_create_options_dict(make_napari_viewer_proxy):
    """Test options dictionary creation from GUI settings."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Set some values
    widget.quality_combo.setCurrentText("balanced")
    widget.smooth_x.setValue(1.5)
    widget.smooth_y.setValue(2.0)
    widget.symmetric_smooth_check.setChecked(False)
    widget.sigma_xy.setValue(1.2)
    widget.sigma_t.setValue(0.3)
    
    # Create options
    options = widget._create_options_dict()
    
    # Check options
    assert options["quality_setting"] == "balanced"
    assert options["alpha"] == (1.5, 2.0)  # Non-symmetric
    assert options["save_w"] == False  # Default changed to False
    assert options["verbose"] == True
    assert options["sigma"] == [[1.2, 1.2, 0.3], [1.2, 1.2, 0.3]]
    
    # Test with export flow enabled
    widget.export_flow_check.setChecked(True)
    options = widget._create_options_dict()
    assert options["save_w"] == True  # Should be True when checkbox is checked
    
    # Test symmetric smoothness
    widget.symmetric_smooth_check.setChecked(True)
    options = widget._create_options_dict()
    assert options["alpha"] == 1.5  # Symmetric, uses x value


def test_get_reference_frames(make_napari_viewer_proxy, sample_video_2d):
    """Test reference frame extraction methods."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Add test data
    viewer.add_image(sample_video_2d, name="test_video")
    widget._update_layer_lists()
    widget.input_combo.setCurrentText("test_video")
    
    # Test frame range
    widget.ref_method_combo.setCurrentText("Frame range")
    widget.ref_start_spin.setValue(2)
    widget.ref_end_spin.setValue(4)
    reference = widget._get_reference_frames()
    assert reference.shape == (32, 32)  # Should be averaged
    
    # Test current frame
    widget.ref_method_combo.setCurrentText("Current frame")
    viewer.dims.current_step = (3,)  # Set current frame
    reference = widget._get_reference_frames()
    assert reference.shape == (32, 32)
    
    # Test average all frames
    widget.ref_method_combo.setCurrentText("Average all frames")
    reference = widget._get_reference_frames()
    assert reference.shape == (32, 32)


def test_logging(make_napari_viewer_proxy):
    """Test that logging works."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Log a message
    widget.log("Test message")
    
    # Check it appears in log
    assert "Test message" in widget.log_text.toPlainText()


def test_start_button_without_data(make_napari_viewer_proxy):
    """Test start button behavior without data."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Mock show_error to capture the call
    with patch('napari_flowreg.flowreg_widget.show_error') as mock_error:
        widget._on_start_clicked()
        
        # Should show error about no input
        mock_error.assert_called()
        assert "No input layer" in str(mock_error.call_args)


@patch('napari_flowreg.flowreg_widget.PYFLOWREG_AVAILABLE', False)
def test_start_without_pyflowreg(make_napari_viewer_proxy):
    """Test start button when pyflowreg is not available."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    with patch('napari_flowreg.flowreg_widget.show_error') as mock_error:
        widget._on_start_clicked()
        
        # Should show error about pyflowreg
        mock_error.assert_called_once()
        assert "PyFlowReg is not installed" in str(mock_error.call_args)


def test_progress_bar_visibility(make_napari_viewer_proxy, sample_video_2d):
    """Test progress bar shows/hides during processing."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    from unittest.mock import MagicMock
    import napari_flowreg.flowreg_widget as widgets_module
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Initially hidden
    assert widget.progress_bar.isHidden()
    
    # Add data
    viewer.add_image(sample_video_2d, name="test_video")
    widget._update_layer_lists()
    widget.input_combo.setCurrentText("test_video")
    
    # Mock pyflowreg availability at module level
    original_available = widgets_module.PYFLOWREG_AVAILABLE
    widgets_module.PYFLOWREG_AVAILABLE = True
    
    try:
        # Create a mock worker that will be returned
        mock_worker = MagicMock()
        mock_worker.returned = MagicMock()
        mock_worker.errored = MagicMock()
        mock_worker.start = MagicMock()
        
        # Mock the _run_motion_correction method directly
        original_method = widget._run_motion_correction
        widget._run_motion_correction = MagicMock(return_value=mock_worker)
        
        # Click start - this should make progress bar visible
        widget._on_start_clicked()
        
        # Progress bar should not be hidden after start
        assert not widget.progress_bar.isHidden()
        assert widget.start_button.isEnabled() == False
        assert widget.cancel_button.isEnabled() == True
        
        # Simulate completion by calling reset
        widget._reset_ui()
        
        # Should be hidden again
        assert widget.progress_bar.isHidden()
        assert widget.start_button.isEnabled() == True
        assert widget.cancel_button.isEnabled() == False
        
        # Restore original method
        widget._run_motion_correction = original_method
    finally:
        # Restore original value
        widgets_module.PYFLOWREG_AVAILABLE = original_available


def test_button_states_during_processing(make_napari_viewer_proxy):
    """Test button enable/disable states during processing."""
    from napari_flowreg.flowreg_widget import FlowRegWidget
    
    viewer = make_napari_viewer_proxy
    widget = FlowRegWidget(viewer)
    
    # Initial state
    assert widget.start_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    
    # During processing (simulate by calling parts of start logic)
    widget.start_button.setEnabled(False)
    widget.cancel_button.setEnabled(True)
    
    assert not widget.start_button.isEnabled()
    assert widget.cancel_button.isEnabled()
    
    # After reset
    widget._reset_ui()
    
    assert widget.start_button.isEnabled()
    assert not widget.cancel_button.isEnabled()