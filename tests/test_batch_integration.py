"""
Tests for batch processing integration in motion_analysis_widget.
"""

import pytest
import numpy as np
from napari_flowreg.motion_analysis import (
    MotionAnalysisWidget,
    create_mode_selector_group,
    add_batch_processing_options,
    handle_mode_change,
    ProcessingMode,
    ModeManager,
    MODE_DEFINITIONS
)


class TestModeDefinitions:
    """Test processing mode definitions."""

    def test_all_modes_defined(self):
        """Test that all modes have definitions."""
        assert ProcessingMode.ONLINE_LOADED in MODE_DEFINITIONS
        assert ProcessingMode.LOADED_BATCH_PROCESS in MODE_DEFINITIONS
        assert ProcessingMode.LOADED_WITH_PRECOMPUTED in MODE_DEFINITIONS

    def test_mode_memory_profiles(self):
        """Test that all modes have memory profiles."""
        for mode, definition in MODE_DEFINITIONS.items():
            assert definition.memory.usage_level in ["minimal", "low", "medium", "high"]
            assert definition.memory.peak_formula is not None
            assert definition.memory.typical_mb > 0

    def test_mode_input_configs(self):
        """Test mode input configurations."""
        online_mode = MODE_DEFINITIONS[ProcessingMode.ONLINE_LOADED]
        assert online_mode.input.requires_flow is True

        batch_mode = MODE_DEFINITIONS[ProcessingMode.LOADED_BATCH_PROCESS]
        assert batch_mode.input.requires_flow is False
        assert batch_mode.input.requires_video is True

    def test_mode_output_configs(self):
        """Test mode output configurations."""
        batch_mode = MODE_DEFINITIONS[ProcessingMode.LOADED_BATCH_PROCESS]
        assert batch_mode.output.save_flow is True
        assert batch_mode.output.save_corrected is True


class TestModeManager:
    """Test mode manager functionality."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = ModeManager()
        assert manager.current_mode == ProcessingMode.ONLINE_LOADED

    def test_get_mode_names(self):
        """Test getting mode display names."""
        manager = ModeManager()
        names = manager.get_mode_names()

        assert len(names) > 0
        assert "Online: Loaded Data" in names
        assert "Batch Process: Loaded Video" in names

    def test_get_mode_by_name(self):
        """Test retrieving mode by display name."""
        manager = ModeManager()

        mode_def = manager.get_mode_by_name("Online: Loaded Data")
        assert mode_def is not None
        assert mode_def.mode == ProcessingMode.ONLINE_LOADED

    def test_set_mode(self):
        """Test setting current mode."""
        manager = ModeManager()
        manager.set_mode(ProcessingMode.LOADED_BATCH_PROCESS)

        assert manager.current_mode == ProcessingMode.LOADED_BATCH_PROCESS

    def test_set_mode_by_name(self):
        """Test setting mode by display name."""
        manager = ModeManager()
        manager.set_mode_by_name("Batch Process: Loaded Video")

        assert manager.current_mode == ProcessingMode.LOADED_BATCH_PROCESS

    def test_validate_inputs_online_mode(self):
        """Test input validation for online mode."""
        manager = ModeManager()
        mode_def = MODE_DEFINITIONS[ProcessingMode.ONLINE_LOADED]

        # Missing flow source
        inputs = {"video_source": "test_video"}
        errors = manager.validate_inputs(mode_def, inputs)
        assert len(errors) > 0
        assert any("flow" in err.lower() for err in errors)

        # Valid inputs
        inputs = {"flow_source": "test_flow", "video_source": "test_video"}
        errors = manager.validate_inputs(mode_def, inputs)
        assert len(errors) == 0

    def test_validate_inputs_batch_mode(self):
        """Test input validation for batch mode."""
        manager = ModeManager()
        mode_def = MODE_DEFINITIONS[ProcessingMode.LOADED_BATCH_PROCESS]

        # Missing video source
        inputs = {"flow_source": "test_flow"}
        errors = manager.validate_inputs(mode_def, inputs)
        assert len(errors) > 0

        # Invalid batch size
        inputs = {
            "video_source": "test_video",
            "batch_size": 5,  # Too small
            "w_output_path": "test.h5",
            "corrected_output_path": "test_corrected.h5"
        }
        errors = manager.validate_inputs(mode_def, inputs)
        assert len(errors) > 0

        # Valid inputs
        inputs = {
            "video_source": "test_video",
            "batch_size": 50,
            "w_output_path": "test.h5",
            "corrected_output_path": "test_corrected.h5"
        }
        errors = manager.validate_inputs(mode_def, inputs)
        assert len(errors) == 0


class TestUIComponents:
    """Test UI component creation."""

    def test_create_mode_selector_group(self, qtbot):
        """Test mode selector group creation."""
        group = create_mode_selector_group()
        qtbot.addWidget(group)

        assert hasattr(group, 'mode_combo')
        assert hasattr(group, 'mode_info')
        assert hasattr(group, 'memory_label')

        # Check combo has items
        assert group.mode_combo.count() > 0

    def test_mode_selector_default(self, qtbot):
        """Test mode selector defaults to online mode."""
        group = create_mode_selector_group()
        qtbot.addWidget(group)

        # Should default to first item (online mode)
        assert group.mode_combo.currentIndex() == 0
        assert "Online" in group.mode_combo.currentText()

    def test_add_batch_processing_options(self, qtbot):
        """Test batch processing options group creation."""
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        parent = QWidget()
        layout = QVBoxLayout()
        parent.setLayout(layout)
        qtbot.addWidget(parent)

        batch_group = add_batch_processing_options(layout)

        assert hasattr(batch_group, 'batch_size_spin')
        assert hasattr(batch_group, 'workers_spin')
        assert hasattr(batch_group, 'save_w_check')
        assert hasattr(batch_group, 'w_path_edit')
        assert hasattr(batch_group, 'load_w_check')

        # Should start hidden
        assert not batch_group.isVisible()

    def test_batch_options_defaults(self, qtbot):
        """Test batch options have sensible defaults."""
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        parent = QWidget()
        layout = QVBoxLayout()
        parent.setLayout(layout)
        qtbot.addWidget(parent)

        batch_group = add_batch_processing_options(layout)

        assert batch_group.batch_size_spin.value() == 50
        assert batch_group.workers_spin.value() == 4
        assert not batch_group.save_w_check.isChecked()
        assert not batch_group.load_w_check.isChecked()


class TestModeChangeHandling:
    """Test mode change behavior."""

    def test_handle_mode_change_online(self, make_napari_viewer, qtbot):
        """Test switching to online mode."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Widget needs to be shown for visibility to work
        widget.show()
        qtbot.waitExposed(widget)

        # Switch to online mode
        handle_mode_change(widget, 0)

        assert widget.current_processing_mode == 0
        assert not widget.batch_processing_group.isVisible()
        assert "HIGH" in widget.memory_label.text()

    def test_handle_mode_change_batch(self, make_napari_viewer, qtbot):
        """Test switching to batch mode."""
        from qtpy.QtWidgets import QApplication

        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Widget needs to be shown for visibility to work
        widget.show()
        qtbot.waitExposed(widget)

        # Switch to batch mode
        handle_mode_change(widget, 1)

        # Force Qt to process all pending events
        QApplication.processEvents()
        qtbot.wait(10)

        assert widget.current_processing_mode == 1
        assert widget.batch_processing_group.isVisible()
        assert "LOW" in widget.memory_label.text()

    def test_handle_mode_change_ui_updates(self, make_napari_viewer, qtbot):
        """Test that UI updates correctly on mode change."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Online mode
        handle_mode_change(widget, 0)
        online_button_text = widget.analyze_button.text()

        # Batch mode
        handle_mode_change(widget, 1)
        batch_button_text = widget.analyze_button.text()

        # Button text should change
        assert online_button_text != batch_button_text
        assert "Process" in batch_button_text


class TestWidgetIntegration:
    """Test integrated widget behavior."""

    def test_widget_has_mode_selector(self, make_napari_viewer, qtbot):
        """Test that widget has mode selector."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        assert hasattr(widget, 'mode_combo')
        assert hasattr(widget, 'batch_processing_group')
        assert hasattr(widget, 'current_processing_mode')

    def test_widget_has_video_combo(self, make_napari_viewer, qtbot):
        """Test that widget has video input dropdown."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        assert hasattr(widget, 'video_combo')

    def test_widget_mode_selector_connected(self, make_napari_viewer, qtbot):
        """Test that mode selector is connected to handler."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        initial_mode = widget.current_processing_mode

        # Change mode via combo
        widget.mode_combo.setCurrentIndex(1)

        # Mode should update
        assert widget.current_processing_mode != initial_mode

    def test_layer_list_includes_videos(self, make_napari_viewer, qtbot):
        """Test that layer list update includes video layers."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Add a video layer
        video = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
        viewer.add_image(video, name="test_video")

        # Video should appear in video combo
        video_items = [widget.video_combo.itemText(i)
                      for i in range(widget.video_combo.count())]
        assert "test_video" in video_items

    def test_analyze_button_online_mode(self, make_napari_viewer, qtbot):
        """Test analyze button behavior in online mode."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Create flow field
        flow = np.random.randn(10, 100, 100, 2).astype(np.float32)
        viewer.add_image(flow, name="w")
        widget.current_flow_layer = viewer.layers["w"]

        # Set to online mode
        widget.current_processing_mode = 0

        # Should not raise error (but may show error notification)
        widget._on_analyze_clicked()


class TestOFOptionsValidation:
    """Test OFOptions validation to prevent parameter errors."""

    def test_ofoptions_rejects_n_jobs_parameter(self):
        """Test that OFOptions correctly rejects n_jobs parameter.

        This test verifies the bug fix where n_jobs was incorrectly passed to OFOptions.
        OFOptions uses Pydantic with extra="forbid", so it should reject unknown parameters.
        """
        pytest.importorskip("pyflowreg", reason="pyflowreg not installed")

        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
        from pydantic import ValidationError

        # Should raise ValidationError when n_jobs is passed
        with pytest.raises(ValidationError) as exc_info:
            OFOptions(
                quality_setting=QualitySetting.BALANCED,
                buffer_size=50,
                n_jobs=4  # This should be rejected
            )

        # Verify error message mentions extra fields not permitted
        assert "extra" in str(exc_info.value).lower() or "forbidden" in str(exc_info.value).lower()

    def test_ofoptions_accepts_valid_parameters(self):
        """Test that OFOptions accepts valid parameters without n_jobs."""
        pytest.importorskip("pyflowreg", reason="pyflowreg not installed")

        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting, OutputFormat

        # Should succeed with valid parameters (no n_jobs)
        options = OFOptions(
            quality_setting=QualitySetting.BALANCED,
            buffer_size=50,
            save_w=True,
            w_output_path="test.h5",
            output_format=OutputFormat.HDF5
        )

        assert options.quality_setting == QualitySetting.BALANCED
        assert options.buffer_size == 50
        assert options.save_w is True

    def test_ofoptions_defaults(self):
        """Test that OFOptions works with minimal parameters."""
        pytest.importorskip("pyflowreg", reason="pyflowreg not installed")

        from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting

        # Should work with just quality_setting
        options = OFOptions(quality_setting=QualitySetting.BALANCED)

        assert options.quality_setting == QualitySetting.BALANCED
        # buffer_size should have a default value
        assert options.buffer_size > 0


@pytest.mark.integration
class TestBatchProcessingIntegration:
    """Integration tests for batch processing.

    Run with: pytest tests/test_batch_integration.py -m integration
    """

    def test_batch_mode_with_synthetic_data(self, make_napari_viewer, qtbot):
        """Test batch processing with synthetic video data."""
        from qtpy.QtWidgets import QApplication

        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Widget needs to be shown for visibility to work
        widget.show()
        qtbot.waitExposed(widget)

        # Create synthetic video
        video = np.random.randint(0, 255, (20, 100, 100), dtype=np.uint8)
        viewer.add_image(video, name="test_video")

        # Switch to batch mode
        widget.mode_combo.setCurrentIndex(1)

        # Force Qt to process all pending events including signals and visibility updates
        QApplication.processEvents()
        qtbot.wait(10)

        # Configure batch options
        widget.batch_processing_group.batch_size_spin.setValue(10)
        widget.batch_processing_group.workers_spin.setValue(2)
        widget.batch_processing_group.load_w_check.setChecked(True)

        # Note: Full batch processing requires pyflowreg
        # This test verifies UI setup, not actual processing
        assert widget.current_processing_mode == 1
        assert widget.batch_processing_group.isVisible()
