"""
FlowReg Motion Correction Widget for napari
Based on the ImageJ Flow Registration GUI design
"""

from typing import Optional, List, Tuple
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSlider, QGridLayout,
    QSplitter, QTextEdit, QProgressBar
)
from qtpy.QtCore import Qt, Signal, QThread
from napari.viewer import Viewer
from napari.layers import Image
from napari.utils.notifications import show_info, show_error
from napari.qt.threading import thread_worker

# Check if pyflowreg is available without importing it (to avoid early Numba import)
try:
    import importlib.util
    spec = importlib.util.find_spec("pyflowreg")
    PYFLOWREG_AVAILABLE = spec is not None
except (ImportError, ModuleNotFoundError):
    PYFLOWREG_AVAILABLE = False


class FlowRegWidget(QWidget):
    """Main widget for FlowReg motion correction in napari."""
    progress_val = Signal(int)
    
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_options = None
        self._init_ui()
        self.progress_val.connect(self.progress_bar.setValue)
        
    def _init_ui(self):
        """Initialize the user interface matching ImageJ design."""
        layout = QVBoxLayout()
        
        # Top section - Main parameters
        main_params_group = self._create_main_parameters_group()
        layout.addWidget(main_params_group)
        
        # Middle section - Channel parameters and reference
        channel_ref_group = self._create_channel_reference_group()
        layout.addWidget(channel_ref_group)
        
        # Bottom section - Control buttons and progress
        control_group = self._create_control_group()
        layout.addWidget(control_group)
        
        # Results/Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def _create_main_parameters_group(self) -> QGroupBox:
        """Create main parameters section like ImageJ."""
        group = QGroupBox("Flow Registration Parameters")
        layout = QGridLayout()
        
        # Input layer selection
        layout.addWidget(QLabel("Input Layer:"), 0, 0)
        self.input_combo = QComboBox()
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        layout.addWidget(self.input_combo, 0, 1, 1, 2)
        
        # Multi-channel checkbox
        self.multi_channel_check = QCheckBox("Multi-channel")
        layout.addWidget(self.multi_channel_check, 0, 3)
        
        # Channel weights (like injection_ch1#, injection_ch2# in ImageJ)
        layout.addWidget(QLabel("Channel 1 weight:"), 1, 0)
        self.ch1_weight = QDoubleSpinBox()
        self.ch1_weight.setRange(0, 1)
        self.ch1_weight.setSingleStep(0.1)
        self.ch1_weight.setValue(0.5)
        layout.addWidget(self.ch1_weight, 1, 1)
        
        layout.addWidget(QLabel("Channel 2 weight:"), 1, 2)
        self.ch2_weight = QDoubleSpinBox()
        self.ch2_weight.setRange(0, 1)
        self.ch2_weight.setSingleStep(0.1)
        self.ch2_weight.setValue(0.5)
        self.ch2_weight.setEnabled(False)  # Enable when multi-channel
        layout.addWidget(self.ch2_weight, 1, 3)
        
        # Symmetric smoothness weight checkbox
        self.symmetric_smooth_check = QCheckBox("Symmetric smoothness weight")
        self.symmetric_smooth_check.setChecked(True)
        layout.addWidget(self.symmetric_smooth_check, 2, 0, 1, 2)
        
        # Smoothness parameters (x, y)
        layout.addWidget(QLabel("Smoothness x:"), 3, 0)
        self.smooth_x = QDoubleSpinBox()
        self.smooth_x.setRange(0.1, 10)
        self.smooth_x.setSingleStep(0.1)
        self.smooth_x.setValue(1.5)
        layout.addWidget(self.smooth_x, 3, 1)
        
        layout.addWidget(QLabel("Smoothness y:"), 3, 2)
        self.smooth_y = QDoubleSpinBox()
        self.smooth_y.setRange(0.1, 10)
        self.smooth_y.setSingleStep(0.1)
        self.smooth_y.setValue(1.5)
        layout.addWidget(self.smooth_y, 3, 3)
        
        # Registration quality dropdown
        layout.addWidget(QLabel("Registration quality:"), 4, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["balanced", "quality", "fast", "custom"])
        self.quality_combo.setCurrentText("balanced")
        self.quality_combo.currentTextChanged.connect(self._on_quality_changed)
        layout.addWidget(self.quality_combo, 4, 1, 1, 2)
        
        # Custom parameters (enabled when quality is "custom")
        layout.addWidget(QLabel("Levels:"), 5, 0)
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(1, 200)
        self.levels_spin.setValue(100)
        self.levels_spin.setEnabled(False)
        layout.addWidget(self.levels_spin, 5, 1)
        
        layout.addWidget(QLabel("Iterations:"), 5, 2)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 200)
        self.iterations_spin.setValue(50)
        self.iterations_spin.setEnabled(False)
        layout.addWidget(self.iterations_spin, 5, 3)
        
        layout.addWidget(QLabel("Eta:"), 6, 0)
        self.eta_spin = QDoubleSpinBox()
        self.eta_spin.setRange(0.1, 1.0)
        self.eta_spin.setSingleStep(0.05)
        self.eta_spin.setValue(0.8)
        self.eta_spin.setEnabled(False)
        layout.addWidget(self.eta_spin, 6, 1)
        
        group.setLayout(layout)
        return group
        
    def _create_channel_reference_group(self) -> QGroupBox:
        """Create channel and reference selection section."""
        group = QGroupBox("Reference Frame Selection")
        layout = QGridLayout()
        
        # Reference selection method
        layout.addWidget(QLabel("Reference method:"), 0, 0)
        self.ref_method_combo = QComboBox()
        self.ref_method_combo.addItems([
            "Frame range",
            "Current frame", 
            "External layer",
            "Average all frames"
        ])
        self.ref_method_combo.currentTextChanged.connect(self._on_ref_method_changed)
        layout.addWidget(self.ref_method_combo, 0, 1, 1, 2)
        
        # Frame range selection (like "Registration Target Frames" in ImageJ)
        layout.addWidget(QLabel("Start frame:"), 1, 0)
        self.ref_start_spin = QSpinBox()
        self.ref_start_spin.setMinimum(0)
        self.ref_start_spin.setMaximum(99999)  # Set high default max, will be updated based on data
        self.ref_start_spin.setValue(0)
        layout.addWidget(self.ref_start_spin, 1, 1)
        
        layout.addWidget(QLabel("End frame:"), 1, 2)
        self.ref_end_spin = QSpinBox()
        self.ref_end_spin.setMinimum(0)
        self.ref_end_spin.setMaximum(99999)  # Set high default max, will be updated based on data
        self.ref_end_spin.setValue(20)
        layout.addWidget(self.ref_end_spin, 1, 3)
        
        # External reference layer (disabled by default)
        layout.addWidget(QLabel("Reference layer:"), 2, 0)
        self.ref_layer_combo = QComboBox()
        self.ref_layer_combo.setEnabled(False)
        layout.addWidget(self.ref_layer_combo, 2, 1, 1, 2)
        
        # Gaussian filter parameters (sigma)
        layout.addWidget(QLabel("Sigma XY:"), 3, 0)
        self.sigma_xy = QDoubleSpinBox()
        self.sigma_xy.setRange(0.1, 5.0)
        self.sigma_xy.setSingleStep(0.1)
        self.sigma_xy.setValue(1.0)
        layout.addWidget(self.sigma_xy, 3, 1)
        
        layout.addWidget(QLabel("Sigma T:"), 3, 2)
        self.sigma_t = QDoubleSpinBox()
        self.sigma_t.setRange(0.0, 2.0)
        self.sigma_t.setSingleStep(0.1)
        self.sigma_t.setValue(0.1)
        layout.addWidget(self.sigma_t, 3, 3)
        
        # Add checkbox for exporting flow field
        self.export_flow_check = QCheckBox("Export flow field")
        self.export_flow_check.setChecked(False)  # Default to NOT exporting flow
        self.export_flow_check.setToolTip("Save displacement field for visualization and analysis")
        layout.addWidget(self.export_flow_check, 4, 0, 1, 2)
        
        group.setLayout(layout)
        return group
        
    def _create_control_group(self) -> QGroupBox:
        """Create control buttons and progress section."""
        group = QGroupBox("Processing")
        layout = QVBoxLayout()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        button_layout.addWidget(self.cancel_button)
        
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self._on_save_settings)
        button_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Settings")
        self.load_button.clicked.connect(self._on_load_settings)
        button_layout.addWidget(self.load_button)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        group.setLayout(layout)
        return group
        
    def _on_input_changed(self):
        """Handle input layer selection change."""
        # Don't update layer lists here - it causes recursion!
        # Only update frame limits
        self._update_frame_limits()
        
    def _on_quality_changed(self, quality: str):
        """Enable/disable custom parameters based on quality setting."""
        is_custom = (quality == "custom")
        self.levels_spin.setEnabled(is_custom)
        self.iterations_spin.setEnabled(is_custom)
        self.eta_spin.setEnabled(is_custom)
        
    def _on_ref_method_changed(self, method: str):
        """Update UI based on reference selection method."""
        is_frame_range = (method == "Frame range")
        is_external = (method == "External layer")
        
        self.ref_start_spin.setEnabled(is_frame_range)
        self.ref_end_spin.setEnabled(is_frame_range)
        self.ref_layer_combo.setEnabled(is_external)
        
    def _update_layer_lists(self):
        """Update layer combo boxes with current napari layers."""
        # Block signals to prevent recursion
        self.input_combo.blockSignals(True)
        self.ref_layer_combo.blockSignals(True)
        
        # Store current selections
        current_input = self.input_combo.currentText()
        current_ref = self.ref_layer_combo.currentText()
        
        # Clear and update
        self.input_combo.clear()
        self.ref_layer_combo.clear()
        
        image_layers = [layer.name for layer in self.viewer.layers 
                       if isinstance(layer, Image)]
        
        self.input_combo.addItems(image_layers)
        self.ref_layer_combo.addItems(image_layers)
        
        # Restore selections if possible
        if current_input in image_layers:
            self.input_combo.setCurrentText(current_input)
        if current_ref in image_layers:
            self.ref_layer_combo.setCurrentText(current_ref)
            
        # Re-enable signals
        self.input_combo.blockSignals(False)
        self.ref_layer_combo.blockSignals(False)
        
    def _update_frame_limits(self):
        """Update frame selection limits based on input data."""
        if self.input_combo.currentIndex() < 0:
            return
            
        layer_name = self.input_combo.currentText()
        if not layer_name:
            return
            
        layer = self.viewer.layers[layer_name]
        if layer.data.ndim >= 3:  # Has time dimension
            n_frames = layer.data.shape[0]
            self.ref_start_spin.setMaximum(n_frames - 1)
            self.ref_end_spin.setMaximum(n_frames - 1)

            self.ref_start_spin.setValue(0)
            self.ref_end_spin.setValue(min(50, n_frames - 1))
                
    def _get_reference_frames(self) -> np.ndarray:
        """Get reference frames based on selected method."""
        method = self.ref_method_combo.currentText()
        layer_name = self.input_combo.currentText()
        layer = self.viewer.layers[layer_name]
        
        if method == "Frame range":
            start = self.ref_start_spin.value()
            end = self.ref_end_spin.value() + 1  # inclusive
            reference_frames = layer.data[start:end]
            reference = np.mean(reference_frames, axis=0)
            
        elif method == "Current frame":
            current_idx = self.viewer.dims.current_step[0] if layer.data.ndim >= 3 else 0
            reference = layer.data[current_idx]
            
        elif method == "External layer":
            ref_layer_name = self.ref_layer_combo.currentText()
            if ref_layer_name:
                reference = self.viewer.layers[ref_layer_name].data
                # Handle multi-frame reference by averaging
                if reference.ndim >= 3 and reference.shape[0] > 1:
                    reference = np.mean(reference, axis=0)
            else:
                raise ValueError("No reference layer selected")
                
        elif method == "Average all frames":
            reference = np.mean(layer.data, axis=0)
            
        else:
            raise ValueError(f"Unknown reference method: {method}")
            
        return reference
        
    def _create_options_dict(self):
        """Create options dictionary from GUI settings."""
        # Get alpha (smoothness)
        if self.symmetric_smooth_check.isChecked():
            alpha = self.smooth_x.value()
        else:
            alpha = (self.smooth_x.value(), self.smooth_y.value())
            
        # Build options dict
        options_dict = {
            "quality_setting": self.quality_combo.currentText(),
            "alpha": alpha,
            "save_w": self.export_flow_check.isChecked(),  # Use checkbox value for save_w
            "output_typename": "double",
            "verbose": True
        }
        
        # Set custom parameters if needed
        if self.quality_combo.currentText() == "custom":
            options_dict["levels"] = self.levels_spin.value()
            options_dict["iterations"] = self.iterations_spin.value()
            options_dict["eta"] = self.eta_spin.value()
            
        # Set sigma for Gaussian filtering
        sigma_xy = self.sigma_xy.value()
        sigma_t = self.sigma_t.value()
        options_dict["sigma"] = [[sigma_xy, sigma_xy, sigma_t], [sigma_xy, sigma_xy, sigma_t]]
        
        # Set channel weights if multi-channel
        if self.multi_channel_check.isChecked():
            options_dict["weight"] = [self.ch1_weight.value(), self.ch2_weight.value()]
            
        return options_dict
        
    @thread_worker
    def _run_motion_correction(self, video_array: np.ndarray, 
                              reference: np.ndarray, 
                              options_dict: dict,
                              progress_callback=None):
        """Run motion correction in a separate thread."""
        try:
            # Import pyflowreg inside the worker to avoid Numba import issues
            from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
            from pyflowreg.motion_correction.compensate_arr import compensate_arr
            
            # Map quality string to enum
            quality_map = {
                "quality": QualitySetting.QUALITY,
                "balanced": QualitySetting.BALANCED,
                "fast": QualitySetting.FAST,
                "custom": QualitySetting.CUSTOM
            }
            
            # Create OFOptions from dict
            quality_setting = options_dict["quality_setting"]
            if quality_setting not in quality_map:
                raise ValueError(f"Invalid quality setting: {quality_setting}")
            options_dict["quality_setting"] = quality_map[quality_setting]
            options = OFOptions(**options_dict)
            
            # Run compensation with progress callback if provided
            registered, flow = compensate_arr(video_array, reference, options, 
                                            progress_callback=progress_callback)
            return registered, flow
        except Exception as e:
            raise e
            
    def _on_start_clicked(self):
        """Handle start button click."""
        if not PYFLOWREG_AVAILABLE:
            show_error("PyFlowReg is not installed")
            return
            
        try:
            # Get input data
            layer_name = self.input_combo.currentText()
            if not layer_name:
                show_error("No input layer selected")
                return
                
            layer = self.viewer.layers[layer_name]
            video_array = layer.data
            
            # Get reference
            self.log("Getting reference frames...")
            reference = self._get_reference_frames()
            
            # Create options dictionary
            self.log("Creating processing options...")
            options_dict = self._create_options_dict()
            self.current_options = options_dict
            
            # Update UI
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Start processing
            self.log(f"Starting motion correction on {layer_name}...")
            self.log(f"Video shape: {video_array.shape}")
            self.log(f"Reference shape: {reference.shape}")

            def update_progress(current_frame, total_frames):
                if not total_frames:
                    self.progress_val.emit(0)  # Emit 0 when no frames
                    return
                self.progress_val.emit(int((current_frame * 100) // total_frames))
            
            worker = self._run_motion_correction(video_array, reference, options_dict, 
                                                progress_callback=update_progress)
            worker.returned.connect(self._on_correction_complete)
            worker.errored.connect(self._on_correction_error)
            worker.start()
            
        except Exception as e:
            show_error(str(e))
            self.log(f"Error: {e}")
            self._reset_ui()
            
    def _on_correction_complete(self, result):
        """Handle successful motion correction."""
        registered, flow = result
        
        self.log("Motion correction complete!")
        self.log(f"Corrected data shape: {registered.shape}")
        
        # Add corrected data to viewer
        # Use rgb=False to ensure single-channel data is not misinterpreted as multichannel
        layer_name = self.input_combo.currentText()
        self.viewer.add_image(
            registered,
            name=f"{layer_name}_corrected",
            rgb=False,  # Explicitly specify this is grayscale, not RGB
            colormap="gray"
        )
        
        # Optionally add flow field visualization
        if flow is not None and self.export_flow_check.isChecked():
            # Create flow magnitude for visualization
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            # Export flow magnitude as grayscale image
            self.viewer.add_image(
                flow_magnitude,
                name=f"{layer_name}_flow_magnitude",
                rgb=False,  # Ensure it's treated as grayscale
                colormap="gray",  # Use grayscale colormap as requested
                visible=True  # Make visible so user can see it
            )
            
            # TODO: Add proper flow color coding visualization using HSV or quiver plots
            # This would show both direction and magnitude of the optical flow
            
            self.log(f"Max displacement: {np.max(flow_magnitude):.2f} pixels")
            self.log(f"Mean displacement: {np.mean(flow_magnitude):.2f} pixels")
        else:
            self.log("Flow field export disabled")
            
        show_info("Motion correction completed successfully!")
        self._reset_ui()
        
    def _on_correction_error(self, error):
        """Handle motion correction error."""
        import traceback
        error_msg = str(error.args[0]) if error.args else str(error)
        tb = traceback.format_exc()
        
        show_error(f"Motion correction failed: {error_msg}")
        self.log(f"Error: {error_msg}")
        self.log(f"Traceback: {tb}")
        self._reset_ui()
        
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        # Note: Actual cancellation would require worker thread management
        self.log("Cancellation requested...")
        self._reset_ui()
        
    def _reset_ui(self):
        """Reset UI after processing."""
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
    def _on_save_settings(self):
        """Save current settings to file."""
        # TODO: Implement settings save
        show_info("Settings save not yet implemented")
        
    def _on_load_settings(self):
        """Load settings from file."""
        # TODO: Implement settings load
        show_info("Settings load not yet implemented")
        
    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        
    def showEvent(self, event):
        """Update layer lists when widget is shown."""
        super().showEvent(event)
        self._update_layer_lists()