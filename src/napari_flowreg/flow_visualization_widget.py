"""
Flow Visualization Widget for napari-flowreg
Provides visualization tools for optical flow fields
"""

from typing import Optional, List
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox
)
from qtpy.QtCore import Qt, Signal
from napari.viewer import Viewer
from napari.layers import Image
from napari.utils.notifications import show_info, show_error
import warnings


class FlowVisualizationWidget(QWidget):
    """Widget for visualizing optical flow fields in napari."""
    
    # Signal emitted when flow layer changes
    flow_layer_changed = Signal(str)
    
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_flow_layer = None
        self.current_secondary_layer = None
        self._init_ui()
        
        # Connect to layer events
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_layers_changed)
        
        # Initial update of layer lists
        self._update_layer_lists()
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Flow field selection
        flow_selection_group = self._create_flow_selection_group()
        layout.addWidget(flow_selection_group)
        
        # Visualization options
        viz_options_group = self._create_visualization_options_group()
        layout.addWidget(viz_options_group)
        
        # Visualization controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def _create_flow_selection_group(self) -> QGroupBox:
        """Create flow field selection section."""
        group = QGroupBox("Flow Field Selection")
        layout = QGridLayout()
        
        # Primary flow field dropdown
        layout.addWidget(QLabel("Flow Field:"), 0, 0)
        self.flow_combo = QComboBox()
        self.flow_combo.setToolTip("Select the image layer containing optical flow data")
        self.flow_combo.currentTextChanged.connect(self._on_flow_layer_changed)
        layout.addWidget(self.flow_combo, 0, 1)
        
        # Secondary image dropdown (for future overlay visualization)
        layout.addWidget(QLabel("Secondary Image:"), 1, 0)
        self.secondary_combo = QComboBox()
        self.secondary_combo.setEnabled(False)  # Grayed out for now
        self.secondary_combo.setToolTip("Secondary image for overlay visualization (future feature)")
        layout.addWidget(self.secondary_combo, 1, 1)
        
        # Info label
        self.flow_info_label = QLabel("No flow field selected")
        self.flow_info_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self.flow_info_label, 2, 0, 1, 2)
        
        group.setLayout(layout)
        return group
        
    def _create_visualization_options_group(self) -> QGroupBox:
        """Create visualization options section."""
        group = QGroupBox("Visualization Options")
        layout = QGridLayout()
        
        # Visualization type dropdown
        layout.addWidget(QLabel("Visualization Type:"), 0, 0)
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Flow Magnitude",
            "Flow Direction (HSV)",
            # Future options to be added:
            # "Quiver Plot",
            # "Streamlines",
            # "Flow Divergence",
            # "Flow Curl"
        ])
        self.viz_type_combo.currentTextChanged.connect(self._on_viz_type_changed)
        layout.addWidget(self.viz_type_combo, 0, 1)
        
        # Magnitude-specific options
        self.mag_options_widget = QWidget()
        mag_layout = QGridLayout()
        
        # Colormap selection
        mag_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "gray", "hot", "jet", "turbo", 
            "plasma", "inferno", "magma", "twilight"
        ])
        self.colormap_combo.setCurrentText("viridis")
        mag_layout.addWidget(self.colormap_combo, 0, 1)
        
        # Auto-scale checkbox
        self.autoscale_check = QCheckBox("Auto-scale")
        self.autoscale_check.setChecked(True)
        self.autoscale_check.toggled.connect(self._on_autoscale_changed)
        mag_layout.addWidget(self.autoscale_check, 1, 0)
        
        # Manual scale controls
        mag_layout.addWidget(QLabel("Min:"), 2, 0)
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(0, 1000)
        self.scale_min_spin.setSingleStep(0.1)
        self.scale_min_spin.setValue(0)
        self.scale_min_spin.setEnabled(False)
        mag_layout.addWidget(self.scale_min_spin, 2, 1)
        
        mag_layout.addWidget(QLabel("Max:"), 3, 0)
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(0, 1000)
        self.scale_max_spin.setSingleStep(0.1)
        self.scale_max_spin.setValue(10)
        self.scale_max_spin.setEnabled(False)
        mag_layout.addWidget(self.scale_max_spin, 3, 1)
        
        self.mag_options_widget.setLayout(mag_layout)
        layout.addWidget(self.mag_options_widget, 1, 0, 1, 2)
        
        group.setLayout(layout)
        return group
        
    def _create_controls_group(self) -> QGroupBox:
        """Create control buttons section."""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self._on_visualize_clicked)
        button_layout.addWidget(self.visualize_button)
        
        self.clear_button = QPushButton("Clear Visualization")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # Statistics display
        self.stats_label = QLabel("Statistics will appear here")
        self.stats_label.setStyleSheet("QLabel { font-family: monospace; }")
        layout.addWidget(self.stats_label)
        
        group.setLayout(layout)
        return group
        
    def _update_layer_lists(self):
        """Update the layer dropdowns with current image layers."""
        # Save current selections
        current_flow = self.flow_combo.currentText()
        current_secondary = self.secondary_combo.currentText()
        
        # Clear combos
        self.flow_combo.clear()
        self.secondary_combo.clear()
        
        # Get all image layers
        image_layers = [layer.name for layer in self.viewer.layers 
                       if isinstance(layer, Image)]
        
        # Add empty option first
        self.flow_combo.addItem("")
        self.secondary_combo.addItem("")
        
        # Add layers to both combos
        self.flow_combo.addItems(image_layers)
        self.secondary_combo.addItems(image_layers)
        
        # Try to auto-select flow field
        flow_layer_found = False
        for layer_name in image_layers:
            if "_flow" in layer_name.lower() or layer_name.lower() == "w":
                self.flow_combo.setCurrentText(layer_name)
                flow_layer_found = True
                break
        
        # Restore previous selection if no auto-select
        if not flow_layer_found and current_flow in image_layers:
            self.flow_combo.setCurrentText(current_flow)
            
        # Restore secondary selection
        if current_secondary in image_layers:
            self.secondary_combo.setCurrentText(current_secondary)
            
    def _on_layers_changed(self, event=None):
        """Handle layer list changes."""
        self._update_layer_lists()
        
    def _on_flow_layer_changed(self, layer_name: str):
        """Handle flow layer selection change."""
        if not layer_name:
            self.flow_info_label.setText("No flow field selected")
            self.flow_info_label.setStyleSheet("QLabel { color: gray; }")
            self.current_flow_layer = None
            return
            
        try:
            layer = self.viewer.layers[layer_name]
            self.current_flow_layer = layer
            
            # Update info label
            shape = layer.data.shape
            dtype = layer.data.dtype
            
            # Check if it looks like flow data
            if len(shape) >= 3 and shape[-1] == 2:
                info_text = f"Flow field: {shape[:-1]} × 2 components, dtype: {dtype}"
                self.flow_info_label.setStyleSheet("QLabel { color: green; }")
            else:
                info_text = f"Warning: Expected (... × 2) flow field, got shape {shape}"
                self.flow_info_label.setStyleSheet("QLabel { color: orange; }")
                
            self.flow_info_label.setText(info_text)
            
            # Emit signal
            self.flow_layer_changed.emit(layer_name)
            
        except KeyError:
            self.flow_info_label.setText("Layer not found")
            self.flow_info_label.setStyleSheet("QLabel { color: red; }")
            self.current_flow_layer = None
            
    def _on_viz_type_changed(self, viz_type: str):
        """Handle visualization type change."""
        # Show/hide options based on visualization type
        if viz_type == "Flow Magnitude":
            self.mag_options_widget.setVisible(True)
        elif viz_type == "Flow Direction (HSV)":
            self.mag_options_widget.setVisible(False)
        else:
            self.mag_options_widget.setVisible(False)
            
    def _on_autoscale_changed(self, checked: bool):
        """Handle auto-scale checkbox change."""
        self.scale_min_spin.setEnabled(not checked)
        self.scale_max_spin.setEnabled(not checked)
        
    def _on_visualize_clicked(self):
        """Generate and display the visualization."""
        if self.current_flow_layer is None:
            show_error("Please select a flow field layer first")
            return
            
        viz_type = self.viz_type_combo.currentText()
        
        if viz_type == "Flow Magnitude":
            self._visualize_flow_magnitude()
        elif viz_type == "Flow Direction (HSV)":
            self._visualize_flow_hsv()
        else:
            show_info(f"Visualization type '{viz_type}' not yet implemented")
            
    def _visualize_flow_magnitude(self):
        """Create and display flow magnitude visualization."""
        if self.current_flow_layer is None:
            return
            
        flow_data = self.current_flow_layer.data
        
        # Check data shape
        if flow_data.ndim < 2:
            show_error("Flow data must be at least 2D")
            return
            
        # Handle different data shapes
        if flow_data.shape[-1] == 2:
            # Assume last dimension is [u, v] components
            u = flow_data[..., 0]
            v = flow_data[..., 1]
        elif flow_data.ndim == 3 and flow_data.shape[0] == 2:
            # Assume first dimension is [u, v] components
            u = flow_data[0]
            v = flow_data[1]
        else:
            # Try to interpret as single component or warn
            show_error(f"Cannot interpret flow data with shape {flow_data.shape}. "
                      f"Expected (..., 2) for [u,v] components")
            return
            
        # Compute magnitude
        magnitude = np.sqrt(u**2 + v**2)
        
        # Compute statistics
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        min_mag = np.min(magnitude)
        std_mag = np.std(magnitude)
        
        # Update statistics display
        stats_text = (
            f"Flow Magnitude Statistics:\n"
            f"  Mean: {mean_mag:.3f} pixels\n"
            f"  Max:  {max_mag:.3f} pixels\n"
            f"  Min:  {min_mag:.3f} pixels\n"
            f"  Std:  {std_mag:.3f} pixels"
        )
        self.stats_label.setText(stats_text)
        
        # Determine contrast limits
        if self.autoscale_check.isChecked():
            contrast_limits = [min_mag, max_mag] if max_mag > min_mag else None
        else:
            contrast_limits = [self.scale_min_spin.value(), self.scale_max_spin.value()]
            
        # Add magnitude layer to viewer
        layer_name = f"{self.current_flow_layer.name}_magnitude"
        
        # Check if layer already exists
        if layer_name in self.viewer.layers:
            # Update existing layer
            self.viewer.layers[layer_name].data = magnitude
            if contrast_limits:
                self.viewer.layers[layer_name].contrast_limits = contrast_limits
            self.viewer.layers[layer_name].colormap = self.colormap_combo.currentText()
        else:
            # Create new layer
            self.viewer.add_image(
                magnitude,
                name=layer_name,
                colormap=self.colormap_combo.currentText(),
                contrast_limits=contrast_limits,
                visible=True
            )
            
        show_info(f"Flow magnitude visualization created: {layer_name}")
        
    def _visualize_flow_hsv(self):
        """Create and display flow direction visualization using HSV color coding."""
        if self.current_flow_layer is None:
            return
            
        flow_data = self.current_flow_layer.data
        
        # Check data shape
        if flow_data.ndim < 2:
            show_error("Flow data must be at least 2D")
            return
            
        # Handle different data shapes
        if flow_data.shape[-1] == 2:
            # Assume last dimension is [u, v] components
            u = flow_data[..., 0]
            v = flow_data[..., 1]
        elif flow_data.ndim == 3 and flow_data.shape[0] == 2:
            # Assume first dimension is [u, v] components
            u = flow_data[0]
            v = flow_data[1]
        else:
            # Try to interpret as single component or warn
            show_error(f"Cannot interpret flow data with shape {flow_data.shape}. "
                      f"Expected (..., 2) for [u,v] components")
            return
            
        try:
            # Import flow_to_color from pyflowreg
            from pyflowreg.util.visualization import flow_to_color

            # Compute maximum flow magnitude across all frames for consistent visualization
            # This ensures all frames use the same color scale
            max_flow_mag = 0.0
            
            # Handle different shapes - check if we have multiple frames
            if flow_data.ndim == 4 and flow_data.shape[-1] == 2:
                # Shape is (T, H, W, 2) - multiple frames
                for frame_idx in range(flow_data.shape[0]):
                    frame_u = flow_data[frame_idx, :, :, 0]
                    frame_v = flow_data[frame_idx, :, :, 1]
                    frame_magnitude = np.sqrt(frame_u**2 + frame_v**2)
                    frame_max = np.max(frame_magnitude)
                    if frame_max > max_flow_mag:
                        max_flow_mag = frame_max
            else:
                # Single frame or different arrangement
                magnitude = np.sqrt(u**2 + v**2)
                max_flow_mag = np.max(magnitude)

            # Convert flow to HSV color representation with consistent max_flow
            # flow_to_color expects (H,W,2) or (T,H,W,2) with u,v in last dimension
            hsv_image = flow_to_color(flow_data, max_flow=max_flow_mag)

            # Compute statistics
            magnitude = np.sqrt(u**2 + v**2)
            mean_mag = np.mean(magnitude)
            max_mag = np.max(magnitude)
            min_mag = np.min(magnitude)
            std_mag = np.std(magnitude)

            # Update statistics display
            stats_text = (
                f"Flow Direction (HSV) Statistics:\n"
                f"  Mean magnitude: {mean_mag:.3f} pixels\n"
                f"  Max magnitude:  {max_mag:.3f} pixels\n"
                f"  Min magnitude:  {min_mag:.3f} pixels\n"
                f"  Std magnitude:  {std_mag:.3f} pixels\n"
                f"  Max flow (for consistent scaling): {max_flow_mag:.3f} pixels\n"
                f"  Hue: Direction, Saturation: Magnitude"
            )
            self.stats_label.setText(stats_text)

            # Add HSV layer to viewer
            layer_name = f"{self.current_flow_layer.name}_hsv"

            # Check if layer already exists
            if layer_name in self.viewer.layers:
                # Update existing layer
                self.viewer.layers[layer_name].data = hsv_image
            else:
                # Create new layer - HSV image is RGB
                self.viewer.add_image(
                    hsv_image,
                    name=layer_name,
                    rgb=True,  # HSV is converted to RGB by flow_to_color
                    visible=True
                )

            show_info(f"Flow direction (HSV) visualization created: {layer_name}")
            
        except ImportError as e:
            show_error(f"Could not import flow_to_color from pyflowreg. Please ensure pyflowreg is installed: {str(e)}")
        except Exception as e:
            show_error(f"Error creating HSV visualization: {str(e)}")
    
    def _on_clear_clicked(self):
        """Clear visualization layers."""
        # Remove any visualization layers we created
        layers_to_remove = []
        for layer in self.viewer.layers:
            if "_magnitude" in layer.name or "_hsv" in layer.name:
                layers_to_remove.append(layer.name)
                
        for layer_name in layers_to_remove:
            self.viewer.layers.remove(layer_name)
            
        if layers_to_remove:
            show_info(f"Removed {len(layers_to_remove)} visualization layer(s)")
        else:
            show_info("No visualization layers to remove")
            
        # Clear statistics
        self.stats_label.setText("Statistics will appear here")