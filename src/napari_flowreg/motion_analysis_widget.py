"""
Motion Analysis Widget for napari-flowreg
Provides analysis tools for optical flow fields
"""

from typing import Optional, List, Tuple, Union
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QGridLayout,
    QCheckBox, QFileDialog, QMessageBox
)
from qtpy.QtCore import Qt, Signal
from napari.viewer import Viewer
from napari.layers import Image, Shapes
from napari.utils.notifications import show_info, show_error
from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.draw import polygon2mask


class MotionAnalysisWidget(QWidget):
    """Widget for analyzing motion patterns from optical flow fields."""
    
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_flow_layer = None
        self.current_roi_layer = None
        self.figure = None
        self.canvas = None
        self.motion_data = None  # Store computed motion data for export
        self._init_ui()
        
        # Connect to layer events
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        
        # Initial update of layer lists
        self._update_layer_lists()
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Data selection
        data_selection_group = self._create_data_selection_group()
        layout.addWidget(data_selection_group)
        
        # Analysis options
        analysis_options_group = self._create_analysis_options_group()
        layout.addWidget(analysis_options_group)
        
        # Controls
        controls_group = self._create_controls_group()
        layout.addWidget(controls_group)
        
        # Plot area
        plot_group = self._create_plot_group()
        layout.addWidget(plot_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def _create_data_selection_group(self) -> QGroupBox:
        """Create data selection section."""
        group = QGroupBox("Data Selection")
        layout = QGridLayout()
        
        # Flow field dropdown (w image)
        layout.addWidget(QLabel("Flow Field (w):"), 0, 0)
        self.flow_combo = QComboBox()
        self.flow_combo.setToolTip("Select the flow field data (displacement field)")
        self.flow_combo.currentTextChanged.connect(self._on_flow_layer_changed)
        layout.addWidget(self.flow_combo, 0, 1)
        
        # ROI selection dropdown (optional)
        layout.addWidget(QLabel("ROI (optional):"), 1, 0)
        self.roi_combo = QComboBox()
        self.roi_combo.setToolTip("Select a Shapes layer to define region of interest")
        self.roi_combo.currentTextChanged.connect(self._on_roi_layer_changed)
        layout.addWidget(self.roi_combo, 1, 1)
        
        # Info label
        self.data_info_label = QLabel("No flow field selected")
        self.data_info_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self.data_info_label, 2, 0, 1, 2)
        
        group.setLayout(layout)
        return group
        
    def _create_analysis_options_group(self) -> QGroupBox:
        """Create analysis options section."""
        group = QGroupBox("Analysis Options")
        layout = QGridLayout()
        
        # Analysis mode
        layout.addWidget(QLabel("Statistic Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["mean", "max", "min"])
        self.mode_combo.setCurrentText("mean")
        self.mode_combo.setToolTip("Statistical measure to compute over ROI or entire image")
        layout.addWidget(self.mode_combo, 0, 1)
        
        # Plot type
        layout.addWidget(QLabel("Plot Type:"), 1, 0)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Motion Magnitude"])
        self.plot_type_combo.setCurrentText("Motion Magnitude")
        self.plot_type_combo.setToolTip("Type of motion analysis to perform")
        layout.addWidget(self.plot_type_combo, 1, 1)
        
        group.setLayout(layout)
        return group
        
    def _create_controls_group(self) -> QGroupBox:
        """Create control buttons section."""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Plot Motion Magnitude")
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        button_layout.addWidget(self.analyze_button)
        
        self.clear_button = QPushButton("Clear Plot")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)
        
        self.export_button = QPushButton("Save as MAT")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)  # Disabled until data is computed
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Statistics display
        self.stats_label = QLabel("Statistics will appear here")
        self.stats_label.setStyleSheet("QLabel { font-family: monospace; }")
        layout.addWidget(self.stats_label)
        
        group.setLayout(layout)
        return group
        
    def _create_plot_group(self) -> QGroupBox:
        """Create plot area."""
        group = QGroupBox("Motion Analysis Plot")
        layout = QVBoxLayout()
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        group.setLayout(layout)
        return group
        
    def _update_layer_lists(self):
        """Update the layer dropdowns with current layers."""
        # Save current selections
        current_flow = self.flow_combo.currentText()
        current_roi = self.roi_combo.currentText()
        
        # Update flow combo
        self.flow_combo.clear()
        self.flow_combo.addItem("")  # Empty option
        
        image_layers = [layer.name for layer in self.viewer.layers 
                       if isinstance(layer, Image)]
        self.flow_combo.addItems(image_layers)
        
        # Try to auto-select flow field
        flow_found = False
        for layer_name in image_layers:
            if "_flow" in layer_name.lower() or layer_name.lower() == "w":
                self.flow_combo.setCurrentText(layer_name)
                flow_found = True
                break
        
        if not flow_found and current_flow in image_layers:
            self.flow_combo.setCurrentText(current_flow)
            
        # Update ROI combo
        self.roi_combo.clear()
        self.roi_combo.addItem("")  # Empty option for no ROI
        
        shapes_layers = [layer.name for layer in self.viewer.layers 
                        if isinstance(layer, Shapes)]
        self.roi_combo.addItems(shapes_layers)
        
        if current_roi in shapes_layers:
            self.roi_combo.setCurrentText(current_roi)
            
    def _on_layers_changed(self, event=None):
        """Handle layer list changes."""
        self._update_layer_lists()
        
    def _on_flow_layer_changed(self, layer_name: str):
        """Handle flow layer selection change."""
        if not layer_name:
            self.data_info_label.setText("No flow field selected")
            self.data_info_label.setStyleSheet("QLabel { color: gray; }")
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
                if len(shape) == 4:  # Video flow (T, H, W, 2)
                    info_text = f"Flow field: {shape[0]} frames, {shape[1]}×{shape[2]} × 2 components"
                else:  # Single frame (H, W, 2)
                    info_text = f"Flow field: {shape[:-1]} × 2 components"
                self.data_info_label.setStyleSheet("QLabel { color: green; }")
            else:
                info_text = f"Warning: Expected (..., 2) flow field, got shape {shape}"
                self.data_info_label.setStyleSheet("QLabel { color: orange; }")
                
            self.data_info_label.setText(info_text)
            
        except KeyError:
            self.data_info_label.setText("Layer not found")
            self.data_info_label.setStyleSheet("QLabel { color: red; }")
            self.current_flow_layer = None
            
    def _on_roi_layer_changed(self, layer_name: str):
        """Handle ROI layer selection change."""
        if not layer_name:
            self.current_roi_layer = None
            return
            
        try:
            layer = self.viewer.layers[layer_name]
            if isinstance(layer, Shapes):
                self.current_roi_layer = layer
            else:
                self.current_roi_layer = None
                show_error(f"Layer {layer_name} is not a Shapes layer")
        except KeyError:
            self.current_roi_layer = None
            
    def _get_roi_mask(self, hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get ROI mask from shapes layer with proper coordinate transformation."""
        if self.current_roi_layer is None or len(self.current_roi_layer.data) == 0:
            return None
        H, W = hw
        shp = self.current_roi_layer
        idxs = list(shp.selected_data) if len(shp.selected_data) > 0 else list(range(len(shp.data)))
        types = getattr(shp, "shape_types", None)
        if types is not None:
            # Only support rectangles and polygons (ellipse control points aren't suitable for polygon fill)
            idxs = [i for i in idxs if types[i] in {"rectangle", "polygon"}]
        mask = np.zeros((H, W), dtype=bool)
        for i in idxs:
            # Get raw shape points - these are in nD coordinates where the last 2 dimensions are spatial (Y, X)
            raw_pts = shp.data[i]
            
            # Extract the spatial coordinates from the last two dimensions
            # For 4D shapes (e.g., from 4D flow layer), this gets columns -2 and -1
            # which contain the actual Y and X coordinates respectively
            ys = raw_pts[:, -2]  # Second-to-last column is Y
            xs = raw_pts[:, -1]  # Last column is X
            
            # Clip to image bounds
            ys = np.clip(ys, 0, H - 1)
            xs = np.clip(xs, 0, W - 1)
            
            # Create polygon for mask
            poly = np.stack([ys, xs], axis=1)
            mask |= polygon2mask((H, W), poly)
            
        return mask if mask.any() else None
        
    def _on_analyze_clicked(self):
        """Perform motion analysis and create plot."""
        if self.current_flow_layer is None:
            show_error("Please select a flow field layer first")
            return
        
        # Ensure viewer is in 2D mode for proper ROI analysis
        if self.viewer.dims.ndisplay != 2:
            show_error("Set viewer to 2D (two displayed axes) before analysis.")
            return
            
        flow = self.current_flow_layer.data
        if flow.ndim == 4 and flow.shape[-1] == 2:
            u, v = flow[..., 0], flow[..., 1]
        elif flow.ndim == 3 and flow.shape[-1] == 2:
            u, v = flow[..., 0], flow[..., 1]
        else:
            show_error(f"Expected (..., 2) flow; got {flow.shape}")
            return
        
        mag = np.hypot(u, v)
        roi_mask = None
        if self.current_roi_layer is not None:
            if mag.ndim == 3:
                roi_mask = self._get_roi_mask(mag.shape[1:3])
            else:
                roi_mask = self._get_roi_mask(mag.shape[0:2])
            
            # Print ROI dimensions in 2D image space
            if roi_mask is not None:
                roi_pixels = np.sum(roi_mask)
                roi_bounds = np.where(roi_mask)
                if len(roi_bounds[0]) > 0:
                    y_min, y_max = roi_bounds[0].min(), roi_bounds[0].max()
                    x_min, x_max = roi_bounds[1].min(), roi_bounds[1].max()
                    print(f"ROI dimensions in 2D image space:")
                    print(f"  - Bounding box: [{y_min}:{y_max+1}, {x_min}:{x_max+1}]")
                    print(f"  - Size: {y_max-y_min+1} x {x_max-x_min+1} pixels")
                    print(f"  - Total pixels: {roi_pixels}")
                    print(f"  - Coverage: {100*roi_pixels/roi_mask.size:.1f}% of image")
                
        # Compute statistics based on mode
        mode = self.mode_combo.currentText()
        
        if mag.ndim == 3:  # Video data (T, H, W)
            n_frames = mag.shape[0]
            motion_values = np.zeros(n_frames)
            
            for t in range(n_frames):
                frame_mag = mag[t]
                
                if roi_mask is not None:
                    frame_mag = frame_mag[roi_mask]
                    
                if mode == "mean":
                    motion_values[t] = np.mean(frame_mag)
                elif mode == "max":
                    motion_values[t] = np.max(frame_mag)
                elif mode == "min":
                    motion_values[t] = np.min(frame_mag)
                    
            # Store for export
            self.motion_data = {
                'motion_magnitude': motion_values,
                'mode': mode,
                'n_frames': n_frames,
                'has_roi': roi_mask is not None
            }
            
            # Create plot
            self._plot_motion_magnitude(motion_values, mode, roi_mask is not None)
            
            # Update statistics
            stats_text = (
                f"Motion Magnitude Analysis ({mode}):\n"
                f"  Frames: {n_frames}\n"
                f"  Overall {mode}: {np.mean(motion_values):.3f} pixels\n"
                f"  Max {mode}: {np.max(motion_values):.3f} pixels\n"
                f"  Min {mode}: {np.min(motion_values):.3f} pixels\n"
                f"  ROI: {'Yes' if roi_mask is not None else 'No'}"
            )
            
        else:  # Single frame
            if roi_mask is not None:
                mag_roi = mag[roi_mask]
            else:
                mag_roi = mag.flatten()
                
            if mode == "mean":
                value = np.mean(mag_roi)
            elif mode == "max":
                value = np.max(mag_roi)
            elif mode == "min":
                value = np.min(mag_roi)
                
            # Store for export
            self.motion_data = {
                'motion_magnitude': value,
                'mode': mode,
                'n_frames': 1,
                'has_roi': roi_mask is not None
            }
            
            # Update statistics (no plot for single frame)
            stats_text = (
                f"Motion Magnitude Analysis ({mode}):\n"
                f"  Single frame\n"
                f"  {mode.capitalize()} magnitude: {value:.3f} pixels\n"
                f"  ROI: {'Yes' if roi_mask is not None else 'No'}"
            )
            show_info(f"Single frame {mode} magnitude: {value:.3f} pixels")
            
        self.stats_label.setText(stats_text)
        
        # Enable export button
        self.export_button.setEnabled(True)
        
    def _plot_motion_magnitude(self, motion_values: np.ndarray, mode: str, has_roi: bool):
        """Create motion magnitude plot."""
        # Clear previous plot
        self.figure.clear()
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Plot motion magnitude over time
        frames = np.arange(len(motion_values))
        ax.plot(frames, motion_values, 'b-', linewidth=2)
        ax.scatter(frames[::10], motion_values[::10], c='red', s=20, zorder=5)  # Mark every 10th point
        
        # Customize plot
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel(f'{mode.capitalize()} Motion Magnitude (pixels)', fontsize=12)
        roi_text = " (ROI)" if has_roi else " (Full Image)"
        ax.set_title(f'Motion Magnitude Analysis - {mode.capitalize()}{roi_text}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for mean
        mean_val = np.mean(motion_values)
        ax.axhline(y=mean_val, color='g', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
        
        # Add legend
        ax.legend(loc='best')
        
        # Tight layout
        self.figure.tight_layout()
        
        # Refresh canvas
        self.canvas.draw()
        
    def _on_clear_clicked(self):
        """Clear the current plot."""
        self.figure.clear()
        self.canvas.draw()
        self.stats_label.setText("Statistics will appear here")
        self.motion_data = None
        self.export_button.setEnabled(False)
        
    def _on_export_clicked(self):
        """Export motion analysis data to MAT file."""
        if self.motion_data is None:
            show_error("No data to export. Please run analysis first.")
            return
            
        # Get save path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Motion Analysis Data",
            "",
            "MAT files (*.mat)"
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            # Import scipy for MAT file saving
            from scipy.io import savemat
            
            # Prepare data for export
            export_data = {
                'motion_magnitude': self.motion_data['motion_magnitude'],
                'statistic_mode': self.motion_data['mode'],
                'n_frames': self.motion_data['n_frames'],
                'has_roi': self.motion_data['has_roi'],
                'flow_layer_name': self.current_flow_layer.name if self.current_flow_layer else 'unknown'
            }
            
            # Add ROI information if available
            if self.current_roi_layer is not None:
                export_data['roi_layer_name'] = self.current_roi_layer.name
                
            # Save to MAT file
            savemat(file_path, export_data)
            
            show_info(f"Motion analysis data saved to {file_path}")
            
        except ImportError:
            show_error("scipy is required for MAT file export. Please install it with: pip install scipy")
        except Exception as e:
            show_error(f"Failed to save MAT file: {str(e)}")