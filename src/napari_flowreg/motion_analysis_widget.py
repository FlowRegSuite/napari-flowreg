"""
Motion Analysis Widget for napari-flowreg
Provides analysis tools for optical flow fields
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QGridLayout,
    QCheckBox, QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QTreeWidget, QTreeWidgetItem, QScrollArea, QSpinBox, QLineEdit
)
from qtpy.QtCore import Qt, Signal
from napari.viewer import Viewer
from napari.layers import Image, Shapes, Points, Tracks
from napari.utils.notifications import show_info, show_error
from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.draw import polygon2mask
from scipy.interpolate import RegularGridInterpolator


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
        self.tracked_points = {}  # Store tracked point trajectories
        self.roi_shapes_data = {}  # Store individual ROI data
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
        
        self.export_button = QPushButton("Export All to MAT")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)  # Disabled until data is computed
        button_layout.addWidget(self.export_button)
        
        self.track_points_button = QPushButton("Track Points")
        self.track_points_button.clicked.connect(self._on_track_points_clicked)
        button_layout.addWidget(self.track_points_button)
        
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
            
    def _get_individual_roi_masks(self, hw: Tuple[int, int]) -> Dict[int, Dict[str, Any]]:
        """Get individual ROI masks from shapes layer, one for each shape."""
        if self.current_roi_layer is None or len(self.current_roi_layer.data) == 0:
            return {}
        
        H, W = hw
        shp = self.current_roi_layer
        idxs = list(shp.selected_data) if len(shp.selected_data) > 0 else list(range(len(shp.data)))
        types = getattr(shp, "shape_types", None)
        
        roi_dict = {}
        for i in idxs:
            # Skip non-polygon shapes (e.g., points, ellipses)
            if types is not None and types[i] not in {"rectangle", "polygon"}:
                continue
                
            mask = np.zeros((H, W), dtype=bool)
            raw_pts = shp.data[i]
            
            # Extract spatial coordinates
            ys = raw_pts[:, -2]
            xs = raw_pts[:, -1]
            
            # Clip to bounds
            ys = np.clip(ys, 0, H - 1)
            xs = np.clip(xs, 0, W - 1)
            
            # Create polygon mask
            poly = np.stack([ys, xs], axis=1)
            mask = polygon2mask((H, W), poly)
            
            if mask.any():
                # Get shape properties for naming
                props = shp.properties if hasattr(shp, 'properties') else {}
                name = f"ROI_{i+1}"
                if 'name' in props and len(props['name']) > i:
                    custom_name = props['name'][i]
                    if custom_name:
                        name = custom_name
                
                roi_dict[i] = {
                    'mask': mask,
                    'name': name,
                    'vertices': poly.tolist(),
                    'shape_type': types[i] if types else 'polygon'
                }
        
        return roi_dict
    
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
        
        # Get individual ROI masks instead of combined mask
        roi_masks_dict = {}
        if self.current_roi_layer is not None:
            if mag.ndim == 3:
                roi_masks_dict = self._get_individual_roi_masks(mag.shape[1:3])
            else:
                roi_masks_dict = self._get_individual_roi_masks(mag.shape[0:2])
        
        # For backward compatibility, create combined mask
        roi_mask = None
        if roi_masks_dict:
            # Combine all masks for legacy code
            roi_mask = np.zeros_like(list(roi_masks_dict.values())[0]['mask'])
            for roi_data in roi_masks_dict.values():
                roi_mask |= roi_data['mask']
            
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
            
            # Compute motion for each ROI separately
            if roi_masks_dict:
                roi_motion_data = {}
                for roi_id, roi_info in roi_masks_dict.items():
                    roi_motion = np.zeros(n_frames)
                    for t in range(n_frames):
                        frame_mag = mag[t]
                        roi_values = frame_mag[roi_info['mask']]
                        
                        if mode == "mean":
                            roi_motion[t] = np.mean(roi_values)
                        elif mode == "max":
                            roi_motion[t] = np.max(roi_values)
                        elif mode == "min":
                            roi_motion[t] = np.min(roi_values)
                    
                    roi_motion_data[roi_id] = {
                        'name': roi_info['name'],
                        'motion_magnitude': roi_motion,
                        'vertices': roi_info['vertices'],
                        'shape_type': roi_info['shape_type']
                    }
                
                # Store individual ROI data
                self.roi_shapes_data = roi_motion_data
                
                # For plotting, use mean of all ROIs (or could plot all)
                all_roi_motions = [data['motion_magnitude'] for data in roi_motion_data.values()]
                motion_values = np.mean(all_roi_motions, axis=0)
            else:
                # No ROI, use full image
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
                'has_roi': roi_mask is not None,
                'individual_rois': self.roi_shapes_data if roi_masks_dict else {}
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
        """Open comprehensive export dialog."""
        dialog = ExportDialog(self.viewer, self)
        
        if dialog.exec_() == QDialog.Accepted:
            options = dialog.get_export_options()
            self._perform_export(options)
    
    def _perform_export(self, options):
        """Perform the actual export based on selected options."""
        try:
            from scipy.io import savemat
            import scipy.io as sio
            from datetime import datetime
            
            export_data = {}
            
            # Add metadata if requested
            if options['include_metadata']:
                export_data['metadata'] = {
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'napari_flowreg_version': '0.1.0',
                    'export_options': str(options)
                }
            
            # Process ROI layers
            for roi_layer in options['roi_layers']:
                roi_name = roi_layer.name.replace(' ', '_')
                
                # Re-extract motion statistics from current viewer state
                if self.current_flow_layer is not None:
                    flow = self.current_flow_layer.data
                    
                    # Extract motion for each shape in the layer
                    roi_data = {}
                    for idx, shape_data in enumerate(roi_layer.data):
                        shape_type = roi_layer.shape_types[idx] if hasattr(roi_layer, 'shape_types') else 'unknown'
                        
                        # Create mask for this shape
                        mask = self._create_shape_mask(shape_data, shape_type, flow.shape[1:3])
                        
                        if mask is not None and np.any(mask):
                            # Calculate motion statistics
                            motion_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                            masked_motion = motion_mag[:, mask]
                            
                            shape_stats = {
                                'shape_type': shape_type,
                                'vertices': np.array(shape_data)
                            }
                            
                            if options['include_time_series']:
                                shape_stats['time_series'] = masked_motion.mean(axis=1)
                            
                            if options['stats']['mean']:
                                shape_stats['mean'] = masked_motion.mean()
                            if options['stats']['max']:
                                shape_stats['max'] = masked_motion.max()
                            if options['stats']['min']:
                                shape_stats['min'] = masked_motion.min()
                            if options['stats']['std']:
                                shape_stats['std'] = masked_motion.std()
                            if options['stats']['median']:
                                shape_stats['median'] = np.median(masked_motion)
                            if options['stats']['percentiles']:
                                shape_stats['percentile_5'] = np.percentile(masked_motion, 5)
                                shape_stats['percentile_95'] = np.percentile(masked_motion, 95)
                            
                            if options['include_roi_masks']:
                                shape_stats['mask'] = mask
                            
                            roi_data[f'shape_{idx}'] = shape_stats
                    
                    export_data[f'roi_{roi_name}'] = roi_data
            
            # Process track layers
            for track_layer in options['track_layers']:
                track_name = track_layer.name.replace(' ', '_')
                export_data[f'tracks_{track_name}'] = {
                    'data': track_layer.data,
                    'properties': track_layer.properties if hasattr(track_layer, 'properties') else {}
                }
            
            # Process flow layers (with options)
            for flow_layer in options['flow_layers']:
                flow_name = flow_layer.name.replace(' ', '_')
                flow_data = flow_layer.data.copy()
                
                # Apply downsampling if requested
                if options['downsample']:
                    factor = options['downsample_factor']
                    flow_data = flow_data[:, ::factor, ::factor, :]
                
                # Apply frame subset if requested
                if options['subset_frames']:
                    start = options['frame_start']
                    end = options['frame_end'] if options['frame_end'] else flow_data.shape[0]
                    flow_data = flow_data[start:end]
                
                # Convert to magnitude only if requested
                if options['magnitude_only']:
                    flow_data = np.sqrt(flow_data[..., 0]**2 + flow_data[..., 1]**2)
                
                export_data[f'flow_{flow_name}'] = flow_data
            
            # Process image layers
            for img_layer in options['image_layers']:
                img_name = img_layer.name.replace(' ', '_')
                export_data[f'image_{img_name}'] = img_layer.data
            
            # Save based on file type
            output_path = options['output_path']
            
            if 'MAT' in options['file_type']:
                # Determine MAT file version
                if 'v7.3' in options['file_type']:
                    # Use HDF5 format for large files
                    savemat(output_path, export_data, 
                           do_compression=options['compress'],
                           format='7.3')
                else:
                    # Use legacy format
                    savemat(output_path, export_data, 
                           do_compression=options['compress'])
            elif 'NPZ' in options['file_type']:
                # Save as NumPy compressed archive
                if options['compress']:
                    np.savez_compressed(output_path, **export_data)
                else:
                    np.savez(output_path, **export_data)
            
            show_info(f"Data exported successfully to {output_path}")
            
        except ImportError:
            show_error("scipy is required for export. Install with: pip install scipy")
        except Exception as e:
            show_error(f"Export failed: {str(e)}")
    
    def _create_shape_mask(self, vertices, shape_type, shape):
        """Create a binary mask from shape vertices."""
        from skimage.draw import polygon2mask
        
        H, W = shape
        
        if shape_type in ('rectangle', 'polygon'):
            # Use polygon2mask for accurate rasterization
            # Ensure vertices are in Y,X order
            if vertices.ndim == 2 and vertices.shape[1] >= 2:
                yx_coords = vertices[:, -2:]  # Last 2 dims are Y,X
                mask = polygon2mask((H, W), yx_coords)
                return mask
        elif shape_type == 'ellipse':
            # For ellipse, vertices represent the bounding box
            # We need to create an ellipse mask
            from skimage.draw import ellipse
            if vertices.shape[0] >= 2:
                # Calculate ellipse parameters from bounding vertices
                y_coords = vertices[:, -2]
                x_coords = vertices[:, -1]
                cy = (y_coords.min() + y_coords.max()) / 2
                cx = (x_coords.min() + x_coords.max()) / 2
                ry = (y_coords.max() - y_coords.min()) / 2
                rx = (x_coords.max() - x_coords.min()) / 2
                
                # Create ellipse mask
                mask = np.zeros((H, W), dtype=bool)
                rr, cc = ellipse(cy, cx, ry, rx, shape=(H, W))
                mask[rr, cc] = True
                return mask
        
        return None
    
    def _on_track_points_clicked(self):
        """Open dialog for point tracking configuration."""
        if self.current_flow_layer is None:
            show_error("Please select a flow field layer first")
            return
            
        # Check if flow is temporal
        flow = self.current_flow_layer.data
        if flow.ndim != 4 or flow.shape[-1] != 2:
            show_error("Point tracking requires temporal flow field (shape: T, H, W, 2)")
            return
        
        # Open point tracking dialog
        dialog = PointTrackingDialog(self.viewer, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_layer = dialog.get_selected_layer()
            track_all = dialog.track_all_checkbox.isChecked()
            
            if track_all:
                self._track_points(None, track_all=True)
            elif selected_layer:
                self._track_points(selected_layer, track_all=False)
    
    def _track_points(self, points_layer_name: Optional[str], track_all: bool = False):
        """Track points through time using optical flow.
        
        Takes seed points (interpreted as positions at frame 0) and creates:
        1. Tracks layer: trajectories across all frames anchored at seed locations
        2. Points layer (optional): all tracked positions across all frames
        
        Uses fixed-reference tracking: position_t = seed + flow_t(seed)
        """
        try:
            if self.current_flow_layer is None:
                show_error("Select a flow field layer first")
                return
            
            flow = self.current_flow_layer.data
            if flow.ndim != 4 or flow.shape[-1] != 2:
                show_error("Flow must be (T,H,W,2)")
                return
            
            T, H, W, _ = flow.shape
            
            # Flow sign: +1 if flow is reference->frame, -1 if frame->reference
            SIGN = +1

            if track_all:
                src_layers = [ly for ly in self.viewer.layers if isinstance(ly, (Points, Shapes))]
            else:
                src_layers = [self.viewer.layers[points_layer_name]]

            any_out = False
            self.tracked_points = {}

            for layer in src_layers:
                if isinstance(layer, Points):
                    coords = np.asarray(layer.data, dtype=float)
                    if coords.ndim != 2 or coords.shape[1] < 2:
                        continue
                    
                    # Extract seed Y,X coordinates (last 2 dimensions)
                    yx0 = coords[:, -2:].copy()
                    
                    # Build Tracks across all T frames, anchored at seed coords
                    tracks = []
                    for pid in range(len(yx0)):
                        y0, x0 = yx0[pid]
                        for t in range(T):
                            # Sample flow at seed location for frame t
                            u = self._bilinear_interpolate(flow[t, :, :, 0], x0, y0)
                            v = self._bilinear_interpolate(flow[t, :, :, 1], x0, y0)
                            
                            # Compute displaced position
                            yt = np.clip(y0 + SIGN * v, 0, H - 1)
                            xt = np.clip(x0 + SIGN * u, 0, W - 1)
                            
                            # Track format: [track_id, time, y, x]
                            tracks.append([pid, float(t), yt, xt])
                    
                    tracks = np.asarray(tracks, dtype=float)
                    track_name = f"{layer.name}_tracked"
                    self.viewer.add_tracks(
                        tracks, 
                        name=track_name, 
                        colormap="turbo", 
                        tail_length=10, 
                        tail_width=2, 
                        head_length=0
                    )
                    self.tracked_points[track_name] = tracks
                    
                    # Optional: Also create Points layer spanning all frames
                    pts_all = []
                    for pid in range(len(yx0)):
                        y0, x0 = yx0[pid]
                        for t in range(T):
                            u = self._bilinear_interpolate(flow[t, :, :, 0], x0, y0)
                            v = self._bilinear_interpolate(flow[t, :, :, 1], x0, y0)
                            # Points format: [time, y, x]
                            pts_all.append([float(t), 
                                          np.clip(y0 + SIGN * v, 0, H - 1), 
                                          np.clip(x0 + SIGN * u, 0, W - 1)])
                    
                    pts_all = np.asarray(pts_all, dtype=float)
                    pts_name = f"{layer.name}_points_all_t"
                    self.viewer.add_points(
                        pts_all, 
                        name=pts_name,
                        size=5,
                        face_color='yellow'
                    )
                    
                    any_out = True

                elif isinstance(layer, Shapes) and hasattr(layer, "shape_types"):
                    # Find point shapes
                    idxs = [i for i, t in enumerate(layer.shape_types) if t in ("point", "points")]
                    if not idxs:
                        continue
                    
                    # Collect points from all point shapes
                    pts = []
                    for i in idxs:
                        d = np.asarray(layer.data[i], dtype=float)
                        if d.ndim == 1:
                            d = d[None, :]
                        pts.append(d)
                    
                    if not pts:
                        continue
                    
                    coords = np.vstack(pts)
                    
                    # Extract seed Y,X coordinates (last 2 dimensions)
                    yx0 = coords[:, -2:].copy()

                    # Build Tracks across all T frames
                    tracks = []
                    for pid in range(len(yx0)):
                        y0, x0 = yx0[pid]
                        for t in range(T):
                            u = self._bilinear_interpolate(flow[t, :, :, 0], x0, y0)
                            v = self._bilinear_interpolate(flow[t, :, :, 1], x0, y0)
                            yt = np.clip(y0 + SIGN * v, 0, H - 1)
                            xt = np.clip(x0 + SIGN * u, 0, W - 1)
                            tracks.append([pid, float(t), yt, xt])
                    
                    tracks = np.asarray(tracks, dtype=float)
                    track_name = f"{layer.name}_tracked"
                    self.viewer.add_tracks(
                        tracks, 
                        name=track_name, 
                        colormap="turbo", 
                        tail_length=10, 
                        tail_width=2, 
                        head_length=0
                    )
                    self.tracked_points[track_name] = tracks
                    
                    # Optional: Points layer
                    pts_all = []
                    for pid in range(len(yx0)):
                        y0, x0 = yx0[pid]
                        for t in range(T):
                            u = self._bilinear_interpolate(flow[t, :, :, 0], x0, y0)
                            v = self._bilinear_interpolate(flow[t, :, :, 1], x0, y0)
                            pts_all.append([float(t), 
                                          np.clip(y0 + SIGN * v, 0, H - 1), 
                                          np.clip(x0 + SIGN * u, 0, W - 1)])
                    
                    pts_all = np.asarray(pts_all, dtype=float)
                    pts_name = f"{layer.name}_points_all_t"
                    self.viewer.add_points(
                        pts_all, 
                        name=pts_name,
                        size=5,
                        face_color='green'
                    )
                    
                    any_out = True

            if any_out:
                self.export_button.setEnabled(True)
                show_info("Created fixed-reference tracks spanning all frames")
            else:
                show_error("No point data found")
                
        except Exception as e:
            show_error(f"Point tracking failed: {str(e)}")
    
    def _bilinear_interpolate(self, field: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation of 2D field at position (x, y)."""
        H, W = field.shape

        # Clip to bounds - allow exact boundary values
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        # Get integer parts
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)

        # Get fractional parts
        fx, fy = x - x0, y - y0

        # Bilinear interpolation
        return (field[y0, x0] * (1 - fx) * (1 - fy) +
                field[y0, x1] * fx * (1 - fy) +
                field[y1, x0] * (1 - fx) * fy +
                field[y1, x1] * fx * fy)


class PointTrackingDialog(QDialog):
    """Dialog for configuring point tracking."""
    
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Track Points Configuration")
        self.setModal(True)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Select a Points layer or Shapes layer with points to track.\n"
            "Points will be tracked through time using the optical flow field."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Layer selection
        layout.addWidget(QLabel("Select Layer:"))
        self.layer_combo = QComboBox()
        self._populate_layers()
        layout.addWidget(self.layer_combo)
        
        # Track all checkbox
        self.track_all_checkbox = QCheckBox("Track all point layers")
        self.track_all_checkbox.stateChanged.connect(self._on_track_all_changed)
        layout.addWidget(self.track_all_checkbox)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.resize(400, 200)
    
    def _populate_layers(self):
        """Populate combo box with point-containing layers."""
        self.layer_combo.clear()
        
        # Add Points layers
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                self.layer_combo.addItem(f"[Points] {layer.name}")
            elif isinstance(layer, Shapes):
                # Check if has point shapes
                if hasattr(layer, 'shape_types'):
                    if 'point' in layer.shape_types or 'points' in layer.shape_types:
                        # Count total points across all point shapes
                        n_points = 0
                        for i, t in enumerate(layer.shape_types):
                            if t in ('point', 'points'):
                                shape_data = layer.data[i]
                                if shape_data.ndim == 1:
                                    n_points += 1
                                else:
                                    n_points += len(shape_data)
                        self.layer_combo.addItem(f"[Shapes] {layer.name} ({n_points} points)")
    
    def _on_track_all_changed(self, state):
        """Handle track all checkbox change."""
        self.layer_combo.setEnabled(state != Qt.Checked)
    
    def get_selected_layer(self) -> Optional[str]:
        """Get the selected layer name."""
        if self.track_all_checkbox.isChecked():
            return None  # Will track all
        
        text = self.layer_combo.currentText()
        if text:
            # Remove prefix and point count
            if text.startswith("[Points] "):
                return text[9:]
            elif text.startswith("[Shapes] "):
                # Remove "[Shapes] " prefix and " (n points)" suffix if present
                name = text[9:]
                if " (" in name:
                    name = name.split(" (")[0]
                return name
        return None


class ExportDialog(QDialog):
    """Comprehensive dialog for exporting motion analysis data."""
    
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.parent_widget = parent
        self.setWindowTitle("Export Motion Analysis Data")
        self.setModal(True)
        self.resize(600, 700)
        
        # Data selection tracking
        self.selected_data = {}
        self.estimated_size = 0
        
        self._init_ui()
        self._scan_viewer_data()
    
    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()
        
        # Create scroll area for main content
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Data Sources Section
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout()
        
        # Flow Fields (unchecked by default)
        self.flow_tree = QTreeWidget()
        self.flow_tree.setHeaderLabel("Flow Field Layers ⚠️ (Large files!)")
        self.flow_tree.itemChanged.connect(self._update_size_estimate)
        data_layout.addWidget(self.flow_tree)
        
        # ROI Motion Statistics
        self.roi_tree = QTreeWidget()
        self.roi_tree.setHeaderLabel("ROI Motion Statistics")
        self.roi_tree.itemChanged.connect(self._update_size_estimate)
        data_layout.addWidget(self.roi_tree)
        
        # Tracked Points
        self.tracks_tree = QTreeWidget()
        self.tracks_tree.setHeaderLabel("Tracked Points")
        self.tracks_tree.itemChanged.connect(self._update_size_estimate)
        data_layout.addWidget(self.tracks_tree)
        
        # Image Layers
        self.image_tree = QTreeWidget()
        self.image_tree.setHeaderLabel("Image Layers")
        self.image_tree.itemChanged.connect(self._update_size_estimate)
        data_layout.addWidget(self.image_tree)
        
        data_group.setLayout(data_layout)
        scroll_layout.addWidget(data_group)
        
        # ROI Statistics Options
        stats_group = QGroupBox("ROI Statistics Options")
        stats_layout = QGridLayout()
        
        self.stat_mean = QCheckBox("Mean")
        self.stat_mean.setChecked(True)
        self.stat_max = QCheckBox("Max")
        self.stat_max.setChecked(True)
        self.stat_min = QCheckBox("Min")
        self.stat_min.setChecked(True)
        self.stat_std = QCheckBox("Std Dev")
        self.stat_std.setChecked(True)
        self.stat_median = QCheckBox("Median")
        self.stat_median.setChecked(True)
        self.stat_percentiles = QCheckBox("Percentiles (5,95)")
        
        self.roi_masks = QCheckBox("ROI Masks")
        self.roi_masks.setChecked(True)
        self.roi_vertices = QCheckBox("ROI Vertices")
        self.roi_vertices.setChecked(True)
        self.time_series = QCheckBox("Time Series")
        self.time_series.setChecked(True)
        
        stats_layout.addWidget(self.stat_mean, 0, 0)
        stats_layout.addWidget(self.stat_max, 0, 1)
        stats_layout.addWidget(self.stat_min, 0, 2)
        stats_layout.addWidget(self.stat_std, 1, 0)
        stats_layout.addWidget(self.stat_median, 1, 1)
        stats_layout.addWidget(self.stat_percentiles, 1, 2)
        stats_layout.addWidget(self.roi_masks, 2, 0)
        stats_layout.addWidget(self.roi_vertices, 2, 1)
        stats_layout.addWidget(self.time_series, 2, 2)
        
        stats_group.setLayout(stats_layout)
        scroll_layout.addWidget(stats_group)
        
        # Flow Field Options
        flow_options_group = QGroupBox("Flow Field Options (if selected above)")
        flow_options_layout = QVBoxLayout()
        
        # Warning label
        warning_label = QLabel("⚠️ Warning: Flow fields can create very large files!\n"
                              "   Consider downsampling or subsetting frames.")
        warning_label.setStyleSheet("QLabel { color: #ff6600; font-weight: bold; }")
        flow_options_layout.addWidget(warning_label)
        
        # Downsampling
        downsample_layout = QHBoxLayout()
        self.downsample_check = QCheckBox("Downsample by factor:")
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setMinimum(2)
        self.downsample_spin.setMaximum(16)
        self.downsample_spin.setValue(4)
        self.downsample_spin.setEnabled(False)
        self.downsample_check.toggled.connect(self.downsample_spin.setEnabled)
        self.downsample_check.toggled.connect(self._update_size_estimate)
        self.downsample_spin.valueChanged.connect(self._update_size_estimate)
        downsample_layout.addWidget(self.downsample_check)
        downsample_layout.addWidget(self.downsample_spin)
        downsample_layout.addStretch()
        flow_options_layout.addLayout(downsample_layout)
        
        # Magnitude only option
        self.magnitude_only = QCheckBox("Flow Magnitude only (reduces size by 50%)")
        self.magnitude_only.setChecked(True)
        self.magnitude_only.toggled.connect(self._update_size_estimate)
        flow_options_layout.addWidget(self.magnitude_only)
        
        # Frame subset
        subset_layout = QHBoxLayout()
        self.subset_check = QCheckBox("Subset frames: Start")
        self.subset_start = QSpinBox()
        self.subset_start.setMinimum(0)
        self.subset_start.setMaximum(9999)
        self.subset_start.setEnabled(False)
        subset_layout.addWidget(self.subset_check)
        subset_layout.addWidget(self.subset_start)
        subset_layout.addWidget(QLabel("End"))
        self.subset_end = QSpinBox()
        self.subset_end.setMinimum(1)
        self.subset_end.setMaximum(9999)
        self.subset_end.setValue(100)
        self.subset_end.setEnabled(False)
        subset_layout.addWidget(self.subset_end)
        subset_layout.addStretch()
        
        self.subset_check.toggled.connect(self.subset_start.setEnabled)
        self.subset_check.toggled.connect(self.subset_end.setEnabled)
        self.subset_check.toggled.connect(self._update_size_estimate)
        self.subset_start.valueChanged.connect(self._update_size_estimate)
        self.subset_end.valueChanged.connect(self._update_size_estimate)
        
        flow_options_layout.addLayout(subset_layout)
        flow_options_group.setLayout(flow_options_layout)
        scroll_layout.addWidget(flow_options_group)
        
        # Export Format Options
        format_group = QGroupBox("Export Format Options")
        format_layout = QVBoxLayout()
        
        # File type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("File Type:"))
        self.file_type = QComboBox()
        self.file_type.addItems(["MAT v7.3 (HDF5)", "MAT v5 (Legacy)", "NPZ (NumPy)"])
        type_layout.addWidget(self.file_type)
        type_layout.addStretch()
        format_layout.addLayout(type_layout)
        
        # Compression
        self.compress_check = QCheckBox("Compress data (recommended for large datasets)")
        self.compress_check.setChecked(True)
        format_layout.addWidget(self.compress_check)
        
        # Metadata
        self.metadata_check = QCheckBox("Include metadata (timestamps, parameters, etc.)")
        self.metadata_check.setChecked(True)
        format_layout.addWidget(self.metadata_check)
        
        # Split files
        self.split_check = QCheckBox("Split into multiple files if > 2GB")
        format_layout.addWidget(self.split_check)
        
        format_group.setLayout(format_layout)
        scroll_layout.addWidget(format_group)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Output path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Output Path:"))
        self.path_edit = QLineEdit()
        from datetime import datetime
        default_name = f"analysis_export_{datetime.now().strftime('%Y-%m-%d')}.mat"
        self.path_edit.setText(default_name)
        path_layout.addWidget(self.path_edit)
        self.browse_btn = QPushButton("📁")
        self.browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        
        # Size estimate
        self.size_label = QLabel("Estimated file size: calculating...")
        layout.addWidget(self.size_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        help_btn = QPushButton("❓ Help")
        help_btn.clicked.connect(self._show_help)
        button_layout.addWidget(help_btn)
        
        refresh_btn = QPushButton("↻ Refresh Data")
        refresh_btn.clicked.connect(self._scan_viewer_data)
        button_layout.addWidget(refresh_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _scan_viewer_data(self):
        """Scan viewer for available data."""
        # Clear existing items
        self.flow_tree.clear()
        self.roi_tree.clear()
        self.tracks_tree.clear()
        self.image_tree.clear()
        
        # Scan for flow fields (unchecked by default)
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and 'flow' in layer.name.lower():
                if hasattr(layer, 'data') and layer.data.ndim == 4 and layer.data.shape[-1] == 2:
                    item = QTreeWidgetItem(self.flow_tree)
                    item.setText(0, f"{layer.name} ({layer.data.shape[0]}×{layer.data.shape[1]}×{layer.data.shape[2]})")
                    item.setCheckState(0, Qt.Unchecked)  # Unchecked by default!
                    item.setData(0, Qt.UserRole, layer)
                    
                    # Add size estimate
                    size_mb = self._estimate_layer_size(layer.data)
                    size_item = QTreeWidgetItem(item)
                    size_item.setText(0, f"~{size_mb:.1f} MB uncompressed")
                    size_item.setFlags(Qt.ItemIsEnabled)
        
        # Scan for ROI shapes (checked by default)
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                item = QTreeWidgetItem(self.roi_tree)
                item.setText(0, f"{layer.name} ({len(layer.data)} shapes)")
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, layer)
        
        # Scan for tracks (checked by default)
        for layer in self.viewer.layers:
            if isinstance(layer, Tracks):
                item = QTreeWidgetItem(self.tracks_tree)
                n_tracks = len(np.unique(layer.data[:, 0]))
                item.setText(0, f"{layer.name} ({n_tracks} tracks)")
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, layer)
        
        # Scan for corrected images
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and 'corrected' in layer.name.lower():
                if hasattr(layer, 'data') and layer.data.ndim >= 3:
                    item = QTreeWidgetItem(self.image_tree)
                    item.setText(0, f"{layer.name} {layer.data.shape}")
                    item.setCheckState(0, Qt.Checked)
                    item.setData(0, Qt.UserRole, layer)
        
        self._update_size_estimate()
    
    def _estimate_layer_size(self, data):
        """Estimate size in MB for numpy array."""
        return data.nbytes / (1024 * 1024)
    
    def _update_size_estimate(self):
        """Update the estimated file size."""
        total_size = 0
        
        # Count checked flow fields
        for i in range(self.flow_tree.topLevelItemCount()):
            item = self.flow_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                layer = item.data(0, Qt.UserRole)
                size = self._estimate_layer_size(layer.data)
                
                # Apply reductions
                if self.downsample_check.isChecked():
                    factor = self.downsample_spin.value()
                    size /= (factor * factor)  # Spatial downsampling
                
                if self.magnitude_only.isChecked():
                    size /= 2  # Only one channel instead of two
                
                if self.subset_check.isChecked():
                    total_frames = layer.data.shape[0]
                    subset_frames = self.subset_end.value() - self.subset_start.value()
                    size *= (subset_frames / total_frames)
                
                total_size += size
        
        # Count other data (much smaller)
        # ROIs: ~1KB per ROI
        roi_count = sum(1 for i in range(self.roi_tree.topLevelItemCount()) 
                       if self.roi_tree.topLevelItem(i).checkState(0) == Qt.Checked)
        total_size += roi_count * 0.001
        
        # Tracks: ~10KB per track layer
        track_count = sum(1 for i in range(self.tracks_tree.topLevelItemCount()) 
                         if self.tracks_tree.topLevelItem(i).checkState(0) == Qt.Checked)
        total_size += track_count * 0.01
        
        # Images
        for i in range(self.image_tree.topLevelItemCount()):
            item = self.image_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                layer = item.data(0, Qt.UserRole)
                total_size += self._estimate_layer_size(layer.data)
        
        # Apply compression estimate
        if self.compress_check.isChecked():
            total_size *= 0.3  # Typical compression ratio
        
        # Update label
        if total_size < 1:
            self.size_label.setText(f"Estimated file size: ~{total_size*1024:.1f} KB")
        elif total_size < 1024:
            self.size_label.setText(f"Estimated file size: ~{total_size:.1f} MB")
        else:
            self.size_label.setText(f"Estimated file size: ~{total_size/1024:.2f} GB")
        
        # Add warning if flow fields are selected
        flow_selected = any(self.flow_tree.topLevelItem(i).checkState(0) == Qt.Checked 
                           for i in range(self.flow_tree.topLevelItemCount()))
        if flow_selected and not self.downsample_check.isChecked():
            self.size_label.setText(self.size_label.text() + " ⚠️ Consider enabling downsampling!")
    
    def _browse_output(self):
        """Browse for output file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Export Data",
            self.path_edit.text(),
            "MAT files (*.mat);;NPZ files (*.npz)"
        )
        if file_path:
            self.path_edit.setText(file_path)
    
    def _show_help(self):
        """Show help information."""
        help_text = """
        <h3>Export Motion Analysis Data</h3>
        
        <h4>Data Sources:</h4>
        <ul>
        <li><b>Flow Fields:</b> Full optical flow data (large files!)</li>
        <li><b>ROI Statistics:</b> Motion statistics within regions of interest</li>
        <li><b>Tracked Points:</b> Point trajectories through time</li>
        <li><b>Image Layers:</b> Corrected video data</li>
        </ul>
        
        <h4>Tips:</h4>
        <ul>
        <li>Flow fields are very large - use downsampling or magnitude-only</li>
        <li>MAT v7.3 format supports files >2GB</li>
        <li>Compression reduces file size by ~70%</li>
        <li>ROI statistics are usually sufficient for analysis</li>
        </ul>
        """
        msg = QMessageBox()
        msg.setWindowTitle("Export Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.exec_()
    
    def get_export_options(self):
        """Get the selected export options."""
        options = {
            'output_path': self.path_edit.text(),
            'file_type': self.file_type.currentText(),
            'compress': self.compress_check.isChecked(),
            'include_metadata': self.metadata_check.isChecked(),
            'split_files': self.split_check.isChecked(),
            
            # Statistics options
            'stats': {
                'mean': self.stat_mean.isChecked(),
                'max': self.stat_max.isChecked(),
                'min': self.stat_min.isChecked(),
                'std': self.stat_std.isChecked(),
                'median': self.stat_median.isChecked(),
                'percentiles': self.stat_percentiles.isChecked(),
            },
            'include_roi_masks': self.roi_masks.isChecked(),
            'include_roi_vertices': self.roi_vertices.isChecked(),
            'include_time_series': self.time_series.isChecked(),
            
            # Flow options
            'downsample': self.downsample_check.isChecked(),
            'downsample_factor': self.downsample_spin.value() if self.downsample_check.isChecked() else 1,
            'magnitude_only': self.magnitude_only.isChecked(),
            'subset_frames': self.subset_check.isChecked(),
            'frame_start': self.subset_start.value() if self.subset_check.isChecked() else 0,
            'frame_end': self.subset_end.value() if self.subset_check.isChecked() else None,
            
            # Selected data
            'flow_layers': [],
            'roi_layers': [],
            'track_layers': [],
            'image_layers': [],
        }
        
        # Collect selected layers
        for i in range(self.flow_tree.topLevelItemCount()):
            item = self.flow_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                options['flow_layers'].append(item.data(0, Qt.UserRole))
        
        for i in range(self.roi_tree.topLevelItemCount()):
            item = self.roi_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                options['roi_layers'].append(item.data(0, Qt.UserRole))
        
        for i in range(self.tracks_tree.topLevelItemCount()):
            item = self.tracks_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                options['track_layers'].append(item.data(0, Qt.UserRole))
        
        for i in range(self.image_tree.topLevelItemCount()):
            item = self.image_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                options['image_layers'].append(item.data(0, Qt.UserRole))
        
        return options