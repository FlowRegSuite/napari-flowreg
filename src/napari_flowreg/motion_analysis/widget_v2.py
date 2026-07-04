"""
Motion Analysis Widget v2 for napari-flowreg
Clean wizard-based interface with separated Statistics/Tracking tabs
"""

import re
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QLabel, QProgressBar, QMessageBox, QFileDialog, QDialog
)
from qtpy.QtCore import Qt
from napari.viewer import Viewer
from napari.layers import Image, Shapes, Points, Tracks
from napari.utils.notifications import show_info, show_error

# Import wizard-based UI components
from .wizard import ProcessingWizard, RunOptionsDialog
from .tabs import AnalysisTabs, LayerSelectorWithPill
from .integration import create_worker_for_widget_v2


def _resolve_export_path(file_path: str, selected_filter: str) -> tuple:
    """Return (final_path, fmt) with fmt in {'mat', 'csv'}.

    A recognized typed extension (.mat/.csv) wins; otherwise the extension
    implied by the selected file-dialog filter is appended.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".mat":
        return file_path, "mat"
    if suffix == ".csv":
        return file_path, "csv"
    fmt = "mat" if "*.mat" in (selected_filter or "") else "csv"
    return f"{file_path}.{fmt}", fmt


def _matlab_safe_name(name: str, used: set) -> str:
    """Sanitize to a valid, unique MATLAB identifier (letter first, word chars only)."""
    safe = re.sub(r"\W", "_", str(name)) or "roi"
    if not safe[0].isalpha():
        safe = "roi_" + safe
    safe = safe[:63]
    base, i = safe, 2
    while safe in used:
        safe = f"{base}_{i}"[:63]
        i += 1
    used.add(safe)
    return safe


def _write_tracks_export(file_path: str, selected_filter: str, tracks: np.ndarray) -> str:
    """Write (N, 4) [track_id, frame, y, x] tracks as CSV or MAT. Returns final path."""
    path, fmt = _resolve_export_path(file_path, selected_filter)
    tracks = np.asarray(tracks)
    if fmt == "csv":
        np.savetxt(
            path, tracks, delimiter=",", comments="",
            header="track_id,frame,y,x",
            fmt=["%d", "%d", "%.6f", "%.6f"],
        )
    else:
        from scipy.io import savemat
        savemat(path, {
            "tracks": tracks,
            "column_names": ["track_id", "frame", "y", "x"],
        })
    return path


def _write_statistics_export(file_path: str, selected_filter: str, motion_data: dict) -> str:
    """Write per-ROI motion statistics as CSV (wide table) or MAT struct. Returns final path."""
    path, fmt = _resolve_export_path(file_path, selected_filter)
    roi_names = list(motion_data.keys())
    series = [
        np.asarray(motion_data[r].get("time_series_mean", []), dtype=float).ravel()
        for r in roi_names
    ]
    if fmt == "csv":
        n = min((len(s) for s in series), default=0)
        header_lines = ["# napari-flowreg ROI motion statistics"]
        for r in roi_names:
            s = motion_data[r]
            header_lines.append(
                f"# {r}: overall_mean={s.get('overall_mean', float('nan')):.6f}, "
                f"max={s.get('max', float('nan')):.6f}, min={s.get('min', float('nan')):.6f}"
            )
        columns = ["frame"] + [r.replace(",", ";").replace("\n", " ") for r in roi_names]
        header = "\n".join(header_lines + [",".join(columns)])
        if n:
            data = np.column_stack([np.arange(n)] + [s[:n] for s in series])
            fmt = ["%d"] + ["%.6f"] * len(roi_names)
        else:
            data = np.empty((0, len(columns)))
            fmt = "%.6f"
        np.savetxt(path, data, delimiter=",", comments="", header=header, fmt=fmt)
    else:
        from scipy.io import savemat
        used = set()
        roi_struct = {}
        for r in roi_names:
            s = motion_data[r]
            roi_struct[_matlab_safe_name(r, used)] = {
                "time_series_mean": np.asarray(s.get("time_series_mean", []), dtype=float),
                "overall_mean": float(s.get("overall_mean", float("nan"))),
                "max": float(s.get("max", float("nan"))),
                "min": float(s.get("min", float("nan"))),
            }
        savemat(path, {"roi_stats": roi_struct, "roi_names": roi_names})
    return path


class MotionAnalysisWidgetV2(QWidget):
    """
    Wizard-based motion analysis widget.

    Features:
    - 3-step wizard for mode selection
    - Separated Statistics/Tracking tabs
    - Explicit layer selectors with status pills
    - Clean language and verb-first buttons
    """

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_flow_layer = None
        self.current_video_layer = None
        self.current_roi_layer = None
        self.current_seed_layer = None
        self.motion_data = None
        self.tracked_points = None
        self.roi_shapes_data = {}
        self._warned_flow_defaults = False
        self._progress_signaler = None
        self._tracks_layer = None

        # Initialize UI
        self._init_ui()

        # Connect to layer events
        self.viewer.layers.events.inserted.connect(self._update_layer_lists)
        self.viewer.layers.events.removed.connect(self._update_layer_lists)

        # Initial update
        self._update_layer_lists()

    def _init_ui(self):
        """Initialize the wizard-based UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Processing Wizard
        self.wizard = ProcessingWizard()
        layout.addWidget(self.wizard)
        self.wizard.signaler.state_changed.connect(self._on_wizard_state_changed)

        # Data Selection
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout()

        # Video layer selector (for loaded-layer workflows)
        self.video_selector = LayerSelectorWithPill(
            "Video layer:", "No video selected"
        )
        self.video_selector.layer_changed.connect(self._on_video_layer_changed)
        data_layout.addWidget(self.video_selector)

        # Displacement field selector with status pill
        self.flow_selector = LayerSelectorWithPill(
            "Displacement field (w):", "No field selected"
        )
        self.flow_selector.layer_changed.connect(self._on_flow_layer_changed)
        data_layout.addWidget(self.flow_selector)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Processing: %p% (%v/%m frames)")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Analysis tabs (Statistics/Tracking)
        self.analysis_tabs = AnalysisTabs()
        layout.addWidget(self.analysis_tabs)

        # Connect tab signals
        self.analysis_tabs.statistics_tab.roi_selector.layer_changed.connect(
            self._on_roi_layer_changed
        )
        self.analysis_tabs.tracking_tab.seed_selector.layer_changed.connect(
            self._on_seed_layer_changed
        )
        self.analysis_tabs.statistics_tab.analyze_button.clicked.connect(
            self._run_statistics_analysis
        )
        self.analysis_tabs.tracking_tab.track_button.clicked.connect(
            self._run_tracking_analysis
        )
        self.analysis_tabs.statistics_tab.export_stats_button.clicked.connect(
            self._export_statistics
        )
        self.analysis_tabs.tracking_tab.export_tracks_button.clicked.connect(
            self._export_tracks
        )

        layout.addStretch()
        self.setLayout(layout)

        # Initialize run options dialog
        self.run_options_dialog = RunOptionsDialog(self)

    def _on_wizard_state_changed(self):
        """Handle wizard state changes."""
        # Update button text based on mode
        # Enable/disable selectors based on input choices
        use_loaded_layer = self.wizard.input_choice == "layer"
        self.video_selector.setEnabled(use_loaded_layer)
        if not use_loaded_layer:
            self.current_video_layer = None
            self.video_selector.set_status("Not required", True)

        requires_flow_layer = (
            self.wizard.flow_choice == "precomputed" and use_loaded_layer
        )
        self.flow_selector.setEnabled(requires_flow_layer)
        if not requires_flow_layer:
            self.current_flow_layer = None
            self.flow_selector.set_status("Not required", True)

    def _prompt_run_options(self) -> Optional[Dict[str, Any]]:
        """Open the run options dialog and return settings."""
        self.run_options_dialog.update_visibility(self.wizard)
        if self.run_options_dialog.exec_() != QDialog.Accepted:
            return None
        return self.run_options_dialog.get_options()

    def _update_layer_lists(self):
        """Update layer dropdowns in selectors."""
        # Update video selector
        current_video = self.video_selector.current_text()
        self.video_selector.clear_items()
        self.video_selector.add_item("")
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.video_selector.add_item(layer.name)
        if current_video:
            self.video_selector.set_current_text(current_video)
        elif self.video_selector.combo.count() > 1:
            first_item = self.video_selector.combo.itemText(1)
            if first_item:
                self.video_selector.set_current_text(first_item)

        # Update displacement field selector
        current_flow = self.flow_selector.current_text()
        self.flow_selector.clear_items()
        self.flow_selector.add_item("")  # Empty option

        # Find image layers that could be flow fields
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                # Check if it looks like a flow field
                if layer.data.ndim >= 3 and layer.data.shape[-1] == 2:
                    self.flow_selector.add_item(layer.name)
                elif "flow" in layer.name.lower() or layer.name.lower() == "w":
                    self.flow_selector.add_item(layer.name)

        # Restore selection if possible
        if current_flow:
            self.flow_selector.set_current_text(current_flow)

        # Update ROI selector
        current_roi = self.analysis_tabs.statistics_tab.roi_selector.current_text()
        self.analysis_tabs.statistics_tab.roi_selector.clear_items()
        self.analysis_tabs.statistics_tab.roi_selector.add_item("")

        # Find shapes layers
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                self.analysis_tabs.statistics_tab.roi_selector.add_item(layer.name)

        if current_roi:
            self.analysis_tabs.statistics_tab.roi_selector.set_current_text(current_roi)
        elif self.analysis_tabs.statistics_tab.roi_selector.combo.count() > 1:
            first_roi = self.analysis_tabs.statistics_tab.roi_selector.combo.itemText(1)
            if first_roi:
                self.analysis_tabs.statistics_tab.roi_selector.set_current_text(first_roi)

        # Update seed selector
        current_seed = self.analysis_tabs.tracking_tab.seed_selector.current_text()
        self.analysis_tabs.tracking_tab.seed_selector.clear_items()
        self.analysis_tabs.tracking_tab.seed_selector.add_item("")

        # Find points and shapes layers with points
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                n_points = len(layer.data) if hasattr(layer, 'data') else 0
                self.analysis_tabs.tracking_tab.seed_selector.add_item(layer.name)
                if layer.name == current_seed:
                    self.analysis_tabs.tracking_tab.seed_selector.set_status(
                        f"{n_points} points", True
                    )
            elif isinstance(layer, Shapes):
                # Check for point shapes
                if hasattr(layer, 'shape_types'):
                    n_points = sum(
                        1 for t in layer.shape_types if t in ('point', 'points')
                    )
                    if n_points > 0:
                        self.analysis_tabs.tracking_tab.seed_selector.add_item(layer.name)
                        if layer.name == current_seed:
                            self.analysis_tabs.tracking_tab.seed_selector.set_status(
                                f"{n_points} points", True
                            )

        if current_seed:
            self.analysis_tabs.tracking_tab.seed_selector.set_current_text(current_seed)
        elif self.analysis_tabs.tracking_tab.seed_selector.combo.count() > 1:
            first_seed = self.analysis_tabs.tracking_tab.seed_selector.combo.itemText(1)
            if first_seed:
                self.analysis_tabs.tracking_tab.seed_selector.set_current_text(first_seed)

    def _on_video_layer_changed(self, layer_name: str):
        """Handle video layer selection change."""
        if not layer_name:
            self.current_video_layer = None
            self.video_selector.set_status("No video selected", False)
            return

        try:
            layer = self.viewer.layers[layer_name]
            self.current_video_layer = layer
            shape = layer.data.shape
            if len(shape) >= 3:
                status = f"{shape[0]} frames"
            else:
                status = f"{shape}"
            self.video_selector.set_status(status, True)
        except KeyError:
            self.current_video_layer = None
            self.video_selector.set_status("Layer not found", False)

    def _on_flow_layer_changed(self, layer_name: str):
        """Handle displacement field selection change."""
        if not layer_name:
            self.current_flow_layer = None
            self.flow_selector.set_status("No field selected", False)
            return

        try:
            layer = self.viewer.layers[layer_name]
            self.current_flow_layer = layer

            # Update status pill
            shape = layer.data.shape
            if len(shape) >= 3 and shape[-1] == 2:
                if len(shape) == 4:  # Temporal
                    status = f"{shape[0]} frames"
                else:  # Single frame
                    status = f"{shape[0]}×{shape[1]}"
                self.flow_selector.set_status(status, True)
            else:
                self.flow_selector.set_status("Invalid shape", False)
        except KeyError:
            self.current_flow_layer = None
            self.flow_selector.set_status("Layer not found", False)

    def _on_roi_layer_changed(self, layer_name: str):
        """Handle ROI layer selection change."""
        if not layer_name:
            self.current_roi_layer = None
            self.analysis_tabs.statistics_tab.roi_selector.set_status(
                "No ROI selected", False
            )
            return

        try:
            layer = self.viewer.layers[layer_name]
            if isinstance(layer, Shapes):
                self.current_roi_layer = layer
                n_shapes = len(layer.data)
                self.analysis_tabs.statistics_tab.roi_selector.set_status(
                    f"{n_shapes} ROIs", True
                )
            else:
                self.current_roi_layer = None
                self.analysis_tabs.statistics_tab.roi_selector.set_status(
                    "Not a Shapes layer", False
                )
        except KeyError:
            self.current_roi_layer = None
            self.analysis_tabs.statistics_tab.roi_selector.set_status(
                "Layer not found", False
            )

    def _on_seed_layer_changed(self, layer_name: str):
        """Handle seed layer selection change."""
        if not layer_name:
            self.current_seed_layer = None
            self.analysis_tabs.tracking_tab.seed_selector.set_status(
                "No seeds selected", False
            )
            return

        try:
            layer = self.viewer.layers[layer_name]
            self.current_seed_layer = layer

            # Count points
            if isinstance(layer, Points):
                n_points = len(layer.data)
                self.analysis_tabs.tracking_tab.seed_selector.set_status(
                    f"{n_points} points", True
                )
            elif isinstance(layer, Shapes):
                n_points = sum(
                    1 for t in layer.shape_types if t in ('point', 'points')
                )
                self.analysis_tabs.tracking_tab.seed_selector.set_status(
                    f"{n_points} points", True
                )
            else:
                self.analysis_tabs.tracking_tab.seed_selector.set_status(
                    "Invalid layer type", False
                )
        except KeyError:
            self.current_seed_layer = None
            self.analysis_tabs.tracking_tab.seed_selector.set_status(
                "Layer not found", False
            )

    def _execute_motion_analysis(self, options: Dict[str, Any]):
        """Shared execution path for statistics/tracking actions."""
        mode_def = self.wizard.get_mode_definition()

        # Check if RAM warning needed
        if self.wizard.requires_confirmation():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("High Memory Usage")
            msg.setText("This operation will load large data into memory.")
            msg.setInformativeText(
                mode_def.warning_message or
                "Ensure you have sufficient RAM available."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            if msg.exec_() != QMessageBox.Ok:
                return

        dialog_flow_params = self._build_flow_parameters_from_options(options)
        flow_params = self._get_flowreg_flow_parameters()
        if flow_params:
            flow_params.update(dialog_flow_params)
        else:
            flow_params = dialog_flow_params

        if flow_params:
            options["flow_params"] = flow_params
            for key in (
                "flow_quality",
                "flow_alpha",
                "flow_sigma_xy",
                "flow_sigma_t",
                "flow_levels",
                "flow_iterations",
                "flow_eta",
            ):
                options.pop(key, None)

            if (
                self.wizard.flow_choice == "compute"
                and not self._warned_flow_defaults
            ):
                show_info(
                    "Using optical-flow parameters from the run options dialog. "
                    "Open the FlowReg widget and run once to reuse different settings."
                )
                self._warned_flow_defaults = True

        options["export_w_viewer"] = bool(options.get("export_w_viewer", False))
        options["save_w"] = bool(options.get("save_w", False))

        # Reset cached results before processing
        enable_roi = bool(options.get("enable_roi_stats", False))
        enable_tracks = bool(options.get("enable_trajectories", False))

        self.motion_data = None
        self.tracked_points = None
        self.analysis_tabs.statistics_tab.export_stats_button.setEnabled(False)
        self.analysis_tabs.tracking_tab.export_tracks_button.setEnabled(False)
        if enable_roi:
            self.analysis_tabs.statistics_tab.stats_label.setText("Processing...")
        else:
            self.analysis_tabs.statistics_tab.stats_label.setText("No statistics computed")
        if enable_tracks:
            self.analysis_tabs.tracking_tab.tracking_info_label.setText("Processing...")
        else:
            self.analysis_tabs.tracking_tab.tracking_info_label.setText("No tracks computed")
        if enable_tracks and self._tracks_layer is not None:
            if self._tracks_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracks_layer)
            self._tracks_layer = None

        # Route to appropriate processor based on wizard state
        try:
            worker = create_worker_for_widget_v2(self, mode_def, options)
        except ValueError as exc:
            self.progress_bar.hide()
            show_error(str(exc))
            return
        except Exception as exc:
            self.progress_bar.hide()
            show_error(f"Failed to start processing: {exc}")
            return

        if worker is None:
            self.progress_bar.hide()
            show_error("Processing mode not yet implemented")
            return

        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # Create and start worker based on mode
        worker.returned.connect(self._on_processing_complete)
        worker.errored.connect(self._on_processing_error)

        if hasattr(worker, "progress_signaler"):
            if self._progress_signaler is not None:
                try:
                    self._progress_signaler.progress_updated.disconnect(self._update_progress)
                except TypeError:
                    pass
                try:
                    self._progress_signaler.tracks_updated.disconnect(self._update_tracks_partial)
                except TypeError:
                    pass
                try:
                    self._progress_signaler.roi_updated.disconnect(self._update_roi_summary)
                except TypeError:
                    pass
            self._progress_signaler = worker.progress_signaler
            self._progress_signaler.progress_updated.connect(self._update_progress)
            if hasattr(self._progress_signaler, "tracks_updated"):
                self._progress_signaler.tracks_updated.connect(self._update_tracks_partial)
            if hasattr(self._progress_signaler, "roi_updated"):
                self._progress_signaler.roi_updated.connect(self._update_roi_summary)
        else:
            self._progress_signaler = None

        worker.start()

    def _update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_processing_complete(self, results: dict):
        """Handle processing completion."""
        self.progress_bar.hide()

        # Add results to viewer if needed
        if results.get('w') is not None and self.wizard.should_load_to_viewer():
            self.viewer.add_image(
                results['w'],
                name="Displacement field (w)",
                colormap="turbo"
            )

        corrected = results.get('corrected')
        if corrected is not None:
            self.viewer.add_image(
                corrected,
                name="Corrected video"
            )
        elif results.get('corrected_path'):
            show_info(f"Corrected video saved to {results['corrected_path']}")

        tracks = results.get('tracks')
        if tracks is not None:
            self._update_tracks_partial(tracks)

        roi_stats = results.get('roi_stats')
        if roi_stats:
            self._update_roi_summary(roi_stats)

        if results.get('w_path'):
            show_info(f"Displacement field saved to {results['w_path']}")

        show_info("Processing complete!")

    def _get_flowreg_flow_parameters(self) -> Optional[Dict[str, Any]]:
        """Fetch current FlowReg options from the main widget if available."""
        try:
            from napari_flowreg.flowreg_widget import FlowRegWidget  # lazy import to avoid cycles
        except ImportError:
            return None

        dock_widgets_attr = getattr(self.viewer.window, "dock_widgets", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            private_widgets = getattr(self.viewer.window, "_dock_widgets", {})
        if not dock_widgets_attr:
            if private_widgets:
                dock_widgets_attr = private_widgets.items()
            else:
                return None

        widget_iter = (
            dock_widgets_attr.items()
            if isinstance(dock_widgets_attr, dict)
            else dock_widgets_attr
        )

        for item in widget_iter:
            if isinstance(item, tuple):
                widget_obj = item[1]
            else:
                widget_obj = item

            if isinstance(widget_obj, str):
                widget_candidate = private_widgets.get(widget_obj)
                if widget_candidate is None:
                    continue
            else:
                widget_candidate = widget_obj

            if hasattr(widget_candidate, "widget"):
                widget_attr = widget_candidate.widget
                widget_candidate = widget_attr() if callable(widget_attr) else widget_attr

            if callable(widget_candidate):
                widget_candidate = widget_candidate()

            if isinstance(widget_candidate, FlowRegWidget):
                options = getattr(widget_candidate, "current_options", None)
                if options:
                    return dict(options)
        return None

    def _on_processing_error(self, error):
        """Handle processing error."""
        self.progress_bar.hide()
        show_error(f"Processing failed: {str(error)}")

    def _has_seed_points(self) -> bool:
        """Check whether any seed points are available in the viewer."""
        if self.current_seed_layer is not None:
            layer = self.current_seed_layer
            if isinstance(layer, Points) and len(layer.data) > 0:
                return True
            if isinstance(layer, Shapes) and hasattr(layer, "shape_types"):
                return any(t in ("point", "points") for t in layer.shape_types)

        for layer in self.viewer.layers:
            if isinstance(layer, Points) and len(layer.data) > 0:
                return True
            if isinstance(layer, Shapes) and hasattr(layer, "shape_types"):
                if any(t in ("point", "points") for t in layer.shape_types):
                    return True
        return False

    def _run_statistics_analysis(self):
        """Run ROI statistics analysis."""
        options = self._prompt_run_options()
        if options is None:
            return

        options["enable_roi_stats"] = True
        options["enable_trajectories"] = False
        self._execute_motion_analysis(options)

    def _run_tracking_analysis(self):
        """Run trajectory tracking."""
        if not self._has_seed_points():
            show_error("No seed points found. Add a Points layer (or Shapes points) before tracking.")
            return

        options = self._prompt_run_options()
        if options is None:
            return

        options["enable_roi_stats"] = False
        options["enable_trajectories"] = True

        self._execute_motion_analysis(options)

    def _update_tracks_partial(self, tracks):
        """Update or create the trajectories layer with partial results."""
        if tracks is None or len(tracks) == 0:
            return
        self.tracked_points = tracks
        if self._tracks_layer is None or self._tracks_layer not in self.viewer.layers:
            self._tracks_layer = self.viewer.add_tracks(
                tracks,
                name="Trajectories",
                colormap="turbo",
            )
        else:
            self._tracks_layer.data = tracks
        self.analysis_tabs.tracking_tab.export_tracks_button.setEnabled(True)
        unique_tracks = len(np.unique(tracks[:, 0])) if tracks.size else 0
        max_frame = tracks[:, 1].max() + 1 if tracks.size else 0
        self.analysis_tabs.tracking_tab.tracking_info_label.setText(
            f"Tracks: {unique_tracks} points over {int(max_frame)} frames"
        )

    def _update_roi_summary(self, summary):
        """Update ROI statistics summary label."""
        if not summary:
            return
        self.motion_data = summary
        lines = []
        for roi_name, stats in summary.items():
            overall = stats.get("overall_mean")
            if overall is not None:
                lines.append(f"{roi_name}: mean {overall:.3f}")
        if lines:
            self.analysis_tabs.statistics_tab.stats_label.setText("\n".join(lines))
        else:
            self.analysis_tabs.statistics_tab.stats_label.setText("Statistics computed.")
        self.analysis_tabs.statistics_tab.export_stats_button.setEnabled(True)

    def _build_flow_parameters_from_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create flow-parameter overrides from the options dialog."""
        quality = str(options.get("flow_quality", "balanced")).lower()
        alpha = options.get("flow_alpha", 1.5)
        sigma_xy = options.get("flow_sigma_xy", 1.0)
        sigma_t = options.get("flow_sigma_t", 0.1)
        levels = options.get("flow_levels", 100)
        iterations = options.get("flow_iterations", 50)
        eta = options.get("flow_eta", 0.8)

        params: Dict[str, Any] = {
            "quality_setting": quality,
            "alpha": alpha,
            "sigma": [[sigma_xy, sigma_xy, sigma_t], [sigma_xy, sigma_xy, sigma_t]],
        }

        if quality == "custom":
            params.update({"levels": levels, "iterations": iterations, "eta": eta})

        return params

    def _export_statistics(self):
        """Export ROI statistics to CSV or MAT."""
        if not self.motion_data:
            show_error("No statistics to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Statistics", "", "MAT files (*.mat);;CSV files (*.csv)"
        )
        if not file_path:
            return

        try:
            final_path = _write_statistics_export(file_path, selected_filter, self.motion_data)
        except Exception as exc:
            show_error(f"Statistics export failed: {exc}")
            return
        show_info(f"Statistics exported to {final_path}")

    def _export_tracks(self):
        """Export trajectory tracks to CSV or MAT."""
        if self.tracked_points is None or len(self.tracked_points) == 0:
            show_error("No tracks to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Tracks", "", "MAT files (*.mat);;CSV files (*.csv)"
        )
        if not file_path:
            return

        try:
            final_path = _write_tracks_export(file_path, selected_filter, self.tracked_points)
        except Exception as exc:
            show_error(f"Track export failed: {exc}")
            return
        show_info(f"Tracks exported to {final_path}")
