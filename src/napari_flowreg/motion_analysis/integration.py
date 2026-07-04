"""
Integration module for adding batch processing modes to motion_analysis_widget.
This module contains the functions and classes that will be integrated into
the existing motion_analysis_widget.py to add new processing modes.
"""

import os
import shutil
import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np

from qtpy.QtWidgets import QComboBox, QLabel, QGroupBox, QVBoxLayout, QSpinBox, QGridLayout, QCheckBox, QLineEdit, QPushButton
from qtpy.QtCore import Qt, Signal, QObject
from napari.utils.notifications import show_info, show_error
from napari.qt.threading import thread_worker

from .modes import ProcessingMode, ModeManager, ModeDefinition

try:
    from pyflowreg.motion_correction.OF_options import QualitySetting
except ImportError:  # pragma: no cover - pyflowreg required at runtime
    QualitySetting = None
from .callbacks import (
    BatchTrajectoryTracker,
    ROIMotionAnalyzer,
    CombinedCallbackDispatcher,
    ProgressTracker,
    WAccumulator,
    extract_roi_masks_from_shapes,
    extract_seed_points_from_layer
)

# Initialize mode manager
_mode_manager = ModeManager()


class ProgressSignaler(QObject):
    """Helper class to emit progress signals from worker threads."""
    progress_updated = Signal(int, int)  # (current, total)
    tracks_updated = Signal(object)
    roi_updated = Signal(object)


def _normalize_flow_params(flow_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(flow_params or {})
    quality_value = params.get("quality_setting")
    if isinstance(quality_value, str):
        if QualitySetting is not None:
            try:
                params["quality_setting"] = QualitySetting(quality_value)
            except ValueError:
                params["quality_setting"] = QualitySetting.BALANCED
        else:
            params["quality_setting"] = quality_value
    return params


def create_mode_selector_group() -> QGroupBox:
    """
    Create the mode selector group box to be added at the top of motion_analysis_widget.
    This will be called in _init_ui() before other groups.
    """
    group = QGroupBox("Processing Mode")
    layout = QVBoxLayout()

    # Mode selector - driven by ModeManager
    mode_combo = QComboBox()
    mode_names = _mode_manager.get_mode_names()
    mode_combo.addItems(mode_names)
    mode_combo.setCurrentIndex(0)  # Default to current behavior (ONLINE_LOADED)

    # Set tooltip from first mode definition
    first_mode = list(_mode_manager.definitions.values())[0]
    mode_combo.setToolTip(first_mode.tooltip)
    layout.addWidget(mode_combo)

    # Mode description
    mode_info = QLabel(first_mode.description)
    mode_info.setWordWrap(True)
    mode_info.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 5px; }")
    layout.addWidget(mode_info)

    # Memory usage indicator
    memory_label = QLabel(f"Memory: {first_mode.memory.usage_level.upper()} ({first_mode.memory.peak_formula})")
    memory_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
    layout.addWidget(memory_label)

    group.setLayout(layout)

    # Store references as attributes for access
    group.mode_combo = mode_combo
    group.mode_info = mode_info
    group.memory_label = memory_label

    return group


def add_batch_processing_options(parent_layout, after_widget_index=1):
    """
    Add batch processing options to existing analysis options group.
    These are hidden by default and shown only for batch modes.
    """
    batch_group = QGroupBox("Batch Processing Options")
    batch_layout = QGridLayout()

    # Batch size
    batch_layout.addWidget(QLabel("Batch Size:"), 0, 0)
    batch_size_spin = QSpinBox()
    batch_size_spin.setRange(10, 500)
    batch_size_spin.setValue(50)
    batch_size_spin.setToolTip("Frames to process at once")
    batch_layout.addWidget(batch_size_spin, 0, 1)

    # Number of workers
    batch_layout.addWidget(QLabel("Workers:"), 1, 0)
    workers_spin = QSpinBox()
    workers_spin.setRange(1, 16)
    workers_spin.setValue(4)
    batch_layout.addWidget(workers_spin, 1, 1)

    # Options for batch mode
    save_w_check = QCheckBox("Save w to disk")
    save_w_check.setToolTip("Save displacement field to HDF5 file")
    batch_layout.addWidget(save_w_check, 2, 0, 1, 2)

    w_path_edit = QLineEdit()
    w_path_edit.setPlaceholderText("Path to save w.h5...")
    w_path_edit.setEnabled(False)
    batch_layout.addWidget(w_path_edit, 3, 0, 1, 2)

    load_w_check = QCheckBox("Load w to viewer (uses memory!)")
    load_w_check.setStyleSheet("QCheckBox { color: orange; }")
    batch_layout.addWidget(load_w_check, 4, 0, 1, 2)

    # Video file input (for file-based modes)
    batch_layout.addWidget(QLabel("Video file:"), 5, 0)
    video_file_path_edit = QLineEdit()
    video_file_path_edit.setPlaceholderText("Path to video file...")
    video_file_path_edit.setEnabled(False)
    batch_layout.addWidget(video_file_path_edit, 5, 1)

    video_file_browse_btn = QPushButton("Browse...")
    video_file_browse_btn.setEnabled(False)
    batch_layout.addWidget(video_file_browse_btn, 5, 2)

    # Load w from file (for mode 2a - analyze pre-computed w.h5)
    batch_layout.addWidget(QLabel("Load w from file:"), 6, 0)
    load_w_path_edit = QLineEdit()
    load_w_path_edit.setPlaceholderText("Path to w.h5 file...")
    load_w_path_edit.setEnabled(False)
    batch_layout.addWidget(load_w_path_edit, 6, 1)

    load_w_browse_btn = QPushButton("Browse...")
    load_w_browse_btn.setEnabled(False)
    batch_layout.addWidget(load_w_browse_btn, 6, 2)

    # Corrected output path (for file-based modes that save corrected video)
    batch_layout.addWidget(QLabel("Save corrected to:"), 7, 0)
    corrected_path_edit = QLineEdit()
    corrected_path_edit.setPlaceholderText("Path to save corrected video...")
    corrected_path_edit.setEnabled(False)
    batch_layout.addWidget(corrected_path_edit, 7, 1)

    corrected_browse_btn = QPushButton("Browse...")
    corrected_browse_btn.setEnabled(False)
    batch_layout.addWidget(corrected_browse_btn, 7, 2)

    batch_group.setLayout(batch_layout)

    # Initially hidden
    batch_group.hide()

    # Store references
    batch_group.batch_size_spin = batch_size_spin
    batch_group.workers_spin = workers_spin
    batch_group.save_w_check = save_w_check
    batch_group.w_path_edit = w_path_edit
    batch_group.load_w_check = load_w_check
    batch_group.video_file_path_edit = video_file_path_edit
    batch_group.video_file_browse_btn = video_file_browse_btn
    batch_group.load_w_path_edit = load_w_path_edit
    batch_group.load_w_browse_btn = load_w_browse_btn
    batch_group.corrected_path_edit = corrected_path_edit
    batch_group.corrected_browse_btn = corrected_browse_btn

    # Connect signals
    save_w_check.toggled.connect(w_path_edit.setEnabled)

    def browse_video_file():
        from qtpy.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select video file", "", "Video files (*.tif *.tiff *.h5 *.hdf5 *.mat)"
        )
        if file_path:
            video_file_path_edit.setText(file_path)

    def browse_w_file():
        from qtpy.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select w.h5 file", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if file_path:
            load_w_path_edit.setText(file_path)

    def browse_corrected_file():
        from qtpy.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save corrected video", "", "HDF5 files (*.h5);;TIFF files (*.tif *.tiff)"
        )
        if file_path:
            corrected_path_edit.setText(file_path)

    video_file_browse_btn.clicked.connect(browse_video_file)
    load_w_browse_btn.clicked.connect(browse_w_file)
    corrected_browse_btn.clicked.connect(browse_corrected_file)

    return batch_group


def handle_mode_change(widget, mode_index: int):
    """
    Handle mode change in the motion_analysis_widget.
    This updates UI visibility and changes behavior of existing functions.
    Driven by ModeDefinition from ModeManager.

    Args:
        widget: The MotionAnalysisWidget instance
        mode_index: Index of selected mode
    """
    # Get mode definition from ModeManager
    mode_names = _mode_manager.get_mode_names()
    if mode_index >= len(mode_names):
        return

    mode_name = mode_names[mode_index]
    mode_def = _mode_manager.get_mode_by_name(mode_name)

    if not mode_def:
        return

    # Update mode info label
    if hasattr(widget, 'mode_info'):
        widget.mode_info.setText(mode_def.description)

    # Update memory label with color coding
    if hasattr(widget, 'memory_label'):
        level = mode_def.memory.usage_level.upper()
        formula = mode_def.memory.peak_formula

        # Color coding based on memory level
        color_map = {
            "MINIMAL": "green",
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red"
        }
        color = color_map.get(level, "gray")

        widget.memory_label.setText(f"Memory: {level} ({formula})")
        widget.memory_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")

    # Show/hide batch processing options based on mode definition
    if hasattr(widget, 'batch_processing_group'):
        # Show batch group if mode uses batch processing
        show_batch = mode_def.processing.batch_processing
        widget.batch_processing_group.setVisible(show_batch)

        # Update visibility of specific batch options based on mode flags
        if show_batch and hasattr(widget.batch_processing_group, 'batch_size_spin'):
            # Batch size always visible when batch processing is used
            pass

        if hasattr(widget.batch_processing_group, 'save_w_check'):
            # Show save w checkbox if mode definition indicates it
            widget.batch_processing_group.save_w_check.setVisible(mode_def.show_export_w_checkbox)

        if hasattr(widget.batch_processing_group, 'w_path_edit'):
            widget.batch_processing_group.w_path_edit.setVisible(mode_def.show_w_path)

        # Show/hide load w from file UI (for modes that need pre-computed w)
        show_load_w = (mode_def.mode in (ProcessingMode.ANALYSIS_PRECOMPUTED, ProcessingMode.LOADED_WITH_PRECOMPUTED, ProcessingMode.FILE_WITH_PRECOMPUTED))
        if hasattr(widget.batch_processing_group, 'load_w_path_edit'):
            widget.batch_processing_group.load_w_path_edit.setVisible(show_load_w)
            widget.batch_processing_group.load_w_path_edit.setEnabled(show_load_w)
        if hasattr(widget.batch_processing_group, 'load_w_browse_btn'):
            widget.batch_processing_group.load_w_browse_btn.setVisible(show_load_w)
            widget.batch_processing_group.load_w_browse_btn.setEnabled(show_load_w)

        # Show/hide video file input (for file-based modes)
        show_video_file = (mode_def.input.source_type == "file")
        if hasattr(widget.batch_processing_group, 'video_file_path_edit'):
            widget.batch_processing_group.video_file_path_edit.setVisible(show_video_file)
            widget.batch_processing_group.video_file_path_edit.setEnabled(show_video_file)
        if hasattr(widget.batch_processing_group, 'video_file_browse_btn'):
            widget.batch_processing_group.video_file_browse_btn.setVisible(show_video_file)
            widget.batch_processing_group.video_file_browse_btn.setEnabled(show_video_file)

        # Show/hide corrected output path (for modes that save corrected video)
        show_corrected_path = mode_def.show_output_path
        if hasattr(widget.batch_processing_group, 'corrected_path_edit'):
            widget.batch_processing_group.corrected_path_edit.setVisible(show_corrected_path)
            widget.batch_processing_group.corrected_path_edit.setEnabled(show_corrected_path)
        if hasattr(widget.batch_processing_group, 'corrected_browse_btn'):
            widget.batch_processing_group.corrected_browse_btn.setVisible(show_corrected_path)
            widget.batch_processing_group.corrected_browse_btn.setEnabled(show_corrected_path)

    # Update flow combo based on requirements
    if hasattr(widget, 'flow_combo'):
        if mode_def.input.requires_flow and not mode_def.processing.compute_flow:
            # Mode needs pre-existing flow
            widget.flow_combo.setEnabled(True)
            widget.flow_combo.setToolTip("Select pre-loaded flow field (w)")
        elif mode_def.processing.compute_flow:
            # Mode will compute flow
            widget.flow_combo.setEnabled(False)
            widget.flow_combo.setToolTip("Flow will be computed during processing")
        else:
            # Flow not needed
            widget.flow_combo.setEnabled(False)
            widget.flow_combo.setToolTip("Not required for this mode")

    # Update video combo based on requirements
    if hasattr(widget, 'video_combo'):
        if mode_def.input.requires_video:
            if hasattr(widget.video_combo, 'setEnabled'):
                widget.video_combo.setEnabled(True)
            widget.video_combo.setToolTip("Select video layer to process")
        else:
            if hasattr(widget.video_combo, 'setEnabled'):
                widget.video_combo.setEnabled(False)
            widget.video_combo.setToolTip("Not required for this mode")

    # Store mode index for processing logic
    widget.current_processing_mode = mode_index

    # Update button text based on mode
    if hasattr(widget, 'analyze_button'):
        if mode_def.processing.compute_flow:
            widget.analyze_button.setText("Process & Analyze")
        elif mode_def.input.requires_flow and mode_def.input.source_type == "file":
            widget.analyze_button.setText("Load & Analyze")
        else:
            widget.analyze_button.setText("Analyze Motion")


def create_batch_processor_for_widget(widget) -> Optional[Callable]:
    """
    Create a batch processor function for the current widget state.
    This will be called when analyze button is clicked in batch mode.
    Uses ModeManager to determine which processor to use.

    Returns:
        Thread worker function or None if not in batch mode
    """
    if not hasattr(widget, 'current_processing_mode'):
        return None

    mode_index = widget.current_processing_mode
    mode_names = _mode_manager.get_mode_names()

    if mode_index >= len(mode_names):
        return None

    mode_name = mode_names[mode_index]
    mode_def = _mode_manager.get_mode_by_name(mode_name)

    if not mode_def:
        return None

    # Route to appropriate worker based on mode
    if mode_def.mode == ProcessingMode.LOADED_BATCH_PROCESS:
        return create_batch_process_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.LOADED_WITH_PRECOMPUTED:
        return create_hybrid_process_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.ANALYSIS_PRECOMPUTED:
        return create_precomputed_analysis_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.FILE_FULL_PIPELINE:
        return create_file_full_pipeline_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.FILE_TO_MEMORY:
        return create_file_to_memory_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.FILE_WITH_PRECOMPUTED:
        return create_file_with_precomputed_worker(widget, mode_def)
    elif mode_def.mode == ProcessingMode.ANALYSIS_STREAMING:
        return create_analysis_streaming_worker(widget, mode_def)
    else:
        # Mode not yet implemented
        return None


def _build_loaded_batch_worker(
    video_layer,
    viewer,
    mode_def: ModeDefinition,
    batch_size: int,
    save_w: bool = False,
    w_dir: Optional[Path] = None,
    load_w: bool = False,
    roi_layer=None,
    seed_layer=None,
    flow_params: Optional[Dict[str, Any]] = None,
    return_corrected: bool = False,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.motion_correction.compensate_arr import compensate_arr
            from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting

            video = video_layer.data
            if video.ndim < 3:
                raise ValueError("Video must be at least 3D (T, H, W)")

            reference = video[0]

            want_load_w = bool(load_w)
            if save_w and w_dir is not None:
                show_info("Saving displacement fields for in-memory videos is not yet supported; ignoring save request.")

            roi_masks = None
            roi_names = None
            if roi_layer is not None:
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    roi_layer, video.shape, target_layer=video_layer
                )

            params = _normalize_flow_params(flow_params)
            params.pop("input_file", None)
            params.pop("output_path", None)
            params.pop("output_file_name", None)
            params.pop("output_format", None)
            params["buffer_size"] = batch_size
            params["save_w"] = False
            params.setdefault("quality_setting", QualitySetting.BALANCED)
            options = OFOptions(**params)

            dispatcher = CombinedCallbackDispatcher()

            T = video.shape[0]
            progress_tracker = ProgressTracker(
                total_frames=T,
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                seed_points = None
                if seed_layer is not None:
                    seed_points = extract_seed_points_from_layer(seed_layer)
                else:
                    for layer in viewer.layers:
                        try:
                            extracted = extract_seed_points_from_layer(layer)
                            if extracted is not None and len(extracted) > 0:
                                seed_points = extracted
                                break
                        except Exception:
                            continue

                if seed_points is not None and len(seed_points) > 0:
                    H, W = video.shape[1:3]
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=T,
                        H=H,
                        W=W,
                    )
                    dispatcher.add_callback(trajectory_tracker)

            analyzer = None
            if roi_masks and len(roi_masks) > 0:
                analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                dispatcher.add_callback(analyzer)

            def wrapped_callback(w_batch, batch_start, batch_end):
                dispatcher.w_callback(w_batch, batch_start, batch_end)
                if trajectory_tracker:
                    tracks_partial = trajectory_tracker.get_tracks_for_napari()
                    progress_signaler.tracks_updated.emit(tracks_partial)
                if analyzer:
                    progress_signaler.roi_updated.emit(analyzer.get_summary_statistics())

            corrected, w = compensate_arr(
                video,
                reference,
                options,
                w_callback=wrapped_callback if dispatcher.callbacks else None,
            )

            if not want_load_w:
                w = None

            if not return_corrected:
                corrected = None

            roi_stats = None
            if analyzer and hasattr(analyzer, "get_summary_statistics"):
                roi_stats = analyzer.get_summary_statistics()

            tracks = None
            if trajectory_tracker:
                tracks = trajectory_tracker.get_tracks_for_napari()

            result = {"roi_stats": roi_stats, "tracks": tracks}
            if want_load_w:
                result["w"] = w
            if corrected is not None:
                result["corrected"] = corrected
            return result

        except Exception as e:
            show_error(f"Batch processing failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_batch_process_worker(widget, mode_def: ModeDefinition):
    video_layer_name = widget.video_combo.currentText() if hasattr(widget, 'video_combo') else widget.flow_combo.currentText()
    if not video_layer_name:
        raise ValueError("No video layer selected")

    video_layer = widget.viewer.layers[video_layer_name]

    batch_size = 50
    load_w = False
    if hasattr(widget, 'batch_processing_group'):
        batch_size = widget.batch_processing_group.batch_size_spin.value()
        load_w = widget.batch_processing_group.load_w_check.isChecked()

    return _build_loaded_batch_worker(
        video_layer=video_layer,
        viewer=widget.viewer,
        mode_def=mode_def,
        batch_size=batch_size,
        save_w=False,
        w_dir=None,
        load_w=load_w,
        roi_layer=getattr(widget, "current_roi_layer", None),
        seed_layer=getattr(widget, "current_seed_layer", None),
        flow_params=None,
        return_corrected=True,
    )


def _build_hybrid_process_worker(
    video_layer,
    viewer,
    w_path: Path,
    batch_size: int,
    mode_def: ModeDefinition,
    roi_layer=None,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        w_reader = None
        try:
            from pyflowreg.util.io.factory import get_video_file_reader

            video = video_layer.data
            if video.ndim < 3:
                raise ValueError("Video must be at least 3D (T, H, W)")

            try:
                w_reader = get_video_file_reader(
                    w_path, buffer_size=batch_size, dataset_names=["u", "v"]
                )
            except Exception:
                w_reader = get_video_file_reader(w_path, buffer_size=batch_size)

            T_w = w_reader.frame_count
            H_w, W_w = w_reader.height, w_reader.width

            T_v, H_v, W_v = video.shape[:3]
            if (T_w != T_v) or (H_w != H_v) or (W_w != W_v):
                raise ValueError(
                    f"w.h5 dimensions ({T_w}x{H_w}x{W_w}) don't match video ({T_v}x{H_v}x{W_v})"
                )

            dispatcher = CombinedCallbackDispatcher()

            progress_tracker = ProgressTracker(
                total_frames=T_v,
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                seed_points = None
                for layer in viewer.layers:
                    try:
                        extracted = extract_seed_points_from_layer(layer)
                        if extracted is not None and len(extracted) > 0:
                            seed_points = extracted
                            break
                    except Exception:
                        continue

                if seed_points is not None and len(seed_points) > 0:
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=T_v,
                        H=H_v,
                        W=W_v,
                    )
                    dispatcher.add_callback(trajectory_tracker)

            analyzer = None
            if roi_layer is not None:
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    roi_layer, video.shape, target_layer=video_layer
                )
                if roi_masks and len(roi_masks) > 0:
                    analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                    dispatcher.add_callback(analyzer)

            for batch_start in range(0, T_v, batch_size):
                batch_end = min(batch_start + batch_size, T_v)
                w_batch = w_reader[slice(batch_start, batch_end)]
                dispatcher.w_callback(w_batch, batch_start, batch_end)
                if trajectory_tracker:
                    progress_signaler.tracks_updated.emit(trajectory_tracker.get_tracks_for_napari())
                if analyzer:
                    progress_signaler.roi_updated.emit(analyzer.get_summary_statistics())

            tracks = None
            if trajectory_tracker:
                tracks = trajectory_tracker.get_tracks_for_napari()

            roi_stats = None
            if analyzer:
                roi_stats = analyzer.get_summary_statistics()

            return {"w": None, "roi_stats": roi_stats, "tracks": tracks}

        except Exception as e:
            show_error(f"Hybrid processing failed: {str(e)}")
            raise
        finally:
            if w_reader is not None:
                try:
                    w_reader.close()
                except Exception:
                    pass

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_hybrid_process_worker(widget, mode_def: ModeDefinition):
    video_layer_name = widget.video_combo.currentText() if hasattr(widget, 'video_combo') else widget.flow_combo.currentText()
    if not video_layer_name:
        raise ValueError("No video layer selected")

    video_layer = widget.viewer.layers[video_layer_name]

    w_path = widget.batch_processing_group.load_w_path_edit.text() if hasattr(widget, 'batch_processing_group') else None
    if not w_path:
        raise ValueError("No w.h5 file path specified")

    batch_size = 50
    if hasattr(widget, 'batch_processing_group'):
        batch_size = widget.batch_processing_group.batch_size_spin.value()

    return _build_hybrid_process_worker(
        video_layer=video_layer,
        viewer=widget.viewer,
        w_path=Path(w_path),
        batch_size=batch_size,
        mode_def=mode_def,
        roi_layer=getattr(widget, "current_roi_layer", None),
    )


def _build_precomputed_analysis_worker(
    w_path: Path,
    batch_size: int,
    viewer,
    mode_def: ModeDefinition,
    roi_layer=None,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        w_reader = None
        try:
            from pyflowreg.util.io.factory import get_video_file_reader

            try:
                w_reader = get_video_file_reader(
                    w_path, buffer_size=batch_size, dataset_names=["u", "v"]
                )
            except Exception:
                w_reader = get_video_file_reader(w_path, buffer_size=batch_size)

            T = w_reader.frame_count
            H, W = w_reader.height, w_reader.width

            dispatcher = CombinedCallbackDispatcher()

            progress_tracker = ProgressTracker(
                total_frames=T,
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                seed_points = None
                for layer in viewer.layers:
                    try:
                        extracted = extract_seed_points_from_layer(layer)
                        if extracted is not None and len(extracted) > 0:
                            seed_points = extracted
                            break
                    except Exception:
                        continue

                if seed_points is not None and len(seed_points) > 0:
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=T,
                        H=H,
                        W=W,
                    )
                    dispatcher.add_callback(trajectory_tracker)

            analyzer = None
            if roi_layer is not None:
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    roi_layer, (T, H, W)
                )
                if roi_masks and len(roi_masks) > 0:
                    analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                    dispatcher.add_callback(analyzer)

            for batch_start in range(0, T, batch_size):
                batch_end = min(batch_start + batch_size, T)
                w_batch = w_reader[slice(batch_start, batch_end)]
                dispatcher.w_callback(w_batch, batch_start, batch_end)
                if trajectory_tracker:
                    progress_signaler.tracks_updated.emit(trajectory_tracker.get_tracks_for_napari())
                if analyzer:
                    progress_signaler.roi_updated.emit(analyzer.get_summary_statistics())

            tracks = None
            if trajectory_tracker:
                tracks = trajectory_tracker.get_tracks_for_napari()

            roi_stats = None
            if analyzer:
                roi_stats = analyzer.get_summary_statistics()

            return {"w": None, "roi_stats": roi_stats, "tracks": tracks}

        except Exception as e:
            show_error(f"Precomputed analysis failed: {str(e)}")
            raise
        finally:
            if w_reader is not None:
                try:
                    w_reader.close()
                except Exception:
                    pass

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_precomputed_analysis_worker(widget, mode_def: ModeDefinition):
    w_path = None
    if hasattr(widget, 'batch_processing_group'):
        w_path = widget.batch_processing_group.load_w_path_edit.text()

    if not w_path:
        raise ValueError("No w.h5 file path specified")

    batch_size = 50
    if hasattr(widget, 'batch_processing_group'):
        batch_size = widget.batch_processing_group.batch_size_spin.value()

    return _build_precomputed_analysis_worker(
        Path(w_path),
        batch_size,
        widget.viewer,
        mode_def,
        roi_layer=getattr(widget, "current_roi_layer", None),
    )


def create_worker_for_widget_v2(widget, mode_def: ModeDefinition, run_options: Dict[str, Any]):
    """
    Build a thread worker for the wizard-based widget.

    Args:
        widget: MotionAnalysisWidgetV2 instance
        mode_def: Selected mode definition
        run_options: Dictionary from RunOptionsDialog.get_options()
    """
    mode = mode_def.mode

    batch_size = run_options.get("batch_size", mode_def.processing.batch_size_default)
    workers = run_options.get("workers", -1)

    enable_traj = bool(run_options.get("enable_trajectories", True))
    enable_roi = bool(run_options.get("enable_roi_stats", True))
    roi_layer = getattr(widget, "current_roi_layer", None) if enable_roi else None
    seed_layer = getattr(widget, "current_seed_layer", None) if enable_traj else None

    flow_params = run_options.get("flow_params")
    save_w = bool(run_options.get("save_w", False))
    w_dir_value = run_options.get("w_dir")
    w_dir = Path(w_dir_value) if w_dir_value else None
    export_w_viewer = bool(run_options.get("export_w_viewer", run_options.get("load_w", False)))
    output_choice = getattr(widget.wizard, "output_choice", "analyze")
    return_corrected = output_choice in {"save", "load"}

    if mode == ProcessingMode.LOADED_BATCH_PROCESS:
        video_layer = getattr(widget, "current_video_layer", None)
        if video_layer is None:
            raise ValueError("Select a video layer before processing")
        return _build_loaded_batch_worker(
            video_layer=video_layer,
            viewer=widget.viewer,
            mode_def=mode_def,
            batch_size=batch_size,
            save_w=save_w,
            w_dir=w_dir,
            load_w=export_w_viewer,
            roi_layer=roi_layer,
            seed_layer=seed_layer,
            flow_params=flow_params,
            return_corrected=return_corrected,
        )

    if mode == ProcessingMode.LOADED_WITH_PRECOMPUTED:
        video_layer = getattr(widget, "current_video_layer", None)
        if video_layer is None:
            raise ValueError("Select a video layer before processing")
        w_path = run_options.get("load_w_path")
        if not w_path:
            raise ValueError("Select a displacement field (w.h5) file")
        return _build_hybrid_process_worker(
            video_layer=video_layer,
            viewer=widget.viewer,
            w_path=Path(w_path),
            batch_size=batch_size,
            mode_def=mode_def,
            roi_layer=roi_layer,
        )

    if mode == ProcessingMode.FILE_FULL_PIPELINE:
        video_path = run_options.get("video_path")
        corrected_path = run_options.get("corrected_path")
        if not video_path or not corrected_path:
            raise ValueError("Video file and corrected output path are required")
        return _build_file_full_pipeline_worker(
            video_path=Path(video_path),
            corrected_path=Path(corrected_path),
            w_output_dir=w_dir,
            save_w=save_w,
            batch_size=batch_size,
            workers=workers,
            flow_params=flow_params,
        )

    if mode == ProcessingMode.FILE_TO_MEMORY:
        video_path = run_options.get("video_path")
        if not video_path:
            raise ValueError("Video file is required")
        return _build_file_to_memory_worker(
            video_path=Path(video_path),
            batch_size=batch_size,
            workers=workers,
            flow_params=flow_params,
            load_w=export_w_viewer,
        )

    if mode == ProcessingMode.FILE_WITH_PRECOMPUTED:
        video_path = run_options.get("video_path")
        w_path = run_options.get("load_w_path")
        corrected_path = run_options.get("corrected_path")
        if not all([video_path, w_path, corrected_path]):
            raise ValueError("Video file, w.h5 file, and corrected output path are required")
        return _build_file_with_precomputed_worker(
            video_path=Path(video_path),
            w_path=Path(w_path),
            corrected_path=Path(corrected_path),
            batch_size=batch_size,
        )

    if mode == ProcessingMode.ANALYSIS_STREAMING:
        video_path = run_options.get("video_path")
        if not video_path:
            raise ValueError("Video file is required")
        return _build_analysis_streaming_worker(
            video_path=Path(video_path),
            batch_size=batch_size,
            workers=workers,
            viewer=widget.viewer,
            mode_def=mode_def,
            roi_layer=roi_layer,
            selected_seed_layer=seed_layer,
            save_w=save_w,
            save_w_path=w_dir,
            load_w=export_w_viewer,
            flow_params=flow_params,
        )

    if mode == ProcessingMode.ANALYSIS_PRECOMPUTED:
        w_path = run_options.get("load_w_path")
        if not w_path:
            raise ValueError("Select a displacement field (w.h5) file")
        return _build_precomputed_analysis_worker(
            w_path=Path(w_path),
            batch_size=batch_size,
            viewer=widget.viewer,
            mode_def=mode_def,
            roi_layer=roi_layer,
        )

    return None


def _build_file_full_pipeline_worker(
    video_path: Path,
    corrected_path: Path,
    w_output_dir: Optional[Path],
    save_w: bool,
    batch_size: int,
    workers: int,
    flow_params: Optional[Dict[str, Any]] = None,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.motion_correction.OF_options import (
                OFOptions,
                OutputFormat,
                QualitySetting,
            )
            from pyflowreg.motion_correction.compensate_recording import (
                BatchMotionCorrector,
                RegistrationConfig,
            )

            corrected_path.parent.mkdir(parents=True, exist_ok=True)

            ext = corrected_path.suffix.lower()
            if ext in (".h5", ".hdf5"):
                output_format = OutputFormat.HDF5
            elif ext in (".tif", ".tiff"):
                output_format = OutputFormat.TIFF
            else:
                raise ValueError("Corrected output must be .h5/.hdf5 or .tif/.tiff")

            params = _normalize_flow_params(flow_params)
            params.update(
                {
                    "input_file": video_path,
                    "output_path": corrected_path.parent,
                    "output_file_name": corrected_path.name,
                    "output_format": output_format,
                    "buffer_size": batch_size,
                    "save_w": bool(save_w),
                }
            )
            params.setdefault("quality_setting", QualitySetting.BALANCED)

            options = OFOptions(**params)

            config = RegistrationConfig(
                n_jobs=workers if workers and workers > 0 else -1,
            )

            pipeline = BatchMotionCorrector(options, config)
            pipeline.register_progress_callback(
                lambda current, total: progress_signaler.progress_updated.emit(
                    current, total or 0
                )
            )

            pipeline.run()

            w_path_result = None
            if save_w:
                produced_w_path = Path(options.output_path) / "w.h5"
                if not produced_w_path.exists():
                    show_info("save_w was requested but the pipeline did not produce w.h5")
                elif w_output_dir:
                    w_output_dir.mkdir(parents=True, exist_ok=True)
                    target = w_output_dir / produced_w_path.name
                    if target != produced_w_path:
                        if target.exists():
                            target.unlink()
                        # shutil.move works across drives, unlike Path.replace
                        shutil.move(str(produced_w_path), str(target))
                    w_path_result = str(target)
                else:
                    w_path_result = str(produced_w_path)

            corrected_file = Path(options.output_path) / options.output_file_name if getattr(options, "output_file_name", None) else corrected_path
            return {
                "corrected_path": str(corrected_file),
                "w_path": w_path_result,
            }

        except Exception as e:
            show_error(f"File pipeline failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_full_pipeline_worker(widget, mode_def: ModeDefinition):
    """
    FILE_FULL_PIPELINE: Open video from file, compute w, save corrected + w to disk.
    No layers added to viewer - only notify paths.
    """
    video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
    if not video_path:
        raise ValueError("No video file path specified")

    corrected_output_path = widget.batch_processing_group.corrected_path_edit.text() if hasattr(widget.batch_processing_group, 'corrected_path_edit') else None
    if not corrected_output_path:
        raise ValueError("Corrected video output path is required")

    save_w = widget.batch_processing_group.save_w_check.isChecked()
    desired_w_path_text = widget.batch_processing_group.w_path_edit.text() if save_w else None

    batch_size = widget.batch_processing_group.batch_size_spin.value()
    workers = widget.batch_processing_group.workers_spin.value()

    video_path = Path(video_path)
    corrected_path = Path(corrected_output_path)
    w_output_dir = None
    if desired_w_path_text:
        desired_path = Path(desired_w_path_text)
        w_output_dir = desired_path if desired_path.is_dir() else desired_path.parent

    return _build_file_full_pipeline_worker(
        video_path,
        corrected_path,
        w_output_dir,
        save_w,
        batch_size,
        workers,
        None,
    )


def _build_file_to_memory_worker(
    video_path: Path,
    batch_size: int,
    workers: int,
    flow_params: Optional[Dict[str, Any]] = None,
    load_w: bool = False,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            import tempfile
            from pyflowreg.motion_correction.compensate_recording import (
                BatchMotionCorrector,
                RegistrationConfig,
            )
            from pyflowreg.motion_correction.OF_options import (
                OFOptions,
                OutputFormat,
                QualitySetting,
            )

            with tempfile.TemporaryDirectory(prefix="flowreg_") as tmp_dir:
                params = _normalize_flow_params(flow_params)
                params.update(
                    {
                        "input_file": video_path,
                        "output_path": Path(tmp_dir),
                        "output_format": OutputFormat.ARRAY,
                        "buffer_size": batch_size,
                        "save_w": True,
                    }
                )
                params.setdefault("quality_setting", QualitySetting.BALANCED)

                options = OFOptions(**params)
                options.save_meta_info = False

                config = RegistrationConfig(
                    n_jobs=workers if workers and workers > 0 else -1,
                )

                pipeline = BatchMotionCorrector(options, config)
                pipeline.register_progress_callback(
                    lambda current, total: progress_signaler.progress_updated.emit(
                        current, total or 0
                    )
                )

                pipeline.run()

                corrected = (
                    pipeline.video_writer.get_array()
                    if pipeline.video_writer is not None
                    else None
                )
                w = (
                    pipeline.w_writer.get_array()
                    if hasattr(pipeline, "w_writer") and pipeline.w_writer is not None
                    else None
                )

                return {"corrected": corrected, "w": w if load_w else None}

        except Exception as e:
            show_error(f"File to memory failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_to_memory_worker(widget, mode_def: ModeDefinition):
    """
    FILE_TO_MEMORY: Process file and load corrected + w to viewer.
    Shows RAM warning if mode_def.requires_confirmation.
    """
    video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
    if not video_path:
        raise ValueError("No video file path specified")

    batch_size = widget.batch_processing_group.batch_size_spin.value()
    workers = widget.batch_processing_group.workers_spin.value()

    return _build_file_to_memory_worker(Path(video_path), batch_size, workers, None, True)


def _build_file_with_precomputed_worker(
    video_path: Path,
    w_path: Path,
    corrected_path: Path,
    batch_size: int,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        video_reader = None
        w_reader = None
        writer = None
        try:
            from pyflowreg.util.io.factory import (
                get_video_file_reader,
                get_video_file_writer,
            )
            from pyflowreg.motion_correction.apply_displacement import apply_displacement_field

            corrected_path.parent.mkdir(parents=True, exist_ok=True)

            video_reader = get_video_file_reader(video_path, buffer_size=batch_size)

            try:
                w_reader = get_video_file_reader(
                    w_path, buffer_size=batch_size, dataset_names=["u", "v"]
                )
            except Exception:
                w_reader = get_video_file_reader(w_path, buffer_size=batch_size)

            ext = corrected_path.suffix.lower()
            if ext in (".h5", ".hdf5"):
                writer = get_video_file_writer(str(corrected_path), "HDF5")
            elif ext in (".tif", ".tiff"):
                writer = get_video_file_writer(str(corrected_path), "TIFF")
            else:
                raise ValueError("Corrected output must be .h5/.hdf5 or .tif/.tiff")

            total_frames = video_reader.frame_count

            if (
                total_frames != w_reader.frame_count
                or video_reader.height != w_reader.height
                or video_reader.width != w_reader.width
            ):
                raise ValueError(
                    f"Dimension mismatch: video {(total_frames, video_reader.height, video_reader.width)} "
                    f"vs w {(w_reader.frame_count, w_reader.height, w_reader.width)}"
                )

            # Process and save corrected video in batches
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                frame_slice = slice(batch_start, batch_end)

                video_batch = video_reader[frame_slice]
                w_batch = w_reader[frame_slice]

                if video_batch.ndim == 3:
                    video_batch = video_batch[..., np.newaxis]
                if w_batch.ndim == 3:
                    w_batch = w_batch[..., np.newaxis]

                corrected_batch = np.zeros_like(video_batch)
                for idx in range(batch_end - batch_start):
                    corrected_batch[idx] = apply_displacement_field(
                        video_batch[idx], w_batch[idx]
                    )

                # Writer expects (T,H,W,C)
                writer.write_frames(corrected_batch)

                progress_signaler.progress_updated.emit(batch_end, total_frames)

            return {"corrected_path": str(corrected_path)}

        except Exception as e:
            show_error(f"File with precomputed failed: {str(e)}")
            raise
        finally:
            # Close the writer first so the corrected file is flushed.
            for handle in (writer, w_reader, video_reader):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_with_precomputed_worker(widget, mode_def: ModeDefinition):
    """
    FILE_WITH_PRECOMPUTED: Open video + w.h5 from file, apply correction, save corrected.
    No layers added.
    """
    video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
    w_path = widget.batch_processing_group.load_w_path_edit.text()
    corrected_output_path = widget.batch_processing_group.corrected_path_edit.text() if hasattr(widget.batch_processing_group, 'corrected_path_edit') else None

    if not video_path or not w_path or not corrected_output_path:
        raise ValueError("Video path, w.h5 path, and output path are required")

    batch_size = widget.batch_processing_group.batch_size_spin.value()

    return _build_file_with_precomputed_worker(
        Path(video_path), Path(w_path), Path(corrected_output_path), batch_size
    )


def _build_analysis_streaming_worker(
    video_path: Path,
    batch_size: int,
    workers: int,
    viewer,
    mode_def: ModeDefinition,
    roi_layer=None,
    selected_seed_layer=None,
    save_w: bool = False,
    save_w_path: Optional[Path] = None,
    load_w: bool = False,
    flow_params: Optional[Dict[str, Any]] = None,
):
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        temp_output_dir: Optional[Path] = None
        try:
            import tempfile
            from pyflowreg.motion_correction.compensate_recording import (
                BatchMotionCorrector,
                RegistrationConfig,
            )
            from pyflowreg.motion_correction.OF_options import (
                OFOptions,
                OutputFormat,
                QualitySetting,
            )
            from pyflowreg.util.io.factory import get_video_file_reader

            reader = get_video_file_reader(video_path, buffer_size=batch_size)
            try:
                total_frames = reader.frame_count
                height, width = reader.height, reader.width
            finally:
                reader.close()

            if save_w:
                if save_w_path:
                    output_dir = Path(save_w_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = video_path.parent
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="flowreg_stream_"))
                temp_output_dir = output_dir

            params = _normalize_flow_params(flow_params)
            params.update(
                {
                    "input_file": video_path,
                    "output_path": output_dir,
                    "output_format": OutputFormat.NULL,
                    "buffer_size": batch_size,
                    "save_w": bool(save_w),
                }
            )
            params.setdefault("quality_setting", QualitySetting.BALANCED)

            options = OFOptions(**params)
            options.save_meta_info = False

            config = RegistrationConfig(
                n_jobs=workers if workers and workers > 0 else -1,
            )

            pipeline = BatchMotionCorrector(options, config)
            pipeline.register_progress_callback(
                lambda current, total: progress_signaler.progress_updated.emit(
                    current, total or total_frames
                )
            )

            dispatcher = CombinedCallbackDispatcher()
            w_accumulator = None
            if load_w:
                w_accumulator = WAccumulator((total_frames, height, width, 2))
                dispatcher.add_callback(w_accumulator)

            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                seed_points = None
                if selected_seed_layer is not None:
                    seed_points = extract_seed_points_from_layer(selected_seed_layer)
                else:
                    for layer in viewer.layers:
                        try:
                            extracted = extract_seed_points_from_layer(layer)
                            if extracted is not None and len(extracted) > 0:
                                seed_points = extracted
                                break
                        except Exception:
                            continue

                if seed_points is not None and len(seed_points) > 0:
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=total_frames,
                        H=height,
                        W=width
                    )
                    dispatcher.add_callback(trajectory_tracker)

            analyzer = None
            if roi_layer is not None:
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    roi_layer, (total_frames, height, width)
                )
                if roi_masks and len(roi_masks) > 0:
                    analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                    dispatcher.add_callback(analyzer)

            if dispatcher.callbacks:
                def wrapped_callback(w_batch, batch_start, batch_end):
                    dispatcher.w_callback(w_batch, batch_start, batch_end)
                    if trajectory_tracker:
                        tracks_partial = trajectory_tracker.get_tracks_for_napari()
                        progress_signaler.tracks_updated.emit(tracks_partial)
                    if analyzer:
                        progress_signaler.roi_updated.emit(analyzer.get_summary_statistics())

                pipeline.register_w_callback(wrapped_callback)

            pipeline.run()

            # options.output_path already is save_w_path (or the video's folder),
            # so the pipeline writes w.h5 directly to its final destination.
            produced_w_path = options.output_path / "w.h5"
            w_path_result = None
            if save_w:
                if produced_w_path.exists():
                    w_path_result = str(produced_w_path)
                else:
                    show_info("save_w was requested but the pipeline did not produce w.h5")

            w_result = None
            if w_accumulator is not None:
                w_result = w_accumulator.get_accumulated()
            if not load_w:
                w_result = None

            tracks = trajectory_tracker.get_tracks_for_napari() if trajectory_tracker else None
            roi_stats = analyzer.get_summary_statistics() if analyzer else None

            return {'tracks': tracks, 'roi_stats': roi_stats, 'w_path': w_path_result, 'w': w_result}

        except Exception as e:
            show_error(f"Analysis streaming failed: {str(e)}")
            raise
        finally:
            if temp_output_dir is not None:
                try:
                    if temp_output_dir.exists():
                        for child in temp_output_dir.iterdir():
                            child.unlink(missing_ok=True)
                        temp_output_dir.rmdir()
                except Exception:
                    pass

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_analysis_streaming_worker(widget, mode_def: ModeDefinition):
    video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
    if not video_path:
        raise ValueError("No video file path specified")

    batch_size = widget.batch_processing_group.batch_size_spin.value()
    workers = widget.batch_processing_group.workers_spin.value()
    save_w = widget.batch_processing_group.save_w_check.isChecked() if hasattr(widget.batch_processing_group, 'save_w_check') else False
    save_w_path_text = widget.batch_processing_group.w_path_edit.text() if save_w and hasattr(widget.batch_processing_group, 'w_path_edit') else None
    save_w_path = Path(save_w_path_text) if save_w_path_text else None
    load_w = widget.batch_processing_group.load_w_check.isChecked() if hasattr(widget.batch_processing_group, 'load_w_check') else False

    return _build_analysis_streaming_worker(
        Path(video_path),
        batch_size,
        workers,
        widget.viewer,
        mode_def,
        roi_layer=getattr(widget, "current_roi_layer", None),
        selected_seed_layer=getattr(widget, "current_seed_layer", None),
        save_w=save_w,
        save_w_path=save_w_path,
        load_w=load_w,
        flow_params=None,
    )
def integrate_batch_processing(widget):
    """
    Main integration function to be called on existing MotionAnalysisWidget.
    This modifies the widget to add batch processing capabilities.

    Args:
        widget: The existing MotionAnalysisWidget instance
    """
    # Add mode selector at the top
    mode_group = create_mode_selector_group()
    widget.layout().insertWidget(0, mode_group)

    # Store references
    widget.mode_combo = mode_group.mode_combo
    widget.mode_info = mode_group.mode_info
    widget.memory_label = mode_group.memory_label

    # Add batch processing options (initially hidden)
    batch_group = add_batch_processing_options(widget.layout(), 2)
    widget.layout().insertWidget(3, batch_group)
    widget.batch_processing_group = batch_group

    # Connect mode change signal
    widget.mode_combo.currentIndexChanged.connect(
        lambda idx: handle_mode_change(widget, idx)
    )

    # Initialize mode
    widget.current_processing_mode = 0

    # Modify analyze button behavior
    original_analyze = widget._on_analyze_clicked

    def new_analyze_clicked():
        """Extended analyze function that handles different modes."""
        if widget.current_processing_mode == 0:
            # Original behavior for online mode
            original_analyze()
        else:
            # Get current mode definition
            mode_names = _mode_manager.get_mode_names()
            if widget.current_processing_mode < len(mode_names):
                mode_name = mode_names[widget.current_processing_mode]
                mode_def = _mode_manager.get_mode_by_name(mode_name)

                # Check if mode requires confirmation
                if mode_def and mode_def.requires_confirmation:
                    from qtpy.QtWidgets import QMessageBox
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("High Memory Usage Warning")
                    msg.setInformativeText(mode_def.warning_message or "This operation will load large data into memory.")
                    msg.setWindowTitle("Confirm Operation")
                    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

                    if msg.exec_() != QMessageBox.Ok:
                        return  # User cancelled

            # Batch processing modes
            worker = create_batch_processor_for_widget(widget)
            if worker:
                # Connect worker signals
                def on_complete(results):
                    if results.get('w') is not None:
                        # Add w to viewer
                        widget.viewer.add_image(
                            results['w'],
                            name="Computed w",
                            colormap="turbo"
                        )
                        # Set as current flow layer
                        widget.current_flow_layer = widget.viewer.layers[-1]
                        # Now run original analyze
                        original_analyze()

                    if results.get('roi_stats'):
                        show_info("ROI statistics computed")

                worker.returned.connect(on_complete)
                worker.errored.connect(lambda e: show_error(f"Processing failed: {e}"))
                worker.start()

    # Replace the analyze function
    widget.analyze_button.clicked.disconnect()
    widget.analyze_button.clicked.connect(new_analyze_clicked)
