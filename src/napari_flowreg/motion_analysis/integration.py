"""
Integration module for adding batch processing modes to motion_analysis_widget.
This module contains the functions and classes that will be integrated into
the existing motion_analysis_widget.py to add new processing modes.
"""

import os
import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
import warnings

from qtpy.QtWidgets import QComboBox, QLabel, QGroupBox, QVBoxLayout, QSpinBox, QGridLayout, QCheckBox, QLineEdit, QPushButton
from qtpy.QtCore import Qt, Signal, QObject
from napari.utils.notifications import show_info, show_error
from napari.qt.threading import thread_worker

from .modes import ProcessingMode, ModeManager, ModeDefinition
from .callbacks import (
    BatchTrajectoryTracker,
    ROIMotionAnalyzer,
    CombinedCallbackDispatcher,
    WAccumulator,
    ProgressTracker,
    extract_roi_masks_from_shapes,
    extract_seed_points_from_layer
)

# Initialize mode manager
_mode_manager = ModeManager()


class ProgressSignaler(QObject):
    """Helper class to emit progress signals from worker threads."""
    progress_updated = Signal(int, int)  # (current, total)


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


def create_batch_process_worker(widget, mode_def: ModeDefinition):
    """
    Create worker for batch processing loaded video.
    Uses mode_def to configure processing behavior.
    """
    # Create progress signaler outside worker so it can be connected before start
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        # Ensure NUMBA compatibility
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.motion_correction.compensate_arr import compensate_arr
            from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat, QualitySetting

            # Get video layer (instead of flow layer)
            video_layer_name = widget.video_combo.currentText() if hasattr(widget, 'video_combo') else widget.flow_combo.currentText()
            if not video_layer_name:
                raise ValueError("No video layer selected")

            video_layer = widget.viewer.layers[video_layer_name]
            video = video_layer.data

            if video.ndim < 3:
                raise ValueError("Video must be at least 3D (T, H, W)")

            # Get reference frame (default to first frame)
            ref_idx = 0
            reference = video[ref_idx] if video.ndim >= 3 else video

            # Get batch processing options
            batch_size = 50
            workers = 4
            save_w_path = None
            load_w = False

            if hasattr(widget, 'batch_processing_group'):
                batch_size = widget.batch_processing_group.batch_size_spin.value()
                workers = widget.batch_processing_group.workers_spin.value()
                if widget.batch_processing_group.save_w_check.isChecked():
                    save_w_path = widget.batch_processing_group.w_path_edit.text()
                load_w = widget.batch_processing_group.load_w_check.isChecked()

            # Determine what user wants
            want_load_w = bool(load_w)        # from the "Load w to viewer" checkbox
            want_save_w = bool(save_w_path)   # from "Save w to disk"

            # Get ROIs for analysis
            roi_masks = None
            roi_names = None
            if widget.current_roi_layer:
                # Pass target_layer for transform-aware ROI extraction
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    widget.current_roi_layer, video.shape, target_layer=video_layer
                )

            # Build options
            # Note: parallelization uses default (all cores) since compensate_arr doesn't accept config parameter
            options = OFOptions(
                quality_setting=QualitySetting.BALANCED,
                buffer_size=batch_size
            )

            # Setup callbacks
            dispatcher = CombinedCallbackDispatcher()

            # Get total frames for progress tracking
            T = video.shape[0] if video.ndim >= 3 else 1

            # Progress tracker for UI updates (uses progress_signaler from outer scope)
            progress_tracker = ProgressTracker(
                total_frames=T,
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            # Trajectory tracker (if seed points exist)
            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                # Try to extract seed points from Points or Shapes layers
                seed_points = None
                for layer in widget.viewer.layers:
                    try:
                        extracted = extract_seed_points_from_layer(layer)
                        if extracted is not None and len(extracted) > 0:
                            seed_points = extracted
                            break
                    except:
                        continue

                if seed_points is not None:
                    H, W = video.shape[1:3] if video.ndim >= 3 else video.shape[:2]
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=T,
                        H=H,
                        W=W
                    )
                    dispatcher.add_callback(trajectory_tracker)

            # ROI analyzer
            analyzer = None
            if roi_masks and len(roi_masks) > 0:
                analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                dispatcher.add_callback(analyzer)

            # W accumulator only if user wants to load w to viewer
            w_accumulator = None
            if want_load_w:
                # Create shape based on video dimensions
                if video.ndim == 4:
                    shape = (*video.shape[:-1], 2)
                else:
                    shape = (*video.shape, 2)
                w_accumulator = WAccumulator(shape)
                dispatcher.add_callback(w_accumulator)

            # Configure output
            options.save_w = want_save_w
            options.output_format = OutputFormat.NULL  # Don't save corrected

            # Run processing
            corrected, w = compensate_arr(
                video,
                reference,
                options,
                w_callback=dispatcher.w_callback if len(dispatcher.callbacks) > 0 else None
            )

            # Only return w if user asked to load it
            if not (want_load_w or want_save_w):
                # User didn't ask to save or load -> don't return huge arrays
                w = None
            elif want_load_w and w_accumulator:
                # User wants w in viewer -> get from accumulator
                w = w_accumulator.get_accumulated()
            elif not want_load_w:
                # User only wants to save, not load -> don't return
                w = None

            # Also return ROI statistics if computed
            roi_stats = None
            if roi_masks and hasattr(analyzer, 'get_summary_statistics'):
                roi_stats = analyzer.get_summary_statistics()

            # Return trajectory tracks if computed
            tracks = None
            if trajectory_tracker:
                tracks = trajectory_tracker.get_tracks_for_napari()

            # Only return w if user requested to load it
            return {'w': w if want_load_w else None, 'roi_stats': roi_stats, 'tracks': tracks}

        except Exception as e:
            show_error(f"Batch processing failed: {str(e)}")
            raise

    worker = process()
    # Attach progress_signaler to worker for connection in widget
    worker.progress_signaler = progress_signaler
    return worker


def create_hybrid_process_worker(widget, mode_def: ModeDefinition):
    """
    Create worker for hybrid mode (video in memory, w from file).
    Uses mode_def to configure processing behavior.
    """
    # Create progress signaler outside worker
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        try:
            import h5py

            # Get video layer
            video_layer_name = widget.video_combo.currentText() if hasattr(widget, 'video_combo') else widget.flow_combo.currentText()
            if not video_layer_name:
                raise ValueError("No video layer selected")

            video_layer = widget.viewer.layers[video_layer_name]
            video = video_layer.data

            if video.ndim < 3:
                raise ValueError("Video must be at least 3D (T, H, W)")

            # Get w file path
            w_path = None
            if hasattr(widget, 'batch_processing_group'):
                w_path = widget.batch_processing_group.load_w_path_edit.text()

            if not w_path:
                raise ValueError("No w.h5 file path specified")

            # Get batch size
            batch_size = 50
            if hasattr(widget, 'batch_processing_group'):
                batch_size = widget.batch_processing_group.batch_size_spin.value()

            # Open HDF5 file and read w dataset
            with h5py.File(w_path, 'r') as f:
                # Typical pyflowreg w.h5 structure: dataset 'w' with shape (T, H, W, 2)
                if 'w' not in f:
                    raise ValueError(f"w.h5 file does not contain 'w' dataset. Available: {list(f.keys())}")

                w_dataset = f['w']
                T_w, H_w, W_w, _ = w_dataset.shape

                # Validate dimensions match video
                T_v, H_v, W_v = video.shape[:3]
                if (T_w != T_v) or (H_w != H_v) or (W_w != W_v):
                    raise ValueError(f"w.h5 dimensions ({T_w}x{H_w}x{W_w}) don't match video ({T_v}x{H_v}x{W_v})")

                # Setup callbacks
                dispatcher = CombinedCallbackDispatcher()

                # Progress tracker
                progress_tracker = ProgressTracker(
                    total_frames=T_v,
                    progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
                )
                dispatcher.add_callback(progress_tracker)

                # Trajectory tracker (if seed points exist)
                trajectory_tracker = None
                if mode_def.processing.trajectory_callback:
                    # Try to extract seed points from Points or Shapes layers
                    seed_points = None
                    for layer in widget.viewer.layers:
                        try:
                            extracted = extract_seed_points_from_layer(layer)
                            if extracted is not None and len(extracted) > 0:
                                seed_points = extracted
                                break
                        except:
                            continue

                    if seed_points is not None:
                        trajectory_tracker = BatchTrajectoryTracker(
                            seed_points=seed_points,
                            total_frames=T_v,
                            H=H_v,
                            W=W_v
                        )
                        dispatcher.add_callback(trajectory_tracker)

                # ROI analyzer with video layer for transform-aware masks
                analyzer = None
                roi_masks = None
                roi_names = None
                if widget.current_roi_layer:
                    roi_masks, roi_names = extract_roi_masks_from_shapes(
                        widget.current_roi_layer, video.shape, target_layer=video_layer
                    )
                    if roi_masks and len(roi_masks) > 0:
                        analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                        dispatcher.add_callback(analyzer)

                # Process w in batches
                for batch_start in range(0, T_v, batch_size):
                    batch_end = min(batch_start + batch_size, T_v)

                    # Read batch from HDF5
                    w_batch = w_dataset[batch_start:batch_end, :, :, :]

                    # Dispatch to callbacks
                    dispatcher.w_callback(w_batch, batch_start, batch_end)

                # Return trajectory tracks if computed
                tracks = None
                if trajectory_tracker:
                    tracks = trajectory_tracker.get_tracks_for_napari()

                # Return ROI statistics if computed
                roi_stats = None
                if analyzer:
                    roi_stats = analyzer.get_summary_statistics()

                return {'w': None, 'roi_stats': roi_stats, 'tracks': tracks}

        except Exception as e:
            show_error(f"Hybrid processing failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_precomputed_analysis_worker(widget, mode_def: ModeDefinition):
    """
    Create worker for analyzing pre-computed w.h5 file without video.
    This is mode 2a from requirements: analyze pre-computed w.h5.
    """
    # Create progress signaler outside worker
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        try:
            import h5py

            # Get w file path
            w_path = None
            if hasattr(widget, 'batch_processing_group'):
                w_path = widget.batch_processing_group.load_w_path_edit.text()

            if not w_path:
                raise ValueError("No w.h5 file path specified")

            # Get batch size
            batch_size = 50
            if hasattr(widget, 'batch_processing_group'):
                batch_size = widget.batch_processing_group.batch_size_spin.value()

            # Open HDF5 file and read w dataset
            with h5py.File(w_path, 'r') as f:
                # Typical pyflowreg w.h5 structure: dataset 'w' with shape (T, H, W, 2)
                if 'w' not in f:
                    raise ValueError(f"w.h5 file does not contain 'w' dataset. Available: {list(f.keys())}")

                w_dataset = f['w']
                T, H, W, _ = w_dataset.shape

                # Setup callbacks
                dispatcher = CombinedCallbackDispatcher()

                # Progress tracker
                progress_tracker = ProgressTracker(
                    total_frames=T,
                    progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
                )
                dispatcher.add_callback(progress_tracker)

                # Trajectory tracker (if seed points exist)
                trajectory_tracker = None
                if mode_def.processing.trajectory_callback:
                    # Try to extract seed points from Points or Shapes layers
                    seed_points = None
                    for layer in widget.viewer.layers:
                        try:
                            extracted = extract_seed_points_from_layer(layer)
                            if extracted is not None and len(extracted) > 0:
                                seed_points = extracted
                                break
                        except:
                            continue

                    if seed_points is not None:
                        trajectory_tracker = BatchTrajectoryTracker(
                            seed_points=seed_points,
                            total_frames=T,
                            H=H,
                            W=W
                        )
                        dispatcher.add_callback(trajectory_tracker)

                # ROI analyzer
                analyzer = None
                roi_masks = None
                roi_names = None
                if widget.current_roi_layer:
                    # Create dummy video shape for ROI extraction
                    video_shape = (T, H, W)
                    roi_masks, roi_names = extract_roi_masks_from_shapes(
                        widget.current_roi_layer, video_shape
                    )
                    if roi_masks and len(roi_masks) > 0:
                        analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                        dispatcher.add_callback(analyzer)

                # Process w in batches
                for batch_start in range(0, T, batch_size):
                    batch_end = min(batch_start + batch_size, T)

                    # Read batch from HDF5
                    w_batch = w_dataset[batch_start:batch_end, :, :, :]

                    # Dispatch to callbacks
                    dispatcher.w_callback(w_batch, batch_start, batch_end)

                # Return trajectory tracks if computed
                tracks = None
                if trajectory_tracker:
                    tracks = trajectory_tracker.get_tracks_for_napari()

                # Return ROI statistics if computed
                roi_stats = None
                if analyzer:
                    roi_stats = analyzer.get_summary_statistics()

                return {'w': None, 'roi_stats': roi_stats, 'tracks': tracks}

        except Exception as e:
            show_error(f"Precomputed analysis failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_full_pipeline_worker(widget, mode_def: ModeDefinition):
    """
    FILE_FULL_PIPELINE: Open video from file, compute w, save corrected + w to disk.
    No layers added to viewer - only notify paths.
    """
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.util.io import load_video
            from pyflowreg.motion_correction.compensate_recording import compensate_recording
            from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat, QualitySetting

            # Get file paths from batch UI
            # TODO: Add video file path input to batch UI
            video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
            if not video_path:
                raise ValueError("No video file path specified")

            w_output_path = widget.batch_processing_group.w_path_edit.text()
            corrected_output_path = widget.batch_processing_group.corrected_path_edit.text() if hasattr(widget.batch_processing_group, 'corrected_path_edit') else None

            if not w_output_path or not corrected_output_path:
                raise ValueError("Output paths for w and corrected video are required")

            # Get batch options
            batch_size = widget.batch_processing_group.batch_size_spin.value()
            workers = widget.batch_processing_group.workers_spin.value()

            # Load video from file
            video, metadata = load_video(video_path)

            # Get reference frame
            reference = video[0] if video.ndim >= 3 else video

            # Build options
            # Note: parallelization uses default (all cores) since compensate_recording doesn't accept config parameter
            options = OFOptions(
                quality_setting=QualitySetting.BALANCED,
                buffer_size=batch_size,
                save_w=True,
                w_output_path=w_output_path,
                output_format=OutputFormat.HDF5 if corrected_output_path.endswith('.h5') else OutputFormat.TIFF,
                output_path=corrected_output_path
            )

            # Progress callback
            progress_tracker = ProgressTracker(
                total_frames=video.shape[0],
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )

            # Run processing
            compensate_recording(
                video,
                reference,
                options,
                progress_callback=lambda c, t: progress_tracker.w_callback(None, 0, c)
            )

            return {'w_path': w_output_path, 'corrected_path': corrected_output_path}

        except Exception as e:
            show_error(f"File pipeline failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_to_memory_worker(widget, mode_def: ModeDefinition):
    """
    FILE_TO_MEMORY: Process file and load corrected + w to viewer.
    Shows RAM warning if mode_def.requires_confirmation.
    """
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.util.io import load_video
            from pyflowreg.motion_correction.compensate_arr import compensate_arr
            from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat, QualitySetting

            video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
            if not video_path:
                raise ValueError("No video file path specified")

            batch_size = widget.batch_processing_group.batch_size_spin.value()
            workers = widget.batch_processing_group.workers_spin.value()

            # Load video
            video, metadata = load_video(video_path)
            reference = video[0]

            # Build options
            # Note: parallelization uses default (all cores) since compensate_arr doesn't accept config parameter
            options = OFOptions(
                quality_setting=QualitySetting.BALANCED,
                buffer_size=batch_size,
                save_w=False,
                output_format=OutputFormat.NULL
            )

            # Setup progress + w accumulator
            dispatcher = CombinedCallbackDispatcher()

            progress_tracker = ProgressTracker(
                total_frames=video.shape[0],
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            # Accumulate w for loading
            w_shape = (*video.shape, 2) if video.ndim == 3 else (*video.shape[:-1], 2)
            w_accumulator = WAccumulator(w_shape)
            dispatcher.add_callback(w_accumulator)

            # Run processing
            corrected, _ = compensate_arr(
                video,
                reference,
                options,
                w_callback=dispatcher.w_callback
            )

            w = w_accumulator.get_accumulated()

            return {'corrected': corrected, 'w': w}

        except Exception as e:
            show_error(f"File to memory failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_file_with_precomputed_worker(widget, mode_def: ModeDefinition):
    """
    FILE_WITH_PRECOMPUTED: Open video + w.h5 from file, apply correction, save corrected.
    No layers added.
    """
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        try:
            import h5py
            from pyflowreg.util.io import load_video, save_video
            from pyflowreg.motion_correction.apply_displacement import apply_displacement_field

            video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
            w_path = widget.batch_processing_group.load_w_path_edit.text()
            corrected_output_path = widget.batch_processing_group.corrected_path_edit.text() if hasattr(widget.batch_processing_group, 'corrected_path_edit') else None

            if not video_path or not w_path or not corrected_output_path:
                raise ValueError("Video path, w.h5 path, and output path are required")

            batch_size = widget.batch_processing_group.batch_size_spin.value()

            # Load video
            video, metadata = load_video(video_path)

            # Open w.h5
            with h5py.File(w_path, 'r') as f:
                w_dataset = f['w']
                T, H, W, _ = w_dataset.shape

                if (T != video.shape[0]) or (H != video.shape[1]) or (W != video.shape[2]):
                    raise ValueError(f"Dimension mismatch: video {video.shape} vs w {w_dataset.shape}")

                # Process and save corrected video in batches
                corrected = np.zeros_like(video)

                for batch_start in range(0, T, batch_size):
                    batch_end = min(batch_start + batch_size, T)

                    # Load w batch
                    w_batch = w_dataset[batch_start:batch_end]

                    # Apply displacement
                    for i, t in enumerate(range(batch_start, batch_end)):
                        corrected[t] = apply_displacement_field(video[t], w_batch[i])

                    # Update progress
                    progress_signaler.progress_updated.emit(batch_end, T)

                # Save corrected video
                save_video(corrected_output_path, corrected, metadata=metadata)

            return {'corrected_path': corrected_output_path}

        except Exception as e:
            show_error(f"File with precomputed failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


def create_analysis_streaming_worker(widget, mode_def: ModeDefinition):
    """
    ANALYSIS_STREAMING: Process from file, compute trajectories/stats only.
    No corrected output, save_w optional, trajectories returned.
    """
    progress_signaler = ProgressSignaler()

    @thread_worker
    def process():
        if sys.platform == "win32":
            os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        try:
            from pyflowreg.util.io import load_video
            from pyflowreg.motion_correction.compensate_arr import compensate_arr
            from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat, QualitySetting

            video_path = widget.batch_processing_group.video_file_path_edit.text() if hasattr(widget.batch_processing_group, 'video_file_path_edit') else None
            if not video_path:
                raise ValueError("No video file path specified")

            batch_size = widget.batch_processing_group.batch_size_spin.value()
            workers = widget.batch_processing_group.workers_spin.value()
            save_w_path = widget.batch_processing_group.w_path_edit.text() if widget.batch_processing_group.save_w_check.isChecked() else None

            # Load video
            video, metadata = load_video(video_path)
            reference = video[0]

            # Build options
            # Note: parallelization uses default (all cores) since compensate_arr doesn't accept config parameter
            options = OFOptions(
                quality_setting=QualitySetting.BALANCED,
                buffer_size=batch_size,
                save_w=bool(save_w_path),
                w_output_path=save_w_path if save_w_path else None,
                output_format=OutputFormat.NULL  # No corrected output
            )

            # Setup callbacks
            dispatcher = CombinedCallbackDispatcher()

            T = video.shape[0]
            H, W = video.shape[1:3]

            # Progress tracker
            progress_tracker = ProgressTracker(
                total_frames=T,
                progress_callback=lambda c, t: progress_signaler.progress_updated.emit(c, t)
            )
            dispatcher.add_callback(progress_tracker)

            # Trajectory tracker
            trajectory_tracker = None
            if mode_def.processing.trajectory_callback:
                seed_points = None
                for layer in widget.viewer.layers:
                    try:
                        extracted = extract_seed_points_from_layer(layer)
                        if extracted is not None and len(extracted) > 0:
                            seed_points = extracted
                            break
                    except:
                        continue

                if seed_points is not None:
                    trajectory_tracker = BatchTrajectoryTracker(
                        seed_points=seed_points,
                        total_frames=T,
                        H=H,
                        W=W
                    )
                    dispatcher.add_callback(trajectory_tracker)

            # ROI analyzer
            analyzer = None
            if widget.current_roi_layer:
                roi_masks, roi_names = extract_roi_masks_from_shapes(
                    widget.current_roi_layer, video.shape
                )
                if roi_masks and len(roi_masks) > 0:
                    analyzer = ROIMotionAnalyzer(roi_masks, roi_names)
                    dispatcher.add_callback(analyzer)

            # Run processing
            _, _ = compensate_arr(
                video,
                reference,
                options,
                w_callback=dispatcher.w_callback if len(dispatcher.callbacks) > 0 else None
            )

            # Collect results
            tracks = trajectory_tracker.get_tracks_for_napari() if trajectory_tracker else None
            roi_stats = analyzer.get_summary_statistics() if analyzer else None

            return {'tracks': tracks, 'roi_stats': roi_stats, 'w_path': save_w_path}

        except Exception as e:
            show_error(f"Analysis streaming failed: {str(e)}")
            raise

    worker = process()
    worker.progress_signaler = progress_signaler
    return worker


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