"""
3-step wizard UI for motion analysis processing mode selection.

This module provides a cleaner, wizard-based interface for selecting:
1. Input source (Loaded layer | File)
2. Flow computation (Compute now | Use precomputed w.h5)
3. Output mode (Analyze only | Correct & save | Load to viewer)

The wizard internally maps these choices to the appropriate ProcessingMode.
"""

from typing import Optional, Tuple
from qtpy.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QDialog, QPushButton,
    QGridLayout, QSpinBox, QCheckBox, QLineEdit, QFileDialog
)
from qtpy.QtCore import Qt, Signal, QObject

from .modes import ProcessingMode, ModeManager


class WizardSignaler(QObject):
    """Signal emitter for wizard state changes."""
    state_changed = Signal()  # Emitted when wizard choices change


class ProcessingWizard(QGroupBox):
    """
    3-step wizard for processing mode selection.

    Provides a clean, step-by-step interface that internally maps to ProcessingMode.
    """

    def __init__(self, parent=None):
        super().__init__("Processing Wizard", parent)

        self.signaler = WizardSignaler()
        self._mode_manager = ModeManager()

        # Step states
        self.input_choice = "layer"  # "layer" or "file"
        self.flow_choice = "compute"  # "compute" or "precomputed"
        self.output_choice = "analyze"  # "analyze", "save", or "load"

        self._init_ui()

    def _init_ui(self):
        """Initialize the wizard UI."""
        layout = QVBoxLayout()

        # Step 1: Input Source
        step1_group = QGroupBox("Step 1 – Input")
        step1_layout = QHBoxLayout()

        self.input_button_group = QButtonGroup()
        self.input_layer_radio = QRadioButton("Loaded layer")
        self.input_file_radio = QRadioButton("File")
        self.input_layer_radio.setChecked(True)

        self.input_button_group.addButton(self.input_layer_radio, 0)
        self.input_button_group.addButton(self.input_file_radio, 1)

        step1_layout.addWidget(self.input_layer_radio)
        step1_layout.addWidget(self.input_file_radio)
        step1_layout.addStretch()
        step1_group.setLayout(step1_layout)

        # Step 2: Flow Computation
        step2_group = QGroupBox("Step 2 – Displacement Field (w)")
        step2_layout = QHBoxLayout()

        self.flow_button_group = QButtonGroup()
        self.flow_compute_radio = QRadioButton("Compute now")
        self.flow_precomputed_radio = QRadioButton("Use precomputed w.h5")
        self.flow_compute_radio.setChecked(True)

        self.flow_button_group.addButton(self.flow_compute_radio, 0)
        self.flow_button_group.addButton(self.flow_precomputed_radio, 1)

        step2_layout.addWidget(self.flow_compute_radio)
        step2_layout.addWidget(self.flow_precomputed_radio)
        step2_layout.addStretch()
        step2_group.setLayout(step2_layout)

        # Step 3: Output Mode
        step3_group = QGroupBox("Step 3 – Output")
        step3_layout = QHBoxLayout()

        self.output_button_group = QButtonGroup()
        self.output_analyze_radio = QRadioButton("Analyze only")
        self.output_save_radio = QRadioButton("Correct & save")
        self.output_load_radio = QRadioButton("Load to viewer")
        self.output_analyze_radio.setChecked(True)

        self.output_button_group.addButton(self.output_analyze_radio, 0)
        self.output_button_group.addButton(self.output_save_radio, 1)
        self.output_button_group.addButton(self.output_load_radio, 2)

        step3_layout.addWidget(self.output_analyze_radio)
        step3_layout.addWidget(self.output_save_radio)
        step3_layout.addWidget(self.output_load_radio)
        step3_layout.addStretch()
        step3_group.setLayout(step3_layout)

        # Memory/IO Estimate
        self.estimate_label = QLabel("Memory: MINIMAL (batch + stats)")
        self.estimate_label.setStyleSheet("QLabel { color: green; font-weight: bold; padding: 5px; }")

        # Add all to main layout
        layout.addWidget(step1_group)
        layout.addWidget(step2_group)
        layout.addWidget(step3_group)
        layout.addWidget(self.estimate_label)

        self.setLayout(layout)

        # Connect signals
        self.input_button_group.buttonClicked.connect(self._on_choice_changed)
        self.flow_button_group.buttonClicked.connect(self._on_choice_changed)
        self.output_button_group.buttonClicked.connect(self._on_choice_changed)

    def _on_choice_changed(self):
        """Handle wizard choice changes."""
        # Update internal state
        self.input_choice = "layer" if self.input_layer_radio.isChecked() else "file"
        self.flow_choice = "compute" if self.flow_compute_radio.isChecked() else "precomputed"

        if self.output_analyze_radio.isChecked():
            self.output_choice = "analyze"
        elif self.output_save_radio.isChecked():
            self.output_choice = "save"
        else:
            self.output_choice = "load"

        # Update memory estimate
        self._update_estimate()

        # Emit signal
        self.signaler.state_changed.emit()

    def _update_estimate(self):
        """Update memory/IO estimate based on current choices."""
        mode_def = self.get_mode_definition()

        if mode_def:
            level = mode_def.memory.usage_level.upper()
            formula = mode_def.memory.peak_formula

            # Color coding
            color_map = {
                "MINIMAL": "green",
                "LOW": "green",
                "MEDIUM": "orange",
                "HIGH": "red"
            }
            color = color_map.get(level, "gray")

            self.estimate_label.setText(f"Memory: {level} ({formula})")
            self.estimate_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; padding: 5px; }}")

    def get_mode_definition(self):
        """
        Map wizard choices to ProcessingMode definition.

        Returns:
            ModeDefinition corresponding to current wizard state
        """
        # Mapping logic
        if self.input_choice == "layer":
            if self.flow_choice == "compute":
                if self.output_choice == "analyze":
                    # Loaded layer + Compute + Analyze -> LOADED_BATCH_PROCESS (analyze mode)
                    return self._mode_manager.definitions[ProcessingMode.LOADED_BATCH_PROCESS]
                elif self.output_choice == "save":
                    # Loaded layer + Compute + Save -> LOADED_BATCH_PROCESS
                    return self._mode_manager.definitions[ProcessingMode.LOADED_BATCH_PROCESS]
                else:  # load
                    # Loaded layer + Compute + Load -> FILE_TO_MEMORY (but from layer)
                    # We'll use a custom variant or LOADED_BATCH_PROCESS with load_w flag
                    return self._mode_manager.definitions[ProcessingMode.LOADED_BATCH_PROCESS]
            else:  # precomputed
                if self.output_choice == "analyze":
                    # Loaded layer + Precomputed + Analyze -> LOADED_WITH_PRECOMPUTED
                    return self._mode_manager.definitions[ProcessingMode.LOADED_WITH_PRECOMPUTED]
                elif self.output_choice == "save":
                    # Loaded layer + Precomputed + Save -> Apply precomputed and save
                    # Use FILE_WITH_PRECOMPUTED as template
                    return self._mode_manager.definitions[ProcessingMode.FILE_WITH_PRECOMPUTED]
                else:  # load
                    # Loaded layer + Precomputed + Load -> LOADED_WITH_PRECOMPUTED (load corrected)
                    return self._mode_manager.definitions[ProcessingMode.LOADED_WITH_PRECOMPUTED]
        else:  # file
            if self.flow_choice == "compute":
                if self.output_choice == "analyze":
                    # File + Compute + Analyze -> ANALYSIS_STREAMING
                    return self._mode_manager.definitions[ProcessingMode.ANALYSIS_STREAMING]
                elif self.output_choice == "save":
                    # File + Compute + Save -> FILE_FULL_PIPELINE
                    return self._mode_manager.definitions[ProcessingMode.FILE_FULL_PIPELINE]
                else:  # load
                    # File + Compute + Load -> FILE_TO_MEMORY
                    return self._mode_manager.definitions[ProcessingMode.FILE_TO_MEMORY]
            else:  # precomputed
                if self.output_choice == "analyze":
                    # File + Precomputed + Analyze -> ANALYSIS_PRECOMPUTED
                    return self._mode_manager.definitions[ProcessingMode.ANALYSIS_PRECOMPUTED]
                elif self.output_choice == "save":
                    # File + Precomputed + Save -> FILE_WITH_PRECOMPUTED
                    return self._mode_manager.definitions[ProcessingMode.FILE_WITH_PRECOMPUTED]
                else:  # load
                    # File + Precomputed + Load -> Load file with precomputed to viewer
                    # Use FILE_TO_MEMORY as template
                    return self._mode_manager.definitions[ProcessingMode.FILE_TO_MEMORY]

        # Fallback
        return self._mode_manager.definitions[ProcessingMode.ONLINE_LOADED]

    def get_processing_mode(self) -> ProcessingMode:
        """Get the ProcessingMode enum value for current wizard state."""
        mode_def = self.get_mode_definition()
        return mode_def.mode if mode_def else ProcessingMode.ONLINE_LOADED

    def requires_video_file(self) -> bool:
        """Check if current wizard state requires video file input."""
        return self.input_choice == "file"

    def requires_flow_file(self) -> bool:
        """Check if current wizard state requires flow file input."""
        return self.flow_choice == "precomputed"

    def requires_output_path(self) -> bool:
        """Check if current wizard state requires output path."""
        return self.output_choice == "save"

    def should_load_to_viewer(self) -> bool:
        """Check if results should be loaded to viewer."""
        return self.output_choice == "load"

    def requires_confirmation(self) -> bool:
        """Check if current mode requires user confirmation (RAM warning)."""
        mode_def = self.get_mode_definition()
        return mode_def.requires_confirmation if mode_def else False


class RunOptionsDialog(QDialog):
    """
    Popover dialog for per-run options.

    Contains:
    - Batch size
    - Save w checkbox + path
    - Corrected output path
    - Trajectory/ROI options
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Options")
        self.setModal(True)

        self._init_ui()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Batch Processing Options
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QGridLayout()

        batch_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(10, 500)
        self.batch_size_spin.setValue(50)
        self.batch_size_spin.setToolTip("Frames to process at once")
        batch_layout.addWidget(self.batch_size_spin, 0, 1)

        batch_layout.addWidget(QLabel("Workers:"), 1, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        batch_layout.addWidget(self.workers_spin, 1, 1)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # Displacement Field (w) Options
        w_group = QGroupBox("Displacement Field (w)")
        w_layout = QVBoxLayout()

        self.save_w_check = QCheckBox("Save displacement field (w) to disk")
        self.save_w_check.setToolTip("Save displacement field to HDF5 file")
        w_layout.addWidget(self.save_w_check)

        w_path_layout = QHBoxLayout()
        w_path_layout.addWidget(QLabel("w.h5 path:"))
        self.w_path_edit = QLineEdit()
        self.w_path_edit.setPlaceholderText("Path to save w.h5...")
        self.w_path_edit.setEnabled(False)
        w_path_layout.addWidget(self.w_path_edit)

        self.w_browse_btn = QPushButton("Browse...")
        self.w_browse_btn.setEnabled(False)
        self.w_browse_btn.clicked.connect(self._browse_w_path)
        w_path_layout.addWidget(self.w_browse_btn)

        w_layout.addLayout(w_path_layout)

        # Load w to viewer option (only for specific modes)
        self.load_w_check = QCheckBox("Load displacement field (w) to viewer (uses memory!)")
        self.load_w_check.setStyleSheet("QCheckBox { color: orange; }")
        w_layout.addWidget(self.load_w_check)

        w_group.setLayout(w_layout)
        layout.addWidget(w_group)

        # Precomputed w.h5 Input (for modes that use precomputed flow)
        load_w_group = QGroupBox("Load Displacement Field (w)")
        load_w_layout = QHBoxLayout()

        load_w_layout.addWidget(QLabel("w.h5 file:"))
        self.load_w_path_edit = QLineEdit()
        self.load_w_path_edit.setPlaceholderText("Path to w.h5 file...")
        load_w_layout.addWidget(self.load_w_path_edit)

        self.load_w_browse_btn = QPushButton("Browse...")
        self.load_w_browse_btn.clicked.connect(self._browse_load_w_path)
        load_w_layout.addWidget(self.load_w_browse_btn)

        load_w_group.setLayout(load_w_layout)
        load_w_group.hide()  # Hidden by default
        layout.addWidget(load_w_group)
        self.load_w_input_group = load_w_group

        # Video File Input (for file-based modes)
        video_group = QGroupBox("Video File")
        video_layout = QHBoxLayout()

        video_layout.addWidget(QLabel("Video file:"))
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Path to video file...")
        video_layout.addWidget(self.video_path_edit)

        self.video_browse_btn = QPushButton("Browse...")
        self.video_browse_btn.clicked.connect(self._browse_video_path)
        video_layout.addWidget(self.video_browse_btn)

        video_group.setLayout(video_layout)
        video_group.hide()  # Hidden by default
        layout.addWidget(video_group)
        self.video_input_group = video_group

        # Corrected Output Path
        corrected_group = QGroupBox("Corrected Output")
        corrected_layout = QHBoxLayout()

        corrected_layout.addWidget(QLabel("Save corrected to:"))
        self.corrected_path_edit = QLineEdit()
        self.corrected_path_edit.setPlaceholderText("Path to save corrected video...")
        corrected_layout.addWidget(self.corrected_path_edit)

        self.corrected_browse_btn = QPushButton("Browse...")
        self.corrected_browse_btn.clicked.connect(self._browse_corrected_path)
        corrected_layout.addWidget(self.corrected_browse_btn)

        corrected_group.setLayout(corrected_layout)
        corrected_group.hide()  # Hidden by default
        layout.addWidget(corrected_group)
        self.corrected_output_group = corrected_group

        # Trajectory Options
        traj_group = QGroupBox("Trajectory Options")
        traj_layout = QVBoxLayout()

        self.enable_trajectories_check = QCheckBox("Compute trajectories")
        self.enable_trajectories_check.setChecked(True)
        traj_layout.addWidget(self.enable_trajectories_check)

        traj_group.setLayout(traj_layout)
        layout.addWidget(traj_group)

        # ROI Options
        roi_group = QGroupBox("ROI Options")
        roi_layout = QVBoxLayout()

        self.enable_roi_stats_check = QCheckBox("Compute ROI statistics")
        self.enable_roi_stats_check.setChecked(True)
        roi_layout.addWidget(self.enable_roi_stats_check)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.save_w_check.toggled.connect(self.w_path_edit.setEnabled)
        self.save_w_check.toggled.connect(self.w_browse_btn.setEnabled)

    def _browse_w_path(self):
        """Browse for w.h5 save path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save displacement field (w)", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if file_path:
            self.w_path_edit.setText(file_path)

    def _browse_load_w_path(self):
        """Browse for w.h5 load path."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select displacement field (w.h5)", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if file_path:
            self.load_w_path_edit.setText(file_path)

    def _browse_video_path(self):
        """Browse for video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video files (*.tif *.tiff *.h5 *.hdf5 *.mat)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)

    def _browse_corrected_path(self):
        """Browse for corrected output path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save corrected video", "", "HDF5 files (*.h5);;TIFF files (*.tif *.tiff)"
        )
        if file_path:
            self.corrected_path_edit.setText(file_path)

    def update_visibility(self, wizard: ProcessingWizard):
        """Update visibility of options based on wizard state."""
        # Show/hide video file input
        self.video_input_group.setVisible(wizard.requires_video_file())

        # Show/hide precomputed flow input
        self.load_w_input_group.setVisible(wizard.requires_flow_file())

        # Show/hide corrected output
        self.corrected_output_group.setVisible(wizard.requires_output_path())

    def get_options(self) -> dict:
        """Get current run options as dictionary."""
        return {
            'batch_size': self.batch_size_spin.value(),
            'workers': self.workers_spin.value(),
            'save_w': self.save_w_check.isChecked(),
            'w_path': self.w_path_edit.text() if self.save_w_check.isChecked() else None,
            'load_w': self.load_w_check.isChecked(),
            'video_path': self.video_path_edit.text(),
            'load_w_path': self.load_w_path_edit.text(),
            'corrected_path': self.corrected_path_edit.text(),
            'enable_trajectories': self.enable_trajectories_check.isChecked(),
            'enable_roi_stats': self.enable_roi_stats_check.isChecked(),
        }
