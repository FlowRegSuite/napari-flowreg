"""
Tabbed analysis panels for Statistics and Tracking.

Separates ROI statistics analysis from trajectory tracking to reduce UI complexity.
"""

from typing import Optional
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QSpinBox, QTabWidget,
    QGridLayout
)
from qtpy.QtCore import Qt, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LayerSelectorWithPill(QWidget):
    """
    Layer selector with status pill showing item count.

    Shows:
    - Dropdown for layer selection
    - Status pill: "N items" or "Not selected"
    """

    layer_changed = Signal(str)  # Emits layer name

    def __init__(self, label_text: str, empty_text: str = "Not selected", parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.empty_text = empty_text

        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        self.label = QLabel(self.label_text)
        layout.addWidget(self.label)

        # Dropdown
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)

        # Status pill
        self.status_pill = QLabel(self.empty_text)
        self.status_pill.setStyleSheet(
            "QLabel { background-color: #ccc; color: #333; padding: 2px 6px; border-radius: 8px; }"
        )
        layout.addWidget(self.status_pill)

        layout.addStretch()
        self.setLayout(layout)

    def _on_selection_changed(self, layer_name: str):
        """Handle selection change."""
        if not layer_name:
            self.status_pill.setText(self.empty_text)
            self.status_pill.setStyleSheet(
                "QLabel { background-color: #ccc; color: #333; padding: 2px 6px; border-radius: 8px; }"
            )
        else:
            # Status will be updated by parent widget
            self.status_pill.setStyleSheet(
                "QLabel { background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 8px; }"
            )

        self.layer_changed.emit(layer_name)

    def set_status(self, text: str, is_valid: bool = True):
        """Update status pill text and style."""
        self.status_pill.setText(text)

        if not text or text == self.empty_text:
            color = "#ccc"
            text_color = "#333"
        elif is_valid:
            color = "#4CAF50"  # Green
            text_color = "white"
        else:
            color = "#ff9800"  # Orange warning
            text_color = "white"

        self.status_pill.setStyleSheet(
            f"QLabel {{ background-color: {color}; color: {text_color}; padding: 2px 6px; border-radius: 8px; }}"
        )

    def clear_items(self):
        """Clear combo box items."""
        self.combo.clear()

    def add_item(self, text: str):
        """Add item to combo box."""
        self.combo.addItem(text)

    def add_items(self, items: list):
        """Add multiple items to combo box."""
        self.combo.addItems(items)

    def current_text(self) -> str:
        """Get current selection."""
        return self.combo.currentText()

    def set_current_text(self, text: str):
        """Set current selection."""
        self.combo.setCurrentText(text)


class StatisticsTab(QWidget):
    """
    Statistics analysis tab.

    Contains:
    - ROI magnitude analysis
    - Divergence summaries
    - Export functionality
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # ROI Selector
        self.roi_selector = LayerSelectorWithPill(
            "ROI layer:", empty_text="No ROI selected"
        )
        layout.addWidget(self.roi_selector)

        # Analysis Options
        options_group = QGroupBox("Analysis Options")
        options_layout = QGridLayout()

        options_layout.addWidget(QLabel("Statistic Mode:"), 0, 0)
        self.stat_mode_combo = QComboBox()
        self.stat_mode_combo.addItems(["mean", "max", "min", "median"])
        self.stat_mode_combo.setCurrentText("mean")
        self.stat_mode_combo.setToolTip("Statistical measure to compute over ROI")
        options_layout.addWidget(self.stat_mode_combo, 0, 1)

        options_layout.addWidget(QLabel("Plot Type:"), 1, 0)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Motion Magnitude", "Divergence", "Curl"])
        self.plot_type_combo.setCurrentText("Motion Magnitude")
        self.plot_type_combo.setToolTip("Type of motion analysis to perform")
        options_layout.addWidget(self.plot_type_combo, 1, 1)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Controls
        controls_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Analyze Motion")
        self.analyze_button.setToolTip("Compute motion statistics for selected ROI")
        controls_layout.addWidget(self.analyze_button)

        self.export_stats_button = QPushButton("Export Statistics")
        self.export_stats_button.setToolTip("Export ROI statistics to file")
        self.export_stats_button.setEnabled(False)
        controls_layout.addWidget(self.export_stats_button)

        layout.addLayout(controls_layout)

        # Statistics Display
        self.stats_label = QLabel("Statistics will appear here")
        self.stats_label.setStyleSheet("QLabel { font-family: monospace; padding: 5px; }")
        layout.addWidget(self.stats_label)

        # Plot Area
        plot_group = QGroupBox("Motion Analysis Plot")
        plot_layout = QVBoxLayout()

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        layout.addStretch()
        self.setLayout(layout)


class TrackingTab(QWidget):
    """
    Tracking analysis tab.

    Contains:
    - Seed layer selection
    - Tracking mode
    - Tail length
    - Track export
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # Seed Layer Selector
        self.seed_selector = LayerSelectorWithPill(
            "Seed layer:", empty_text="No seeds selected"
        )
        layout.addWidget(self.seed_selector)

        # Tracking Options
        options_group = QGroupBox("Tracking Options")
        options_layout = QGridLayout()

        options_layout.addWidget(QLabel("Tracking Mode:"), 0, 0)
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems(["Forward (flow applied)", "Backward (flow inverted)"])
        self.tracking_mode_combo.setToolTip("Direction of trajectory tracking")
        options_layout.addWidget(self.tracking_mode_combo, 0, 1)

        options_layout.addWidget(QLabel("Tail Length:"), 1, 0)
        self.tail_length_spin = QSpinBox()
        self.tail_length_spin.setRange(1, 100)
        self.tail_length_spin.setValue(10)
        self.tail_length_spin.setToolTip("Number of frames to show in track tail")
        options_layout.addWidget(self.tail_length_spin, 1, 1)

        options_layout.addWidget(QLabel("Tail Width:"), 2, 0)
        self.tail_width_spin = QSpinBox()
        self.tail_width_spin.setRange(1, 10)
        self.tail_width_spin.setValue(2)
        self.tail_width_spin.setToolTip("Width of track tail visualization")
        options_layout.addWidget(self.tail_width_spin, 2, 1)

        # Track all checkbox
        self.track_all_check = QCheckBox("Track all point layers")
        self.track_all_check.setToolTip("Track all Points and Shapes layers with points")
        options_layout.addWidget(self.track_all_check, 3, 0, 1, 2)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Controls
        controls_layout = QHBoxLayout()

        self.track_button = QPushButton("Track Points")
        self.track_button.setToolTip("Compute trajectories for selected seed points")
        controls_layout.addWidget(self.track_button)

        self.export_tracks_button = QPushButton("Export Tracks")
        self.export_tracks_button.setToolTip("Export track data to file")
        self.export_tracks_button.setEnabled(False)
        controls_layout.addWidget(self.export_tracks_button)

        layout.addLayout(controls_layout)

        # Tracking Info
        self.tracking_info_label = QLabel("No tracks computed")
        self.tracking_info_label.setStyleSheet("QLabel { font-family: monospace; padding: 5px; }")
        layout.addWidget(self.tracking_info_label)

        layout.addStretch()
        self.setLayout(layout)


class AnalysisTabs(QTabWidget):
    """
    Combined tabbed interface for Statistics and Tracking.

    Provides clean separation between:
    - Statistics: ROI magnitude, divergence summaries, export
    - Tracking: seed layer selection, tracking mode, tail length, track export
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create tabs
        self.statistics_tab = StatisticsTab()
        self.tracking_tab = TrackingTab()

        # Add tabs
        self.addTab(self.statistics_tab, "Statistics")
        self.addTab(self.tracking_tab, "Tracking")

        # Set style
        self.setDocumentMode(True)  # Cleaner look on some platforms
