"""
Motion Analysis module for napari-flowreg.

Provides tools for analyzing optical flow fields, tracking trajectories,
and computing motion statistics using batch processing callbacks.
"""

# Main widget
from .widget import MotionAnalysisWidget

# Callback system for batch processing
from .callbacks import (
    BatchTrajectoryTracker,
    ROIMotionAnalyzer,
    WAccumulator,
    ProgressTracker,
    CombinedCallbackDispatcher,
    extract_roi_masks_from_shapes,
    extract_seed_points_from_layer,
)

# Mode definitions and management
from .modes import (
    ProcessingMode,
    ModeDefinition,
    ModeManager,
    MODE_DEFINITIONS,
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    MemoryProfile,
)

# Integration utilities
from .integration import (
    create_mode_selector_group,
    add_batch_processing_options,
    handle_mode_change,
    create_batch_processor_for_widget,
)

__all__ = [
    # Widget
    "MotionAnalysisWidget",
    # Callbacks
    "BatchTrajectoryTracker",
    "ROIMotionAnalyzer",
    "WAccumulator",
    "ProgressTracker",
    "CombinedCallbackDispatcher",
    "extract_roi_masks_from_shapes",
    "extract_seed_points_from_layer",
    # Modes
    "ProcessingMode",
    "ModeDefinition",
    "ModeManager",
    "MODE_DEFINITIONS",
    "InputConfig",
    "ProcessingConfig",
    "OutputConfig",
    "MemoryProfile",
    # Integration
    "create_mode_selector_group",
    "add_batch_processing_options",
    "handle_mode_change",
    "create_batch_processor_for_widget",
]
