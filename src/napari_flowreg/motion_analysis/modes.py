"""
Processing mode definitions for motion analysis widget.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List


class ProcessingMode(Enum):
    """Processing modes for motion analysis."""

    # Current functionality (keep backward compatible)
    ONLINE_LOADED = auto()

    # New memory-efficient modes
    LOADED_BATCH_PROCESS = auto()
    LOADED_WITH_PRECOMPUTED = auto()

    # File-based modes
    FILE_FULL_PIPELINE = auto()
    FILE_TO_MEMORY = auto()
    FILE_WITH_PRECOMPUTED = auto()

    # Analysis modes
    ANALYSIS_STREAMING = auto()
    ANALYSIS_PRECOMPUTED = auto()


@dataclass
class InputConfig:
    """Input configuration for a processing mode."""
    source_type: Literal["layer", "file", "both"]
    requires_video: bool = True
    requires_flow: bool = False
    allows_roi: bool = True


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    compute_flow: bool = False
    compute_correction: bool = False
    batch_processing: bool = False
    batch_size_default: int = 50
    batch_size_min: int = 10
    batch_size_max: int = 500
    allow_parallel: bool = True
    trajectory_callback: bool = False


@dataclass
class OutputConfig:
    """Output configuration."""
    save_flow: bool = False
    save_corrected: bool = False
    save_trajectories: bool = False
    load_flow_to_viewer: bool = False
    load_corrected_to_viewer: bool = False
    load_trajectories_to_viewer: bool = True


@dataclass
class MemoryProfile:
    """Memory usage profile."""
    usage_level: Literal["minimal", "low", "medium", "high"]
    peak_formula: str
    typical_mb: int  # For 512x512x1000 uint8 video


@dataclass
class ModeDefinition:
    """Complete definition of a processing mode."""
    mode: ProcessingMode
    display_name: str
    description: str
    tooltip: str

    input: InputConfig
    processing: ProcessingConfig
    output: OutputConfig
    memory: MemoryProfile

    # UI hints
    show_batch_size: bool = False
    show_w_path: bool = False
    show_output_path: bool = False
    show_export_w_checkbox: bool = False
    show_save_corrected_checkbox: bool = False

    # Validation
    requires_confirmation: bool = False
    warning_message: Optional[str] = None


# Define all modes
MODE_DEFINITIONS = {
    ProcessingMode.ONLINE_LOADED: ModeDefinition(
        mode=ProcessingMode.ONLINE_LOADED,
        display_name="Online: Loaded Data",
        description="Analyze motion from viewer data",
        tooltip="Use data already loaded in napari viewer. Both video and flow field must be in memory.",
        input=InputConfig(source_type="layer", requires_flow=True),
        processing=ProcessingConfig(),
        output=OutputConfig(load_trajectories_to_viewer=True),
        memory=MemoryProfile("high", "video + w", 2300),
    ),

    ProcessingMode.LOADED_BATCH_PROCESS: ModeDefinition(
        mode=ProcessingMode.LOADED_BATCH_PROCESS,
        display_name="Batch Process: Loaded Video",
        description="Process loaded video in batches, save w & corrected",
        tooltip="Process uint8 video from viewer in batches. Saves memory by streaming to disk. Computes trajectories on-the-fly from w batches.",
        input=InputConfig(source_type="layer", requires_flow=False),
        processing=ProcessingConfig(
            compute_flow=True,
            compute_correction=True,
            batch_processing=True,
            batch_size_default=50,
            trajectory_callback=True
        ),
        output=OutputConfig(
            save_flow=True,
            save_corrected=True,
            load_trajectories_to_viewer=True
        ),
        memory=MemoryProfile("low", "uint8_video + batch", 300),
        show_batch_size=True,
        show_w_path=True,
        show_output_path=True,
        show_export_w_checkbox=True,
        show_save_corrected_checkbox=True,
    ),

    ProcessingMode.LOADED_WITH_PRECOMPUTED: ModeDefinition(
        mode=ProcessingMode.LOADED_WITH_PRECOMPUTED,
        display_name="Analyze: Video + w.h5",
        description="Use loaded video with flow file",
        tooltip="Combine video from viewer with pre-computed flow field from disk.",
        input=InputConfig(source_type="both", requires_flow=True),
        processing=ProcessingConfig(batch_processing=True),
        output=OutputConfig(load_trajectories_to_viewer=True),
        memory=MemoryProfile("medium", "video + batch", 350),
        show_batch_size=True,
    ),

    ProcessingMode.FILE_FULL_PIPELINE: ModeDefinition(
        mode=ProcessingMode.FILE_FULL_PIPELINE,
        display_name="File: Full Pipeline",
        description="Complete flow-registration: compute & save w + corrected",
        tooltip="Run complete flow-registration pipeline. Saves both displacement field and motion-corrected video.",
        input=InputConfig(source_type="file"),
        processing=ProcessingConfig(
            compute_flow=True,
            compute_correction=True,
            batch_processing=True,
            batch_size_default=50
        ),
        output=OutputConfig(
            save_flow=True,
            save_corrected=True,
            save_trajectories=True
        ),
        memory=MemoryProfile("minimal", "3 * batch", 150),
        show_batch_size=True,
        show_w_path=True,
        show_output_path=True,
    ),

    ProcessingMode.FILE_TO_MEMORY: ModeDefinition(
        mode=ProcessingMode.FILE_TO_MEMORY,
        display_name="Process & Load",
        description="Process file and load results",
        tooltip="Process video file and load corrected result to viewer.",
        input=InputConfig(source_type="file"),
        processing=ProcessingConfig(
            compute_flow=True,
            compute_correction=True,
            batch_processing=True
        ),
        output=OutputConfig(
            load_flow_to_viewer=True,
            load_corrected_to_viewer=True
        ),
        memory=MemoryProfile("high", "output_video + w", 3000),
        show_batch_size=True,
        requires_confirmation=True,
        warning_message="This will load full results into memory. Ensure sufficient RAM.",
    ),

    ProcessingMode.FILE_WITH_PRECOMPUTED: ModeDefinition(
        mode=ProcessingMode.FILE_WITH_PRECOMPUTED,
        display_name="Apply Flow Field",
        description="Apply existing flow to video",
        tooltip="Use pre-computed flow field to correct video motion.",
        input=InputConfig(source_type="file", requires_flow=True),
        processing=ProcessingConfig(
            compute_correction=True,
            batch_processing=True
        ),
        output=OutputConfig(save_corrected=True),
        memory=MemoryProfile("minimal", "2 * batch", 100),
        show_batch_size=True,
        show_output_path=True,
    ),

    ProcessingMode.ANALYSIS_STREAMING: ModeDefinition(
        mode=ProcessingMode.ANALYSIS_STREAMING,
        display_name="Stream Analysis",
        description="Compute trajectories only",
        tooltip="Process video to extract motion trajectories without full correction.",
        input=InputConfig(source_type="file"),
        processing=ProcessingConfig(
            compute_flow=True,
            batch_processing=True,
            trajectory_callback=True
        ),
        output=OutputConfig(
            save_flow=True,
            save_trajectories=True,
            load_trajectories_to_viewer=True
        ),
        memory=MemoryProfile("minimal", "batch + trajectories", 60),
        show_batch_size=True,
        show_w_path=True,
    ),

    ProcessingMode.ANALYSIS_PRECOMPUTED: ModeDefinition(
        mode=ProcessingMode.ANALYSIS_PRECOMPUTED,
        display_name="Analyze w.h5",
        description="Extract statistics from flow",
        tooltip="Analyze pre-computed flow field to extract motion statistics.",
        input=InputConfig(
            source_type="file",
            requires_video=False,
            requires_flow=True
        ),
        processing=ProcessingConfig(batch_processing=True),
        output=OutputConfig(
            save_trajectories=True,
            load_trajectories_to_viewer=True
        ),
        memory=MemoryProfile("minimal", "batch + stats", 30),
        show_batch_size=True,
    ),
}


class ModeManager:
    """Manages processing modes and their configurations."""

    def __init__(self):
        self.definitions = MODE_DEFINITIONS
        self.current_mode = ProcessingMode.ONLINE_LOADED

    def get_mode_names(self) -> List[str]:
        """Get display names for dropdown."""
        return [defn.display_name for defn in self.definitions.values()]

    def get_mode_by_name(self, name: str) -> Optional[ModeDefinition]:
        """Get mode definition by display name."""
        for defn in self.definitions.values():
            if defn.display_name == name:
                return defn
        return None

    def get_current_definition(self) -> ModeDefinition:
        """Get current mode definition."""
        return self.definitions[self.current_mode]

    def set_mode(self, mode: ProcessingMode):
        """Set current processing mode."""
        self.current_mode = mode

    def set_mode_by_name(self, name: str):
        """Set mode by display name."""
        mode_def = self.get_mode_by_name(name)
        if mode_def:
            self.current_mode = mode_def.mode

    def validate_inputs(self, mode: ModeDefinition, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs for a given mode. Returns list of errors."""
        errors = []

        if mode.input.requires_video and not inputs.get("video_source"):
            errors.append("Video source is required")

        if mode.input.requires_flow and not inputs.get("flow_source"):
            errors.append("Flow field source is required")

        if mode.processing.batch_processing:
            batch_size = inputs.get("batch_size", mode.processing.batch_size_default)
            if batch_size < mode.processing.batch_size_min:
                errors.append(f"Batch size must be at least {mode.processing.batch_size_min}")
            if batch_size > mode.processing.batch_size_max:
                errors.append(f"Batch size cannot exceed {mode.processing.batch_size_max}")

        if mode.output.save_flow and not inputs.get("w_output_path"):
            errors.append("Flow output path is required")

        if mode.output.save_corrected and not inputs.get("corrected_output_path"):
            errors.append("Corrected video output path is required")

        return errors