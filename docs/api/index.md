# API Reference

Complete API documentation for napari-flowreg.

## Contents

```{toctree}
:maxdepth: 2

widgets
callbacks
modes
integration
```

## Main Modules

### Widgets

Interactive napari widgets for motion correction and analysis.

- `FlowRegWidget` – Main motion-correction widget that mirrors the PyFlowReg GUI.
- `MotionAnalysisWidgetV2` – Wizard-based motion analysis widget with streaming callbacks.
- `MotionAnalysisWidget` – Legacy analysis widget retained for backward compatibility.

### Callbacks

Batch processing callbacks for zero-copy analysis.

- `BatchTrajectoryTracker` - Track points through flow field batches
- `ROIMotionAnalyzer` - Compute ROI statistics from flow batches
- `WAccumulator` - Accumulate displacement fields in memory
- `CombinedCallbackDispatcher` - Route callbacks to multiple analyzers

### Modes

Processing mode definitions and management.

- `ProcessingMode` - Enum of available processing modes
- `ModeDefinition` - Complete mode configuration
- `ModeManager` - Mode validation and state management

### Integration

Integration functions for adding batch processing to widgets.

- `create_mode_selector_group()` - Create mode UI
- `handle_mode_change()` - Update widget on mode change
- `create_batch_processor_for_widget()` - Create batch processor

## Quick Links

- [Widgets API](widgets.md)
- [Callbacks API](callbacks.md)
- [Modes API](modes.md)
- [Integration API](integration.md)

## Usage Examples

### Using Widgets Programmatically

```python
import napari
from napari_flowreg import FlowRegWidget
from napari_flowreg.motion_analysis import MotionAnalysisWidgetV2

viewer = napari.Viewer()
viewer.open("video.tif")

flowreg = FlowRegWidget(viewer)
viewer.window.add_dock_widget(flowreg, name="FlowReg", area="right")

analysis = MotionAnalysisWidgetV2(viewer)
viewer.window.add_dock_widget(analysis, name="Analysis", area="right")
```

### Using Callbacks

```python
from napari_flowreg.motion_analysis import (
    BatchTrajectoryTracker,
    ROIMotionAnalyzer,
    CombinedCallbackDispatcher
)
from pyflowreg.motion_correction import compensate_arr, OFOptions

# Setup callbacks
dispatcher = CombinedCallbackDispatcher()

# Add trajectory tracker
seed_points = np.array([[50, 50], [100, 100]])
tracker = BatchTrajectoryTracker(seed_points, T, H, W)
dispatcher.add_callback(tracker)

# Add ROI analyzer
roi_masks = [mask1, mask2]
analyzer = ROIMotionAnalyzer(roi_masks, ["ROI_1", "ROI_2"])
dispatcher.add_callback(analyzer)

# Run with callbacks
registered, w = compensate_arr(
    video,
    reference,
    OFOptions(),
    w_callback=dispatcher.w_callback
)

# Get results
tracks = tracker.get_tracks_for_napari()
roi_stats = analyzer.get_summary_statistics()
```

### Working with Modes

```python
from napari_flowreg.motion_analysis import ModeManager, ProcessingMode

manager = ModeManager()

# Get available modes
mode_names = manager.get_mode_names()

# Switch modes
manager.set_mode(ProcessingMode.LOADED_BATCH_PROCESS)

# Validate inputs
mode_def = manager.get_current_definition()
errors = manager.validate_inputs(mode_def, {
    "video_source": "my_video",
    "batch_size": 50
})
```

## See Also

- [User Guide](../user_guide/index.md)
- [Quick Start](../quickstart.md)
- [PyFlowReg API](https://pyflowreg.readthedocs.io/)
