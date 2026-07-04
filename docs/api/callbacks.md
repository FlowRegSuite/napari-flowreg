# Callbacks API

The callback system enables zero-copy processing of displacement fields during batch computation.

## BatchTrajectoryTracker

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.BatchTrajectoryTracker
   :members:
   :undoc-members:
   :show-inheritance:
```

Tracks seed points through displacement field batches without requiring the full field in memory.

### Example

```python
import numpy as np
from napari_flowreg.motion_analysis import BatchTrajectoryTracker

# Initialize with seed points
seed_points = np.array([[50.0, 50.0], [100.0, 100.0]])
tracker = BatchTrajectoryTracker(seed_points, total_frames=100, H=256, W=256)

# Process w batches (called by pyflowreg during computation)
tracker.w_callback(w_batch, batch_start=0, batch_end=50)
tracker.w_callback(w_batch2, batch_start=50, batch_end=100)

# Get napari-compatible tracks
tracks = tracker.get_tracks_for_napari()  # Shape: (n_points * n_frames, 4)
```

## ROIMotionAnalyzer

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.ROIMotionAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

Computes per-ROI motion statistics during batch processing.

### Example

```python
from napari_flowreg.motion_analysis import ROIMotionAnalyzer

# Create ROI masks
roi_masks = [mask1, mask2]  # Binary masks
roi_names = ["ROI_1", "ROI_2"]

analyzer = ROIMotionAnalyzer(roi_masks, roi_names)

# Process batches (called by pyflowreg)
analyzer.w_callback(w_batch, batch_start=0, batch_end=50)

# Get statistics
stats = analyzer.get_summary_statistics()
# {'ROI_1': {'overall_mean': ..., 'time_series_mean': [...]}, ...}
```

## WAccumulator

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.WAccumulator
   :members:
   :undoc-members:
   :show-inheritance:
```

Accumulates displacement field batches in memory for optional post-processing.

## ProgressTracker

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.ProgressTracker
   :members:
   :undoc-members:
   :show-inheritance:
```

Tracks progress during batch processing using w callbacks for UI updates.

### Example

```python
from napari_flowreg.motion_analysis import ProgressTracker

def update_ui(current, total):
    print(f"Progress: {current}/{total} frames")

tracker = ProgressTracker(total_frames=100, progress_callback=update_ui)

# Process batches (called by pyflowreg)
tracker.w_callback(w_batch, batch_start=0, batch_end=50)  # Prints: Progress: 50/100 frames
tracker.w_callback(w_batch2, batch_start=50, batch_end=100)  # Prints: Progress: 100/100 frames

# Get current progress
current, total = tracker.get_progress()
percentage = tracker.get_percentage()
```

## CombinedCallbackDispatcher

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.CombinedCallbackDispatcher
   :members:
   :undoc-members:
   :show-inheritance:
```

Routes callbacks to multiple analyzers for simultaneous processing.

### Example

```python
from napari_flowreg.motion_analysis import (
    CombinedCallbackDispatcher,
    BatchTrajectoryTracker,
    ROIMotionAnalyzer
)
from pyflowreg.motion_correction import compensate_arr

# Setup dispatcher
dispatcher = CombinedCallbackDispatcher()

# Add multiple callbacks
tracker = BatchTrajectoryTracker(seed_points, T, H, W)
analyzer = ROIMotionAnalyzer(roi_masks, roi_names)

dispatcher.add_callback(tracker)
dispatcher.add_callback(analyzer)

# Use with pyflowreg
registered, w = compensate_arr(
    video,
    reference,
    options,
    w_callback=dispatcher.w_callback  # All callbacks receive w batches
)
```

## Helper Functions

### extract_roi_masks_from_shapes

```{eval-rst}
.. autofunction:: napari_flowreg.motion_analysis.extract_roi_masks_from_shapes
```

### extract_seed_points_from_layer

```{eval-rst}
.. autofunction:: napari_flowreg.motion_analysis.extract_seed_points_from_layer
```