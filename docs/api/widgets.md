# Widgets API

## FlowRegWidget

```{eval-rst}
.. autoclass:: napari_flowreg.FlowRegWidget
   :members:
   :undoc-members:
   :show-inheritance:
```

## MotionAnalysisWidgetV2

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.MotionAnalysisWidgetV2
   :members:
   :undoc-members:
   :show-inheritance:
```

### Key Methods and Hooks

- `_prompt_run_options()` – opens the run options dialog and captures configuration.
- `_execute_motion_analysis()` – shared entry point for tracking/statistics actions.
- `_run_tracking_analysis()` – validates seeds and launches trajectory callbacks.
- `_run_statistics_analysis()` – launches ROI statistics callbacks.
- `_update_tracks_partial()` – refreshes the napari `Tracks` layer incrementally.
- `_update_roi_summary()` – updates per-ROI summaries and enables export controls.

## MotionAnalysisWidget (legacy)

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.MotionAnalysisWidget
   :members:
   :undoc-members:
   :show-inheritance:
```

### Key Methods

#### Motion Analysis

- `_on_analyze_clicked()`: Main entry point for motion analysis
- `_perform_original_analysis()`: Performs traditional online analysis
- `_get_roi_mask()`: Extracts ROI masks from Shapes layers
- `_get_individual_roi_masks()`: Extracts per-shape ROI masks

#### Point Tracking

- `_track_points()`: Tracks seed points through flow field
- `_bilinear_interpolate()`: Sub-pixel interpolation of flow values
- `_on_track_points_clicked()`: Opens point tracking dialog

#### Export

- `_on_export_clicked()`: Opens comprehensive export dialog
- `_perform_export()`: Executes data export to MAT/NPZ

## Dialog Widgets

### PointTrackingDialog

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.widget.PointTrackingDialog
   :members:
   :undoc-members:
   :show-inheritance:
```

### ExportDialog

```{eval-rst}
.. autoclass:: napari_flowreg.motion_analysis.widget.ExportDialog
   :members:
   :undoc-members:
   :show-inheritance:
```
