# Motion Analysis Widget v2

The **Flow-Registration Analysis (Experimental)** widget provides streaming motion analysis on top of PyFlowReg. It combines a 3-step wizard, explicit layer selectors, and dedicated tabs for **Tracking** and **Statistics**.

![Screenshot placeholder: motion analysis widget](../screenshots/motion_analysis_v2.png)

## Accessing the widget

- Open from the napari menu: **Plugins → napari-flowreg → Flow-Registration Analysis (Experimental)**.
- Or instantiate manually:

```python
from napari_flowreg.motion_analysis import MotionAnalysisWidgetV2
widget = MotionAnalysisWidgetV2(viewer)
viewer.window.add_dock_widget(widget, name="Flow-Registration Analysis")
```

The widget assumes PyFlowReg ≥ 0.1.0a5 and shares optical-flow parameters with the primary FlowReg widget when it is open.

## Wizard overview

The wizard summarises how the run will acquire data, compute flow, and deliver outputs.

| Step | Options | Behaviour |
|------|---------|-----------|
| **1 – Input** | `Loaded layer` or `File` | Choose between the current napari `Image` layer and a file path supplied in the run options dialog. |
| **2 – Displacement field (w)** | `Compute now` or `Use precomputed w.h5` | Compute optical flow during the run or reuse a saved displacement field. |
| **3 – Output** | `Analyze only`, `Correct & save`, `Load to viewer` | Keep everything in callbacks, stream corrected data to disk, or load corrected outputs back into napari. |

The coloured banner below the wizard estimates the peak memory footprint for the selected combination. When “Load to viewer” is chosen, the widget will warn before allocating large arrays.

## Layer selectors

Under the wizard, the **Data Selection** panel contains status-aware drop-downs:

- **Video layer** – auto-populated with napari `Image` layers; required for *Loaded layer* workflows.
- **Displacement field (w)** – optional `Image` layer with shape `(T, H, W, 2)` when reusing precomputed flows.
- **ROI layer** (Statistics tab) – accepts `Shapes` layers.
- **Seed layer** (Tracking tab) – accepts `Points` layers and `Shapes` layers that contain point primitives.

Each selector shows a pill with the number of frames/ROIs/points detected.

## Run options dialog

Pressing **Track Points** or **Analyze Motion** raises the run options dialog. It configures both PyFlowReg’s `OFOptions` and the analysis callbacks.

### Batch processing

- **Batch size** – number of frames per optical-flow chunk (10–500, default 50).
- **Workers** – number of parallel worker processes used by file-based pipelines (-1–64, default -1 = all cores). Loaded-layer runs stream batches in-process and ignore this value.

### Displacement handling

- `Save displacement field (w) to disk` – enables `OFOptions.save_displacement` via `output_path`. The field is written by PyFlowReg; the plugin never serialises it directly.
- `Output folder` – directory used for `OFOptions.output_path` when saving.
- `Add displacement field (w) to viewer` – accumulates the streamed field and loads the full array into napari after processing. Use this only when your workstation has enough RAM and set wizard Step 3 to **Load to viewer**; otherwise the layer is not added.
- `w.h5 file` – path to an existing displacement field when “Use precomputed w.h5” is selected in the wizard.

### Video IO

- `Video file` – path required when the wizard Input is set to File.
- `Save corrected to` – path required when Output is “Correct & save”. PyFlowReg picks the container (HDF5 or TIFF) based on the extension.

### Optical-flow parameters

- `Quality` – presets (`balanced`, `quality`, `fast`, `custom`).
- `Alpha`, `Sigma (xy/t)` – smoothing and regularisation parameters.
- `Levels`, `Iterations`, `Eta` – only enabled in **custom** mode.
- If the FlowReg widget is open and has been run, its latest options are loaded and merged with any overrides from this dialog.

## Tracking workflow

1. Add a `Points` layer (or a `Shapes` layer with point primitives). The widget refuses to start tracking when no seeds are available.
2. Select the desired seed layer from the **Tracking** tab.
3. Wizard suggestion: **Loaded layer → Compute now → Analyze only** for in-memory streaming (no corrected output).
4. Click **Track Points**. Adjust batch size, choose whether to save/export the displacement field, and press **OK**. To materialise the field as a layer, change wizard Step 3 to **Load to viewer** and enable “Add displacement field (w) to viewer”.
5. Partial trajectories appear as soon as each batch finishes. The summary label reports the number of unique tracks and frames covered.
6. Click **Export Tracks** to save the trajectories. CSV files contain one row per point with columns `track_id,frame,y,x`; MAT files contain a `tracks` matrix with the same columns plus a `column_names` list.

The tracking tab intentionally exposes only the “Forward (flow applied)” mode. Streaming trajectories in the natural acquisition direction is numerically more stable; if you need backward-style analysis, run the solver on a reversed stack.

## ROI statistics workflow

1. Draw ROIs in a `Shapes` layer. Rectangles and polygons are supported; ellipses are rendered as polygons internally.
2. Switch to the **Statistics** tab, choose the ROI layer, statistic, and metric (Magnitude, Divergence, Curl).
3. Wizard suggestion: the same **Loaded layer → Compute now → Analyze only** combination keeps processing lean. Switch Step 3 to **Load to viewer** if you also want the displacement field layer.
4. Click **Analyze Motion** and confirm the options.
5. The summary label lists per-ROI means. The embedded plot displays the selected metric over time. Click **Export Statistics** to save the results: CSV files contain a wide table (`frame` column plus one column per ROI) with the overall mean/max/min in `#`-comment header lines; MAT files contain a `roi_stats` struct (fields `time_series_mean`, `overall_mean`, `max`, `min` per ROI, names sanitised to MATLAB identifiers) and a `roi_names` cell array.

## File-based pipelines

Choose Input → File when you do not want to preload video frames into napari. The worker reads directly from disk using PyFlowReg’s readers. Combine with Output → Correct & save to create corrected videos on disk. When Input → File and Output → Analyze only are selected, the widget performs streaming analysis without creating napari layers.

## Tips and best practices

- Always review the memory warning before selecting “Load to viewer”; reconstructing a full displacement field can allocate multiple gigabytes.
- Saving `w.h5` requires an output directory. The filename is created by PyFlowReg (`compensated.w.h5` by default); leave the field unchecked if you only need on-the-fly analysis.
- For reproducible parameter sets, open the FlowReg widget once and press **Run**. The analysis dialog will pick up the same `OFOptions` until they change.
- When reusing precomputed flows, make sure the spatial dimensions of the `w.h5` match the selected video layer or file. The widget validates the shape before processing.
- The progress bar reports processed frames out of the total. If processing stops early, check the napari console for errors raised by the worker.

## Troubleshooting

- **“No seed points found”** – add a `Points` layer with at least one point or select an existing layer in the Tracking tab.
- **“Failed to start processing: …”** – usually indicates a mismatch between wizard settings and available inputs (missing file path, incompatible shapes).
- **“Processing failed: … QualitySetting …”** – ensure PyFlowReg ≥ 0.1.0a5 is installed; earlier versions do not expose `QualitySetting` to the plugin.
- **Displacement not appearing in viewer** – verify that “Add displacement field (w) to viewer” is enabled and that the wizard output mode is “Load to viewer”. The worker only emits full `w` arrays when requested.
