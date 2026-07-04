# Quick Start

Get motion correction and analysis running in napari-flowreg in just a few minutes.

## Launch the widgets

1. Start napari: `napari`
2. Open the FlowReg widgets from the menu:
   - **Plugins → napari-flowreg → Flow-Registration** for the main motion-correction panel.
   - **Plugins → napari-flowreg → Flow-Registration Analysis (Experimental)** for the new wizard-based analysis widget.

Both widgets can live side-by-side in the dock. The analysis widget can also be added programmatically:

```python
import napari
from napari_flowreg.motion_analysis import MotionAnalysisWidgetV2

viewer = napari.Viewer()
viewer.open("my_2p_video.tif")
widget = MotionAnalysisWidgetV2(viewer)
viewer.window.add_dock_widget(widget, name="Flow-Registration Analysis")
napari.run()
```

## Compute motion correction (optional)

Use the **Flow-Registration** widget when you want interactive previews of motion correction or to run the classic PyFlowReg pipeline. Configure the optical-flow parameters there and click **Run Motion Correction** to create a corrected video and `w.h5`. The analysis widget will reuse the last options from FlowReg automatically; if FlowReg was never opened, the analysis dialog falls back to its own defaults.

## Motion analysis workflow

The **Flow-Registration Analysis** widget focuses on ROI statistics and trajectory tracking. Its layout has three parts:

1. **Wizard** – choose how data should be processed.
2. **Selectors** – pick the video layer and optional pre-computed displacement field.
3. **Tabs** – run tracking or ROI statistics (Tracking is first, Statistics second).

### Step 1 – Prepare layers

- Add your raw video as an `Image` layer.
- (Optional) Draw ROIs in a `Shapes` layer.
- (Optional) Place seed points in a `Points` layer for tracking. The widget refuses to start tracking if no seeds exist.

### Step 2 – Configure the wizard

Select one option per step:

| Step | Choices | Notes |
|------|---------|-------|
| **Input** | Loaded layer / File | *Loaded layer* uses the currently selected napari video. *File* prompts for a file path in the run options dialog. |
| **Displacement field (w)** | Compute now / Use precomputed w.h5 | *Compute now* streams flow batches through callbacks. *Precomputed* expects a path to an existing `w.h5`. |
| **Output** | Analyze only / Correct & save / Load to viewer | *Analyze only* keeps everything in callbacks. *Correct & save* requires an output path. *Load to viewer* reconstructs the corrected video (and optionally the flow) as napari layers. |

The colour-coded banner under the wizard estimates the peak memory footprint based on your selections.

### Step 3 – Pick analysis action

- **Tracking tab → Track Points** – computes trajectories for the selected seed layer (default tab).
- **Statistics tab → Analyze Motion** – computes ROI time series and summaries.

Clicking either button opens the **Run Options** dialog before any work starts.

## Run options dialog

The dialog mirrors the FlowReg configuration and adds analysis-specific choices:

- **Batch size** – number of frames processed per optical-flow batch (default 50).
- **Workers** – parallel worker processes for file-based pipelines (default -1 = all cores). In-memory (loaded layer) runs ignore this.
- **Displacement field (w)** –
  - `Save displacement field (w) to disk`: enables automatic `w.h5` creation in the chosen output folder. The plugin never writes the file directly; it delegates to PyFlowReg.
  - `Output folder`: directory passed to `OFOptions.output_path` when saving is enabled.
- `Add displacement field (w) to viewer`: rebuilds the full field from streaming callbacks and adds it as a layer after processing (uses RAM). This requires setting wizard Step 3 to **Load to viewer**.
- **Video file** – shown when the wizard is set to *File* input.
- **w.h5 file** – shown when reusing a precomputed flow.
- **Corrected output** – shown when *Correct & save* is chosen.
- **Optical-flow parameters** – quality preset, alpha, sigma, and optional custom levels/iterations/eta. These override FlowReg defaults when the main widget is not supplying options.

Press **OK** to launch processing or **Cancel** to abort.

## Tracking example

1. Draw or load a `Points` layer (`t`, `y`, `x` or `y`, `x`). Only the spatial coordinates are required, the widget will expand them across time.
2. Set the wizard to **Loaded layer → Compute now → Analyze only** for streaming callbacks without saving.
3. Click **Track Points**, adjust the run options (for example, batch size 75). If you also want the displacement field layer, change Step 3 of the wizard to **Load to viewer** and enable “Add displacement field (w) to viewer” before confirming.
4. Watch the progress bar update per batch. Partial track updates appear as soon as each batch finishes.
5. On completion, a `Tracks` layer named “Trajectories” is created. Click **Export Tracks** to save the trajectories as CSV (columns `track_id,frame,y,x`) or MATLAB `.mat` (a `tracks` matrix with the same columns).

## ROI statistics example

1. Create a `Shapes` layer with polygons or rectangles covering the regions of interest.
2. Wizard selection: **Loaded layer → Compute now → Analyze only** (same as above).
3. Switch to the **Statistics** tab, pick the ROI layer from the selector, choose the statistic (mean/median/max/min) and metric (motion magnitude, divergence, curl). To load the displacement field after processing, change Step 3 of the wizard to **Load to viewer** and enable “Add displacement field (w) to viewer” in the dialog.
4. Click **Analyze Motion** and confirm the run options.
5. The label under the buttons lists per-ROI summaries, and the embedded matplotlib canvas plots the chosen metric. Click **Export Statistics** to save the per-ROI time series and summary values as CSV or MATLAB `.mat`.

## Working with files

- Choose **Input → File** when the video is not already loaded. Supply the path in the Run Options dialog.
- Choose **Displacement field → Use precomputed w.h5** and provide a path when you only need to analyse an existing flow.
- Choose **Output → Correct & save** to let PyFlowReg stream the corrected result to disk (the dialog will ask for an output path). Combine with “Save displacement field (w) to disk” if you want the corresponding `w.h5`.

## Tips

- The analysis widget never writes `w.h5` itself; it only configures `OFOptions` so PyFlowReg handles the export.
- Streaming callbacks keep memory usage low. Only enable “Add displacement field (w) to viewer” when you need the entire field in napari.
- If the wizard warns about high memory usage, reconsider the output mode or disable the displacement export.
- Open the FlowReg widget once per session if you want the analysis dialog to inherit customised optical-flow presets.
