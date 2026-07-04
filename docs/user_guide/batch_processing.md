# Batch Processing Guide

The Motion Analysis widget v2 streams PyFlowReg in batches so you can analyse large recordings without exhausting RAM. This guide explains how the wizard maps to concrete processing modes, which options influence throughput, and how displacement fields are handled.

## Wizard combinations

The three wizard steps describe the processing pipeline. Each combination maps to a `ProcessingMode` behind the scenes.

| Input | w | Output | Uses | Notes |
|-------|---|--------|------|-------|
| Loaded layer | Compute now | Analyze only | `LOADED_BATCH_PROCESS` | Default for interactive tracking/statistics. Reads frames from the viewer in chunks, streams callbacks, never writes intermediate files. |
| Loaded layer | Compute now | Load to viewer | `LOADED_BATCH_PROCESS` + export | Same as above but reconstructs the full displacement field (and optionally corrected video) after processing. Enable **Add displacement field (w) to viewer** in the dialog to materialise the array. |
| Loaded layer | Precomputed w.h5 | Analyze only | `LOADED_WITH_PRECOMPUTED` | Reuses a flow loaded as an `Image` layer or from disk. Great for re-running analyses with different ROIs or seed points. |
| File | Compute now | Analyze only | `ANALYSIS_STREAMING` | Reads directly from disk. No need to load the video into napari first. |
| File | Compute now | Correct & save | `FILE_FULL_PIPELINE` | End-to-end PyFlowReg run writing corrected data and optional `w.h5`. Ideal for unattended batch jobs. |
| File | Precomputed w.h5 | Analyze only | `ANALYSIS_PRECOMPUTED` | Streams flow from disk to callbacks, skipping recomputation. |
| File | Precomputed w.h5 | Correct & save | `FILE_WITH_PRECOMPUTED` | Applies a precomputed flow to the input file and saves the corrected result. |

The widget automatically enables or disables layer selectors and file pickers based on the selection.

## Memory expectations

The banner below the wizard reports the peak memory estimate. Typical usage patterns:

- **Analyze only** – Minimal allocations. Only the current batch and callback accumulators are in memory.
- **Load to viewer** – Requires enough RAM for the entire displacement field (and corrected video, if present). The run options dialog warns before allocating.
- **Correct & save** – Memory footprint similar to Analyze only; corrected data is streamed directly to disk.

## Displacement field handling

- Saving `w.h5` is controlled by the **Save displacement field (w) to disk** checkbox in the run options dialog. Set an output folder; PyFlowReg writes the displacement file there using its configured naming scheme.
- Exporting to napari requires two things: the wizard Output set to *Load to viewer* and **Add displacement field (w) to viewer** enabled. The widget accumulates the streamed batches via callbacks and adds the full `Image` layer at the end.
- When working purely in memory, leave both options unchecked. The callbacks still receive each batch and can compute trajectories/statistics without storing the entire field.

## Streaming callbacks

napari-flowreg wires PyFlowReg’s callback system so that each finished batch immediately updates the UI:

- `tracks_updated` – emits the current trajectory array; the Tracks layer is refreshed incrementally.
- `roi_updated` – emits per-ROI summaries; the Statistics tab label and plot update live.
- `progress_updated` – updates the progress bar.

No intermediate files are created; only the final results requested in the dialog are materialised.

## Common scenarios

### Track loaded video with minimal RAM

1. Load your video into napari (or drag-and-drop).
2. Wizard: **Loaded layer → Compute now → Analyze only**.
3. Click **Track Points**, set a batch size (e.g. 75), leave both displacement checkboxes unchecked, and press **OK**.
4. Watch tracks populate as each batch completes.

### Track and visualise the displacement field

1. Same as above, but change wizard Step 3 to **Load to viewer**.
2. In the run options enable **Add displacement field (w) to viewer** so the batches are accumulated into a full array.
3. Keep an eye on the memory warning; large volumes may need several gigabytes.
4. After processing, a “Displacement field (w)” `Image` layer appears alongside the `Tracks` layer.

### Batch process from disk and save results

1. Wizard: **File → Compute now → Correct & save**.
2. In the dialog specify:
   - `Video file` – the input recording.
   - `Save corrected to` – destination file (HDF5 or TIFF based on extension).
   - Optionally enable **Save displacement field (w) to disk** and set the output folder.
3. Press **OK**. The widget streams progress; corrected data and `w.h5` are written by PyFlowReg.

### Reanalyse with precomputed w.h5

1. Wizard: **Loaded layer → Use precomputed w.h5 → Analyze only**.
2. Load the previously generated `w.h5` as a layer or provide the file path in the dialog.
3. Run tracking or statistics without recomputing flow.

## Tips

- Batch size is the main knob for balancing speed vs. memory. Larger batches reduce solver overhead but increase RAM usage.
- There is no “workers” spinner—the executor configuration follows the defaults from PyFlowReg and the FlowReg widget.
- Ensure the ROI and seed selectors point to the desired layers before launching a run; they are captured at start time.
- When supplying file paths, double-check permissions. The widget surfaces PyFlowReg exceptions, so path errors appear in the napari console and as notifications.
- If you only need tracks, you can disable both displacement checkboxes to avoid writing any auxiliary data.
