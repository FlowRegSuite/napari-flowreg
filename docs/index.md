# napari-flowreg

**Interactive Motion Correction for 2-Photon Microscopy in napari**

napari-flowreg is a napari plugin that provides interactive visualization and motion correction for 2-photon microscopy data using the PyFlowReg optical flow engine. It brings high-accuracy variational optical flow motion correction into the napari viewer with an intuitive graphical interface.

## Key Features

- **FlowReg widget** for motion correction parameter tuning and preview
- **Wizard-driven motion analysis** for choosing input/flow/output pipelines
- **Streaming ROI statistics and trajectory tracking** without materializing w.h5
- **Batch-friendly callbacks** that accumulate results as frames are processed
- **Optional displacement export** to disk or the workspace when needed
- **Multi-format I/O** supporting HDF5, TIFF, MAT files
- **Flow field visualization** with magnitude, direction, and quiver plots
- **Seamless integration** with napari's layer system

## Quick Example

```python
import napari
from napari_flowreg import FlowRegWidget

# Launch napari and load your data
viewer = napari.Viewer()
viewer.open("my_2p_video.tif")

# Add FlowReg widget
viewer.window.add_plugin_dock_widget("napari-flowreg", "FlowReg")
```

![Screenshot placeholder: napari-flowreg main interface](screenshots/main_interface.png)

## Motion Analysis Modes

The **Flow-Registration Analysis (Experimental)** widget uses a three-step wizard
to describe the processing route:

1. **Input** — work from a loaded napari layer or supply a file path.
2. **Displacement field (w)** — compute the flow now or reuse an on-disk `w.h5`.
3. **Output** — analyse only, write corrected data to disk, or load outputs back into napari.

Each combination maps to an execution mode that picks the right PyFlowReg worker.
Common choices include:

| Input        | Flow         | Output        | What happens                                                                 |
|--------------|--------------|---------------|-------------------------------------------------------------------------------|
| Loaded layer | Compute now  | Analyze only  | Stream batches from the viewer, track/compute stats via callbacks, keep RAM low. |
| Loaded layer | Compute now  | Load to viewer| Same as above but reconstruct the full displacement field for visualization.  |
| File         | Compute now  | Correct & save| Invoke PyFlowReg on disk files, writing corrected data and optional `w.h5`.   |
| File         | Precomputed  | Analyze only  | Read a `w.h5` from disk and run analysis without recomputing flow.           |

See the [Batch Processing Guide](user_guide/batch_processing.md) for detailed memory notes and configuration tips.

## Citation

If you use napari-flowreg in your research, please cite:

> "Pyflowreg," (in preparation), 2025.

and

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, "Software for Non-Parametric Image Registration of 2-Photon Imaging Data," J Biophotonics, 2022. [doi:10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

## Related Projects

Part of the [FlowRegSuite](https://github.com/FlowRegSuite) ecosystem:
- [PyFlowReg](https://github.com/FlowRegSuite/pyflowreg) - Python optical flow engine
- [Flow-Registration (MATLAB)](https://github.com/FlowRegSuite/flow_registration) - Original MATLAB implementation

## Documentation

```{toctree}
:maxdepth: 2

installation
quickstart
api/index
user_guide/index
changelog
```

## License

napari-flowreg is released under the CC BY-NC-SA 4.0 license.
