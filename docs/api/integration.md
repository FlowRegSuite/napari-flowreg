# Integration API

Helper functions wire PyFlowReg workers into napari widgets and expose streaming callbacks.

```{eval-rst}
.. automodule:: napari_flowreg.motion_analysis.integration
   :members:
   :undoc-members:
   :show-inheritance:
```

## Example: creating a worker for the v2 widget

```python
from napari_flowreg.motion_analysis.integration import create_worker_for_widget_v2
from napari_flowreg.motion_analysis.modes import ModeManager, ProcessingMode

manager = ModeManager()
mode_def = manager.definitions[ProcessingMode.LOADED_BATCH_PROCESS]
options = {"batch_size": 50, "enable_trajectories": True}
worker = create_worker_for_widget_v2(widget, mode_def, options)
```
