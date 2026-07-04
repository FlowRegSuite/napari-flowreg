# Modes API

Processing-mode definitions drive the wizard experience in the motion analysis widget.

```{eval-rst}
.. automodule:: napari_flowreg.motion_analysis.modes
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
from napari_flowreg.motion_analysis.modes import ModeManager, ProcessingMode

manager = ModeManager()
definition = manager.definitions[ProcessingMode.LOADED_BATCH_PROCESS]
print(definition.display_name)
```
