# Installation

## Requirements

- Python 3.10 or later
- napari 0.4.18 or later
- PyFlowReg

## Install from PyPI

```bash
pip install napari-flowreg
```

## Install from Source

For development:

```bash
git clone https://github.com/FlowRegSuite/napari-flowreg.git
cd napari-flowreg
pip install -e ".[dev]"
```

## Verify Installation

Launch napari and check that FlowReg appears in the Plugins menu:

```bash
napari
```

Or verify programmatically:

```python
import napari
viewer = napari.Viewer()
# Check Plugins > napari-flowreg > FlowReg
```

## Dependencies

`pip install napari-flowreg` pulls in:

- `pyflowreg[vis]` (core engine + visualisation helpers)
- `qtpy`
- `numpy`
- `scipy>=1.10`
- `matplotlib`

You still need a napari installation in your environment. We recommend installing napari first (for example `pip install "napari[all]"` or `conda install -c conda-forge napari`) and then adding napari-flowreg.

## Platform-Specific Notes

### Windows

On Windows, napari-flowreg automatically configures Numba to avoid SVML issues when running in Qt contexts.

### macOS

No special configuration needed.

### Linux

No special configuration needed.

## GPU Support

PyFlowReg can use CUDA acceleration if available. Install with:

```bash
pip install pyflowreg[cuda]
```

## Troubleshooting

### Plugin Not Appearing

If napari-flowreg doesn't appear in the Plugins menu:

1. Check installation: `pip list | grep napari-flowreg`
2. Verify napari version: `napari --version` (need ≥ 0.4.18)
3. Try reinstalling: `pip uninstall napari-flowreg && pip install napari-flowreg`

### Import Errors

If you see import errors, ensure all dependencies are installed:

```bash
pip install napari pyflowreg qtpy
```

### Performance Issues

For better performance:

1. Install with CUDA support (if GPU available)
2. Ensure you have sufficient RAM for your video size
3. Consider using batch processing mode for large videos

## Development Setup

For contributing to napari-flowreg:

```bash
# Clone and install in editable mode
git clone https://github.com/FlowRegSuite/napari-flowreg.git
cd napari-flowreg
pip install -e ".[dev,docs]"

# Run tests
pytest

# Build documentation
cd docs
make html
```
