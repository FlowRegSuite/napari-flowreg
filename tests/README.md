# napari-flowreg Test Suite

This directory contains the pytest-based test suite for the napari-flowreg plugin.

## Test Structure

```
tests/
├── conftest.py                  # Pytest configuration, fixtures, and hermetic environment setup
├── test_manifest.py             # Plugin manifest and registration tests
├── test_widget.py               # Widget functionality and UI interaction tests
├── test_io.py                   # Reader/writer round-trip tests
├── test_integration.py         # End-to-end motion correction tests (slow)
└── README.md                    # This file
```

## Prerequisites

### Install Test Dependencies

```bash
# Install the plugin with test dependencies
pip install -e ".[testing]"

# Or install test dependencies separately
pip install pytest pytest-cov pytest-qt npe2 PySide6 h5py tifffile psutil tomli
```

### Optional: Install pyflowreg for Integration Tests

```bash
pip install pyflowreg
```

Note: Some tests will be skipped if pyflowreg is not installed.

## Running Tests

### Quick Test Run (Fast Tests Only)

```bash
# Run all fast tests, skip slow integration tests
pytest tests/ -m "not slow"
```

### Run All Tests

```bash
# Run complete test suite including slow tests
pytest tests/
```

### Run Specific Test Categories

```bash
# Run only manifest validation tests
pytest tests/test_manifest.py

# Run only widget tests (requires Qt)
pytest tests/test_widget.py

# Run only I/O tests
pytest tests/test_io.py

# Run only slow integration tests
pytest tests/ -m "slow"

# Run tests requiring Qt event loop
pytest tests/ -m "qt"
```

### Run with Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=napari_flowreg --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

### Verbose Output and Debugging

```bash
# Show test names as they run
pytest tests/ -v

# Show detailed output
pytest tests/ -vv

# Don't capture stdout (see print statements)
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb
```

## Test Categories

### Manifest Tests (`test_manifest.py`)
- **Fast**: < 1 second
- Validates `napari.yaml` exists and is well-formed
- Checks all `python_name` references are valid
- Verifies plugin is discoverable by npe2
- Validates `pyproject.toml` entry points

### Widget Tests (`test_widget.py`)
- **Fast**: 1-2 seconds
- **Requires**: Qt backend (PySide6/PyQt5)
- Tests widget creation and dock integration
- Validates UI element presence and wiring
- Tests layer list updates
- Verifies parameter controls enable/disable logic
- Tests options dictionary creation
- Validates error handling without pyflowreg

### I/O Tests (`test_io.py`)
- **Fast**: 1-2 seconds
- Tests reader accepts/rejects appropriate file types
- Validates HDF5 and TIFF round-trip preservation
- Tests multi-channel data handling
- Verifies metadata preservation
- Tests handling of corrupt files

### Integration Tests (`test_integration.py`)
- **Slow**: 10-30 seconds per test
- **Requires**: pyflowreg installation
- Tests end-to-end motion correction on small datasets
- Validates multiprocessing with SVML disabled
- Tests widget with real processing
- Verifies memory efficiency
- Tests error handling in worker threads

## Test Fixtures

Key fixtures provided by `conftest.py`:

### Environment
- `hermetic_environment`: Ensures tests run in isolated environment with SVML disabled

### Viewers
- `make_napari_viewer_proxy`: Creates headless napari viewer (newer napari)
- `make_napari_viewer`: Creates headless napari viewer (older napari fallback)

### Test Data
- `sample_video_2d`: 10 frames, 32x32 pixels with synthetic motion
- `sample_video_3d`: 10 frames, 16x32x32 voxels
- `sample_video_multichannel`: 10 frames, 32x32 pixels, 2 channels
- `temp_directory`: Temporary directory for file I/O tests
- `mock_pyflowreg_options`: Mock options dict for testing without pyflowreg

## CI/CD Configuration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[testing]"
        pip install pyflowreg  # For integration tests
    
    - name: Run fast tests
      run: pytest tests/ -m "not slow" --cov=napari_flowreg
    
    - name: Run slow tests (nightly only)
      if: github.event_name == 'schedule'
      run: pytest tests/ -m "slow"
```

### Local Development Workflow

```bash
# During development - run fast tests frequently
pytest tests/ -m "not slow" -x

# Before committing - run all tests with coverage
pytest tests/ --cov=napari_flowreg --cov-report=term-missing

# Check specific functionality after changes
pytest tests/test_widget.py -v  # After widget changes
pytest tests/test_manifest.py    # After manifest changes
```

## Hermetic Test Environment

The test suite enforces a hermetic environment to prevent "works on my machine" issues:

1. **SVML Disabled**: `NUMBA_DISABLE_INTEL_SVML=1` prevents Numba JIT crashes
2. **Generic CPU**: `NUMBA_CPU_NAME=generic` ensures consistent codegen
3. **Isolated Cache**: Temporary Numba cache prevents cross-test pollution
4. **Spawn Method**: Forces spawn on POSIX to mirror Windows semantics

This configuration is automatically applied via the `hermetic_environment` fixture.

## Platform-Specific Notes

### Windows
- Tests run with spawn multiprocessing by default
- SVML is forcefully disabled to prevent Qt context issues
- DLL directory additions handled automatically

### Linux
- Install `pytest-xvfb` for headless Qt testing in CI
- May need `xvfb-run` prefix for Qt tests without display

### macOS
- Qt tests may require display access
- Consider using GitHub Actions' built-in display

## Troubleshooting

### "No Qt bindings found"
```bash
pip install PySide6>=6.5
```

### "pyflowreg not installed" (skipped tests)
```bash
pip install pyflowreg
```

### "LLVM ERROR: Symbol not found: __svml_pow8"
The hermetic environment fixture should prevent this. If it occurs:
1. Ensure tests are run through pytest (not directly)
2. Check that `NUMBA_DISABLE_INTEL_SVML=1` is set
3. Clear Numba cache: `rm -rf ~/.numba_cache`

