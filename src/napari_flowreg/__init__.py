"""
napari-flowreg: A napari plugin for Flow-Registration motion correction.
"""

import os
import sys
from pathlib import Path

# Fix for Windows SVML/Numba issue when running in Qt context
# Must be done before any Numba imports
if sys.platform == "win32":
    os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")
    os.environ.setdefault("NUMBA_CPU_NAME", "generic")
    cache_root = Path.home() / ".napari-flowreg" / "numba-cache-nosvml"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root))
    dll = Path(sys.prefix) / "Library" / "bin"
    if dll.exists():
        os.add_dll_directory(str(dll))

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .flowreg_widget import FlowRegWidget

__all__ = ["FlowRegWidget"]