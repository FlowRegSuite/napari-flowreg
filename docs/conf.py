# Configuration file for Sphinx documentation

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "napari-flowreg"
author = "Philipp Flotho"
copyright = "2025, Philipp Flotho"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/FlowRegSuite/napari-flowreg",
    "logo": {"text": "napari-flowreg"},
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "linkify",
    "dollarmath",
]
myst_heading_anchors = 3
myst_url_schemes = ["http", "https"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = False
napoleon_use_rtype = False

autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autosummary_generate = True

# Mock imports that are unavailable or un-importable on Read the Docs:
# napari is not a package dependency; qtpy raises without a Qt binding;
# matplotlib's qt5agg backend needs Qt at import time; skimage is only a
# transitive dependency; pyflowreg pulls the numba/llvmlite stack and is
# documented via intersphinx instead.
autodoc_mock_imports = [
    "napari",
    "qtpy",
    "matplotlib",
    "skimage",
    "pyflowreg",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "napari": ("https://napari.org/stable/", None),
    "pyflowreg": ("https://pyflowreg.readthedocs.io/en/latest/", None),
}

extlinks = {
    "napari": ("https://napari.org/stable/%s", ""),
}

bibtex_bibfiles = ["references.bib"]

nitpicky = True
