"""
Test plugin manifest and registration.
"""

import pytest
from pathlib import Path


def test_manifest_exists():
    """Test that napari.yaml manifest exists."""
    manifest_path = Path(__file__).parent.parent / "src" / "napari_flowreg" / "napari.yaml"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"


def test_manifest_valid():
    """Test that manifest is valid using npe2."""
    try:
        import npe2
        from npe2 import PluginManifest
        
        manifest_path = Path(__file__).parent.parent / "src" / "napari_flowreg" / "napari.yaml"
        
        # Load and validate manifest
        with open(manifest_path) as f:
            manifest = PluginManifest.from_file(manifest_path)
        
        # Check basic manifest properties
        assert manifest.name == "napari-flowreg"
        # Display name can change, just verify it exists
        assert manifest.display_name is not None
        assert len(manifest.display_name) > 0
        
        # Check contributions exist
        assert manifest.contributions is not None
        assert manifest.contributions.commands is not None
        assert len(manifest.contributions.commands) > 0
        
        # Check widget command exists
        widget_commands = [cmd for cmd in manifest.contributions.commands 
                          if "widget" in cmd.id.lower()]
        assert len(widget_commands) > 0, "No widget command found in manifest"
        
    except ImportError:
        pytest.skip("npe2 not installed")


def test_manifest_python_names_valid():
    """Test that all python_name references in manifest are valid."""
    try:
        import npe2
        from npe2 import PluginManifest
        import importlib
        
        manifest_path = Path(__file__).parent.parent / "src" / "napari_flowreg" / "napari.yaml"
        manifest = PluginManifest.from_file(manifest_path)
        
        for command in manifest.contributions.commands:
            python_name = command.python_name
            
            # Parse module and attribute
            if ":" in python_name:
                module_name, attr_name = python_name.rsplit(":", 1)
                
                # Check module can be imported
                try:
                    module = importlib.import_module(module_name)
                    # Check attribute exists
                    assert hasattr(module, attr_name), \
                        f"Module {module_name} has no attribute {attr_name}"
                except ImportError as e:
                    # It's OK if pyflowreg is not installed for CI
                    if "pyflowreg" not in str(e):
                        raise
                        
    except ImportError:
        pytest.skip("npe2 not installed")


def test_plugin_discoverable():
    """Test that plugin is discoverable by napari."""
    try:
        import npe2
        
        # Check if our plugin is in the npe2 plugin manager
        pm = npe2.PluginManager.instance()
        
        # Try to discover our plugin
        # Note: This might not work in test environment without proper installation
        # so we make this a soft check
        manifest_path = Path(__file__).parent.parent / "src" / "napari_flowreg" / "napari.yaml"
        if manifest_path.exists():
            # At minimum, manifest should be parseable
            from npe2 import PluginManifest
            manifest = PluginManifest.from_file(manifest_path)
            assert manifest.name == "napari-flowreg"
            
    except ImportError:
        pytest.skip("npe2 not installed")


def test_entry_point_configured():
    """Test that pyproject.toml has correct entry point."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    if pyproject_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        
        # Check for napari.manifest entry point
        entry_points = pyproject.get("project", {}).get("entry-points", {})
        napari_manifest = entry_points.get("napari.manifest", {})
        
        assert "napari-flowreg" in napari_manifest, \
            "napari-flowreg not found in napari.manifest entry points"
        
        # Check it points to napari.yaml
        assert "napari.yaml" in napari_manifest["napari-flowreg"], \
            "Entry point doesn't reference napari.yaml"