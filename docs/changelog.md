# Changelog

All notable changes to napari-flowreg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Batch processing modes for memory-efficient motion analysis
  - Online mode: analyze pre-loaded flow fields
  - Batch process mode: compute flow from video in memory-efficient batches
  - Hybrid mode (planned): combine in-memory video with flow fields streamed from disk
  - Stream mode (planned): full file-to-file processing pipeline
- Zero-copy trajectory tracking via PyFlowReg callback system
- ROI motion analysis with on-the-fly statistics computation during batch processing
- Mode selector UI with dynamic memory usage indicators
- `BatchTrajectoryTracker` for tracking seed points through flow field batches
- `ROIMotionAnalyzer` for computing per-ROI statistics during batch processing
- `WAccumulator` for accumulating displacement fields in memory
- `CombinedCallbackDispatcher` for routing callbacks to multiple analyzers
- Processing mode definitions and validation system
- Integration functions for adding batch processing to existing widgets
- Comprehensive test suite for batch processing callbacks and integration
- ReadTheDocs documentation structure with user guides and API reference

### Changed
- Motion Analysis Widget now supports multiple processing modes with dynamic UI
- Enhanced memory management for large dataset processing

## [0.1.0a3] - 2025-09-28

### Added
- Point tracking functionality in Motion Analysis Widget
- Bilinear interpolation for sub-pixel accurate point tracking
- Track visualization with napari Tracks layers
- Points layer visualization spanning all frames
- Comprehensive motion analysis test suite with bilinear interpolation tests

### Fixed
- Windows CI test failures caused by Numba SVML symbol resolution issues
- PyQt6 API compatibility issues in QSignalSpy usage
- Progress callback test flakiness across different Qt backends
- Cross-platform test reliability and stability
- Viewer.add_image patching in tests for proper napari integration
- Dock widgets iteration compatibility

### Changed
- Migrated test suite to PyQt6 for consistency across platforms
- Improved Windows CI workflow with explicit platform handling
- Enhanced test robustness with platform-specific handling
- Disabled Numba SVML on Windows to prevent Qt context conflicts

## [0.1.0a2] - 2025-08-25

### Added
- Python 3.13 support in project configuration
- CI/CD status badges to README
- Comprehensive test workflow for continuous integration
- PyPI-compatible README with corrected image paths for package repository

### Fixed
- PyPI deployment workflow configuration
- Image path rendering in PyPI package description
- HTML image tag compatibility in CI pipeline
- Version string formatting in setuptools-scm configuration

### Changed
- Refactored test suite for improved maintainability
- Improved package metadata for PyPI presentation

## [0.1.0a1] - 2025-08-22

### Added
- Initial alpha release of napari-flowreg
- FlowReg Widget for interactive motion correction
  - Integration with PyFlowReg optical flow engine
  - Real-time parameter configuration
  - Progress bar with callback handling
- Motion Analysis Widget for flow field analysis
  - ROI-based motion statistics
  - Multi-ROI support with individual shape tracking
  - Export functionality to MAT/NPZ formats
- Flow field visualization modes:
  - Middlebury color coding (HSV for direction and magnitude)
  - Flow divergence visualization
  - Quiver plot overlays with configurable density
- Multi-channel video support with automatic handling
- File format support: HDF5, TIFF, MAT, MDF
- Comprehensive test suite with pytest
- CI/CD pipeline with automated PyPI deployment
- Auto-versioning with setuptools-scm

### Fixed
- Numba parallelization conflicts in Qt GUI context on Windows
- Python 3.10 compatibility with tomllib (fallback to tomli package)
- macOS CI vispy OpenGL segmentation faults in headless environment
- Ubuntu CI dependency resolution with updated Mesa packages

### Changed
- Moved napari to optional dependencies following napari plugin best practices
- Organized package metadata and project URLs
- Established development workflow and testing infrastructure

## Initial Development - 2025-07

### Added
- Project repository structure and configuration
- Basic napari plugin framework
- FlowReg branding and logo
- Initial documentation structure

[Unreleased]: https://github.com/FlowRegSuite/napari-flowreg/compare/v0.1.0a3...HEAD
[0.1.0a3]: https://github.com/FlowRegSuite/napari-flowreg/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/FlowRegSuite/napari-flowreg/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/FlowRegSuite/napari-flowreg/releases/tag/v0.1.0a1