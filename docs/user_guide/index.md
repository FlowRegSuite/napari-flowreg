# User Guide

This guide covers common workflows and features in napari-flowreg.

## Contents

```{toctree}
:maxdepth: 2

motion_analysis
batch_processing
```

## Overview

napari-flowreg provides two main widgets:

1. **Flow-Registration** – the primary motion-correction widget that mirrors the PyFlowReg GUI.
2. **Flow-Registration Analysis (Experimental)** – the wizard-based motion analysis widget introduced in v2 with streaming callbacks, ROI statistics, and trajectory tracking.

Both widgets integrate with napari's layer system and can share optical-flow settings.

## Getting Help

- Check the [Quick Start](../quickstart.md) for basic usage
- See [API Reference](../api/index.md) for detailed function documentation
- Report issues on [GitHub](https://github.com/FlowRegSuite/napari-flowreg/issues)
