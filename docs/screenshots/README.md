# Documentation screenshots

The docs reference two PNGs from this directory that still need to be
captured. Commit them here — the repository `.gitignore` explicitly
re-includes `docs/screenshots/*.png` despite the global `*.png` rule.

General settings: napari light theme, window about 1400 px wide, PNG
format, no personal paths visible in title bars or file fields.

## main_interface.png (referenced by `docs/index.md`)

- Start napari and load a sample 2-photon frame or a short video as an
  Image layer.
- Dock the main widget: *Plugins → napari-flowreg → Flow-Registration*.
- Show the canvas with the 2P frame and the FlowReg parameter panel
  docked on the right; leave the default parameter values visible.

## motion_analysis_v2.png (referenced by `docs/user_guide/motion_analysis.md`)

- Dock *Plugins → napari-flowreg → Flow-Registration Analysis (Experimental)*.
- Make sure the 3-step wizard (Input / Displacement field / Output), the
  Data Selection selectors with their status pills, and the
  Tracking/Statistics tabs are all visible; select a video layer so the
  pill shows a frame count.
- Crop to the docked widget plus a slice of the canvas for context.
