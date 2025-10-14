"""
Motion analysis callbacks for napari-flowreg.
These classes provide analysis functionality using pyflowreg's callback system.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import warnings


class BatchTrajectoryTracker:
    """
    Computes trajectories on-the-fly from w batches using pyflowreg callbacks.
    Zero additional I/O - processes w while it's in memory during pipeline execution.
    """

    def __init__(
        self,
        seed_points: np.ndarray,  # (N, 2) in Y,X format
        total_frames: int,
        H: int,
        W: int,
        flow_sign: int = 1,
    ):  # +1 if w is reference->frame, -1 if frame->reference
        """
        Initialize tracker with seed points.

        Args:
            seed_points: Initial positions at reference frame (N, 2) [y, x]
            total_frames: Total number of frames to track
            H, W: Video dimensions
            flow_sign: Sign convention for displacement field
        """
        self.seeds = seed_points
        self.n_points = len(seed_points)
        self.total_frames = total_frames
        self.H = H
        self.W = W
        self.flow_sign = flow_sign

        # Pre-allocate trajectory storage
        self.trajectories = np.zeros((self.n_points, total_frames, 2), dtype=np.float32)

        # Initialize all frames with seed positions (will update as batches arrive)
        for t in range(total_frames):
            self.trajectories[:, t] = seed_points

        # Track which frames have been processed
        self.processed_frames = np.zeros(total_frames, dtype=bool)

    def w_callback(self, w_batch: np.ndarray, batch_start: int, batch_end: int):
        """
        Callback for displacement field batches.
        This is called by pyflowreg's BatchMotionCorrector after computing w but before writing to disk.

        Args:
            w_batch: Displacement field (batch_size, H, W, 2) with [u, v]
            batch_start: Starting frame index in video
            batch_end: Ending frame index in video
        """
        batch_size = batch_end - batch_start

        for t_local in range(batch_size):
            t_global = batch_start + t_local

            if t_global >= self.total_frames:
                continue

            # For each seed point, interpolate displacement
            for p_idx in range(self.n_points):
                y0, x0 = self.seeds[p_idx]

                # Bilinear interpolation of displacement at seed position
                u = self._bilinear_interpolate(w_batch[t_local, :, :, 0], x0, y0)
                v = self._bilinear_interpolate(w_batch[t_local, :, :, 1], x0, y0)

                # Apply displacement with sign convention
                yt = np.clip(y0 + self.flow_sign * v, 0, self.H - 1)
                xt = np.clip(x0 + self.flow_sign * u, 0, self.W - 1)

                self.trajectories[p_idx, t_global] = [yt, xt]

            self.processed_frames[t_global] = True

    def _bilinear_interpolate(self, field: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation of 2D field at position (x, y)."""
        H, W = field.shape
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
        fx, fy = x - x0, y - y0

        return (
            field[y0, x0] * (1 - fx) * (1 - fy)
            + field[y0, x1] * fx * (1 - fy)
            + field[y1, x0] * (1 - fx) * fy
            + field[y1, x1] * fx * fy
        )

    def get_tracks_for_napari(self) -> np.ndarray:
        """
        Format trajectories for napari Tracks layer.

        Returns:
            Array of shape (N*T, 4) with columns [track_id, time, y, x]
        """
        tracks = []
        for pid in range(self.n_points):
            for t in range(self.total_frames):
                if self.processed_frames[t]:
                    y, x = self.trajectories[pid, t]
                    tracks.append([pid, t, y, x])
        return np.array(tracks, dtype=np.float32)

    def get_motion_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute motion statistics from trajectories.

        Returns:
            Dictionary with statistics arrays
        """
        # Compute displacement magnitudes
        if self.n_points > 0 and np.any(self.processed_frames):
            # Only compute for processed frames
            valid_frames = np.where(self.processed_frames)[0]
            if len(valid_frames) > 1:
                displacements = np.diff(
                    self.trajectories[:, valid_frames, :], axis=1
                )  # (N, T-1, 2)
                magnitudes = np.sqrt(np.sum(displacements**2, axis=2))  # (N, T-1)

                return {
                    "mean_displacement": np.mean(magnitudes, axis=0),  # Per frame
                    "max_displacement": np.max(magnitudes, axis=0),
                    "per_point_mean": np.mean(magnitudes, axis=1),  # Per point
                    "trajectories": self.trajectories,
                    "processed_frames": self.processed_frames,
                }

        return {
            "mean_displacement": np.array([]),
            "max_displacement": np.array([]),
            "per_point_mean": np.array([]),
            "trajectories": self.trajectories,
            "processed_frames": self.processed_frames,
        }


class ROIMotionAnalyzer:
    """
    Computes motion statistics for ROIs using w batch callbacks.
    """

    def __init__(self, roi_masks: List[np.ndarray], roi_names: List[str]):
        """
        Initialize analyzer with ROI masks.

        Args:
            roi_masks: List of boolean masks (H, W)
            roi_names: Names for each ROI
        """
        self.roi_masks = roi_masks
        self.roi_names = roi_names
        self.n_rois = len(roi_masks)

        # Storage for statistics
        self.frame_stats = []  # Will store per-frame stats

    def w_callback(self, w_batch: np.ndarray, batch_start: int, batch_end: int):
        """
        Compute statistics from w batch.

        Args:
            w_batch: Displacement field (batch_size, H, W, 2)
            batch_start: Starting frame index
            batch_end: Ending frame index
        """
        batch_size = batch_end - batch_start

        for t_local in range(batch_size):
            t_global = batch_start + t_local

            # Compute magnitude
            u = w_batch[t_local, :, :, 0]
            v = w_batch[t_local, :, :, 1]
            magnitude = np.sqrt(u**2 + v**2)

            # Compute stats for each ROI
            frame_roi_stats = {"frame": t_global, "rois": {}}

            for roi_idx, (mask, name) in enumerate(zip(self.roi_masks, self.roi_names)):
                roi_values = magnitude[mask]

                if len(roi_values) > 0:
                    frame_roi_stats["rois"][name] = {
                        "mean": float(np.mean(roi_values)),
                        "max": float(np.max(roi_values)),
                        "min": float(np.min(roi_values)),
                        "std": float(np.std(roi_values)),
                        "median": float(np.median(roi_values)),
                    }

            self.frame_stats.append(frame_roi_stats)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all frames."""
        summary = {}

        for roi_name in self.roi_names:
            roi_means = [
                f["rois"][roi_name]["mean"]
                for f in self.frame_stats
                if roi_name in f["rois"]
            ]

            if roi_means:
                summary[roi_name] = {
                    "time_series_mean": np.array(roi_means),
                    "overall_mean": np.mean(roi_means),
                    "max": np.max(roi_means),
                    "min": np.min(roi_means),
                }

        return summary


class WAccumulator:
    """
    Accumulates displacement field w in memory using callbacks.
    Use with caution - requires significant memory for large videos.
    """

    def __init__(self, shape: tuple):
        """
        Initialize accumulator.

        Args:
            shape: Expected shape (T, H, W, 2) for the full w field
        """
        self.w_accumulated = np.zeros(shape, dtype=np.float32)
        self.shape = shape

    def w_callback(self, w_batch: np.ndarray, batch_start: int, batch_end: int):
        """
        Accumulate w batch in memory.

        Args:
            w_batch: Displacement field batch (batch_size, H, W, 2)
            batch_start: Starting frame index
            batch_end: Ending frame index
        """
        if batch_end <= self.shape[0]:
            self.w_accumulated[batch_start:batch_end] = w_batch
        else:
            warnings.warn(f"Batch end {batch_end} exceeds expected size {self.shape[0]}")

    def get_accumulated(self) -> np.ndarray:
        """Get the accumulated w field."""
        return self.w_accumulated


class ProgressTracker:
    """
    Tracks progress during batch processing using w callbacks.
    Allows progress updates to be sent to UI.
    """

    def __init__(self, total_frames: int, progress_callback: Optional[callable] = None):
        """
        Initialize progress tracker.

        Args:
            total_frames: Total number of frames to process
            progress_callback: Optional callback function(current, total) to report progress
        """
        self.total_frames = total_frames
        self.frames_processed = 0
        self.progress_callback = progress_callback

    def w_callback(self, w_batch: np.ndarray, batch_start: int, batch_end: int):
        """
        Track progress from w batch.

        Args:
            w_batch: Displacement field batch (batch_size, H, W, 2)
            batch_start: Starting frame index
            batch_end: Ending frame index
        """
        self.frames_processed = min(batch_end, self.total_frames)

        # Call progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.frames_processed, self.total_frames)

    def get_progress(self) -> tuple:
        """Get current progress as (current, total)."""
        return (self.frames_processed, self.total_frames)

    def get_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_frames == 0:
            return 0.0
        return 100.0 * self.frames_processed / self.total_frames


class CombinedCallbackDispatcher:
    """
    Dispatches w callbacks to multiple analyzers.
    This is the main callback registered with pyflowreg.
    """

    def __init__(self):
        """Initialize dispatcher with empty callback list."""
        self.callbacks = []

    def add_callback(self, callback):
        """Add a callback to the dispatch list."""
        if hasattr(callback, 'w_callback'):
            self.callbacks.append(callback.w_callback)
        elif callable(callback):
            self.callbacks.append(callback)

    def w_callback(self, w_batch: np.ndarray, batch_start: int, batch_end: int):
        """
        Dispatch w batch to all registered callbacks.

        Args:
            w_batch: Displacement field batch
            batch_start: Starting frame index
            batch_end: Ending frame index
        """
        for callback in self.callbacks:
            try:
                callback(w_batch, batch_start, batch_end)
            except Exception as e:
                warnings.warn(f"Callback error: {str(e)}")


def extract_roi_masks_from_shapes(shapes_layer, video_shape, target_layer=None):
    """
    Convert napari Shapes layer to binary masks.

    Args:
        shapes_layer: napari Shapes layer
        video_shape: Shape of the video (T, H, W) or (T, H, W, C)
        target_layer: Optional target layer for coordinate transforms

    Returns:
        Tuple of (masks, names) where masks is list of binary arrays

    Note:
        If target_layer is provided and both layers have transforms, coordinates
        will be converted from Shapes space → world → target space.
        Otherwise, raw data coordinates are used (assumes aligned coordinate spaces).
    """
    from skimage.draw import polygon2mask

    H, W = video_shape[1:3] if len(video_shape) >= 3 else video_shape[:2]

    roi_masks = []
    roi_names = []

    for idx, shape_data in enumerate(shapes_layer.data):
        shape_type = shapes_layer.shape_types[idx] if hasattr(shapes_layer, 'shape_types') else 'polygon'

        if shape_type in ('rectangle', 'polygon'):
            # Apply coordinate transforms if available
            if target_layer is not None and hasattr(shapes_layer, 'data_to_world') and hasattr(target_layer, 'world_to_data'):
                # Transform: Shapes data → world → target data
                world_pts = shapes_layer.data_to_world(shape_data)
                if isinstance(world_pts, tuple):
                    world_pts = np.stack(world_pts, axis=1)
                data_pts = target_layer.world_to_data(world_pts)
                if isinstance(data_pts, tuple):
                    data_pts = np.stack(data_pts, axis=1)
                ys = data_pts[:, -2]
                xs = data_pts[:, -1]
            else:
                # No transforms - use raw coordinates
                ys = shape_data[:, -2]
                xs = shape_data[:, -1]

            # Create polygon and convert to mask
            poly = np.stack([ys, xs], axis=1)
            mask = polygon2mask((H, W), poly)
            roi_masks.append(mask)

            # Get name from properties if available
            name = f"ROI_{idx}"
            if hasattr(shapes_layer, 'properties') and 'name' in shapes_layer.properties:
                if idx < len(shapes_layer.properties['name']):
                    custom_name = shapes_layer.properties['name'][idx]
                    if custom_name:
                        name = custom_name
            roi_names.append(name)

    return roi_masks, roi_names


def extract_seed_points_from_layer(layer):
    """
    Extract seed points from a napari Points or Shapes layer.

    Args:
        layer: napari Points or Shapes layer

    Returns:
        Array of shape (N, 2) with [y, x] coordinates
    """
    from napari.layers import Points, Shapes

    if isinstance(layer, Points):
        # Direct extraction from Points layer
        coords = np.asarray(layer.data, dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            # Extract Y, X from last 2 dimensions
            return coords[:, -2:].copy()

    elif isinstance(layer, Shapes):
        # Extract point shapes
        points = []
        if hasattr(layer, 'shape_types'):
            for idx, shape_type in enumerate(layer.shape_types):
                if shape_type in ('point', 'points'):
                    shape_data = np.asarray(layer.data[idx], dtype=float)
                    if shape_data.ndim == 1:
                        shape_data = shape_data[None, :]
                    # Extract Y, X coordinates
                    points.append(shape_data[:, -2:])

        if points:
            return np.vstack(points)

    return np.array([])  # Empty array if no points found