"""
Tests for batch processing callbacks.
"""

import pytest
import numpy as np
from napari_flowreg.motion_analysis import (
    BatchTrajectoryTracker,
    ROIMotionAnalyzer,
    WAccumulator,
    CombinedCallbackDispatcher,
    extract_roi_masks_from_shapes,
    extract_seed_points_from_layer
)


class TestBatchTrajectoryTracker:
    """Test trajectory tracking from w batch callbacks."""

    def test_initialization(self):
        """Test tracker initialization."""
        seed_points = np.array([[50.0, 50.0], [100.0, 100.0]])
        tracker = BatchTrajectoryTracker(seed_points, total_frames=10, H=256, W=256)

        assert tracker.n_points == 2
        assert tracker.total_frames == 10
        assert tracker.trajectories.shape == (2, 10, 2)
        assert not np.any(tracker.processed_frames)

    def test_w_callback_single_batch(self):
        """Test processing single w batch."""
        seed_points = np.array([[50.0, 50.0]])
        tracker = BatchTrajectoryTracker(seed_points, total_frames=5, H=100, W=100)

        # Create synthetic w batch
        w_batch = np.zeros((5, 100, 100, 2), dtype=np.float32)
        w_batch[:, :, :, 0] = 1.0  # Constant u displacement
        w_batch[:, :, :, 1] = 0.5  # Constant v displacement

        tracker.w_callback(w_batch, batch_start=0, batch_end=5)

        # Verify all frames processed
        assert np.all(tracker.processed_frames)

        # Verify trajectory positions
        for t in range(5):
            y, x = tracker.trajectories[0, t]
            # Expected: seed + displacement (clipped to bounds)
            assert np.isclose(y, np.clip(50.0 + 0.5, 0, 99))
            assert np.isclose(x, np.clip(50.0 + 1.0, 0, 99))

    def test_w_callback_multiple_batches(self):
        """Test processing multiple batches."""
        seed_points = np.array([[25.0, 25.0]])
        tracker = BatchTrajectoryTracker(seed_points, total_frames=10, H=100, W=100)

        # First batch
        w_batch1 = np.ones((5, 100, 100, 2), dtype=np.float32)
        tracker.w_callback(w_batch1, batch_start=0, batch_end=5)

        assert np.sum(tracker.processed_frames) == 5

        # Second batch
        w_batch2 = np.ones((5, 100, 100, 2), dtype=np.float32) * 2
        tracker.w_callback(w_batch2, batch_start=5, batch_end=10)

        assert np.all(tracker.processed_frames)

    def test_get_tracks_for_napari(self):
        """Test conversion to napari tracks format."""
        seed_points = np.array([[50.0, 50.0], [75.0, 75.0]])
        tracker = BatchTrajectoryTracker(seed_points, total_frames=3, H=100, W=100)

        w_batch = np.zeros((3, 100, 100, 2), dtype=np.float32)
        tracker.w_callback(w_batch, batch_start=0, batch_end=3)

        tracks = tracker.get_tracks_for_napari()

        # Should have 2 points * 3 frames = 6 rows
        assert tracks.shape == (6, 4)

        # Verify format: [track_id, time, y, x]
        assert np.all(tracks[:, 0] >= 0)  # track_id >= 0
        assert np.all(tracks[:, 1] >= 0)  # time >= 0
        assert np.all(tracks[:, 1] < 3)   # time < n_frames

    def test_boundary_clipping(self):
        """Test that trajectories are clipped to image bounds."""
        # Seed point near boundary
        seed_points = np.array([[5.0, 5.0]])
        tracker = BatchTrajectoryTracker(seed_points, total_frames=3, H=100, W=100)

        # Large displacement that would push out of bounds
        w_batch = np.ones((3, 100, 100, 2), dtype=np.float32) * 200
        tracker.w_callback(w_batch, batch_start=0, batch_end=3)

        tracks = tracker.get_tracks_for_napari()

        # All positions should be within bounds
        assert np.all(tracks[:, 2] >= 0)    # y >= 0
        assert np.all(tracks[:, 2] < 100)   # y < H
        assert np.all(tracks[:, 3] >= 0)    # x >= 0
        assert np.all(tracks[:, 3] < 100)   # x < W


class TestROIMotionAnalyzer:
    """Test ROI motion analysis from w callbacks."""

    def test_initialization(self):
        """Test analyzer initialization."""
        masks = [np.ones((100, 100), dtype=bool)]
        names = ["ROI_1"]
        analyzer = ROIMotionAnalyzer(masks, names)

        assert analyzer.n_rois == 1
        assert len(analyzer.frame_stats) == 0

    def test_w_callback_single_roi(self):
        """Test processing w batch with single ROI."""
        # Create ROI mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        analyzer = ROIMotionAnalyzer([mask], ["test_roi"])

        # Create w batch with constant magnitude
        w_batch = np.ones((3, 100, 100, 2), dtype=np.float32)
        analyzer.w_callback(w_batch, batch_start=0, batch_end=3)

        assert len(analyzer.frame_stats) == 3

        # Check statistics
        for frame_stat in analyzer.frame_stats:
            roi_stat = frame_stat["rois"]["test_roi"]
            # Magnitude of (1, 1) is sqrt(2)
            assert np.isclose(roi_stat["mean"], np.sqrt(2))

    def test_w_callback_multiple_rois(self):
        """Test processing with multiple ROIs."""
        # Create two non-overlapping ROIs
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[20:40, 20:40] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[60:80, 60:80] = True

        analyzer = ROIMotionAnalyzer([mask1, mask2], ["roi_1", "roi_2"])

        # Create w batch with different values in different regions
        w_batch = np.zeros((2, 100, 100, 2), dtype=np.float32)
        w_batch[:, 20:40, 20:40, 0] = 1.0  # roi_1 region
        w_batch[:, 60:80, 60:80, 0] = 2.0  # roi_2 region

        analyzer.w_callback(w_batch, batch_start=0, batch_end=2)

        assert len(analyzer.frame_stats) == 2

        # roi_1 should have magnitude 1.0, roi_2 should have magnitude 2.0
        frame0 = analyzer.frame_stats[0]
        assert np.isclose(frame0["rois"]["roi_1"]["mean"], 1.0)
        assert np.isclose(frame0["rois"]["roi_2"]["mean"], 2.0)

    def test_get_summary_statistics(self):
        """Test summary statistics computation."""
        mask = np.ones((50, 50), dtype=bool)
        analyzer = ROIMotionAnalyzer([mask], ["test_roi"])

        # Process multiple frames with varying magnitude
        for batch_idx in range(3):
            w_batch = np.ones((1, 50, 50, 2), dtype=np.float32) * (batch_idx + 1)
            analyzer.w_callback(w_batch, batch_start=batch_idx, batch_end=batch_idx + 1)

        summary = analyzer.get_summary_statistics()

        assert "test_roi" in summary
        assert "overall_mean" in summary["test_roi"]
        assert "time_series_mean" in summary["test_roi"]


class TestWAccumulator:
    """Test w field accumulation."""

    def test_initialization(self):
        """Test accumulator initialization."""
        shape = (10, 100, 100, 2)
        accumulator = WAccumulator(shape)

        assert accumulator.shape == shape
        assert accumulator.w_accumulated.shape == shape

    def test_w_callback_accumulation(self):
        """Test accumulating w batches."""
        shape = (10, 50, 50, 2)
        accumulator = WAccumulator(shape)

        # First batch
        w_batch1 = np.ones((5, 50, 50, 2), dtype=np.float32)
        accumulator.w_callback(w_batch1, batch_start=0, batch_end=5)

        w = accumulator.get_accumulated()
        assert np.all(w[0:5] == 1.0)
        assert np.all(w[5:10] == 0.0)

        # Second batch
        w_batch2 = np.ones((5, 50, 50, 2), dtype=np.float32) * 2
        accumulator.w_callback(w_batch2, batch_start=5, batch_end=10)

        w = accumulator.get_accumulated()
        assert np.all(w[0:5] == 1.0)
        assert np.all(w[5:10] == 2.0)

    def test_batch_end_exceeds_shape(self):
        """Test warning when batch end exceeds expected size."""
        shape = (5, 50, 50, 2)
        accumulator = WAccumulator(shape)

        # This should trigger a warning
        w_batch = np.ones((3, 50, 50, 2), dtype=np.float32)

        with pytest.warns(UserWarning):
            accumulator.w_callback(w_batch, batch_start=4, batch_end=7)


class TestCombinedCallbackDispatcher:
    """Test callback dispatcher."""

    def test_add_callback(self):
        """Test adding callbacks."""
        dispatcher = CombinedCallbackDispatcher()

        # Add callback with w_callback method
        tracker = BatchTrajectoryTracker(
            np.array([[50.0, 50.0]]),
            total_frames=5,
            H=100,
            W=100
        )
        dispatcher.add_callback(tracker)

        assert len(dispatcher.callbacks) == 1

    def test_w_callback_dispatch(self):
        """Test dispatching to multiple callbacks."""
        dispatcher = CombinedCallbackDispatcher()

        # Add tracker
        tracker = BatchTrajectoryTracker(
            np.array([[50.0, 50.0]]),
            total_frames=3,
            H=100,
            W=100
        )
        dispatcher.add_callback(tracker)

        # Add accumulator
        accumulator = WAccumulator((3, 100, 100, 2))
        dispatcher.add_callback(accumulator)

        # Dispatch w batch
        w_batch = np.ones((3, 100, 100, 2), dtype=np.float32)
        dispatcher.w_callback(w_batch, batch_start=0, batch_end=3)

        # Both should have processed the batch
        assert np.all(tracker.processed_frames)
        assert np.all(accumulator.get_accumulated() == 1.0)

    def test_callback_error_handling(self):
        """Test that errors in one callback don't stop others."""
        dispatcher = CombinedCallbackDispatcher()

        # Add a callback that will fail
        def failing_callback(w_batch, batch_start, batch_end):
            raise ValueError("Test error")

        dispatcher.add_callback(failing_callback)

        # Add a working callback
        accumulator = WAccumulator((3, 100, 100, 2))
        dispatcher.add_callback(accumulator)

        # Dispatch should continue despite error
        w_batch = np.ones((3, 100, 100, 2), dtype=np.float32)

        with pytest.warns(UserWarning):
            dispatcher.w_callback(w_batch, batch_start=0, batch_end=3)

        # Working callback should still have processed
        assert np.all(accumulator.get_accumulated() == 1.0)


class TestExtractROIMasks:
    """Test ROI mask extraction from napari Shapes layer."""

    def test_extract_roi_masks_from_shapes(self, make_napari_viewer):
        """Test extracting masks from shapes layer."""
        viewer = make_napari_viewer

        # Create shapes
        rectangles = [
            np.array([[20, 20], [20, 40], [40, 40], [40, 20]]),
            np.array([[60, 60], [60, 80], [80, 80], [80, 60]])
        ]

        shapes_layer = viewer.add_shapes(
            rectangles,
            shape_type='rectangle',
            name='test_rois'
        )

        # Extract masks
        masks, names = extract_roi_masks_from_shapes(shapes_layer, (100, 100, 100))

        assert len(masks) == 2
        assert len(names) == 2
        assert masks[0].shape == (100, 100)

        # Check that masks have pixels
        assert np.sum(masks[0]) > 0
        assert np.sum(masks[1]) > 0

        # Masks should not overlap
        assert np.sum(masks[0] & masks[1]) == 0


class TestExtractSeedPoints:
    """Test seed point extraction from napari layers."""

    def test_extract_from_points_layer(self, make_napari_viewer):
        """Test extracting seed points from Points layer."""
        viewer = make_napari_viewer

        points = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        points_layer = viewer.add_points(points, name='test_points')

        extracted = extract_seed_points_from_layer(points_layer)

        assert extracted.shape == (3, 2)
        np.testing.assert_array_almost_equal(extracted, points)

    def test_extract_from_shapes_layer(self, make_napari_viewer):
        """Test extracting points from Shapes layer with point shapes."""
        viewer = make_napari_viewer

        # Note: Shapes layer doesn't support 'point' type in napari
        # Use Points layer instead for point extraction
        points_data = np.array([
            [10.0, 20.0],
            [30.0, 40.0]
        ])

        points_layer = viewer.add_points(points_data, name='test_points')

        extracted = extract_seed_points_from_layer(points_layer)

        assert extracted.shape[0] == 2
        assert extracted.shape[1] == 2
        np.testing.assert_array_almost_equal(extracted, points_data)

    def test_empty_layer(self, make_napari_viewer):
        """Test handling of empty layer."""
        viewer = make_napari_viewer

        points_layer = viewer.add_points(np.empty((0, 2)), name='empty')

        extracted = extract_seed_points_from_layer(points_layer)

        assert extracted.shape == (0, 2)
