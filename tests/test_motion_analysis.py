"""
Focused tests for Motion Analysis Widget.
Tests actual functionality, not mocked behavior.
"""

import pytest
import numpy as np
from napari_flowreg.motion_analysis_widget import MotionAnalysisWidget


class TestBilinearInterpolation:
    """Test the bilinear interpolation function."""

    def test_exact_grid_points(self, make_napari_viewer, qtbot):
        """Test interpolation at exact grid points returns exact values."""
        # Create a widget instance with a real napari viewer
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Simple 2x2 field
        field = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        # Test exact grid points - should return exact values
        # Note: _bilinear_interpolate takes (field, x, y) where x is column, y is row
        assert widget._bilinear_interpolate(field, 0, 0) == 1.0  # field[0,0]
        assert widget._bilinear_interpolate(field, 1, 0) == 2.0  # field[0,1]
        assert widget._bilinear_interpolate(field, 0, 1) == 3.0  # field[1,0]
        assert widget._bilinear_interpolate(field, 1, 1) == 4.0  # field[1,1]

    def test_center_interpolation(self, make_napari_viewer, qtbot):
        """Test interpolation at center of grid."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        field = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        # Center point should be average of all four corners
        center_val = widget._bilinear_interpolate(field, 0.5, 0.5)
        assert np.isclose(center_val, 2.5)

    def test_edge_interpolation(self, make_napari_viewer, qtbot):
        """Test interpolation along edges."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        field = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        # Along top edge
        edge_val = widget._bilinear_interpolate(field, 0.5, 0)
        assert np.isclose(edge_val, 1.5)  # Average of 1 and 2

        # Along left edge
        edge_val = widget._bilinear_interpolate(field, 0, 0.5)
        assert np.isclose(edge_val, 2.0)  # Average of 1 and 3

    def test_boundary_handling(self, make_napari_viewer, qtbot):
        """Test that interpolation handles boundaries correctly."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        field = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        # Test points at or beyond boundaries
        # The function clips x to [0, W-1] and y to [0, H-1]
        # For a 3x3 field, max x is 2, max y is 2

        # Test negative coordinates get clipped to 0
        assert widget._bilinear_interpolate(field, -0.5, 0.5) == widget._bilinear_interpolate(field, 0, 0.5)

        # Test coordinates beyond max get clipped to max valid index
        # 3.5 gets clipped to 2, 1.5 stays as is
        val_clipped = widget._bilinear_interpolate(field, 3.5, 1.5)
        val_max = widget._bilinear_interpolate(field, 2, 1.5)
        assert np.isclose(val_clipped, val_max)

        # Test corner values - exact grid points
        assert widget._bilinear_interpolate(field, 0, 0) == 1.0
        assert widget._bilinear_interpolate(field, 2, 2) == 9.0

    def test_fractional_interpolation(self, make_napari_viewer, qtbot):
        """Test interpolation with various fractional positions."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Create a gradient field for predictable interpolation
        field = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0]
        ])

        # Test various fractional positions
        # At (0.25, 0.25), should be weighted average
        val = widget._bilinear_interpolate(field, 0.25, 0.25)
        expected = (0.0 * 0.75 * 0.75 +  # (0,0) weight
                   1.0 * 0.25 * 0.75 +   # (0,1) weight
                   1.0 * 0.75 * 0.25 +   # (1,0) weight
                   2.0 * 0.25 * 0.25)    # (1,1) weight
        assert np.isclose(val, expected)

        # At (1.5, 1.5), center of bottom-right 2x2 subgrid
        val = widget._bilinear_interpolate(field, 1.5, 1.5)
        expected = (2.0 + 3.0 + 3.0 + 4.0) / 4.0  # Average of 2,3,3,4
        assert np.isclose(val, expected)

    def test_large_field(self, make_napari_viewer, qtbot):
        """Test interpolation on larger field."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)
        qtbot.addWidget(widget)

        # Create 100x100 field with known pattern
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        field = X + Y  # Simple sum pattern

        # Test interpolation maintains the pattern
        # At position (50, 50) should get value close to field[50, 50]
        val = widget._bilinear_interpolate(field, 50, 50)
        assert np.isclose(val, field[50, 50])

        # At fractional position
        val = widget._bilinear_interpolate(field, 25.5, 25.5)
        # Should be between field[25,25] and field[26,26]
        assert field[25, 25] <= val <= field[26, 26]


@pytest.mark.integration
class TestPointTrackingIntegration:
    """Integration tests for point tracking with real napari viewer.

    Run with: pytest test_motion_analysis.py -m integration
    """

    def test_constant_flow_tracking(self, make_napari_viewer):
        """Test tracking points through constant flow field."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)

        # Create constant flow field: every frame has same displacement
        T, H, W = 5, 100, 100
        flow = np.zeros((T, H, W, 2))
        flow[..., 0] = 1.0  # U component (x displacement)
        flow[..., 1] = 0.5  # V component (y displacement)

        # Add flow to viewer
        viewer.add_image(flow, name="test_flow")
        widget.current_flow_layer = viewer.layers["test_flow"]

        # Add test points
        initial_points = np.array([
            [25.0, 25.0],
            [50.0, 50.0],
            [75.0, 75.0]
        ])
        viewer.add_points(initial_points, name="test_points")

        # Track points
        widget._track_points("test_points", track_all=False)

        # Verify tracks were created
        assert "test_points_tracked" in [layer.name for layer in viewer.layers]

        # Get tracks data
        tracks_layer = viewer.layers["test_points_tracked"]
        tracks_data = tracks_layer.data

        # Verify all points are tracked across all frames
        for pid in range(3):
            point_tracks = tracks_data[tracks_data[:, 0] == pid]
            assert len(point_tracks) == T  # One position per frame

            # Verify positions match expected constant displacement
            for t in range(T):
                track = point_tracks[point_tracks[:, 1] == t]
                assert len(track) == 1

                # Expected position: initial + displacement
                # Note: flow is sampled at initial position (fixed-reference)
                expected_y = initial_points[pid, 0] + 0.5
                expected_x = initial_points[pid, 1] + 1.0

                np.testing.assert_allclose(
                    track[0, 2:4],  # [y, x] columns
                    [expected_y, expected_x],
                    rtol=1e-5
                )

    def test_zero_flow_tracking(self, make_napari_viewer):
        """Test tracking with zero flow - points should remain stationary."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)

        # Create zero flow field
        T, H, W = 3, 100, 100
        flow = np.zeros((T, H, W, 2))

        viewer.add_image(flow, name="zero_flow")
        widget.current_flow_layer = viewer.layers["zero_flow"]

        # Add test point
        initial_point = np.array([[50.0, 50.0]])
        viewer.add_points(initial_point, name="stationary_point")

        # Track point
        widget._track_points("stationary_point", track_all=False)

        # Verify point remains at same position
        tracks_layer = viewer.layers["stationary_point_tracked"]
        tracks_data = tracks_layer.data

        for track in tracks_data:
            np.testing.assert_allclose(
                track[2:4],  # [y, x] position
                initial_point[0],
                rtol=1e-5
            )

    def test_boundary_point_tracking(self, make_napari_viewer):
        """Test that points near boundaries are handled correctly."""
        viewer = make_napari_viewer
        widget = MotionAnalysisWidget(viewer)

        # Create flow that pushes points toward boundaries
        T, H, W = 3, 100, 100
        flow = np.zeros((T, H, W, 2))
        flow[..., 0] = 30.0  # Large x displacement
        flow[..., 1] = 30.0  # Large y displacement

        viewer.add_image(flow, name="boundary_flow")
        widget.current_flow_layer = viewer.layers["boundary_flow"]

        # Add points near boundaries
        boundary_points = np.array([
            [5.0, 5.0],      # Near top-left
            [95.0, 95.0],    # Near bottom-right
            [5.0, 95.0],     # Near top-right
            [95.0, 5.0]      # Near bottom-left
        ])
        viewer.add_points(boundary_points, name="boundary_points")

        # Track points
        widget._track_points("boundary_points", track_all=False)

        # Verify all tracked positions stay within bounds
        tracks_layer = viewer.layers["boundary_points_tracked"]
        tracks_data = tracks_layer.data

        # All y coordinates should be in [0, 99]
        assert np.all(tracks_data[:, 2] >= 0)
        assert np.all(tracks_data[:, 2] < 100)

        # All x coordinates should be in [0, 99]
        assert np.all(tracks_data[:, 3] >= 0)
        assert np.all(tracks_data[:, 3] < 100)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-k", "not Integration"])

    # For integration tests, run with:
    # pytest test_motion_analysis.py --integration -v