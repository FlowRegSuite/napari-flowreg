"""
Tests for the wizard-based Motion Analysis V2 widget.

Covers the export helpers, the RunOptionsDialog options contract,
and regression tests for the ndarray-truthiness fixes in
_on_processing_complete and _export_tracks.
"""

import numpy as np
import pytest

from napari_flowreg.motion_analysis.widget_v2 import (
    _matlab_safe_name,
    _resolve_export_path,
    _write_statistics_export,
    _write_tracks_export,
)


SAMPLE_TRACKS = np.array(
    [
        [0, 0, 1.5, 2.5],
        [0, 1, 1.6, 2.4],
        [1, 0, 10.0, 20.0],
        [1, 1, 10.5, 19.5],
    ],
    dtype=np.float32,
)

SAMPLE_STATS = {
    "ROI 1": {
        "time_series_mean": np.array([0.1, 0.2, 0.3]),
        "overall_mean": 0.2,
        "max": 0.3,
        "min": 0.1,
    },
    "ROI 2": {
        "time_series_mean": np.array([1.0, 2.0, 3.0]),
        "overall_mean": 2.0,
        "max": 3.0,
        "min": 1.0,
    },
}


class TestExportHelpers:
    """Pure-function export helpers - no Qt or napari required."""

    def test_resolve_typed_extension_wins_over_filter(self):
        path, fmt = _resolve_export_path("out.csv", "MAT files (*.mat)")
        assert path == "out.csv"
        assert fmt == "csv"

    def test_resolve_mat_filter_appends_extension(self):
        path, fmt = _resolve_export_path("out", "MAT files (*.mat)")
        assert path == "out.mat"
        assert fmt == "mat"

    def test_resolve_csv_filter_appends_extension(self):
        path, fmt = _resolve_export_path("out", "CSV files (*.csv)")
        assert path == "out.csv"
        assert fmt == "csv"

    def test_resolve_unknown_extension_uses_filter(self):
        path, fmt = _resolve_export_path("out.txt", "CSV files (*.csv)")
        assert path == "out.txt.csv"
        assert fmt == "csv"

    def test_matlab_safe_name_sanitizes(self):
        used = set()
        assert _matlab_safe_name("ROI 1!", used) == "ROI_1_"

    def test_matlab_safe_name_letter_first(self):
        used = set()
        assert _matlab_safe_name("1st roi", used).startswith("roi_")

    def test_matlab_safe_name_dedupes(self):
        used = set()
        first = _matlab_safe_name("roi", used)
        second = _matlab_safe_name("roi", used)
        assert first == "roi"
        assert second == "roi_2"

    def test_tracks_csv_roundtrip(self, tmp_path):
        out = tmp_path / "tracks.csv"
        final = _write_tracks_export(str(out), "CSV files (*.csv)", SAMPLE_TRACKS)
        assert final == str(out)
        lines = out.read_text().strip().splitlines()
        assert lines[0] == "track_id,frame,y,x"
        assert len(lines) == 1 + len(SAMPLE_TRACKS)
        loaded = np.loadtxt(out, delimiter=",", skiprows=1)
        np.testing.assert_allclose(loaded, SAMPLE_TRACKS, rtol=1e-5)

    def test_tracks_mat_roundtrip(self, tmp_path):
        scipy_io = pytest.importorskip("scipy.io")
        out = tmp_path / "tracks.mat"
        _write_tracks_export(str(out), "MAT files (*.mat)", SAMPLE_TRACKS)
        contents = scipy_io.loadmat(str(out))
        assert contents["tracks"].shape == (4, 4)
        np.testing.assert_allclose(contents["tracks"], SAMPLE_TRACKS, rtol=1e-5)

    def test_statistics_csv_layout(self, tmp_path):
        out = tmp_path / "stats.csv"
        _write_statistics_export(str(out), "CSV files (*.csv)", SAMPLE_STATS)
        lines = out.read_text().strip().splitlines()
        summary_lines = [ln for ln in lines if ln.startswith("#")]
        assert len(summary_lines) == 1 + len(SAMPLE_STATS)
        assert any("overall_mean=0.200000" in ln for ln in summary_lines)
        header = [ln for ln in lines if not ln.startswith("#")][0]
        assert header == "frame,ROI 1,ROI 2"
        data = np.loadtxt(out, delimiter=",", comments="#", skiprows=len(summary_lines) + 1)
        assert data.shape == (3, 3)
        np.testing.assert_allclose(data[:, 1], [0.1, 0.2, 0.3])

    def test_statistics_mat_layout(self, tmp_path):
        scipy_io = pytest.importorskip("scipy.io")
        out = tmp_path / "stats.mat"
        _write_statistics_export(str(out), "MAT files (*.mat)", SAMPLE_STATS)
        contents = scipy_io.loadmat(str(out))
        assert "roi_stats" in contents
        assert "roi_names" in contents
        struct = contents["roi_stats"]
        assert "ROI_1" in struct.dtype.names
        assert "ROI_2" in struct.dtype.names


class TestRunOptionsContract:
    """The dict produced by RunOptionsDialog.get_options()."""

    def test_get_options_keys_and_defaults(self, qtbot):
        from napari_flowreg.motion_analysis.wizard import RunOptionsDialog

        dialog = RunOptionsDialog()
        qtbot.addWidget(dialog)
        opts = dialog.get_options()

        expected = {
            "batch_size", "workers", "save_w", "w_dir", "export_w_viewer",
            "video_path", "load_w_path", "corrected_path", "flow_quality",
            "flow_alpha", "flow_sigma_xy", "flow_sigma_t", "flow_levels",
            "flow_iterations", "flow_eta",
        }
        assert expected <= set(opts)
        assert opts["workers"] == -1
        # export_w_viewer is the canonical key; the duplicate was removed
        assert "load_w" not in opts


class TestWidgetV2Regressions:
    """Regression tests for the ndarray-truthiness fixes."""

    @pytest.fixture
    def widget(self, make_napari_viewer, qtbot):
        from napari_flowreg.motion_analysis.widget_v2 import MotionAnalysisWidgetV2

        viewer = make_napari_viewer
        widget = MotionAnalysisWidgetV2(viewer)
        qtbot.addWidget(widget)
        return widget

    def test_processing_complete_with_w_ndarray(self, widget):
        widget.wizard.output_choice = "load"
        w = np.zeros((3, 8, 8, 2), dtype=np.float32)
        # Pre-fix this raised "truth value of an array is ambiguous"
        widget._on_processing_complete({"w": w})
        assert "Displacement field (w)" in [layer.name for layer in widget.viewer.layers]

    def test_processing_complete_empty_results(self, widget):
        widget._on_processing_complete({})

    def test_export_tracks_with_ndarray(self, widget, tmp_path, monkeypatch):
        from napari_flowreg.motion_analysis import widget_v2 as widget_v2_module

        target = tmp_path / "tracks.csv"
        monkeypatch.setattr(
            widget_v2_module.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: (str(target), "CSV files (*.csv)")),
        )
        widget.tracked_points = SAMPLE_TRACKS
        # Pre-fix the guard raised "truth value of an array is ambiguous"
        widget._export_tracks()
        assert target.exists()
        assert target.read_text().startswith("track_id,frame,y,x")

    def test_export_tracks_none_guard(self, widget, monkeypatch):
        from napari_flowreg.motion_analysis import widget_v2 as widget_v2_module

        def _fail(*args, **kwargs):
            pytest.fail("File dialog must not open when there are no tracks")

        monkeypatch.setattr(
            widget_v2_module.QFileDialog, "getSaveFileName", staticmethod(_fail)
        )
        widget.tracked_points = None
        widget._export_tracks()
