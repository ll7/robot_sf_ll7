"""Direct unit coverage for :mod:`robot_sf.data_analysis.recording_analysis`.

These tests lock the public contracts of the recording-analysis helpers without
exercising the underlying KDE/matplotlib stack or opening any GUI:

- :func:`extract_pedestrian_positions` validates and concatenates per-state
  pedestrian positions.
- :func:`kde_plot_grid_creation` builds the evaluation grid from map bounds and
  a requested resolution.
- :func:`visualize_kde_of_pedestrians_on_map` forwards transposed samples and
  the bandwidth method to ``gaussian_kde``, plots a normalized density, and
  applies the map bounds.

``gaussian_kde`` and ``matplotlib.pyplot`` are mocked at the module boundary so
the visualization orchestration can be asserted without ever creating a figure.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_sf.data_analysis import recording_analysis
from robot_sf.data_analysis.recording_analysis import (
    extract_pedestrian_positions,
    kde_plot_grid_creation,
    visualize_kde_of_pedestrians_on_map,
)


def _state(pedestrian_positions):
    """Build a lightweight synthetic state exposing only pedestrian positions.

    The extractor only reads ``state.pedestrian_positions``, so a minimal stub
    is sufficient and avoids constructing a full ``VisualizableSimState``.
    """
    return SimpleNamespace(pedestrian_positions=pedestrian_positions)


# --------------------------------------------------------------------------- #
# extract_pedestrian_positions
# --------------------------------------------------------------------------- #


class TestExtractPedestrianPositions:
    """Contract tests for pedestrian-position extraction and validation."""

    def test_concatenates_valid_positions_across_states(self):
        """Positions from multiple states are concatenated into shape (n, 2)."""
        states = [
            _state([[0.0, 1.0], [2.0, 3.0]]),
            _state([[4.0, 5.0]]),
        ]

        result = extract_pedestrian_positions(states)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        assert np.array_equal(result, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))

    @pytest.mark.parametrize("states", [[], [_state([])], [_state([]), _state([])]])
    def test_returns_empty_when_no_positions(self, states):
        """Missing or empty pedestrian positions yield an empty array."""
        result = extract_pedestrian_positions(states)

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.size == 0

    @pytest.mark.parametrize(
        "malformed_position",
        [
            [5.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [[0.0, 1.0], [2.0, 3.0]],
            np.zeros((2, 2, 2)),
            np.array(1.0),
        ],
        ids=[
            "one_coordinate",
            "three_coordinates",
            "four_coordinates",
            "rank_2",
            "rank_3",
            "scalar",
        ],
    )
    def test_returns_empty_for_malformed_positions(self, malformed_position):
        """Every shape other than a two-coordinate vector returns an empty array."""
        result = extract_pedestrian_positions([_state([malformed_position])])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.size == 0

    def test_non_2d_row_discards_all_positions(self):
        """Validation fails the whole batch, including otherwise-valid rows."""
        result = extract_pedestrian_positions(
            [_state([[0.0, 1.0], [2.0, 3.0, 4.0]])],
        )

        assert result.shape == (0,)
        assert result.size == 0

    def test_nested_coordinate_array_discards_all_positions(self):
        """A nested coordinate array is not a single two-dimensional point."""
        result = extract_pedestrian_positions(
            [_state([[[0.0, 1.0], [2.0, 3.0]]])],
        )

        assert result.shape == (0,)
        assert result.size == 0

    def test_ragged_coordinate_returns_controlled_empty_array(self):
        """Ragged nested input is rejected without leaking NumPy shape errors."""
        result = extract_pedestrian_positions(
            [_state([[[0.0, 1.0], [2.0]]])],
        )

        assert result.shape == (0,)
        assert result.size == 0


# --------------------------------------------------------------------------- #
# kde_plot_grid_creation
# --------------------------------------------------------------------------- #


class TestKdePlotGridCreation:
    """Contract tests for the KDE evaluation-grid construction."""

    def test_default_resolution_mesh_and_eval_shapes(self):
        """The default grid yields 100x100 meshes and a (2, 10000) eval stack."""
        grid_xx, grid_yy, grid_points = kde_plot_grid_creation(-1.0, 1.0, 0.0, 10.0)

        assert grid_xx.shape == (100, 100)
        assert grid_yy.shape == (100, 100)
        assert grid_points.shape == (2, 100 * 100)

    def test_grid_boundary_values_match_requested_bounds(self):
        """Mesh boundaries equal the requested x/y min and max values."""
        grid_xx, grid_yy, _ = kde_plot_grid_creation(-1.0, 1.0, 0.0, 10.0)

        assert np.isclose(grid_xx[0, 0], -1.0)
        assert np.isclose(grid_xx[0, -1], 1.0)
        assert np.isclose(grid_yy[0, 0], 0.0)
        assert np.isclose(grid_yy[-1, 0], 10.0)

    @pytest.mark.parametrize("number_of_grid_points", [1, 2, 25, 250])
    def test_requested_resolution_drives_shapes(self, number_of_grid_points):
        """``number_of_grid_points`` controls the per-axis resolution."""
        n = number_of_grid_points

        grid_xx, grid_yy, grid_points = kde_plot_grid_creation(
            0.0,
            4.0,
            -2.0,
            2.0,
            number_of_grid_points=number_of_grid_points,
        )

        assert grid_xx.shape == (n, n)
        assert grid_yy.shape == (n, n)
        assert grid_points.shape == (2, n * n)

    def test_flattened_eval_points_span_bounds_by_axis(self):
        """Row 0 carries x coordinates and row 1 carries y coordinates."""
        grid_xx, grid_yy, grid_points = kde_plot_grid_creation(
            0.0,
            4.0,
            -2.0,
            2.0,
            number_of_grid_points=25,
        )

        assert np.array_equal(grid_points[0], grid_xx.ravel())
        assert np.array_equal(grid_points[1], grid_yy.ravel())
        assert np.isclose(grid_points[0].min(), 0.0)
        assert np.isclose(grid_points[0].max(), 4.0)
        assert np.isclose(grid_points[1].min(), -2.0)
        assert np.isclose(grid_points[1].max(), 2.0)


# --------------------------------------------------------------------------- #
# visualize_kde_of_pedestrians_on_map
# --------------------------------------------------------------------------- #


def _run_visualize(positions, bounds, *, bw_method=None):
    """Invoke the visualizer with mocked KDE/matplotlib and return the spies.

    Returns a namespace of the injected mocks so individual tests can assert on
    the orchestration without exercising the real plotting stack.
    """
    map_def = MagicMock()
    map_def.get_map_bounds.return_value = bounds

    kde_instance = MagicMock(name="pedestrian_kde")
    # Non-uniform flat density over the default 100x100 grid so preservation and
    # normalization are both verifiable.
    kde_instance.return_value = np.arange(1, 10001, dtype=float)
    kde_cls = MagicMock(name="gaussian_kde", return_value=kde_instance)

    fig = MagicMock(name="fig")
    ax = MagicMock(name="ax")
    plt_mock = MagicMock(name="plt")
    plt_mock.subplots.return_value = (fig, ax)

    visualize_kwargs = {}
    if bw_method is not None:
        visualize_kwargs["kde_bandwith_method"] = bw_method

    with (
        patch.object(recording_analysis, "plt", plt_mock),
        patch.object(recording_analysis, "gaussian_kde", kde_cls),
    ):
        visualize_kde_of_pedestrians_on_map(positions, map_def, **visualize_kwargs)

    return SimpleNamespace(
        map_def=map_def,
        kde_cls=kde_cls,
        kde_instance=kde_instance,
        fig=fig,
        ax=ax,
        plt=plt_mock,
    )


class TestVisualizeKdeOfPedestriansOnMap:
    """Contract tests for the KDE visualization orchestration."""

    def test_forwards_transposed_samples_and_bandwidth(self):
        """``gaussian_kde`` receives the transposed samples and bandwidth."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        spies = _run_visualize(positions, (0.0, 10.0, -5.0, 5.0), bw_method="silverman")

        spies.kde_cls.assert_called_once()
        call = spies.kde_cls.call_args
        forwarded = np.asarray(call.args[0])
        assert forwarded.shape == (2, 3)
        assert np.array_equal(forwarded, positions.T)
        assert call.kwargs == {"bw_method": "silverman"}

    def test_default_bandwidth_is_scott(self):
        """The default bandwidth method is ``scott``."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])

        spies = _run_visualize(positions, (0.0, 1.0, 0.0, 1.0))

        assert spies.kde_cls.call_args.kwargs == {"bw_method": "scott"}

    def test_kde_evaluated_on_flattened_grid_from_bounds(self):
        """The KDE is evaluated on grid points spanning the map bounds."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])

        spies = _run_visualize(positions, (0.0, 10.0, -5.0, 5.0))

        eval_points = np.asarray(spies.kde_instance.call_args.args[0])
        assert eval_points.shape == (2, 100 * 100)
        assert np.isclose(eval_points[0].min(), 0.0)
        assert np.isclose(eval_points[0].max(), 10.0)
        assert np.isclose(eval_points[1].min(), -5.0)
        assert np.isclose(eval_points[1].max(), 5.0)

    def test_contour_receives_normalized_density(self):
        """The density passed to ``contourf`` is normalized to sum to one."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        spies = _run_visualize(positions, (0.0, 10.0, -5.0, 5.0))

        contour_args = spies.ax.contourf.call_args
        grid_xx, grid_yy, kde_vals = (
            contour_args.args[0],
            contour_args.args[1],
            contour_args.args[2],
        )
        assert np.asarray(grid_xx).shape == (100, 100)
        assert np.asarray(grid_yy).shape == (100, 100)
        kde_vals = np.asarray(kde_vals)
        assert kde_vals.shape == (100, 100)
        assert np.isclose(kde_vals.sum(), 1.0)
        raw_density = np.arange(1, 10001, dtype=float)
        expected_density = (raw_density / raw_density.sum()).reshape(100, 100)
        assert np.allclose(kde_vals, expected_density)

    def test_map_bounds_applied_to_axes_and_obstacles_plotted(self):
        """Axes limits use the map bounds and obstacles are plotted on the axis."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        bounds = (0.0, 10.0, -5.0, 5.0)

        spies = _run_visualize(positions, bounds)

        spies.map_def.get_map_bounds.assert_called_once_with()
        spies.ax.set_xlim.assert_called_once_with(bounds[0], bounds[1])
        spies.ax.set_ylim.assert_called_once_with(bounds[2], bounds[3])
        spies.map_def.plot_map_obstacles.assert_called_once_with(spies.ax)

    def test_does_not_open_gui(self):
        """The figure is created and shown through the mocked pyplot surface."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])

        spies = _run_visualize(positions, (0.0, 1.0, 0.0, 1.0))

        spies.plt.subplots.assert_called_once()
        spies.plt.show.assert_called_once()
