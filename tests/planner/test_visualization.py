"""Headless unit tests for planner plot composition.

Locks the wiring of the public ``plot_global_plan`` and ``plot_visibility_graph``
composition helpers and their private collaborators in
``robot_sf.planner.visualization`` without ever opening a GUI window or writing
an image.

Matplotlib ``pyplot``, the Shapely ``plot_polygon`` helper, and the figure/axes
objects are replaced with ``unittest.mock.MagicMock`` stand-ins, while the
planner, map definition, and visibility graph are tiny in-memory fakes built
from lightweight classes. Real Shapely geometries are used only for the pure
coordinate math (e.g. ``LineString.xy``) that the plotter reads; nothing is
rendered because every drawing target is a mock.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from shapely.geometry import LineString, Polygon

from robot_sf.planner import visualization as vis_mod
from robot_sf.planner.visualization import (
    _init_axes,
    _plot_obstacles,
    _plot_path,
    _plot_pois,
    _plot_visibility_graph_edges,
    _plot_visibility_graph_vertices,
    _save_figure,
    plot_global_plan,
    plot_visibility_graph,
)


def _square(x: float, y: float, side: float = 1.0) -> Polygon:
    """Return a small axis-aligned square polygon for obstacle fakes."""
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


class FakeGraph:
    """Minimal stand-in for a NetworkX visibility graph consumed by the plotter."""

    def __init__(self, *, nodes=None, edges=None, truthy: bool = True) -> None:
        """Store node/edge iterables and the truthiness used for empty-graph guards."""
        self._nodes = list(nodes or [])
        self._edges = list(edges or [])
        self._truthy = truthy

    def nodes(self):
        """Return a fresh list copy of the graph nodes."""
        return list(self._nodes)

    def edges(self):
        """Return a fresh list copy of the graph edges."""
        return list(self._edges)

    def __bool__(self) -> bool:
        """Return the configured truthiness so falsy graph fakes can be exercised."""
        return self._truthy


def make_planner(
    *,
    width: float = 10.0,
    height: float = 6.0,
    obstacles: list[Polygon] | None = None,
    poi_positions: list[tuple[float, float]] | None = None,
    poi_labels: dict[str, str] | None = None,
    graph: object | None = None,
) -> SimpleNamespace:
    """Build a tiny planner fake matching the visualization public surface.

    ``graph`` is assigned verbatim to the private ``_graph`` attribute: pass
    ``None`` for the no-graph path, a holder with ``networkx_graph=None`` for the
    not-built path, or a holder wrapping a :class:`FakeGraph` for the active path.
    """
    map_def = SimpleNamespace(
        width=width,
        height=height,
        poi_positions=list(poi_positions or []),
        poi_labels=dict(poi_labels or {}),
    )
    return SimpleNamespace(
        map_def=map_def,
        _graph=graph,
        build_inflated_obstacles=lambda: list(obstacles or []),
    )


def _scatter_calls_with_label(ax: MagicMock, label: str) -> list:
    """Return scatter call records whose keyword ``label`` matches."""
    return [call for call in ax.scatter.call_args_list if call.kwargs.get("label") == label]


@pytest.fixture
def mock_mpl(monkeypatch) -> SimpleNamespace:
    """Replace pyplot + plot_polygon with mocks and yield fresh fig/ax stand-ins."""
    fig = MagicMock(name="figure")
    ax = MagicMock(name="axes")
    ax.figure = fig
    plt_mock = MagicMock(name="plt")
    plt_mock.subplots.return_value = (fig, ax)
    plot_polygon_mock = MagicMock(name="plot_polygon")
    monkeypatch.setattr(vis_mod, "plt", plt_mock)
    monkeypatch.setattr(vis_mod, "plot_polygon", plot_polygon_mock)
    return SimpleNamespace(plt=plt_mock, fig=fig, ax=ax, plot_polygon=plot_polygon_mock)


class TestInitAxes:
    """Cover supplied-vs-created axes, bounds wiring, title, and y-flip behavior."""

    def test_creates_new_axes_when_none_supplied(self, mock_mpl):
        planner = make_planner(width=12.0, height=7.0)
        figure, axes = _init_axes(planner, ax=None, title=None, flip_y=True)

        assert figure is mock_mpl.fig
        assert axes is mock_mpl.ax
        mock_mpl.plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_mpl.ax.set_xlim.assert_called_once_with(0, 12.0)
        mock_mpl.ax.set_ylim.assert_called_once_with(0, 7.0)
        mock_mpl.ax.set_aspect.assert_called_once_with("equal", adjustable="box")
        mock_mpl.ax.invert_yaxis.assert_called_once()

    def test_reuses_supplied_axes_without_creating_figure(self, mock_mpl):
        planner = make_planner()
        supplied_ax = mock_mpl.ax
        figure, axes = _init_axes(planner, ax=supplied_ax, title=None, flip_y=True)

        mock_mpl.plt.subplots.assert_not_called()
        # Caller-owned axes/figure are reused, not replaced.
        assert axes is supplied_ax
        assert figure is supplied_ax.figure

    def test_caller_owned_axes_are_not_closed(self, mock_mpl):
        planner = make_planner()
        _init_axes(planner, ax=mock_mpl.ax, title=None, flip_y=True)

        # The composition helpers must never close a caller-provided figure.
        mock_mpl.plt.close.assert_not_called()

    def test_flip_y_false_skips_axis_inversion(self, mock_mpl):
        planner = make_planner()
        _init_axes(planner, ax=None, title=None, flip_y=False)

        mock_mpl.ax.invert_yaxis.assert_not_called()

    @pytest.mark.parametrize(
        "title, expected", [(None, "Global Planner Route"), ("Route X", "Route X")]
    )
    def test_title_resolution(self, mock_mpl, title, expected):
        planner = make_planner()
        _init_axes(planner, ax=None, title=title, flip_y=True)

        mock_mpl.ax.set_title.assert_called_once_with(expected)


class TestPlotObstacles:
    """Obstacles delegate to Shapely's plot_polygon, skipping empty geometries."""

    def test_obstacles_delegate_to_plot_polygon(self, mock_mpl):
        obstacles = [_square(0.0, 0.0), _square(3.0, 3.0)]
        _plot_obstacles(obstacles, mock_mpl.ax)

        assert mock_mpl.plot_polygon.call_count == len(obstacles)
        rendered = [call.args[0] for call in mock_mpl.plot_polygon.call_args_list]
        assert rendered == obstacles
        for call in mock_mpl.plot_polygon.call_args_list:
            assert call.kwargs["ax"] is mock_mpl.ax
            assert call.kwargs["add_points"] is False

    def test_empty_polygons_are_skipped(self, mock_mpl):
        obstacles = [Polygon(), _square(0.0, 0.0)]
        _plot_obstacles(obstacles, mock_mpl.ax)

        # Only the non-empty polygon is rendered.
        assert mock_mpl.plot_polygon.call_count == 1
        assert mock_mpl.plot_polygon.call_args.args[0] == obstacles[1]

    def test_empty_obstacle_list_draws_nothing(self, mock_mpl):
        _plot_obstacles([], mock_mpl.ax)

        mock_mpl.plot_polygon.assert_not_called()


class TestPlotPois:
    """POI markers and labels are drawn only when POIs are present."""

    def test_pois_render_marker_and_labels(self, mock_mpl):
        positions = [(1.0, 1.0), (4.0, 3.0)]
        labels = {"p0": "Start", "p1": "Goal"}
        planner = make_planner(poi_positions=positions, poi_labels=labels)
        _plot_pois(planner, mock_mpl.ax)

        poi_scatter = _scatter_calls_with_label(mock_mpl.ax, "POIs")
        assert len(poi_scatter) == 1
        assert poi_scatter[0].args[0] == [1.0, 4.0]
        assert poi_scatter[0].args[1] == [1.0, 3.0]
        # One annotation per POI label.
        assert mock_mpl.ax.annotate.call_count == len(labels)
        annotated_labels = [call.args[0] for call in mock_mpl.ax.annotate.call_args_list]
        assert annotated_labels == ["Start", "Goal"]

    def test_no_pois_returns_early(self, mock_mpl):
        planner = make_planner(poi_positions=[], poi_labels={})
        _plot_pois(planner, mock_mpl.ax)

        assert _scatter_calls_with_label(mock_mpl.ax, "POIs") == []
        mock_mpl.ax.annotate.assert_not_called()


class TestPlotPath:
    """Path line, start/goal markers, and via points are styled distinctly."""

    def test_path_line_uses_planned_path_styling(self, mock_mpl):
        path = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        _plot_path(path, via_points=[], ax=mock_mpl.ax)

        mock_mpl.ax.plot.assert_called_once()
        call = mock_mpl.ax.plot.call_args
        xs, ys = call.args
        # Coordinates are wired straight from the path geometry.
        assert [float(v) for v in xs] == [0.0, 1.0, 2.0]
        assert [float(v) for v in ys] == [0.0, 1.0, 0.0]
        assert call.kwargs["label"] == "planned path"
        assert call.kwargs["color"] == "#2563eb"
        assert call.kwargs["linewidth"] == 2.5

    def test_start_and_goal_markers_present(self, mock_mpl):
        path = [(0.0, 0.0), (5.0, 5.0)]
        _plot_path(path, via_points=[], ax=mock_mpl.ax)

        assert len(_scatter_calls_with_label(mock_mpl.ax, "start")) == 1
        goal_calls = _scatter_calls_with_label(mock_mpl.ax, "goal")
        assert len(goal_calls) == 1
        assert goal_calls[0].kwargs["marker"] == "X"
        # The path-point scatter is intentionally unlabeled.
        assert _scatter_calls_with_label(mock_mpl.ax, "via POIs") == []

    def test_via_points_rendered_when_provided(self, mock_mpl):
        path = [(0.0, 0.0), (5.0, 5.0)]
        via = [(2.0, 1.0), (4.0, 3.0)]
        _plot_path(path, via_points=via, ax=mock_mpl.ax)

        via_calls = _scatter_calls_with_label(mock_mpl.ax, "via POIs")
        assert len(via_calls) == 1
        # zip(*via_points) forwards tuples, so compare via list() for resilience.
        assert list(via_calls[0].args[0]) == [2.0, 4.0]
        assert list(via_calls[0].args[1]) == [1.0, 3.0]
        assert via_calls[0].kwargs["marker"] == "D"


class TestSaveFigure:
    """Save delegation creates parent dirs and forwards dpi/bbox to savefig."""

    def test_save_figure_delegates_to_savefig(self, tmp_path):
        fig = MagicMock(name="figure")
        target = tmp_path / "nested" / "dir" / "route.png"

        _save_figure(fig, target)

        assert target.parent.exists()
        fig.savefig.assert_called_once_with(target, dpi=200, bbox_inches="tight")

    def test_save_figure_accepts_str_target(self, tmp_path):
        fig = MagicMock(name="figure")
        target_str = str(tmp_path / "out.png")

        _save_figure(fig, target_str)

        fig.savefig.assert_called_once_with(Path(target_str), dpi=200, bbox_inches="tight")

    def test_existing_parent_dir_is_not_an_error(self, tmp_path):
        fig = MagicMock(name="figure")
        target = tmp_path / "out.png"  # parent already exists

        _save_figure(fig, target)  # must not raise on exist_ok mkdir

        fig.savefig.assert_called_once()


class TestPlotGlobalPlan:
    """Composition of obstacles + POIs + path with save/show boundaries."""

    def test_empty_path_raises_value_error(self, mock_mpl):
        planner = make_planner()
        with pytest.raises(ValueError, match="path must not be empty"):
            plot_global_plan(planner, [], show=False)

    def test_composition_calls_each_helper_in_order(self, mock_mpl):
        obstacles = [_square(0.0, 0.0)]
        positions = [(1.0, 1.0), (4.0, 3.0)]
        labels = {"p0": "A", "p1": "B"}
        planner = make_planner(obstacles=obstacles, poi_positions=positions, poi_labels=labels)
        path = [(0.0, 0.0), (5.0, 5.0)]

        figure = plot_global_plan(planner, path, show=False)

        assert figure is mock_mpl.fig
        # Obstacle, POI, and path rendering all dispatched.
        mock_mpl.plot_polygon.assert_called_once()
        assert len(_scatter_calls_with_label(mock_mpl.ax, "POIs")) == 1
        mock_mpl.ax.plot.assert_called_once()  # the path line only
        # Composition tail: legend, grid, tight layout.
        mock_mpl.ax.legend.assert_called_once_with(loc="upper right", frameon=True)
        mock_mpl.ax.grid.assert_called_once()
        mock_mpl.fig.tight_layout.assert_called_once()
        # No GUI and no image by default.
        mock_mpl.plt.show.assert_not_called()
        mock_mpl.fig.savefig.assert_not_called()

    def test_show_true_invokes_pyplot_show(self, mock_mpl):
        planner = make_planner()
        plot_global_plan(planner, [(0.0, 0.0), (1.0, 1.0)], show=True)

        mock_mpl.plt.show.assert_called_once()

    def test_save_path_delegates_to_figure_savefig(self, mock_mpl, tmp_path):
        planner = make_planner()
        target = tmp_path / "route.png"

        plot_global_plan(planner, [(0.0, 0.0), (1.0, 1.0)], save_path=target, show=False)

        mock_mpl.fig.savefig.assert_called_once_with(target, dpi=200, bbox_inches="tight")

    def test_supplied_axes_are_reused_not_replaced(self, mock_mpl):
        planner = make_planner()
        supplied_ax = mock_mpl.ax

        figure = plot_global_plan(planner, [(0.0, 0.0), (1.0, 1.0)], ax=supplied_ax, show=False)

        mock_mpl.plt.subplots.assert_not_called()
        assert figure is supplied_ax.figure
        # Caller-owned figure must survive intact.
        mock_mpl.plt.close.assert_not_called()


class TestVisibilityGraphHelpers:
    """Direct coverage for the edge/vertex rendering helpers and empty cases."""

    def test_edges_plot_one_line_per_edge(self, mock_mpl):
        graph = FakeGraph(
            nodes=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            edges=[((0.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (2.0, 0.0))],
        )

        _plot_visibility_graph_edges(graph, mock_mpl.ax)

        assert mock_mpl.ax.plot.call_count == 2
        first = mock_mpl.ax.plot.call_args_list[0]
        assert first.args[0] == [0.0, 1.0]
        assert first.args[1] == [0.0, 1.0]
        assert first.kwargs["color"] == "#9ca3af"

    def test_edges_empty_collection_skips_plotting(self, mock_mpl):
        graph = FakeGraph(nodes=[(0.0, 0.0)], edges=[])

        _plot_visibility_graph_edges(graph, mock_mpl.ax)

        mock_mpl.ax.plot.assert_not_called()

    def test_falsy_graph_edges_skipped(self, mock_mpl):
        graph = FakeGraph(edges=[((0.0, 0.0), (1.0, 1.0))], truthy=False)

        _plot_visibility_graph_edges(graph, mock_mpl.ax)

        mock_mpl.ax.plot.assert_not_called()

    def test_vertices_plot_single_scatter(self, mock_mpl):
        nodes = [(0.0, 0.0), (1.0, 1.0)]
        graph = FakeGraph(nodes=nodes)

        _plot_visibility_graph_vertices(graph, mock_mpl.ax)

        mock_mpl.ax.scatter.assert_called_once()
        call = mock_mpl.ax.scatter.call_args
        assert call.args[0] == [0.0, 1.0]
        assert call.args[1] == [0.0, 1.0]
        assert "2 nodes" in call.kwargs["label"]

    def test_vertices_empty_collection_skips_plotting(self, mock_mpl):
        graph = FakeGraph(nodes=[])

        _plot_visibility_graph_vertices(graph, mock_mpl.ax)

        mock_mpl.ax.scatter.assert_not_called()


class TestPlotVisibilityGraph:
    """Visibility graph composition with graph-presence branches and save."""

    def test_graph_rendered_when_present(self, mock_mpl):
        obstacles = [_square(0.0, 0.0)]
        graph = FakeGraph(
            nodes=[(0.0, 0.0), (1.0, 1.0)],
            edges=[((0.0, 0.0), (1.0, 1.0))],
        )
        planner = make_planner(obstacles=obstacles, graph=SimpleNamespace(networkx_graph=graph))

        figure = plot_visibility_graph(planner, show=False)

        assert figure is mock_mpl.fig
        mock_mpl.plot_polygon.assert_called_once()  # obstacles
        assert mock_mpl.ax.plot.call_count == 1  # one visibility edge
        mock_mpl.ax.scatter.assert_called_once()  # vertex scatter (no POIs configured)
        mock_mpl.ax.legend.assert_called_once()
        mock_mpl.fig.tight_layout.assert_called_once()

    def test_no_graph_skips_visibility_rendering(self, mock_mpl):
        planner = make_planner(graph=None)

        plot_visibility_graph(planner, show=False)

        # Only obstacles/POIs would draw; with none configured, nothing plots.
        mock_mpl.ax.plot.assert_not_called()
        mock_mpl.ax.scatter.assert_not_called()

    def test_not_built_graph_skips_visibility_rendering(self, mock_mpl):
        planner = make_planner(graph=SimpleNamespace(networkx_graph=None))

        plot_visibility_graph(planner, show=False)

        mock_mpl.ax.plot.assert_not_called()
        mock_mpl.ax.scatter.assert_not_called()

    def test_empty_edges_keeps_vertices(self, mock_mpl):
        graph = FakeGraph(nodes=[(0.0, 0.0), (1.0, 1.0)], edges=[])
        planner = make_planner(graph=SimpleNamespace(networkx_graph=graph))

        plot_visibility_graph(planner, show=False)

        mock_mpl.ax.plot.assert_not_called()
        mock_mpl.ax.scatter.assert_called_once()  # vertices still rendered

    def test_save_path_delegates_to_figure_savefig(self, mock_mpl, tmp_path):
        graph = FakeGraph(nodes=[(0.0, 0.0)], edges=[])
        planner = make_planner(graph=SimpleNamespace(networkx_graph=graph))
        target = tmp_path / "graph.png"

        plot_visibility_graph(planner, save_path=target, show=False)

        mock_mpl.fig.savefig.assert_called_once_with(target, dpi=200, bbox_inches="tight")

    def test_supplied_axes_are_reused(self, mock_mpl):
        graph = FakeGraph(nodes=[(0.0, 0.0)], edges=[])
        planner = make_planner(graph=SimpleNamespace(networkx_graph=graph))
        supplied_ax = mock_mpl.ax

        figure = plot_visibility_graph(planner, ax=supplied_ax, show=False)

        mock_mpl.plt.subplots.assert_not_called()
        assert figure is supplied_ax.figure
        mock_mpl.plt.close.assert_not_called()

    def test_show_true_invokes_pyplot_show(self, mock_mpl):
        planner = make_planner(graph=None)

        plot_visibility_graph(planner, show=True)

        mock_mpl.plt.show.assert_called_once()


def test_path_geometry_roundtrip_matches_linestring_xy():
    """Sanity guard: the path coordinates the plotter forwards equal LineString.xy."""
    path = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]
    line = LineString(path)

    assert [float(v) for v in line.xy[0]] == [p[0] for p in path]
    assert [float(v) for v in line.xy[1]] == [p[1] for p in path]
