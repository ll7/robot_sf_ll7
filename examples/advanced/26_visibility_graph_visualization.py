"""Visualize the visibility graph structure used by GlobalPlanner.

This example shows:
- How the planner constructs obstacle corners as graph vertices
- Direct visibility edges between non-colliding points
- The difference between raw obstacles and the graph's inflated boundaries
- Path planning within this graph structure

The visibility graph is a directed acyclic graph (DAG) where:
- Nodes are obstacle corners (after inflation for collision avoidance)
- Edges connect nodes that have direct line-of-sight
- The shortest path is found using this sparse graph

This is much more efficient than checking every point in the map.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from robot_sf.common.logging import configure_logging
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, plot_visibility_graph


def _ensure_interactive_backend() -> None:
    """Switch away from headless Agg when possible to show the plot interactively."""
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        return
    for candidate in ("MacOSX", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def main() -> None:
    """Visualize the visibility graph structure on a sample map."""
    configure_logging()
    _ensure_interactive_backend()

    map_path = Path("maps/svg_maps/MIT_corridor.svg")
    map_def = convert_map(str(map_path))

    # Create planner with collision-safe inflation
    planner = GlobalPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.25,
            min_safe_clearance=0.5,
            enable_smoothing=False,
        ),
    )

    # Plan a path to trigger graph construction
    start = (5.0, 5.0)
    goal = (45.0, 25.0)
    _ = planner.plan(start, goal)  # Triggers graph construction

    print("\nVisibility graph constructed with:")
    if planner._graph and planner._graph.networkx_graph:
        graph = planner._graph.networkx_graph
        print(f"  Nodes (graph vertices): {len(graph.nodes())}")
        print(f"  Edges (visibility connections): {len(graph.edges())}")
    else:
        print("  Graph not available")

    # Visualize the visibility graph
    print("\nGenerating visibility graph plot...")
    plot_visibility_graph(
        planner,
        title=f"Visibility Graph on {map_path.name}",
        save_path=Path("output/plots/visibility_graph.png"),
        show=True,
        flip_y=True,
    )

    print("✓ Visibility graph visualization complete!")


if __name__ == "__main__":
    main()
