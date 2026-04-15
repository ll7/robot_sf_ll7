"""Ensure polygon obstacles mark interior occupancy cells."""

from shapely.geometry import Polygon

from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy_grid import (
    GridChannel,
    GridConfig,
    OccupancyGrid,
    POIQuery,
    POIQueryType,
)


def test_polygon_fill_marks_interior_cells():
    """Occupancy grid should fill polygon interiors, not just edges."""
    config = GridConfig(
        resolution=1.0,
        width=5.0,
        height=5.0,
        channels=[GridChannel.OBSTACLES],
    )
    grid = OccupancyGrid(config)

    square = Obstacle([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    obstacle_lines = [((x1, y1), (x2, y2)) for x1, x2, y1, y2 in square.lines]

    grid.generate(
        obstacles=obstacle_lines,
        obstacle_polygons=[square.vertices],
        pedestrians=[],
        robot_pose=((0.0, 0.0), 0.0),
    )

    inside = grid.query(POIQuery(x=1.0, y=1.0, query_type=POIQueryType.POINT))
    outside = grid.query(POIQuery(x=4.0, y=4.0, query_type=POIQueryType.POINT))

    assert inside.occupancy > 0.0
    assert outside.occupancy == 0.0


def test_polygon_hole_remains_free() -> None:
    """Occupancy grid should respect polygon holes when filling interiors."""
    config = GridConfig(
        resolution=1.0,
        width=6.0,
        height=6.0,
        channels=[GridChannel.OBSTACLES],
    )
    grid = OccupancyGrid(config)

    obstacle = Obstacle.from_geometry(
        Polygon(
            [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            holes=[[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]],
        )
    )
    obstacle_lines = [((x1, y1), (x2, y2)) for x1, x2, y1, y2 in obstacle.lines]

    grid.generate(
        obstacles=obstacle_lines,
        obstacle_polygons=obstacle.iter_polygons(),
        pedestrians=[],
        robot_pose=((0.0, 0.0), 0.0),
    )

    solid = grid.query(POIQuery(x=0.5, y=0.5, query_type=POIQueryType.POINT))
    hole = grid.query(POIQuery(x=2.0, y=2.0, query_type=POIQueryType.POINT))

    assert solid.occupancy > 0.0
    assert hole.occupancy == 0.0
