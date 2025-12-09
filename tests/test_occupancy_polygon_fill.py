"""Ensure polygon obstacles mark interior occupancy cells."""

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
