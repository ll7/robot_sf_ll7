import json
from typing import Tuple, List

from svgelements import SVG, Point, Path


Vec2D = Tuple[float, float]
ColorRGB = Tuple[int, int, int]


def paths_of_svg(svg: SVG) -> List[Path]:
    return [e for e in svg.elements() if isinstance(e, Path)]


def filter_paths_by_color(paths: List[Path], color: ColorRGB) -> List[Path]:
    red, green, blue = color
    paths = [e for e in paths \
             if e.fill.red == red \
                and e.fill.green == green \
                and e.fill.blue == blue]
    return paths


def points_of_paths(paths: List[Path]) -> List[List[Vec2D]]:
    all_lines = []
    for path in paths:
        points: List[Point] = list(path.as_points())
        all_lines.append([(p.x, p.y) for p in points])
    return all_lines


def serialize_mapjson(poly_points: List[List[Vec2D]]) -> str:
    obstacles_list = [{ 'ID': str(i), 'Vertex': points } for i, points in enumerate(poly_points)]
    obstacles = dict()
    for obstacle in obstacles_list:
        obs_name = obstacle['ID']
        obstacles[f'obstacle_{obs_name}'] = obstacle

    all_points = [p for points in poly_points for p in points]
    x_margin = [min([x for x, y in all_points]), max([x for x, y in all_points])]
    y_margin = [min([y for x, y in all_points]), max([y for x, y in all_points])]

    map_obj = {
        'Obstacles': obstacles,
        'x_margin': x_margin,
        'y_margin': y_margin
    }

    return json.dumps(map_obj)


def main():
    svg = SVG.parse('map_small.svg')
    house_color = (217, 208, 201)
    paths = paths_of_svg(svg)
    paths = filter_paths_by_color(paths, house_color)
    poly_points = points_of_paths(paths)
    map_json = serialize_mapjson(poly_points)
    with open('map.json', 'w') as file:
        file.write(map_json)


if __name__ == '__main__':
    main()
