HELP_MSG = """This tool converts SVG maps from OpenStreetMap to JSON maps that can be imported
into the RobotSF simulator.

USAGE
  python3 svg_conv.py <osm_input_file.svg> <output_file.json>

The converter extracts all polygons of given colors, which is currently
only the borders of houses (brown). Feel free to modify this script to support
your own use case."""

import os
import sys
import json
from typing import Tuple, List

from svgelements import SVG, Point, Path


Vec2D = Tuple[float, float]
ColorRGB = Tuple[int, int, int]


def paths_of_svg(svg: SVG) -> List[Path]:
    return [e for e in svg.elements() if isinstance(e, Path)]


def filter_paths_by_color(paths: List[Path], color: ColorRGB) -> List[Path]:
    red, green, blue = color
    paths = [
        e
        for e in paths
        if e.fill.red == red and e.fill.green == green and e.fill.blue == blue
    ]
    return paths


def points_of_paths(paths: List[Path]) -> List[List[Vec2D]]:
    all_lines = []
    for path in paths:
        points: List[Point] = list(path.as_points())
        all_lines.append([(p.x, p.y) for p in points])
    return all_lines


def serialize_mapjson(poly_points: List[List[Vec2D]]) -> str:
    obstacles = poly_points
    all_points = [p for points in poly_points for p in points]
    x_margin = [min([x for x, _ in all_points]), max([x for x, _ in all_points])]
    y_margin = [min([y for _, y in all_points]), max([y for _, y in all_points])]

    map_obj = {"obstacles": obstacles, "x_margin": x_margin, "y_margin": y_margin}

    return json.dumps(map_obj)


def scale(p: Vec2D, factor: float) -> Vec2D:
    return p[0] * factor, p[1] * factor


def convert_map(input_svg_file: str, output_json_file: str):
    svg = SVG.parse(input_svg_file)
    house_color = (217, 208, 201)
    paths = paths_of_svg(svg)
    paths = filter_paths_by_color(paths, house_color)
    poly_points = points_of_paths(paths)
    poly_points = [[scale(p, 1.62) for p in poly] for poly in poly_points]
    map_json = serialize_mapjson(poly_points)
    with open(output_json_file, "w") as file:
        file.write(map_json)


def main():
    def file_exists(path):
        return os.path.exists(path) and os.path.isfile(path)

    def has_fileext(path, ext):
        return "." + path.split(".")[-1] == ext

    if (
        len(sys.argv) == 3
        and file_exists(sys.argv[1])
        and has_fileext(sys.argv[1], ".svg")
        and has_fileext(sys.argv[2], ".json")
    ):
        convert_map(input_svg_file=sys.argv[1], output_json_file=sys.argv[2])
    else:
        print(HELP_MSG)


if __name__ == "__main__":
    main()
