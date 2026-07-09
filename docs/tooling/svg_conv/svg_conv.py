"""SVG to JSON map converter for RobotSF simulator.

This module provides functionality to convert SVG maps from OpenStreetMap
to JSON format that can be imported into the RobotSF simulator. It extracts
polygon data from SVG files and converts them to obstacle representations.
"""

import json
import os
import sys

from svgelements import SVG, Path, Point  # type: ignore  # members not statically resolvable

from robot_sf.common.types import RgbColor, Vec2D

HELP_MSG = """This tool converts SVG maps from OpenStreetMap to JSON maps that can be imported
into the RobotSF simulator.

USAGE
  python3 svg_conv.py <osm_input_file.svg> <output_file.json>

The converter extracts all polygons of given colors, which is currently
only the borders of houses (brown). Feel free to modify this script to support
your own use case."""


def paths_of_svg(svg: SVG) -> list[Path]:
    """Extract all path elements from an SVG document.

    Args:
        svg: The SVG document to extract paths from.

    Returns:
        A list of Path elements from the SVG.
    """
    return [e for e in svg.elements() if isinstance(e, Path)]


def filter_paths_by_color(paths: list[Path], color: RgbColor) -> list[Path]:
    """Filter paths by a specific RGB color.

    Args:
        paths: List of Path elements to filter.
        color: RGB color tuple (red, green, blue) to match.

    Returns:
        A list of paths that have the specified fill color.
    """
    red, green, blue = color
    paths = [
        e for e in paths if e.fill.red == red and e.fill.green == green and e.fill.blue == blue
    ]
    return paths


def points_of_paths(paths: list[Path]) -> list[list[Vec2D]]:
    """Extract points from paths as coordinate tuples.

    Args:
        paths: List of Path elements to extract points from.

    Returns:
        A list of lists, where each inner list contains (x, y) coordinate tuples
        representing the points of a path.
    """
    all_lines = []
    for path in paths:
        points: list[Point] = list(path.as_points())
        all_lines.append([(p.x, p.y) for p in points])
    return all_lines


def serialize_mapjson(poly_points: list[list[Vec2D]]) -> str:
    """Serialize polygon points to JSON map format.

    Args:
        poly_points: List of polygon point sequences, where each polygon is a list
                    of (x, y) coordinate tuples.

    Returns:
        A JSON string containing obstacles and margin information.
    """
    obstacles = poly_points
    all_points = [p for points in poly_points for p in points]
    x_margin = [min(x for x, _ in all_points), max(x for x, _ in all_points)]
    y_margin = [min(y for _, y in all_points), max(y for _, y in all_points)]

    map_obj = {"obstacles": obstacles, "x_margin": x_margin, "y_margin": y_margin}

    return json.dumps(map_obj)


def scale(p: Vec2D, factor: float) -> Vec2D:
    """Scale a 2D point by a factor.

    Args:
        p: A 2D point (x, y) to scale.
        factor: The scaling factor to apply to both coordinates.

    Returns:
        A new 2D point with coordinates scaled by the factor.
    """
    return p[0] * factor, p[1] * factor


def convert_map(input_svg_file: str, output_json_file: str):
    """Convert an SVG file to a JSON map file.

    Args:
        input_svg_file: Path to the input SVG file.
        output_json_file: Path to the output JSON file.
    """
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
    """Main entry point for the SVG-to-JSON converter.

    Parses command line arguments and performs the conversion if valid.
    Expects two arguments: input SVG file path and output JSON file path.
    """

    def file_exists(path):
        """Check if a file exists at the given path.

        Args:
            path: The file path to check.

        Returns:
            True if the file exists and is a regular file, False otherwise.
        """
        return os.path.exists(path) and os.path.isfile(path)

    def has_fileext(path, ext):
        """Check if a file has the specified extension.

        Args:
            path: The file path to check.
            ext: The file extension to look for (without the dot).

        Returns:
            True if the file has the specified extension, False otherwise.
        """
        return "." + path.split(".")[-1] == ext

    if (
        len(sys.argv) == 3
        and file_exists(sys.argv[1])
        and has_fileext(sys.argv[1], ".svg")
        and has_fileext(sys.argv[2], ".json")
    ):
        convert_map(input_svg_file=sys.argv[1], output_json_file=sys.argv[2])
    else:
        pass


if __name__ == "__main__":
    main()
