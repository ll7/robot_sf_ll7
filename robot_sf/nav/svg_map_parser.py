"""SVG map parser for converting labeled Inkscape maps to MapDefinition objects.

This module provides utilities to parse SVG files created in Inkscape with specific
labeling conventions into structured MapDefinition objects used by the simulation.
It extracts obstacles, routes, spawn/goal zones, and pedestrian definitions from
SVG elements (paths, rectangles, circles) based on their inkscape:label attributes.

Supported SVG Elements and Labels:
    Paths:
        - 'obstacle': Closed polygon obstacles
        - 'ped_route_<spawn>_<goal>': Pedestrian navigation routes
        - 'robot_route_<spawn>_<goal>': Robot navigation routes
        - 'crowded_zone': High pedestrian density areas

    Rectangles:
        - 'obstacle': Rectangular obstacles
        - 'robot_spawn_zone': Robot starting areas
        - 'ped_spawn_zone': Pedestrian starting areas
        - 'robot_goal_zone': Robot target areas
        - 'ped_goal_zone': Pedestrian target areas
        - 'ped_crowded_zone': Crowded pedestrian zones
        - 'bound': Additional map boundaries

    Circles:
        - 'single_ped_<id>_start': Individual pedestrian start position
        - 'single_ped_<id>_goal': Individual pedestrian goal position

Typical Usage:
    # Load a single SVG map
    map_def = convert_map('maps/svg_maps/example.svg')

    # Load all SVG maps from a directory
    maps = load_svg_maps('maps/svg_maps/', pattern='*.svg')
"""

import re
import xml.etree.ElementTree as ET
from math import atan2, ceil, cos, dist, pi, radians, sin, sqrt
from pathlib import Path

import numpy as np
from loguru import logger
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import explain_validity, make_valid

from robot_sf.common.errors import raise_fatal_with_remedy
from robot_sf.common.types import Line2D, Rect, Zone
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.nav_types import SvgCircle, SvgPath, SvgRectangle
from robot_sf.nav.obstacle import Obstacle, obstacle_from_svgrectangle


class SvgMapConverter:
    """Manages conversion of labeled SVG maps to MapDefinition objects.

    This class orchestrates the complete parsing pipeline: loading SVG XML, extracting
    labeled elements (paths, rectangles, circles), categorizing them by label, and
    assembling the final MapDefinition with all obstacles, routes, zones, and pedestrians.

    Attributes:
        svg_file_str: Path to the SVG file being parsed.
        svg_root: Parsed XML root element of the SVG document.
        path_info: Extracted path elements with coordinates and labels.
        rect_info: Extracted rectangle elements with dimensions and labels.
        circle_info: Extracted circle elements with centers and labels.
        map_definition: Final assembled MapDefinition object.

    Example:
        >>> converter = SvgMapConverter("maps/example.svg")
        >>> map_def = converter.get_map_definition()
        >>> len(map_def.obstacles)
        5
    """

    svg_file_str: str
    svg_root: ET.Element
    path_info: list
    rect_info: list
    circle_info: list
    map_definition: MapDefinition
    _ROUTE_ONLY_ZONE_EDGE: float = 1.0
    _INDEX_FALLBACK_ZONE_EDGE: float = 0.1
    _CURVE_MAX_STEP: float = 0.75

    def __init__(self, svg_file: str):
        """Initialize the SVG map converter and perform full parsing.

        Loads the SVG file, extracts all labeled elements, and constructs the
        MapDefinition object. All parsing happens during initialization.

        Args:
            svg_file: Path to the SVG file to parse.

        Raises:
            FileNotFoundError: If svg_file does not exist.
            xml.etree.ElementTree.ParseError: If SVG file has invalid XML syntax.
            ValueError: If map validation fails (e.g., no robot routes defined).
        """
        self.svg_file_str = svg_file
        self._load_svg_root()

        self._get_svg_info()
        self._info_to_mapdefintion()

    def _load_svg_root(self):
        """Load and parse the SVG file's XML root element.

        Reads the SVG file and parses it into an ElementTree structure. Provides
        actionable error messages for common failure modes (missing file, invalid XML).

        Raises:
            FileNotFoundError: If svg_file_str path does not exist (raised via raise_fatal_with_remedy).
            xml.etree.ElementTree.ParseError: If SVG has invalid XML syntax (raised via raise_fatal_with_remedy).
        """
        logger.debug(f"Loading the root of the SVG file: {self.svg_file_str}")

        # Parse the SVG file with actionable error handling
        try:
            svg_tree = ET.parse(self.svg_file_str)
            self.svg_root = svg_tree.getroot()
        except FileNotFoundError:
            raise_fatal_with_remedy(
                f"Map file not found: {self.svg_file_str}",
                f"Place SVG map at '{self.svg_file_str}' or check available maps in maps/svg_maps/",
            )
        except ET.ParseError as e:
            raise_fatal_with_remedy(
                f"Invalid SVG format in {self.svg_file_str}: {e}",
                "Ensure the SVG file is valid XML (use Inkscape or check XML syntax)",
            )

    @staticmethod
    def _append_point(points: list[tuple[float, float]], point: tuple[float, float]) -> None:
        """Append a point unless it duplicates the latest waypoint."""
        normalized = (float(point[0]), float(point[1]))
        if not points or dist(points[-1], normalized) > 1e-9:
            points.append(normalized)

    @staticmethod
    def _estimate_segment_count(
        polyline_length: float,
        *,
        max_step: float,
        min_segments: int = 2,
        max_segments: int = 256,
    ) -> int:
        """Estimate interpolation segments from control-polygon length.

        Returns:
            int: Segment count bounded by ``min_segments`` and ``max_segments``.
        """
        if polyline_length <= 0:
            return min_segments
        return max(min_segments, min(max_segments, ceil(polyline_length / max_step)))

    @staticmethod
    def _sample_cubic_curve(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        *,
        max_step: float,
    ) -> list[tuple[float, float]]:
        """Return sampled points for a cubic bezier segment, excluding the start point."""
        control_len = dist(p0, p1) + dist(p1, p2) + dist(p2, p3)
        segments = SvgMapConverter._estimate_segment_count(control_len, max_step=max_step)
        samples: list[tuple[float, float]] = []
        for i in range(1, segments + 1):
            t = i / segments
            omt = 1.0 - t
            x = omt**3 * p0[0] + 3.0 * omt**2 * t * p1[0] + 3.0 * omt * t**2 * p2[0] + t**3 * p3[0]
            y = omt**3 * p0[1] + 3.0 * omt**2 * t * p1[1] + 3.0 * omt * t**2 * p2[1] + t**3 * p3[1]
            samples.append((x, y))
        return samples

    @staticmethod
    def _sample_quadratic_curve(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        max_step: float,
    ) -> list[tuple[float, float]]:
        """Return sampled points for a quadratic bezier segment, excluding the start point."""
        control_len = dist(p0, p1) + dist(p1, p2)
        segments = SvgMapConverter._estimate_segment_count(control_len, max_step=max_step)
        samples: list[tuple[float, float]] = []
        for i in range(1, segments + 1):
            t = i / segments
            omt = 1.0 - t
            x = omt**2 * p0[0] + 2.0 * omt * t * p1[0] + t**2 * p2[0]
            y = omt**2 * p0[1] + 2.0 * omt * t * p1[1] + t**2 * p2[1]
            samples.append((x, y))
        return samples

    @staticmethod
    def _sample_arc_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        rx: float,
        ry: float,
        x_axis_rotation_deg: float,
        large_arc: bool,
        sweep: bool,
        *,
        max_step: float,
    ) -> list[tuple[float, float]]:
        """Return sampled points for an SVG arc segment, excluding the start point."""
        if dist(start, end) <= 1e-12:
            return []
        if abs(rx) <= 1e-12 or abs(ry) <= 1e-12:
            return [end]

        rx = abs(rx)
        ry = abs(ry)
        x1, y1 = start
        x2, y2 = end
        phi = radians(x_axis_rotation_deg % 360.0)
        cos_phi = cos(phi)
        sin_phi = sin(phi)

        dx2 = (x1 - x2) / 2.0
        dy2 = (y1 - y2) / 2.0
        x1p = cos_phi * dx2 + sin_phi * dy2
        y1p = -sin_phi * dx2 + cos_phi * dy2

        lambda_ratio = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
        if lambda_ratio > 1.0:
            scale = sqrt(lambda_ratio)
            rx *= scale
            ry *= scale

        num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
        den = rx**2 * y1p**2 + ry**2 * x1p**2
        if den <= 1e-18:
            return [end]

        sign = -1.0 if large_arc == sweep else 1.0
        coef = sign * sqrt(max(0.0, num / den))
        cxp = coef * (rx * y1p / ry)
        cyp = coef * (-ry * x1p / rx)

        cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
        cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

        ux = (x1p - cxp) / rx
        uy = (y1p - cyp) / ry
        vx = (-x1p - cxp) / rx
        vy = (-y1p - cyp) / ry

        theta_1 = atan2(uy, ux)
        delta_theta = atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        if not sweep and delta_theta > 0:
            delta_theta -= 2.0 * pi
        elif sweep and delta_theta < 0:
            delta_theta += 2.0 * pi

        if abs(delta_theta) <= 1e-12:
            return [end]

        arc_len = abs(delta_theta) * max(rx, ry)
        segments = SvgMapConverter._estimate_segment_count(
            arc_len,
            max_step=max_step,
            min_segments=2,
        )

        samples: list[tuple[float, float]] = []
        for i in range(1, segments + 1):
            theta = theta_1 + delta_theta * i / segments
            cos_theta = cos(theta)
            sin_theta = sin(theta)
            x = cx + cos_phi * rx * cos_theta - sin_phi * ry * sin_theta
            y = cy + sin_phi * rx * cos_theta + cos_phi * ry * sin_theta
            samples.append((x, y))

        # Keep deterministic exact endpoint to avoid tiny floating point drift.
        samples[-1] = end
        return samples

    @staticmethod
    def _reflect_point(
        current: tuple[float, float],
        control: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Reflect a control point around the current position.

        Returns:
            tuple[float, float]: Reflected control point. If ``control`` is None,
                returns ``current`` unchanged.
        """
        if control is None:
            return current
        return (2.0 * current[0] - control[0], 2.0 * current[1] - control[1])

    @staticmethod
    def _split_arc_flag_token(tokens: list[str], idx: int) -> None:
        """Split compact arc-flag tokens like ``01`` into separate values.

        SVG arc syntax allows two adjacent boolean flags (large-arc, sweep).
        Some authoring tools emit them without separators, which the generic
        numeric tokenization reads as a single token (e.g. ``"01"``). This
        helper rewrites that token into ``["0", "1"]`` in-place.
        """
        if idx >= len(tokens):
            return
        token = tokens[idx]
        if len(token) == 2 and set(token) <= {"0", "1"}:
            tokens[idx : idx + 1] = [token[0], token[1]]

    @staticmethod
    def _parse_path_coordinates(  # noqa: C901, PLR0912, PLR0915
        path_d: str,
    ) -> tuple[tuple[float, float], ...]:
        """Parse SVG path commands into ordered waypoint coordinates.

        Supports absolute and relative variants of ``M L H V C S Q T A Z``.
        Curves and arcs are flattened into deterministic waypoint samples.

        Returns:
            tuple[tuple[float, float], ...]: Ordered waypoints extracted from path commands.
        """
        token_pattern = re.compile(
            r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?",
        )
        tokens = token_pattern.findall(path_d)
        if not tokens:
            return ()

        def is_command(token: str) -> bool:
            return len(token) == 1 and token.isalpha()

        idx = 0
        command: str | None = None
        current = (0.0, 0.0)
        subpath_start: tuple[float, float] | None = None
        last_cubic_ctrl: tuple[float, float] | None = None
        last_quad_ctrl: tuple[float, float] | None = None
        points: list[tuple[float, float]] = []

        def read_number() -> float:
            nonlocal idx
            if idx >= len(tokens):
                raise ValueError("Unexpected end of path data while reading numeric parameter.")
            token = tokens[idx]
            if is_command(token):
                raise ValueError(f"Expected numeric path parameter, found command {token!r}.")
            idx += 1
            return float(token)

        def has_more_numbers() -> bool:
            return idx < len(tokens) and not is_command(tokens[idx])

        while idx < len(tokens):
            token = tokens[idx]
            if is_command(token):
                command = token
                idx += 1
                if command in {"Z", "z"}:
                    if subpath_start is not None:
                        SvgMapConverter._append_point(points, subpath_start)
                        current = subpath_start
                    command = None
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    continue

            if command is None:
                raise ValueError("Path data is missing an initial SVG command.")

            if command in {"M", "m"}:
                x = read_number()
                y = read_number()
                if command == "m":
                    current = (current[0] + x, current[1] + y)
                else:
                    current = (x, y)
                subpath_start = current
                SvgMapConverter._append_point(points, current)

                line_command = "l" if command == "m" else "L"
                while has_more_numbers():
                    x = read_number()
                    y = read_number()
                    if line_command == "l":
                        current = (current[0] + x, current[1] + y)
                    else:
                        current = (x, y)
                    SvgMapConverter._append_point(points, current)
                command = line_command
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if command in {"L", "l"}:
                while has_more_numbers():
                    x = read_number()
                    y = read_number()
                    if command == "l":
                        current = (current[0] + x, current[1] + y)
                    else:
                        current = (x, y)
                    SvgMapConverter._append_point(points, current)
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if command in {"H", "h"}:
                while has_more_numbers():
                    x = read_number()
                    current = (current[0] + x, current[1]) if command == "h" else (x, current[1])
                    SvgMapConverter._append_point(points, current)
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if command in {"V", "v"}:
                while has_more_numbers():
                    y = read_number()
                    current = (current[0], current[1] + y) if command == "v" else (current[0], y)
                    SvgMapConverter._append_point(points, current)
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if command in {"C", "c"}:
                while has_more_numbers():
                    x1 = read_number()
                    y1 = read_number()
                    x2 = read_number()
                    y2 = read_number()
                    x = read_number()
                    y = read_number()
                    if command == "c":
                        p1 = (current[0] + x1, current[1] + y1)
                        p2 = (current[0] + x2, current[1] + y2)
                        p3 = (current[0] + x, current[1] + y)
                    else:
                        p1 = (x1, y1)
                        p2 = (x2, y2)
                        p3 = (x, y)
                    for point in SvgMapConverter._sample_cubic_curve(
                        current,
                        p1,
                        p2,
                        p3,
                        max_step=SvgMapConverter._CURVE_MAX_STEP,
                    ):
                        SvgMapConverter._append_point(points, point)
                    current = p3
                    last_cubic_ctrl = p2
                    last_quad_ctrl = None
                continue

            if command in {"S", "s"}:
                while has_more_numbers():
                    x2 = read_number()
                    y2 = read_number()
                    x = read_number()
                    y = read_number()
                    p1 = SvgMapConverter._reflect_point(current, last_cubic_ctrl)
                    if command == "s":
                        p2 = (current[0] + x2, current[1] + y2)
                        p3 = (current[0] + x, current[1] + y)
                    else:
                        p2 = (x2, y2)
                        p3 = (x, y)
                    for point in SvgMapConverter._sample_cubic_curve(
                        current,
                        p1,
                        p2,
                        p3,
                        max_step=SvgMapConverter._CURVE_MAX_STEP,
                    ):
                        SvgMapConverter._append_point(points, point)
                    current = p3
                    last_cubic_ctrl = p2
                    last_quad_ctrl = None
                continue

            if command in {"Q", "q"}:
                while has_more_numbers():
                    x1 = read_number()
                    y1 = read_number()
                    x = read_number()
                    y = read_number()
                    if command == "q":
                        p1 = (current[0] + x1, current[1] + y1)
                        p2 = (current[0] + x, current[1] + y)
                    else:
                        p1 = (x1, y1)
                        p2 = (x, y)
                    for point in SvgMapConverter._sample_quadratic_curve(
                        current,
                        p1,
                        p2,
                        max_step=SvgMapConverter._CURVE_MAX_STEP,
                    ):
                        SvgMapConverter._append_point(points, point)
                    current = p2
                    last_quad_ctrl = p1
                    last_cubic_ctrl = None
                continue

            if command in {"T", "t"}:
                while has_more_numbers():
                    x = read_number()
                    y = read_number()
                    control = SvgMapConverter._reflect_point(current, last_quad_ctrl)
                    if command == "t":
                        p2 = (current[0] + x, current[1] + y)
                    else:
                        p2 = (x, y)
                    for point in SvgMapConverter._sample_quadratic_curve(
                        current,
                        control,
                        p2,
                        max_step=SvgMapConverter._CURVE_MAX_STEP,
                    ):
                        SvgMapConverter._append_point(points, point)
                    current = p2
                    last_quad_ctrl = control
                    last_cubic_ctrl = None
                continue

            if command in {"A", "a"}:
                while has_more_numbers():
                    rx = read_number()
                    ry = read_number()
                    x_axis_rotation = read_number()
                    SvgMapConverter._split_arc_flag_token(tokens, idx)
                    large_arc_flag = read_number()
                    SvgMapConverter._split_arc_flag_token(tokens, idx)
                    sweep_flag = read_number()
                    x = read_number()
                    y = read_number()
                    if command == "a":
                        end = (current[0] + x, current[1] + y)
                    else:
                        end = (x, y)
                    for point in SvgMapConverter._sample_arc_segment(
                        current,
                        end,
                        rx,
                        ry,
                        x_axis_rotation,
                        large_arc_flag >= 0.5,
                        sweep_flag >= 0.5,
                        max_step=SvgMapConverter._CURVE_MAX_STEP,
                    ):
                        SvgMapConverter._append_point(points, point)
                    current = end
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                continue

            raise ValueError(f"Unsupported SVG path command {command!r}.")

        return tuple(points)

    def _parse_path_element(
        self, path: ET.Element, coordinate_pattern: re.Pattern
    ) -> SvgPath | None:
        """Parse a single SVG path element into a SvgPath object.

        Returns:
            SvgPath | None: Parsed path with coordinates and labels, or None if no coordinates found.
        """
        input_string = path.attrib.get("d")
        if not input_string:
            return None

        try:
            coordinates = self._parse_path_coordinates(input_string)
        except ValueError as parse_error:
            logger.debug(
                "Falling back to regex coordinate extraction for path %s: %s",
                path.attrib.get("id"),
                parse_error,
            )
            coordinates = ()

        if len(coordinates) < 2:
            filtered_coordinates = coordinate_pattern.findall(input_string)
            # If both structured parsing and fallback regex fail, the path has
            # no usable coordinates and should be dropped entirely.
            if not filtered_coordinates and not coordinates:
                logger.warning("No coordinates found for path: %s", path.attrib.get("id"))
                return None
            regex_coordinates = tuple(
                map(tuple, np.array(filtered_coordinates, dtype=float).tolist()),
            )
            if len(regex_coordinates) > len(coordinates):
                coordinates = regex_coordinates

        label = path.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label")
        path_id = path.attrib.get("id")
        if label is None:
            label = path_id
        if label is None:
            logger.warning(
                "Path element missing both inkscape:label and id attribute; using empty string",
            )
            label = ""

        return SvgPath(
            coordinates=coordinates,
            label=label,
            id=path_id or "",
        )

    def _parse_rect_element(self, rect: ET.Element) -> SvgRectangle:
        """Parse a single SVG rectangle element into a SvgRectangle object.

        Returns:
            SvgRectangle: Parsed rectangle with position, dimensions, label, and ID.
        """
        rect_label = rect.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label")
        rect_id = rect.attrib.get("id")
        if rect_label is None and rect_id is None:
            logger.warning(
                "Rectangle element missing both inkscape:label and id attribute; using empty string",
            )

        return SvgRectangle(
            float(rect.attrib.get("x")),
            float(rect.attrib.get("y")),
            float(rect.attrib.get("width")),
            float(rect.attrib.get("height")),
            rect_label or rect_id or "",
            rect_id or "",
        )

    def _parse_circle_element(self, circle: ET.Element) -> SvgCircle | None:
        """Parse a single SVG circle element into a SvgCircle object.

        Returns:
            SvgCircle | None: Parsed circle with center, radius, label, and ID, or None if parsing fails.
        """
        try:
            cx = float(circle.attrib.get("cx", 0))
            cy = float(circle.attrib.get("cy", 0))
            r = float(circle.attrib.get("r", 0))
            circle_label = circle.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label")
            circle_id = circle.attrib.get("id")
            circle_class = circle.attrib.get("class", "")
            if circle_label is None and circle_id is None:
                logger.warning(
                    "Circle element missing both inkscape:label and id attribute; using empty string",
                )

            return SvgCircle(
                cx,
                cy,
                r,
                circle_label or circle_class or circle_id or "",
                circle_id or "",
                circle_class,
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse circle {circle.attrib.get('id')}: {e}")
            return None

    def _get_svg_info(self):
        """
        Extracts path and rectangle information from an SVG file.

        It is important that the SVG file uses absolute coordinates for the paths.

        This method finds all 'path' and 'rect' elements in the SVG file and extracts their
        coordinates, labels, and ids. The information is stored in the 'path_info' and 'rect_info'
        attributes of the SvgMapConverter instance.

        For 'path' elements, the 'd' attribute is parsed to extract the coordinates. Each path is
        represented as a SvgPath object with the following attributes:
        - 'coordinates': a tuple of (x, y) waypoints
        - 'label': the 'inkscape:label' attribute of the path
        - 'id': the 'id' attribute of the path

        For 'rect' elements, the 'x', 'y', 'width', and 'height' attributes are extracted.
        Each rectangle is
        represented as a SvgRectangle object with the following attributes:
        - 'x': the 'x' attribute of the rectangle
        - 'y': the 'y' attribute of the rectangle
        - 'width': the 'width' attribute of the rectangle
        - 'height': the 'height' attribute of the rectangle
        - 'label': the 'inkscape:label' attribute of the rectangle
        - 'id': the 'id' attribute of the rectangle

        If the SVG root is not loaded, an error is logged and the method returns None.
        """
        if self.svg_root is None:
            logger.error("SVG root not loaded")
            return

        namespaces = {
            "svg": "http://www.w3.org/2000/svg",
            "inkscape": "http://www.inkscape.org/namespaces/inkscape",
        }

        paths = self.svg_root.findall(".//svg:path", namespaces)
        logger.debug(f"Found {len(paths)} paths in the SVG file")
        rects = self.svg_root.findall(".//svg:rect", namespaces)
        logger.debug(f"Found {len(rects)} rects in the SVG file")
        circles = self.svg_root.findall(".//svg:circle", namespaces)
        logger.debug(f"Found {len(circles)} circles in the SVG file")

        coordinate_pattern = re.compile(r"([+-]?[0-9]*\.?[0-9]+)[, ]([+-]?[0-9]*\.?[0-9]+)")

        path_info = [
            p for path in paths if (p := self._parse_path_element(path, coordinate_pattern))
        ]
        rect_info = [self._parse_rect_element(rect) for rect in rects]
        circle_info = [c for circle in circles if (c := self._parse_circle_element(circle))]

        logger.debug(f"Parsed {len(path_info)} paths in the SVG file")
        self.path_info = path_info
        logger.debug(f"Parsed {len(rect_info)} rects in the SVG file")
        self.rect_info = rect_info
        logger.debug(f"Parsed {len(circle_info)} circles in the SVG file")
        self.circle_info = circle_info

    def _apply_viewbox_offset(self, offset_x: float, offset_y: float) -> None:
        """Shift all parsed SVG elements by the viewBox origin to normalize to (0,0)."""
        if abs(offset_x) < 1e-9 and abs(offset_y) < 1e-9:
            return

        offset = np.array([offset_x, offset_y])
        for path in self.path_info:
            coords = np.asarray(path.coordinates, dtype=float) - offset
            path.coordinates = tuple(map(tuple, coords.tolist()))

        for rect in self.rect_info:
            rect.x -= offset_x
            rect.y -= offset_y

        for circle in self.circle_info:
            circle.cx -= offset_x
            circle.cy -= offset_y

    def _process_obstacle_path(self, path: SvgPath) -> Obstacle:
        """Process a path labeled as obstacle.

        Returns:
            Obstacle: Processed obstacle with closed polygon vertices.
        """
        # SvgPath.coordinates is a Tuple[Vec2D]; convert to list of tuples
        vertices = list(path.coordinates)

        if not np.array_equal(vertices[0], vertices[-1]):
            logger.warning(
                f"Closing polygon: first and last vertices of obstacle <{path.id}> differ",
            )
            vertices.append(vertices[0])

        # Validate obstacle geometry to flag self-intersections or degenerate shapes early.
        poly = Polygon(vertices)
        if not poly.is_valid or poly.is_empty:
            logger.warning(
                "Obstacle path id={pid} produced invalid polygon (valid={valid}, empty={empty}): {reason}",
                pid=path.id,
                valid=poly.is_valid,
                empty=poly.is_empty,
                reason=explain_validity(poly),
            )
            repaired = self._repair_invalid_obstacle_polygon(poly)
            if repaired is not None:
                vertices = list(repaired.exterior.coords)
                logger.info(
                    "Repaired invalid obstacle path id={pid} via make_valid (area={area:.3f}).",
                    pid=path.id,
                    area=repaired.area,
                )
            else:
                logger.warning(
                    "Failed to repair invalid obstacle path id={pid}; using raw vertices.",
                    pid=path.id,
                )

        return Obstacle(vertices)

    @staticmethod
    def _repair_invalid_obstacle_polygon(polygon: Polygon) -> Polygon | None:
        """Repair an invalid obstacle polygon and return the largest usable polygon.

        Args:
            polygon: Raw polygon built from SVG obstacle vertices.

        Returns:
            Polygon | None: Largest valid polygon candidate or ``None`` when no
                polygon geometry can be recovered.
        """
        repaired = make_valid(polygon)
        candidates = SvgMapConverter._polygon_members(repaired)
        if not candidates:
            return None
        candidate = max(candidates, key=lambda part: part.area)
        if candidate.is_empty or candidate.area <= 0.0:
            return None
        if candidate.is_valid:
            return candidate

        buffered = candidate.buffer(0)
        buffered_candidates = SvgMapConverter._polygon_members(buffered)
        if not buffered_candidates:
            return None
        buffered_best = max(buffered_candidates, key=lambda part: part.area)
        if buffered_best.is_empty or buffered_best.area <= 0.0:
            return None
        return buffered_best if buffered_best.is_valid else None

    @staticmethod
    def _polygon_members(geometry) -> list[Polygon]:
        """Collect polygon members from Polygon/MultiPolygon/GeometryCollection.

        Returns:
            list[Polygon]: Non-empty polygon members extracted from ``geometry``.
        """
        if isinstance(geometry, Polygon):
            return [] if geometry.is_empty else [geometry]
        if isinstance(geometry, MultiPolygon):
            return [polygon for polygon in geometry.geoms if not polygon.is_empty]
        if isinstance(geometry, GeometryCollection):
            members: list[Polygon] = []
            for child in geometry.geoms:
                members.extend(SvgMapConverter._polygon_members(child))
            return members
        return []

    def _process_route_path(
        self,
        path: SvgPath,
        spawn_zones: list[Rect],
        goal_zones: list[Rect],
        *,
        route_kind: str,
        route_only_mode: bool = False,
    ) -> GlobalRoute:
        """Process a path labeled as route (pedestrian or robot).

        Returns:
            GlobalRoute: Route with spawn/goal zones and waypoints extracted from path.
        """
        vertices = list(path.coordinates)
        if len(vertices) < 2:
            raise ValueError(
                "Route path must provide at least two waypoints; got "
                f"{len(vertices)} for path id={path.id!r} label={path.label!r}"
            )
        spawn, goal = self.__get_path_number(path.label)

        # Defensive fallback: if spawn/goal indices are out of range (or missing zones), create
        # minimal synthetic zones around first/last waypoint so downstream logic still works.
        svg_name = Path(self.svg_file_str).name

        def _safe_zone(index: int, zones: list[Rect], waypoint, kind: str) -> Rect:
            """Get a zone by index or create a synthetic fallback zone around waypoint.

            Args:
                index: Index into zones list.
                zones: Available spawn or goal zones.
                waypoint: Fallback (x, y) position if index is invalid.
                kind: Description string ('spawn' or 'goal') for logging.

            Returns:
                Rect: Zone from list if valid index, otherwise a deterministic synthetic zone.
            """
            if zones and 0 <= index < len(zones):
                return zones[index]
            if route_only_mode and not zones:
                logger.debug(
                    "SVG file '{svg}' route path '{pid}' uses {rk} route-only mode; deriving {k} zone from endpoint.",
                    svg=svg_name,
                    pid=path.id,
                    rk=route_kind,
                    k=kind,
                )
                return self._synthetic_zone_from_waypoint(
                    waypoint,
                    edge=self._ROUTE_ONLY_ZONE_EDGE,
                )
            logger.warning(
                "SVG file '{svg}' route path '{pid}' refers to {k} index {idx} but only {avail} zones available; creating synthetic zone.",
                svg=svg_name,
                pid=path.id,
                k=kind,
                idx=index,
                avail=len(zones),
            )
            return self._synthetic_zone_from_waypoint(
                waypoint,
                edge=self._INDEX_FALLBACK_ZONE_EDGE,
            )

        spawn_zone = _safe_zone(spawn, spawn_zones, vertices[0], "spawn")
        goal_zone = _safe_zone(goal, goal_zones, vertices[-1], "goal")

        return GlobalRoute(
            spawn_id=spawn,
            goal_id=goal,
            waypoints=vertices,
            spawn_zone=spawn_zone,
            goal_zone=goal_zone,
            source_path_id=path.id,
            source_label=path.label,
        )

    @staticmethod
    def _synthetic_zone_from_waypoint(
        waypoint: tuple[float, float],
        *,
        edge: float,
    ) -> Rect:
        """Create a deterministic right-triangle zone anchored at a waypoint.

        Route-only maps intentionally omit explicit spawn/goal rectangles and rely on
        route endpoints to define respawn anchors. This helper centralizes synthetic
        zone construction so route-only behavior and malformed-index fallback can use
        different zone sizes while sharing the same geometry convention.

        Returns:
            Rect: Synthetic triangular zone anchored at ``waypoint`` with side length ``edge``.
        """
        wx, wy = waypoint
        return ((wx, wy), (wx + edge, wy), (wx + edge, wy + edge))

    def _process_crowded_zone_path(self, path: SvgPath) -> Zone:
        """Process a path labeled as crowded zone.

        Returns:
            Zone: Zone polygon defined by path coordinates.
        """
        return Zone(list(path.coordinates))

    def _process_paths(
        self,
        obstacles: list[Obstacle],
        robot_spawn_zones: list[Rect],
        ped_spawn_zones: list[Rect],
        robot_goal_zones: list[Rect],
        ped_goal_zones: list[Rect],
        ped_crowded_zones: list[Rect],
    ) -> tuple[list[GlobalRoute], list[GlobalRoute]]:
        """Convert parsed paths into routes, obstacles, or crowded zones.

        Args:
            obstacles: Mutable obstacle list to append parsed obstacle paths to.
            robot_spawn_zones: Spawn zones used to resolve robot routes.
            ped_spawn_zones: Spawn zones used to resolve pedestrian routes.
            robot_goal_zones: Goal zones used to resolve robot routes.
            ped_goal_zones: Goal zones used to resolve pedestrian routes.
            ped_crowded_zones: Crowded zones list mutated when crowded paths are parsed.

        Returns:
            tuple[list[GlobalRoute], list[GlobalRoute]]: Parsed robot and pedestrian routes
                (robot_routes, ped_routes) extracted from the SVG paths.
        """
        robot_routes: list[GlobalRoute] = []
        ped_routes: list[GlobalRoute] = []
        ped_route_only_mode = not ped_spawn_zones and not ped_goal_zones
        robot_route_only_mode = not robot_spawn_zones and not robot_goal_zones

        if ped_route_only_mode and any("ped_route" in path.label for path in self.path_info):
            logger.info(
                "SVG file '{svg}' has pedestrian routes without ped_spawn_zone/ped_goal_zone "
                "rectangles; enabling route-only mode.",
                svg=Path(self.svg_file_str).name,
            )
        if robot_route_only_mode and any("robot_route" in path.label for path in self.path_info):
            logger.info(
                "SVG file '{svg}' has robot routes without robot_spawn_zone/robot_goal_zone "
                "rectangles; enabling route-only mode.",
                svg=Path(self.svg_file_str).name,
            )

        path_processors = {
            "obstacle": lambda p: obstacles.append(self._process_obstacle_path(p)),
            "ped_route": lambda p: ped_routes.append(
                self._process_route_path(
                    p,
                    ped_spawn_zones,
                    ped_goal_zones,
                    route_kind="ped",
                    route_only_mode=ped_route_only_mode,
                ),
            ),
            "robot_route": lambda p: robot_routes.append(
                self._process_route_path(
                    p,
                    robot_spawn_zones,
                    robot_goal_zones,
                    route_kind="robot",
                    route_only_mode=robot_route_only_mode,
                ),
            ),
            "crowded_zone": lambda p: ped_crowded_zones.append(self._process_crowded_zone_path(p)),
        }

        for path in self.path_info:
            label = path.label
            if "ped_route" in label:
                processor = path_processors["ped_route"]
            elif "robot_route" in label:
                processor = path_processors["robot_route"]
            else:
                processor = path_processors.get(label)

            if processor:
                processor(path)
            else:
                logger.error(f"Unknown label <{label}> in id <{path.id}>")

        return robot_routes, ped_routes

    def _process_single_pedestrians_from_circles(self) -> list[SinglePedestrianDefinition]:
        """
        Process circles labeled as single pedestrian markers.

        Circles should be labeled as:
        - "single_ped_<id>_start" for start positions
        - "single_ped_<id>_goal" for goal positions

        Returns:
            list[SinglePedestrianDefinition]: List of pedestrian definitions with matched
                start/goal positions, filtered to exclude incomplete pairs.
        """
        # Group circles by pedestrian ID
        ped_data: dict[str, dict[str, tuple[float, float]]] = {}

        # Use regex to robustly parse labels that may contain underscores in the ID
        pattern = re.compile(r"^single_ped_(?P<pid>.+)_(?P<kind>start|goal)$")
        for circle in self.circle_info:
            if not circle.label:
                continue

            # Circles can represent multiple concepts (e.g., POIs via class="poi"). Only treat
            # circles explicitly labeled as single pedestrian markers as candidates here.
            if not circle.label.startswith("single_ped_"):
                continue

            m = pattern.match(circle.label)
            if not m:
                logger.warning(
                    "Invalid single pedestrian circle label '{lab}' (id={cid}); "
                    "expected 'single_ped_<id>_(start|goal)'",
                    lab=circle.label,
                    cid=circle.id_,
                )
                continue

            ped_id = m.group("pid")
            marker_type = m.group("kind")  # "start" or "goal"

            if ped_id not in ped_data:
                ped_data[ped_id] = {}

            # Detect and warn about duplicate markers
            if marker_type in ped_data[ped_id]:
                logger.warning(
                    "Duplicate single_ped marker for id={pid} kind={k}; "
                    "keeping first, ignoring circle id={cid}",
                    pid=ped_id,
                    k=marker_type,
                    cid=circle.id_,
                )
                continue

            ped_data[ped_id][marker_type] = circle.get_center()

        # Create SinglePedestrianDefinition objects
        single_pedestrians = []
        for ped_id, data in ped_data.items():
            if "start" not in data:
                logger.warning(
                    f"Single pedestrian '{ped_id}' has no start marker; skipping",
                )
                continue

            # Goal is optional (pedestrian may be static or use trajectory from JSON)
            single_pedestrians.append(
                SinglePedestrianDefinition(
                    id=ped_id,
                    start=data["start"],
                    goal=data.get("goal"),
                    trajectory=None,  # Trajectories must come from JSON config
                ),
            )

        if single_pedestrians:
            logger.debug(f"Parsed {len(single_pedestrians)} single pedestrian(s) from circles")

        return single_pedestrians

    def _process_pois(self) -> tuple[list[tuple[float, float]], dict[str, str]]:
        """Extract POI positions and labels from parsed circles.

        Returns:
            tuple[list[tuple[float, float]], dict[str, str]]: POI positions and labels keyed
                by circle identifier.
        """
        poi_positions: list[tuple[float, float]] = []
        poi_labels: dict[str, str] = {}
        for circle in self.circle_info:
            is_poi = "poi" in circle.cls.split() or circle.label == "poi"
            if not is_poi:
                continue
            poi_positions.append(circle.get_center())
            poi_labels[circle.id_] = circle.label if circle.label else circle.id_
        return poi_positions, poi_labels

    def _log_map_warnings(
        self,
        obstacles: list[Obstacle],
        ped_routes: list[GlobalRoute],
        ped_crowded_zones: list[Rect],
    ) -> None:
        """Log non-fatal warnings for missing optional map elements."""
        if not obstacles:
            logger.warning("No obstacles found in the SVG file")
        if not ped_routes:
            logger.debug("No pedestrian routes found in the SVG file (optional)")
        if not ped_crowded_zones:
            logger.debug("No crowded zones found in the SVG file (optional)")

    @staticmethod
    def _zone_center(zone: Zone) -> tuple[float, float]:
        """Return the centroid of a rectangular zone."""
        xs = [p[0] for p in zone]
        ys = [p[1] for p in zone]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _ensure_robot_routes(
        self,
        robot_routes: list[GlobalRoute],
        robot_spawn_zones: list[Rect],
        robot_goal_zones: list[Rect],
    ) -> list[GlobalRoute]:
        """Ensure at least one robot route exists, generating straight-line defaults if absent.

        Returns:
            Parsed routes when present, otherwise synthesized straight-line routes between
            spawn and goal zone centers.
        """
        if robot_routes:
            return robot_routes

        if not robot_spawn_zones or not robot_goal_zones:
            logger.warning(
                "No robot routes found and spawn/goal zones are missing; planner may fail without routes."
            )
            return []

        logger.warning(
            "No robot routes found; generating straight-line routes between spawn and goal zone centers.",
        )
        fallback_routes: list[GlobalRoute] = []
        for spawn_idx, spawn_zone in enumerate(robot_spawn_zones):
            spawn_center = self._zone_center(spawn_zone)
            for goal_idx, goal_zone in enumerate(robot_goal_zones):
                goal_center = self._zone_center(goal_zone)
                fallback_routes.append(
                    GlobalRoute(
                        spawn_idx,
                        goal_idx,
                        [spawn_center, goal_center],
                        spawn_zone,
                        goal_zone,
                    )
                )

        return fallback_routes

    @staticmethod
    def _parse_zone_index(label: str | None, rect_id: str | None) -> tuple[str | None, int | None]:
        """Parse a zone label/id into a zone type and optional index.

        Returns:
            tuple[str | None, int | None]: Zone type and optional index (None if unindexed).
        """
        candidates = [label, rect_id]
        for value in candidates:
            if not value:
                continue
            match = re.match(
                r"^(robot_spawn_zone|ped_spawn_zone|robot_goal_zone|ped_goal_zone)_(\d+)$",
                value,
            )
            if match:
                return match.group(1), int(match.group(2))
        for value in candidates:
            if not value:
                continue
            match = re.match(
                r"^(robot_spawn_zone|ped_spawn_zone|robot_goal_zone|ped_goal_zone)$",
                value,
            )
            if match:
                return match.group(1), None
        return None, None

    @staticmethod
    def _assign_zone(
        zone_type: str,
        zone_index: int | None,
        zone: Rect,
        indexed_zones: dict[str, dict[int, Rect]],
        unindexed_zones: dict[str, list[Rect]],
    ) -> None:
        """Assign a zone into indexed/unindexed buckets, warning on duplicates."""
        if zone_index is None:
            unindexed_zones[zone_type].append(zone)
            return
        if zone_index in indexed_zones[zone_type]:
            logger.warning(
                "Duplicate %s index %d in SVG; keeping first, ignoring duplicate.",
                zone_type,
                zone_index,
            )
            return
        indexed_zones[zone_type][zone_index] = zone

    @staticmethod
    def _assemble_zones(
        zone_type: str,
        indexed_zones: dict[str, dict[int, Rect]],
        unindexed_zones: dict[str, list[Rect]],
    ) -> list[Rect]:
        """Build an ordered zone list, honoring explicit indices when present.

        Returns:
            list[Rect]: Ordered zones for the requested zone type.
        """
        indexed = indexed_zones[zone_type]
        unindexed = unindexed_zones[zone_type]
        if not indexed:
            return list(unindexed)

        result: list[Rect] = []
        max_index = max(indexed.keys())
        for idx in range(max_index + 1):
            if idx in indexed:
                result.append(indexed[idx])
            elif unindexed:
                result.append(unindexed.pop(0))
            else:
                logger.warning(
                    "Missing %s index %d with no unindexed zones available.",
                    zone_type,
                    idx,
                )
        result.extend(unindexed)
        return result

    def _process_rects(
        self,
        width: float,
        height: float,
    ) -> tuple[
        list[Obstacle],
        list[Rect],
        list[Rect],
        list[Rect],
        list[Line2D],
        list[Rect],
        list[Rect],
    ]:
        """
        Process rectangle elements from the SVG file and create game objects.

        Categorizes rectangles based on their labels and converts them into appropriate game
        objects:
        - Obstacles: Walls and static barriers
        - Spawn zones: Starting positions for robots and pedestrians
        - Goal zones: Target areas for robots and pedestrians
        - Bounds: Map boundaries (automatically creates edges plus any additional bounds)
        - Crowded zones: Areas with high pedestrian density

        Args:
            width (float): Width of the SVG map
            height (float): Height of the SVG map

        Returns:
            Tuple containing:
            - obstacles: List of Obstacle objects
            - robot_spawn_zones: List of rectangle zones where robots can spawn
            - ped_spawn_zones: List of rectangle zones where pedestrians can spawn
            - robot_goal_zones: List of rectangle zones marking robot goals
            - bounds: List of line segments defining map boundaries
            - ped_goal_zones: List of rectangle zones marking pedestrian goals
            - ped_crowded_zones: List of rectangle zones with high pedestrian density

        Note:
            Map bounds are automatically created for the edges of the map, additional bounds
            can be defined in the SVG file using the 'bound' label.
        """
        obstacles: list[Obstacle] = []
        robot_spawn_zones: list[Rect] = []
        ped_spawn_zones: list[Rect] = []
        robot_goal_zones: list[Rect] = []
        bounds: list[Line2D] = [
            (0, width, 0, 0),  # bottom
            (0, width, height, height),  # top
            (0, 0, 0, height),  # left
            (width, width, 0, height),  # right
        ]
        # logger.debug(f"Bounds: {bounds}")
        ped_crowded_zones: list[Rect] = []

        zone_types = {
            "robot_spawn_zone",
            "ped_spawn_zone",
            "robot_goal_zone",
            "ped_goal_zone",
        }
        indexed_zones: dict[str, dict[int, Rect]] = {z: {} for z in zone_types}
        unindexed_zones: dict[str, list[Rect]] = {z: [] for z in zone_types}

        for rect in self.rect_info:
            zone_type, zone_index = self._parse_zone_index(rect.label, rect.id_)
            if zone_type:
                self._assign_zone(
                    zone_type,
                    zone_index,
                    rect.get_zone(),
                    indexed_zones,
                    unindexed_zones,
                )
                continue

            if rect.label == "bound":
                bounds.append(rect.get_zone())
            elif rect.label == "obstacle":
                obstacles.append(obstacle_from_svgrectangle(rect))
            elif rect.label == "ped_crowded_zone":
                ped_crowded_zones.append(rect.get_zone())
            else:
                # Downgrade to debug: some maps may contain decorative/helper rects without a
                # semantic inkscape:label relevant to the simulation. Treat them as ignorable
                # instead of erroring which previously caused noisy logs and could mask real
                # issues. (Feature 131 hardening)
                logger.debug(
                    "Ignoring non-simulation rectangle id={id} label={lab}",
                    id=rect.id_,
                    lab=rect.label,
                )

        robot_spawn_zones = self._assemble_zones("robot_spawn_zone", indexed_zones, unindexed_zones)
        ped_spawn_zones = self._assemble_zones("ped_spawn_zone", indexed_zones, unindexed_zones)
        robot_goal_zones = self._assemble_zones("robot_goal_zone", indexed_zones, unindexed_zones)
        ped_goal_zones = self._assemble_zones("ped_goal_zone", indexed_zones, unindexed_zones)

        return (
            obstacles,
            robot_spawn_zones,
            ped_spawn_zones,
            robot_goal_zones,
            bounds,
            ped_goal_zones,
            ped_crowded_zones,
        )

    def _info_to_mapdefintion(self) -> None:
        """
        Create a MapDefinition object from the path and rectangle information.
        """

        def _parse_svg_length(value: str | None) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                match = re.match(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", value)
                return float(match.group(1)) if match else None

        width: float | None = _parse_svg_length(self.svg_root.attrib.get("width"))
        height: float | None = _parse_svg_length(self.svg_root.attrib.get("height"))
        view_min_x = 0.0
        view_min_y = 0.0

        view_box = self.svg_root.attrib.get("viewBox") or self.svg_root.attrib.get("viewbox")
        if view_box:
            parts = view_box.replace(",", " ").split()
            if len(parts) == 4:
                vb_min_x, vb_min_y, vb_width, vb_height = parts
                view_min_x = _parse_svg_length(vb_min_x) or 0.0
                view_min_y = _parse_svg_length(vb_min_y) or 0.0
                width = width or _parse_svg_length(vb_width)
                height = height or _parse_svg_length(vb_height)

        if width is None or height is None:
            msg = "SVG width/height missing and could not be inferred from viewBox."
            raise ValueError(msg)

        # Normalize coordinates when the SVG viewBox does not start at (0,0).
        self._apply_viewbox_offset(view_min_x, view_min_y)

        # Process rectangles first
        (
            obstacles,
            robot_spawn_zones,
            ped_spawn_zones,
            robot_goal_zones,
            bounds,
            ped_goal_zones,
            ped_crowded_zones,
        ) = self._process_rects(width, height)

        robot_routes, ped_routes = self._process_paths(
            obstacles,
            robot_spawn_zones,
            ped_spawn_zones,
            robot_goal_zones,
            ped_goal_zones,
            ped_crowded_zones,
        )

        # Process single pedestrians from circles
        # Circles labeled like "single_ped_<id>_start" or "single_ped_<id>_goal"
        single_pedestrians = self._process_single_pedestrians_from_circles()

        self._log_map_warnings(obstacles, ped_routes, ped_crowded_zones)
        robot_routes = self._ensure_robot_routes(robot_routes, robot_spawn_zones, robot_goal_zones)

        poi_positions, poi_labels = self._process_pois()

        logger.debug("Creating MapDefinition object")
        self.map_definition = MapDefinition(
            width,
            height,
            obstacles,
            robot_spawn_zones,
            ped_spawn_zones,
            robot_goal_zones,
            bounds,
            robot_routes,
            ped_goal_zones,
            ped_crowded_zones,
            ped_routes,
            single_pedestrians,
            poi_positions=poi_positions,
            poi_labels=poi_labels,
        )
        logger.debug(f"MapDefinition object created: {type(self.map_definition)}")

    def get_map_definition(self) -> MapDefinition:
        """
        Return the MapDefinition object.

        Returns:
            MapDefinition: The validated map definition created from SVG parsing.
        """
        # verify that the map definition is the correct type
        try:
            assert isinstance(self.map_definition, MapDefinition)
        except AssertionError:
            raise TypeError(
                f"Map definition is not of type MapDefinition: {type(self.map_definition)}",
            )
        return self.map_definition

    def __get_path_number(self, route: str) -> tuple[int, int]:
        """Extract spawn and goal zone indices from a route label.

        Routes have labels of the form 'ped_route_<spawn>_<goal>' or 'robot_route_<spawn>_<goal>'.
        This method extracts the first two numeric segments as spawn and goal indices.

        Args:
            route: Route label string (e.g., 'ped_route_0_1' or 'robot_route_2_3').

        Returns:
            Tuple of (spawn_index, goal_index). Defaults to (0, 0) if no numbers found.

        Example:
            >>> self.__get_path_number("ped_route_3_7")
            (3, 7)
        """
        numbers = re.findall(r"\d+", route)
        if numbers:
            spawn = int(numbers[0])
            goal = int(numbers[1])
        else:
            spawn = 0
            goal = 0
        return spawn, goal


def convert_map(svg_file: str) -> MapDefinition | None:
    """Create a MapDefinition object from an SVG file.

    Parses the SVG file and converts labeled elements into a structured MapDefinition.
    Most errors are caught and logged (returning None), but validation errors
    (e.g., missing robot routes) are re-raised to allow caller fallback logic.

    Args:
        svg_file: Path to the SVG file to convert.

    Returns:
        MapDefinition object on successful conversion, or None if parsing fails
        (except for validation errors which are re-raised).

    Raises:
        ValueError: If map validation fails (e.g., no robot routes defined).

    Example:
        >>> map_def = convert_map("maps/svg_maps/hallway.svg")
        >>> if map_def:
        ...     print(f"Loaded map with {len(map_def.obstacles)} obstacles")
    """

    logger.debug(f'Converting SVG map "{svg_file}" to MapDefinition object.')

    try:
        converter = SvgMapConverter(svg_file)
        assert isinstance(converter.map_definition, MapDefinition)
        md: MapDefinition = converter.map_definition
        logger.debug(
            "SVG map {svg_file} converted: robot_routes={rr} ped_routes={pr} spawn_zones={sz} goal_zones={gz}",
            svg_file=svg_file,
            rr=len(md.robot_routes),
            pr=len(md.ped_routes),
            sz=len(md.robot_spawn_zones),
            gz=len(md.robot_goal_zones),
        )
        return md
    except ValueError as ve:
        # Propagate the specific validation error (e.g., zero robot routes) so caller can fallback.
        logger.error(f"SVG validation error: {ve}")
        raise
    except AssertionError:
        logger.error("Error converting SVG file: MapDefinition object not created.")
        logger.exception("Assertion failure during SVG conversion")
    except (OSError, RuntimeError, TypeError) as e:
        logger.error(f"Unexpected error converting SVG file: {e}")
        logger.exception("Unhandled exception in convert_map")
    return None


def _load_single_svg(file_path: Path, strict: bool) -> dict[str, MapDefinition]:
    """Load a single SVG file and return it as a dictionary entry.

    Args:
        file_path: Path to the SVG file to load.
        strict: If True, raise exceptions on conversion errors; if False, log warnings
            and return empty dict for invalid files.

    Returns:
        Dictionary with single entry {filename_stem: MapDefinition} on success,
        or empty dict if conversion fails (in non-strict mode).

    Raises:
        ValueError: If file is not an SVG or conversion fails (strict mode only).
        OSError: If file cannot be read (strict mode only).
    """
    if file_path.suffix.lower() != ".svg":
        raise ValueError(f"Expected an SVG file, got: {file_path}")
    try:
        md = convert_map(str(file_path))
        if md is None:
            raise ValueError(f"Failed to convert SVG file: {file_path}")
        return {file_path.stem: md}
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        if strict:
            raise
        logger.warning("Skipping invalid SVG map: {f} ({err})", f=str(file_path), err=exc)
        return {}


def _load_svg_directory(dir_path: Path, pattern: str, strict: bool) -> dict[str, MapDefinition]:
    """Load all matching SVG files from a directory.

    Scans the directory for SVG files matching the pattern, converts each to a
    MapDefinition, and returns them as a dictionary keyed by filename stem.

    Args:
        dir_path: Path to directory containing SVG files.
        pattern: Glob pattern for matching files (e.g., '*.svg' or 'map_*.svg').
        strict: If True, raise exceptions on conversion errors; if False, skip
            invalid files with warnings.

    Returns:
        Dictionary mapping filename stems to MapDefinition objects. Only valid
        maps are included.

    Raises:
        ValueError: If no SVG files found matching pattern, or no valid maps loaded.

    Example:
        >>> maps = _load_svg_directory(Path("maps/svg_maps"), "*.svg", strict=False)
        >>> len(maps)
        12
    """
    svg_files = sorted(dir_path.glob(pattern))
    if not svg_files:
        raise ValueError(f"No SVG files found in directory {dir_path} with pattern '{pattern}'")
    out: dict[str, MapDefinition] = {}
    for f in svg_files:
        if f.suffix.lower() != ".svg":
            continue
        out.update(_load_single_svg(f, strict))
    if not out:
        raise ValueError(f"No valid SVG maps loaded from directory {dir_path}")
    logger.info("Loaded {n} SVG map(s) from {dir}", n=len(out), dir=str(dir_path))
    return out


def load_svg_maps(
    path: str,
    pattern: str = "*.svg",
    strict: bool = False,
) -> dict[str, MapDefinition]:
    """Load SVG map(s) from a file or directory into a dictionary.

    Flexible loader that handles both single SVG files and directories of SVGs.
    Returns a dictionary keyed by filename stem (without .svg extension) for easy
    access and programmatic map selection.

    Args:
        path: Path to SVG file or directory containing SVG files.
        pattern: Glob pattern for matching files when path is a directory.
            Ignored if path points to a single file. Defaults to '*.svg'.
        strict: If True, raise exceptions on conversion errors; if False,
            skip invalid maps with warnings. Defaults to False.

    Returns:
        Dictionary mapping filename stems to parsed MapDefinition objects.
        Empty dict if no valid maps found (non-strict mode only).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no valid maps loaded (strict mode) or path validation fails.

    Example:
        >>> # Load single map
        >>> maps = load_svg_maps("maps/svg_maps/hallway.svg")
        >>> hallway_map = maps["hallway"]
        >>>
        >>> # Load all maps from directory
        >>> maps = load_svg_maps("maps/svg_maps/", pattern="map_*.svg")
        >>> for name, map_def in maps.items():
        ...     print(f"{name}: {len(map_def.obstacles)} obstacles")
    """
    p = Path(path)
    if not p.exists():  # pragma: no cover
        raise FileNotFoundError(f"Path does not exist: {path}")
    return _load_single_svg(p, strict) if p.is_file() else _load_svg_directory(p, pattern, strict)
