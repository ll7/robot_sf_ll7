"""get a labled svg map and parse it to a map definition object"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.common.errors import raise_fatal_with_remedy
from robot_sf.common.types import Line2D, Rect, Zone
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.nav_types import SvgCircle, SvgPath, SvgRectangle
from robot_sf.nav.obstacle import Obstacle, obstacle_from_svgrectangle


class SvgMapConverter:
    """This class manages the conversion of a labeled svg map to a map definition object"""

    svg_file_str: str
    svg_root: ET.Element
    path_info: list
    rect_info: list
    circle_info: list
    map_definition: MapDefinition

    def __init__(self, svg_file: str):
        """
        Initialize the SvgMapConverter.
        """
        self.svg_file_str = svg_file
        self._load_svg_root()

        self._get_svg_info()
        self._info_to_mapdefintion()

    def _load_svg_root(self):
        """
        Load the root of the SVG file.
        """
        logger.info(f"Loading the root of the SVG file: {self.svg_file_str}")

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

    def _parse_path_element(
        self, path: ET.Element, coordinate_pattern: re.Pattern
    ) -> SvgPath | None:
        """Parse a single SVG path element into a SvgPath object."""
        input_string = path.attrib.get("d")
        if not input_string:
            return None

        filtered_coordinates = coordinate_pattern.findall(input_string)
        if not filtered_coordinates:
            logger.warning("No coordinates found for path: %s", path.attrib.get("id"))
            return None

        np_coordinates = np.array(filtered_coordinates, dtype=float)

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
            coordinates=np_coordinates,
            label=label,
            id=path_id or "",
        )

    def _parse_rect_element(self, rect: ET.Element) -> SvgRectangle:
        """Parse a single SVG rectangle element into a SvgRectangle object."""
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
        """Parse a single SVG circle element into a SvgCircle object."""
        try:
            cx = float(circle.attrib.get("cx", 0))
            cy = float(circle.attrib.get("cy", 0))
            r = float(circle.attrib.get("r", 0))
            circle_label = circle.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label")
            circle_id = circle.attrib.get("id")
            if circle_label is None and circle_id is None:
                logger.warning(
                    "Circle element missing both inkscape:label and id attribute; using empty string",
                )

            return SvgCircle(
                cx,
                cy,
                r,
                circle_label or circle_id or "",
                circle_id or "",
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
        - 'coordinates': a numpy array of shape (n, 2) containing the x and y coordinates
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

    def _process_obstacle_path(self, path: SvgPath) -> Obstacle:
        """Process a path labeled as obstacle."""
        # SvgPath.coordinates is a Tuple[Vec2D]; convert to list of tuples
        vertices = list(path.coordinates)

        if not np.array_equal(vertices[0], vertices[-1]):
            logger.warning(
                f"Closing polygon: first and last vertices of obstacle <{path.id}> differ",
            )
            vertices.append(vertices[0])

        return Obstacle(vertices)

    def _process_route_path(
        self,
        path: SvgPath,
        spawn_zones: list[Rect],
        goal_zones: list[Rect],
    ) -> GlobalRoute:
        """Process a path labeled as route (pedestrian or robot)."""
        vertices = list(path.coordinates)
        spawn, goal = self.__get_path_number(path.label)

        # Defensive fallback: if spawn/goal indices are out of range (or missing zones), create
        # minimal synthetic zones around first/last waypoint so downstream logic still works.
        def _safe_zone(index: int, zones: list[Rect], waypoint, kind: str) -> Rect:
            if zones and 0 <= index < len(zones):
                return zones[index]
            logger.warning(
                "SVG route path '{pid}' refers to {k} index {idx} but only {avail} zones available; creating synthetic zone.",
                pid=path.id,
                k=kind,
                idx=index,
                avail=len(zones),
            )
            wx, wy = waypoint
            return ((wx, wy), (wx + 0.1, wy), (wx + 0.1, wy + 0.1))

        spawn_zone = _safe_zone(spawn, spawn_zones, vertices[0], "spawn")
        goal_zone = _safe_zone(goal, goal_zones, vertices[-1], "goal")

        return GlobalRoute(
            spawn_id=spawn,
            goal_id=goal,
            waypoints=vertices,
            spawn_zone=spawn_zone,
            goal_zone=goal_zone,
        )

    def _process_crowded_zone_path(self, path: SvgPath) -> Zone:
        """Process a path labeled as crowded zone."""
        return Zone(list(path.coordinates))

    def _process_single_pedestrians_from_circles(self) -> list[SinglePedestrianDefinition]:
        """
        Process circles labeled as single pedestrian markers.

        Circles should be labeled as:
        - "single_ped_<id>_start" for start positions
        - "single_ped_<id>_goal" for goal positions

        Returns a list of SinglePedestrianDefinition objects.
        """
        # Group circles by pedestrian ID
        ped_data: dict[str, dict[str, tuple[float, float]]] = {}

        # Use regex to robustly parse labels that may contain underscores in the ID
        pattern = re.compile(r"^single_ped_(?P<pid>.+)_(?P<kind>start|goal)$")
        for circle in self.circle_info:
            if not circle.label:
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
        logger.debug(f"Bounds: {bounds}")
        ped_goal_zones: list[Rect] = []
        ped_crowded_zones: list[Rect] = []

        for rect in self.rect_info:
            if rect.label == "robot_spawn_zone":
                robot_spawn_zones.append(rect.get_zone())
            elif rect.label == "ped_spawn_zone":
                ped_spawn_zones.append(rect.get_zone())
            elif rect.label == "robot_goal_zone":
                robot_goal_zones.append(rect.get_zone())
            elif rect.label == "bound":
                bounds.append(rect.get_zone())
            elif rect.label == "ped_goal_zone":
                ped_goal_zones.append(rect.get_zone())
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
        width: float = float(self.svg_root.attrib.get("width"))
        height: float = float(self.svg_root.attrib.get("height"))

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

        # Initialize routes
        robot_routes: list[GlobalRoute] = []
        ped_routes: list[GlobalRoute] = []

        # Process paths
        path_processors = {
            "obstacle": lambda p: obstacles.append(self._process_obstacle_path(p)),
            "ped_route": lambda p: ped_routes.append(
                self._process_route_path(p, ped_spawn_zones, ped_goal_zones),
            ),
            "robot_route": lambda p: robot_routes.append(
                self._process_route_path(p, robot_spawn_zones, robot_goal_zones),
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

        # Process single pedestrians from circles
        # Circles labeled like "single_ped_<id>_start" or "single_ped_<id>_goal"
        single_pedestrians = self._process_single_pedestrians_from_circles()

        # Log warnings / validation for required / optional elements
        if not obstacles:
            logger.warning("No obstacles found in the SVG file")
        if not ped_routes:
            logger.warning("No pedestrian routes found in the SVG file")
        if not ped_crowded_zones:
            logger.debug("No crowded zones found in the SVG file (optional)")

        if not robot_routes:
            # Hard validation: we cannot produce robot start positions => downstream division by zero.
            raise ValueError(
                "SVG map conversion produced zero robot routes. Ensure at least one 'robot_route_*_*' path label exists.",
            )

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
        )
        logger.debug(f"MapDefinition object created: {type(self.map_definition)}")

    def get_map_definition(self) -> MapDefinition:
        """
        Return the MapDefinition object.
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
        # routes have a label of the form 'ped_route_<spawn>_<goal>'
        numbers = re.findall(r"\d+", route)
        if numbers:
            spawn = int(numbers[0])
            goal = int(numbers[1])
        else:
            spawn = 0
            goal = 0
        return spawn, goal


def convert_map(svg_file: str):
    """Create MapDefinition from svg file.

    Returns None on conversion failure; raises no exceptions outward (they are logged) except
    for the explicit validation error on missing robot routes which is also logged then rethrown
    so callers can decide to fallback to a default map pool.
    """

    logger.debug("Converting SVG map to MapDefinition object.")
    logger.info(f"SVG file: {svg_file}")

    try:
        converter = SvgMapConverter(svg_file)
        assert isinstance(converter.map_definition, MapDefinition)
        md: MapDefinition = converter.map_definition
        logger.info(
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
    """Load one or many SVG maps into a dict keyed by filename stem."""
    p = Path(path)
    if not p.exists():  # pragma: no cover
        raise FileNotFoundError(f"Path does not exist: {path}")
    return _load_single_svg(p, strict) if p.is_file() else _load_svg_directory(p, pattern, strict)
