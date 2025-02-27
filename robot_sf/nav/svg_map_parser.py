"""get a labled svg map and parse it to a map definition object"""

import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
from loguru import logger

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.nav_types import SvgPath, SvgRectangle
from robot_sf.nav.obstacle import Obstacle, obstacle_from_svgrectangle
from robot_sf.util.types import Line2D, Rect, Zone


class SvgMapConverter:
    """This class manages the conversion of a labeled svg map to a map definition object"""

    svg_file_str: str
    svg_root: ET.Element
    path_info: list
    rect_info: list
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

        # Parse the SVG file
        svg_tree = ET.parse(self.svg_file_str)
        self.svg_root = svg_tree.getroot()

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
        # check that the svg root is loaded
        if self.svg_root is None:
            logger.error("SVG root not loaded")
            return

        # Define the SVG and Inkscape namespaces
        namespaces = {
            "svg": "http://www.w3.org/2000/svg",
            "inkscape": "http://www.inkscape.org/namespaces/inkscape",
        }

        # Find all 'path' elements in the SVG file
        paths = self.svg_root.findall(".//svg:path", namespaces)
        logger.info(f"Found {len(paths)} paths in the SVG file")
        rects = self.svg_root.findall(".//svg:rect", namespaces)
        logger.info(f"Found {len(rects)} rects in the SVG file")

        # Initialize an empty list to store the path information
        path_info = []
        rect_info = []

        # Compile the regex pattern for performance
        coordinate_pattern = re.compile(r"([+-]?[0-9]*\.?[0-9]+)[, ]([+-]?[0-9]*\.?[0-9]+)")

        # Iterate over each 'path' element
        for path in paths:
            # Extract the 'd' attribute (coordinates), 'inkscape:label' and 'id'
            input_string = path.attrib.get("d")
            if not input_string:
                continue  # Skip paths without the 'd' attribute

            # Find all matching coordinates
            filtered_coordinates = coordinate_pattern.findall(input_string)
            if not filtered_coordinates:
                logger.warning("No coordinates found for path: %s", id)
                continue

            # Convert the matched strings directly into a numpy array of floats
            np_coordinates = np.array(filtered_coordinates, dtype=float)

            # Append the information to the list
            path_info.append(
                SvgPath(
                    coordinates=np_coordinates,
                    label=path.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label"),
                    id=path.attrib.get("id"),
                )
            )

        # Iterate over each 'rect' element
        for rect in rects:
            # Extract the attributes and append the information to the list
            rect_info.append(
                SvgRectangle(
                    float(rect.attrib.get("x")),
                    float(rect.attrib.get("y")),
                    float(rect.attrib.get("width")),
                    float(rect.attrib.get("height")),
                    rect.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label"),
                    rect.attrib.get("id"),
                )
            )

        logger.info(f"Parsed {len(path_info)} paths in the SVG file")
        self.path_info = path_info
        logger.info(f"Parsed {len(rect_info)} rects in the SVG file")
        self.rect_info = rect_info

    def _process_obstacle_path(self, path: SvgPath) -> Obstacle:
        """Process a path labeled as obstacle."""
        vertices = path.coordinates.tolist()

        if not np.array_equal(vertices[0], vertices[-1]):
            logger.warning(
                f"Closing polygon: first and last vertices of obstacle <{path.id}> differ"
            )
            vertices.append(vertices[0])

        return Obstacle(vertices)

    def _process_route_path(
        self, path: SvgPath, spawn_zones: List[Rect], goal_zones: List[Rect]
    ) -> GlobalRoute:
        """Process a path labeled as route (pedestrian or robot)."""
        vertices = path.coordinates.tolist()
        spawn, goal = self.__get_path_number(path.label)

        return GlobalRoute(
            spawn_id=spawn,
            goal_id=goal,
            waypoints=vertices,
            spawn_zone=spawn_zones[spawn] if spawn_zones else (vertices[0], 0, 0),
            goal_zone=goal_zones[goal] if goal_zones else (vertices[-1], 0, 0),
        )

    def _process_crowded_zone_path(self, path: SvgPath) -> Zone:
        """Process a path labeled as crowded zone."""
        return Zone(path.coordinates.tolist())

    def _process_rects(
        self, width: float, height: float
    ) -> Tuple[
        List[Obstacle], List[Rect], List[Rect], List[Rect], List[Line2D], List[Rect], List[Rect]
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
        obstacles: List[Obstacle] = []
        robot_spawn_zones: List[Rect] = []
        ped_spawn_zones: List[Rect] = []
        robot_goal_zones: List[Rect] = []
        bounds: List[Line2D] = [
            (0, width, 0, 0),  # bottom
            (0, width, height, height),  # top
            (0, 0, 0, height),  # left
            (width, width, 0, height),  # right
        ]
        logger.debug(f"Bounds: {bounds}")
        ped_goal_zones: List[Rect] = []
        ped_crowded_zones: List[Rect] = []

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
                logger.error(f"Unknown label <{rect.label}> in id <{rect.id_}>")

        return (
            obstacles,
            robot_spawn_zones,
            ped_spawn_zones,
            robot_goal_zones,
            bounds,
            ped_goal_zones,
            ped_crowded_zones,
        )

    def _info_to_mapdefintion(self) -> MapDefinition:
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
        robot_routes: List[GlobalRoute] = []
        ped_routes: List[GlobalRoute] = []

        # Process paths
        path_processors = {
            "obstacle": lambda p: obstacles.append(self._process_obstacle_path(p)),
            "ped_route": lambda p: ped_routes.append(
                self._process_route_path(p, ped_spawn_zones, ped_goal_zones)
            ),
            "robot_route": lambda p: robot_routes.append(
                self._process_route_path(p, robot_spawn_zones, robot_goal_zones)
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

        # Log warnings if required elements are missing
        if not obstacles:
            logger.warning("No obstacles found in the SVG file")
        if not ped_routes:
            logger.warning("No routes found in the SVG file")
        if not ped_crowded_zones:
            logger.warning("No crowded zones found in the SVG file")

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
                f"Map definition is not of type MapDefinition: {type(self.map_definition)}"
            )
        return self.map_definition

    def __get_path_number(self, route: str) -> Tuple[int, int]:
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
    """Create MapDefinition from svg file."""

    logger.info("Converting SVG map to MapDefinition object.")
    logger.info(f"SVG file: {svg_file}")

    try:
        converter = SvgMapConverter(svg_file)
        assert isinstance(converter.map_definition, MapDefinition)
        return converter.map_definition
    except AssertionError:
        logger.error("Error converting SVG file: MapDefinition object not created.")
        logger.error(f"type converter.map_definition: {type(converter.map_definition)}")
    except Exception as e:
        logger.error(f"Error converting SVG file: {e}")
        return None
