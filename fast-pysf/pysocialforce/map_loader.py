"""
map_loader.py
"""

import json
import logging
from pathlib import Path

from pysocialforce.map_config import GlobalRoute, MapDefinition, Obstacle, Zone

# Configure logger
logger = logging.getLogger(__name__)


def load_map(file_path: str | Path) -> MapDefinition:
    """Load map data from the given file path.
    
    Args:
        file_path: Path to the JSON map file (string or Path object)
    
    Returns:
        MapDefinition object containing obstacles, routes, and crowded zones
    
    Raises:
        json.JSONDecodeError: If the file contains invalid JSON
        FileNotFoundError: If the file does not exist
        KeyError: If required keys are missing from the map data
    """

    # Initialize empty lists for map components
    obstacles: list[Obstacle] = []
    routes: list[GlobalRoute] = []
    crowded_zones: list[Zone] = []

    with open(file_path, encoding="utf-8") as file:
        try:
            map_json = json.load(file)

            # Load obstacles if they exist
            if "obstacles" in map_json:
                obstacles = [Obstacle(o["vertices"]) for o in map_json["obstacles"]]
            else:
                logger.warning("No obstacles found in map file")

            # Load pedestrian routes if they exist
            if "ped_routes" in map_json:
                routes = [GlobalRoute(r["waypoints"]) for r in map_json["ped_routes"]]
                # Add reversed routes if reversible
                routes += [
                    GlobalRoute(list(reversed(r["waypoints"])))
                    for r in map_json["ped_routes"]
                    if r.get("reversible", False)
                ]
            else:
                logger.warning("No pedestrian routes found in map file")

            # Load crowded zones if they exist
            if "crowded_zones" in map_json:
                crowded_zones = [tuple(z["zone_rect"]) for z in map_json["crowded_zones"]]
            else:
                logger.warning("No crowded zones found in map file")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from map file: {file_path}")
            raise
        except KeyError as e:
            logger.warning(f"Key {e} not found in map file")
        except Exception as e:
            logger.error(f"An error occurred while loading the map: {e}")
            raise

    return MapDefinition(obstacles, routes, crowded_zones)
