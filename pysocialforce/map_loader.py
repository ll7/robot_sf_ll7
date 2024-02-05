"""
map_loader.py
"""
import logging
import json
from pysocialforce.map_config import MapDefinition, GlobalRoute, Obstacle

# Configure logger
logger = logging.getLogger(__name__)

def load_map(file_path: str) -> MapDefinition:
    """Load map data from the given file path."""

    # Initialize empty lists for map components
    obstacles, routes, crowded_zones = [], [], []

    with open(file_path, 'r', encoding="utf-8") as file:
        try:
            map_json = json.load(file)

            # Load obstacles if they exist
            if 'obstacles' in map_json:
                obstacles = [Obstacle(o["vertices"]) for o in map_json['obstacles']]
            else:
                logger.warning("No obstacles found in map file")

            # Load pedestrian routes if they exist
            if 'ped_routes' in map_json:
                routes = [GlobalRoute(r['waypoints']) for r in map_json['ped_routes']]
                # Add reversed routes if reversible
                routes += [GlobalRoute(list(reversed(r['waypoints'])))
                           for r in map_json['ped_routes'] if r.get("reversible", False)]
            else:
                logger.warning("No pedestrian routes found in map file")

            # Load crowded zones if they exist
            if 'crowded_zones' in map_json:
                crowded_zones = [tuple(z["zone_rect"]) for z in map_json['crowded_zones']]
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
