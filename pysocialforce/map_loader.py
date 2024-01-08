import json
from pysocialforce.map_config import \
    MapDefinition, GlobalRoute, Obstacle


def load_map(file_path: str) -> MapDefinition:
    """Load map data from the given file path."""

    with open(file_path, 'r') as file:
        map_json = json.load(file)

    obstacles = [Obstacle(o["vertices"]) for o in map_json['obstacles']]
    routes = [GlobalRoute(r['waypoints']) for r in map_json['ped_routes']]
    routes += [GlobalRoute(list(reversed(r['waypoints'])))
               for r in map_json['ped_routes'] if r["reversible"]]
    crowded_zones = [tuple(z["zone_rect"]) for z in map_json['crowded_zones']]

    return MapDefinition(obstacles, routes, crowded_zones)
