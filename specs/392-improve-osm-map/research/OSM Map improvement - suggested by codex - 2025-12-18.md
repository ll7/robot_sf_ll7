**OSM Plan**

- Move away from color-based SVG parsing and ingest raw OSM (PBF/XML) via a parser (e.g., pyrosm or osmnx) to produce Shapely geometries; keep robot_sf/nav/svg_map_parser.py only for legacy hand-edited maps.
- Build a “drivable mask” from OSM tags: buffer line geometries (default ~3 m, override with width/lanes/sidewalk/cycleway tags) for footway/cycleway/path/bridleway/pedestrian/service/residential/unclassified; union with area polygons where highway=* and area=yes (pedestrian/footway/path/service/residential/unclassified).
- Derive obstacles from building/water/landuse=forest/grass etc.; snap/clean with shapely.make_valid and buffer(0) to avoid self-intersections, and optionally shrink by a small epsilon to prevent touching the drivable mask.
- Define a consistent local metric frame: pick an origin (e.g., min lon/lat of the bbox), project to UTM, store scale/offset alongside the MapDefinition; export a background raster (OSM tiles or rendered vector) using the same transform so sim_view.py can overlay accurately.
- Emit MapDefinition artifacts directly: allowed-area polygons (or a raster mask) plus obstacles, spawn/goal zones sampled from allowed areas, and optional inferred crossings where buffered paths intersect obstacles; keep outputs under maps/osm_generated/ with the origin/CRS metadata.

Next steps:

1. Prototype a scripts/osm_to_mapdef.py that reads a local PBF, applies the tag filters/buffering above, and writes a MapDefinition + optional SVG/PNG preview.
2. Decide storage: add an allowed_areas field to MapDefinition (preferred) or bake it into obstacles as negative space.
3. Validate on maps/osm_svg_maps/uni_campus_1350.svg equivalent by comparing drivable mask vs. manual SVG, and iterate on tag/width heuristics.