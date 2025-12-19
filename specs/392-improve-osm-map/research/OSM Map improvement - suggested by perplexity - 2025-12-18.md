A robust approach is to move from SVG color heuristics to an OSM-native pipeline that (1) queries and filters ways/areas by tags, (2) buffers line geometries into drivable polygons, (3) resolves topology/self-intersections, and then (4) exports a clean MapDefinition (plus a raster OSM background if desired). [1][2][3][4]

## Overall architecture

Use an OSM processing stack (e.g., `osmium` + `shapely`/`pygeos` or `osmnx`) to generate a “drivable area” polygon layer from OSM extracts, and treat SVG/Inkscape purely as an optional visualization/editor on top, not as the ground truth. [3][4]

Core steps:
- Download or clip OSM data (PBF or XML) to your map region.
- Build two layers:
  - Line-based: pedestrian/low-speed roads, sidewalks, paths (buffered to 3 m width if no `width=*`).
  - Area-based: `highway=* + area=yes` / `area:highway=*` polygons. [1][5][3]
- Union and clean these layers into one or more “drivable area” polygons and “non-drivable” polygons (water, buildings, etc.).
- Convert to your MapDefinition structures directly (no SVG round-trip), and optionally render a background raster from OSM tiles or a carto style.

This directly fixes scale precision, self-intersecting SVG issues, and the “driveable = complement of obstacles” problem because the drivable region is explicitly constructed from OSM semantics. [1][5][3][4]

## Tag filters for drivable lines/areas

For “lines where we can drive”, define an OSM filter roughly as:
- Include ways with:
  - `highway=footway`, `highway=path`, `highway=cycleway`, `highway=bridleway`, `highway=pedestrian`, and possibly low-speed `highway=service|residential|unclassified` depending on your scenario. [1][6][3]
- Exclude stairs and non-accessible segments:
  - `highway=steps`
  - or `incline=up/down` with `step_count=*` if present.
- Sidewalk-specific:
  - `footway=sidewalk` on `highway=footway` or separate ways mapped as sidewalks. [6][7]

For areas:
- Include polygons where:
  - `highway=pedestrian|footway|path|service|residential|unclassified` with `area=yes`. [1][5][3]
  - or `area:highway=footway|pedestrian|path` for mapped sidewalk/footway areas. [1][5]
- Exclude obstacles:
  - `building=*`, `amenity=parking` (if you do not want driving through lots), `natural=water`, `landuse=construction`, etc. [3]

For each line, buffer:
- If `width` tag exists, use $$ w = \text{width} $$; else default to 3 m total width (1.5 m on each side). [2][3]
- Optionally maintain type-specific widths using a small lookup table (like SidewalKreator does). [2]

This gives you a polygonal representation directly tied to OSM semantics rather than color-based SVG. [1][2][3]

## Geometric cleaning and topology

Use a geometry engine (e.g., `shapely`) to:
- Buffer all selected line-ways, union with area-ways to get `drivable_multi_polygon`.
- Fix self-intersections and sliver polygons with `buffer(0)`, `unary_union`, and `polygonize` if needed. [2][4]
- Subtract “obstacle” polygons (buildings, water, private areas) from the drivable regions:
  - `drivable_clean = drivable_multi_polygon.difference(obstacles_union)`
- Optionally decompose into connected components to become multiple MapDefinitions or zones. [4]

Because your MapDefinition already stores polygons, this step can map 1:1 to your internal `Obstacle`, `DrivableArea`, `CrowdedZone`, etc., classes without SVG at all. [8]

## Integration with MapDefinition and visualization

Implementation strategy for `robot_sf_ll7`:
- Add a new module, e.g., `robot_sf/nav/osm_map_builder.py`, which:
  - Accepts an OSM file or bounding box + Overpass query.
  - Runs the tag filters, buffering, and cleaning to produce:
    - `drivable_polygons: list[Polygon]`
    - `obstacle_polygons: list[Polygon]`
    - optional `water_polygons`, `building_polygons`, etc.
  - Converts these into `MapDefinition` instances directly, bypassing SVG parsing. [8]
- Keep `svg_map_parser.py` for hand-crafted maps and small tweaks, but treat it as a separate path.

For background imagery:
- Render a static PNG from OSM tiles or from a carto-style using the same bounding box and scale you use for coordinates, and store an affine transform in MapDefinition metadata.
- In your sim viewer, draw the OSM raster as background plus the vector drivable area overlay.

This gives your “OSM-based architecture” where OSM is the primary source and SVG is simply a UI option (e.g., for drawing extra zones on top of an exported vector layer). [4]

## Practical tooling choices

Concrete, low-friction stack:
- Use `osmnx` to fetch and filter networks and polygons; it already wraps Overpass and `networkx` + `shapely`. [4]
- Mirror SidewalKreator’s idea: keep a small YAML/JSON with default widths per `highway` type and apply them when `width` is missing. [2]
- For performance and reproducibility, consider:
  - Pre-downloading PBFs with `osmium` and using `pyosmium` for low-level parsing if you outgrow `osmnx`. [3][4]

This design addresses your pain points (precision, semantics, polygon robustness) and keeps the mental model simple: OSM → semantic filtering → buffered polygons → cleaned drivable area → MapDefinition, with optional overlays for crowded/spawn routes either programmatically or via a thin SVG/GUI layer. [1][2][5][3][4]

Sources
[1] Tag:highway=footway - OpenStreetMap Wiki https://wiki.openstreetmap.org/wiki/Tag:highway=footway
[2] OSM Sidewalkreator - A QGIS plugin for automated sidewalk ... https://media.ccc.de/v/state-of-the-map-2022-academic-track-19556-osm-sidewalkreator-a-qgis-plugin-for-automated-sidewalk-drawing-for-osm
[3] Map features - OpenStreetMap Wiki https://wiki.openstreetmap.org/wiki/Map_features
[4] 6 Retrieving Data From OpenStreetMap - GitHub Pages https://gdsl-ul.github.io/wma/labs/w07_OSM.html
[5] Clarification of footway area mapping - General talk https://community.openstreetmap.org/t/clarification-of-footway-area-mapping/130974
[6] Tag:footway=sidewalk - OpenStreetMap Wiki https://wiki.openstreetmap.org/wiki/Tag:footway=sidewalk
[7] OpenSidewalks Rules For Mapping Pedestrian Pathway ... https://tcat.cs.washington.edu/wp-content/uploads/OpenSidewalks_mapping_rulesForMapping.pdf
[8] robot_sf_ll7 https://github.com/ll7/robot_sf_ll7
[9] Multi-Modal 3D Scene Graph Updater for Shared and Dynamic Environments https://arxiv.org/pdf/2411.02938.pdf
[10] Robotic Template Library https://arxiv.org/pdf/2107.00324.pdf
[11] Robotic Template Library http://openresearchsoftware.metajnl.com/articles/10.5334/jors.353/galley/512/download/
[12] Tag Map: A Text-Based Map for Spatial Reasoning and Navigation with
  Large Language Models https://arxiv.org/html/2409.15451
[13] Empowering Robot Path Planning with Large Language Models: osmAG Map
  Topology & Hierarchy Comprehension with LLMs http://arxiv.org/pdf/2403.08228.pdf
[14] Visual Language Maps for Robot Navigation https://arxiv.org/pdf/2210.05714.pdf
[15] Do Visual-Language Grid Maps Capture Latent Semantics? https://arxiv.org/html/2403.10117v2
[16] Time is on my sight: scene graph filtering for dynamic environment
  perception in an LLM-driven robot http://arxiv.org/pdf/2411.15027.pdf
[17] Generate a inkscape template or plugin for map generation https://callonbrainly.blogspot.com/?page=en-git-ll7-robot-sf-ll7-1765526970152
[18] ROS Developers LIVE Class #73: Why my Robot Map is not Correct https://www.youtube.com/watch?v=RgQQCDMSbOQ
[19] Demo 1 - Mapping & Navigation - Stretch Docs https://docs.hello-robot.com/0.3/getting_started/demos_mapping_and_navigation/
[20] GitHub - Lash-L/RoborockCustomMap https://github.com/Lash-L/RoborockCustomMap
[21] ROS Answers: Open Source Q&A Forum https://question1975.rssing.com/chan-34076341/all_p2.html
[22] Loading... https://lerobot-visualize-dataset.hf.space/yaak-ai/L2D/episode_77
[23] Support the sidewalk tag on highways · Issue #1762 · openstreetmap/iD https://github.com/openstreetmap/iD/issues/1762
[24] ROS1云课→28机器人代价地图配置 https://blog.csdn.net/ZhangRelay/article/details/126861763
[25] From Sideline to Center: The Potential for OpenStreetMap to Seize ... https://blog.mapillary.com/update/2019/02/12/potential-for-openstreetmap-to-seize-the-curb.html
[26] [Solved] ROSbot 2 Pro - ROS2 Foxy. Mapping Error https://community.husarion.com/t/solved-rosbot-2-pro-ros2-foxy-mapping-error/1558
[27] ROS Navigation Real Robot Project Map https://get-help.theconstruct.ai/t/ros-navigation-real-robot-project-map/25657
[28] Appropriately mapping sidewalks (footways) on OSM https://www.reddit.com/r/openstreetmap/comments/iiwvus/appropriately_mapping_sidewalks_footways_on_osm/
[29] GitHub - TheOnceAndFutureSmalltalker/ros_map_editor: Qt based tool for editing maps generated by ROS gmapping package. https://github.com/TheOnceAndFutureSmalltalker/ros_map_editor
