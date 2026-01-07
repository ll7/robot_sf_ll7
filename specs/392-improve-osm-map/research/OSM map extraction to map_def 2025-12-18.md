# OSM map extraction to map_def

- [[OSM Map improvement - suggested by copilot - 2025-12-18]]
- [[OSM Map improvement - suggested by codex - 2025-12-18]]
- [[OSM Map improvement - suggested by chatgpt - 2025-12-18]]
- [[OSM Map improvement - suggested by perplexity - 2025-12-18]]

Dealing with open street map files is currently not optimal in <https://github.com/ll7/robot_sf_ll7>.

## Current Process

Process is described in `docs/SVG_MAP_EDITOR.md`

This workflow shows how to include a lake as an obstacle when working with OSM-based exports such as `maps/osm_svg_maps/uni_campus_1350.svg`.

1. Export the SVG from the OSM website and keep the scale factor in the filename (e.g., `_1350`) so downstream tools can reuse it. If you only need buildings, run `examples/example_osm_svg_to_obstacle_svg.py` on the export. The script filters elements by the building color string and applies the scale factor to produce a building-only SVG.
2. The lake in this example does not share the building color, so it is skipped as an obstacle. To include it, open the OSM SVG in Inkscape (or edit the XML directly), locate the lake path, and change its color to exactly match the building color string. Reading the color value from the SVG source and pasting it into the lake’s style can be faster than using the GUI.
3. After recoloring, the lake becomes one large obstacle with no crossings. To create crossings, split the shape: use the **Eraser** tool to cut the lake where crossings should exist (it will still be a single path), then use the **Shape Builder** tool to click each desired section and confirm to turn the pieces into separate objects.
4. Select each new lake section, open **Object Properties**, set the **Label** to `obstacle`, and click **Set** so the parser treats every piece as an obstacle.
5. Save the updated SVG. You now have a lake represented as multiple obstacle objects that allow crossings where you split the shape.

After generating the base obstacle svg, we add crowded zones, spawn zones and routes manually. Sometimes adding the lake as obstacle and cutting it in pieces to visualize the routes.

After adding these areas described in `docs/SVG_MAP_EDITOR.md` we use the `robot_sf/nav/svg_map_parser.py` to generate MapDefintions that include everything as python objects. Mainly the MapDefintions.


## Problems with the current process

- we only extract obstacles by color string from the open street map
- Obstacles are partially formatted wrong.
- The area we can drive through does not match real world pathes, instead it is the area between obstacles.
- Obstacles can sometimes not be handled as polygons correctly, as the svg defintion has intersections with themselves.
- I would prefer a osm map based architechture
- The scaling of the svg is not very precise

## Solutions to consider

What I would like to have is a more robust way of using open street map data to create driveable areas for robot that can technically on sidewalks, but can not handle stairs.

If I look at the open street map specification, I see data for the follwing:

- lines: <https://wiki.openstreetmap.org/wiki/OpenStreetMap_Carto/Lines>
- areas: <https://wiki.openstreetmap.org/wiki/OpenStreetMap_Carto/Areas>

Maybe we should implement an ability to define allowed areas as map definition with the open street map as image background.

So in my imagination, we use the entire open street map data, but then we can filter areas that are allowed to be used for our robot to navigate.

### Lines where we can drive

If no further information is available, the lines should be considered to be 3m wide and be an driveable area.

- Roads and ways for non-motorized vehicles
	- everything but steps
	- highway=bridleway
	- highway=cycleway / highway=path + bicycle=designated
	- highway=footway / highway=path

### Areas where we can drive

- highway=pedestrian + area=yes / highway=footway + area=yes / highway=path + area=yes
- highway=service + area=yes / highway=residential + area=yes / highway=unclassified + area=yes
- highway=pedestrian + area=yes / highway=footway + area=yes / highway=path + area=yes

## Task

evaluate the best approach to implement this