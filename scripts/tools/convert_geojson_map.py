"""CLI wrapper for converting annotated OSM/GeoJSON into a Robot SF segment map."""

from robot_sf.nav.geojson_map_builder import main

if __name__ == "__main__":
    raise SystemExit(main())
