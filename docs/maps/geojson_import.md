# Importing an OpenStreetMap GeoJSON map

This converter turns an annotated OpenStreetMap (OSM) GeoJSON extract into the YAML map format
Robot SF already loads for scenarios. It projects longitude/latitude into a local metre-based frame,
uses OSM tags to form walkable areas and obstacles, and writes obstacle polygons that the runtime
uses as collision line segments.

This is an import tool, not a benchmark claim: an imported location still needs scenario review and
certification before it is used as benchmark evidence.

## Convert a public extract

Download a public GeoJSON extract using a source whose licence permits your intended use. Copy
[`geojson_import_provenance.example.yaml`](../../configs/maps/geojson_import_provenance.example.yaml),
record its source URL, query/bounds in the citation, download date, licence, and raw-file SHA-256.
Then run the provenance checker and converter together:

```bash
uv run python scripts/validation/check_geojson_import.py \
  campus.geojson campus.provenance.yaml maps/imported/campus.yaml
```

`--line-buffer-m 1.5` is the default half-width applied to tagged LineString footways and obstacles.
The input follows standard GeoJSON (WGS84 when no coordinate reference system is declared); the
output coordinates are metres in a local Universal Transverse Mercator (UTM) frame.

The converter recognises OSM `highway` values such as `footway`, `path`, and `pedestrian`, plus
`footway=sidewalk|crossing` and the existing OSM obstacle tags (`building`, water, cliff, and tree).
Use `robot_sf_role=walkable` or `robot_sf_role=obstacle` when source tags are not sufficient.

## Required scenario annotations

Geometry alone cannot identify a safe robot task, so runnable maps require explicit annotation. Add
these properties to GeoJSON features (both `robot_sf_role` and `robot_sf:role` are accepted):

| Geometry | Required properties | Meaning |
| --- | --- | --- |
| Polygon | `robot_sf_role: robot_spawn`, `robot_sf_id: west` | Robot start zone. |
| Polygon | `robot_sf_role: robot_goal`, `robot_sf_id: east` | Robot goal zone. |
| LineString | `robot_sf_role: robot_route`, `robot_sf_spawn: west`, `robot_sf_goal: east` | Route joining the named robot zones. |
| Polygon | `robot_sf_role: ped_spawn` or `ped_goal`, `robot_sf_id: ...` | Optional pedestrian zones. |
| LineString | `robot_sf_role: ped_route`, `robot_sf_spawn: ...`, `robot_sf_goal: ...` | Optional pedestrian route. |

Zone identifiers are sorted before YAML indexes are assigned, so the same annotated extract produces
stable route references. Zones are converted to their minimum rotated rectangles because the existing
map schema represents spawn and goal areas as rectangles.

The command fails rather than inventing a robot spawn, goal, or route. Fix the missing annotation and
rerun it; do not treat a geometry-only conversion as a runnable or benchmark-ready scenario.

## Verify the generated map

```bash
uv run python -c "from pathlib import Path; from robot_sf.training.scenario_loader import resolve_map_definition; assert resolve_map_definition('maps/imported/campus.yaml', scenario_path=Path('maps/imported/campus.yaml'))"
```

For a real import, keep the source data outside Git unless its licence and repository review allow
tracking it. The checker accepts only `exploratory_only`: importing geometry alone does not establish
benchmark evidence. Commit the small YAML map and provenance manifest only after the source,
annotations, and scenario use have been reviewed.
