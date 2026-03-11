# SVG Inspection Workflow

[← Back to Documentation Index](../README.md)

This guide defines a practical, repeatable workflow for debugging SVG maps when
rendered geometry and parser behavior diverge.

## Why this exists

Robot SF currently parses SVG paths with regex-based waypoint extraction. Some
SVG path commands (for example `H`, `V`, curves, relative commands) can look
correct in an editor but produce unexpected waypoints at runtime.

## Recommended workflow

1. Fast text inspection
   - Check map labels and route ids:
   - `rg -n "ped_route|robot_route|spawn_zone|goal_zone|obstacle" maps/svg_maps/<map>.svg`

2. Structural/semantic inspection (new helper)
   - Single map:
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps/<map>.svg --show-routes`
   - Batch map family:
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps --pattern "classic_*.svg" --strict warning`

3. JSON report for tooling or PR artifacts
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps --json output/validation/svg_inspection.json`

4. Existing map verification (complements this tool)
   - `uv run python scripts/validation/verify_maps.py --scope all --mode local`

5. Visual editor pass
   - Open SVG in Inkscape to inspect route nodes, labels, and obstacle overlap.

6. Lock behavior with tests
   - Add or update tests under `tests/maps/` and `tests/test_svg_classic_maps_format.py`.

## What `svg_inspect.py` checks

- Route-only mode detection:
  - routes present but corresponding spawn/goal rectangles absent
- Route index consistency:
  - route labels pointing to out-of-range zone indices
- Obstacle-interior crossing:
  - route line crosses obstacle interior (`crosses` / `within`)
- Risky path commands for parser:
  - `H/V`, curves (`C/Q/S/T/A`), and relative commands (`m/l/...`)

## Useful Python tooling

- `robot_sf.maps.verification.svg_inspection`:
  - Reusable report API used by the CLI helper.
- `xml.etree.ElementTree`:
  - Raw SVG attribute and path command extraction.
- `shapely`:
  - Route-vs-obstacle geometry checks.
- Optional external tools for deeper visual/debug workflows:
  - `svgpathtools` for full SVG path evaluation and resampling.
  - `cairosvg` for deterministic raster exports (`SVG -> PNG/PDF`) in CI artifacts.
  - `lxml` for advanced XML querying/transform pipelines.

## Policy recommendations

- Benchmark/canonical maps should prefer explicit spawn/goal zones.
- Route-only maps are supported for rapid prototyping, but validate with
  `svg_inspect.py` before training or benchmark runs.
