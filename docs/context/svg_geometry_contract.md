# SVG geometry contract: legacy vs corrected execution (Issue #8314)

Plain-language summary: the SVG map loader used to silently ignore authored
`transform` attributes, so some maps loaded in a different place than drawn.
There are now two named geometry contracts — `legacy` (old positions, for
comparing with old results) and `corrected` (drawn positions) — and every loaded
map records which one produced it.

Date: 2026-09-03

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/8314>

## What was wrong

`SvgMapConverter` (`robot_sf/nav/svg_map_parser.py`) read `x`/`y`/`d` attributes
directly and never consulted ancestor `transform` attributes. Five tracked maps
carry exactly one pure `translate(...)` group each:

- `maps/svg_maps/classic_bottleneck*.svg` (3 maps): `translate(0,-4.3651647)`,
- `maps/svg_maps/classic_t_intersection.svg`: `translate(0.36645698,-0.15705299)`,
- `maps/svg_maps/planner_test_simple.svg`: `translate(-2.9886965,-0.03735871)`.

On the bottleneck maps the ignored offset moved pedestrian geometry 4.37 m: the
pedestrian spawn overlapped the robot goal by 96.4 %, and the pedestrian goal
extended 2.135 m past the map `viewBox` into the northern boundary wall.

## The two contracts

- `legacy` (default): ancestor transforms are ignored, preserving the
  historical transform-ignoring geometry and simulation inputs. All existing
  call sites keep this behavior unless they opt in.
- `corrected`: nested ancestor `translate(...)` transforms apply to parsed
  paths, rectangles, and circles. Any other transform class (`scale`, `rotate`,
  `skew`, `matrix`) or malformed transform fails closed with `ValueError`
  instead of being silently ignored.

Selection: `SvgMapConverter(svg_file, geometry_contract=...)`,
`convert_map(svg_file, geometry_contract=...)`, scenario key
`map_geometry_contract` (default `"legacy"`), resolved through
`resolve_map_definition`. Unknown contract names fail closed.

## Compatibility boundary

- Every `MapDefinition` records `svg_geometry_contract`. The label distinguishes
  legacy and corrected executions; downstream consumers must partition them or
  establish explicit compatibility before pooling rows as comparable evidence.
- Existing frozen results and figures that ran on the legacy loader remain
  labeled legacy/as-run and are not retroactively corrected.
- The low-tier bottleneck cell authored no pedestrian markers (`diss#2144`);
  transform support shifts existing elements only and adds none. This is
  covered by a regression test asserting an empty pedestrian list for
  `classic_bottleneck.svg` under both contracts.

## Reproducibility commands

```bash
uv run pytest tests/nav/test_svg_transform_contract.py -q
uv run pytest tests/test_svg_classic_maps_format.py tests/maps/test_route_clearance_maps.py -q
```
