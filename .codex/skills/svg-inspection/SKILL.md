---
name: svg-inspection
description: "Inspect and debug SVG maps for parser-facing issues (route-only mode, zone index mismatches, risky path commands, and obstacle-crossing routes) using reusable Robot SF helpers."
---

# SVG Inspection

## Overview

Use this skill when SVG maps behave unexpectedly at runtime or emit map parser
warnings. The workflow combines fast text inspection, semantic checks via a
dedicated CLI helper, and targeted follow-up tests.

## Procedure

1. Quick label scan
   - Run:
   - `rg -n "ped_route|robot_route|spawn_zone|goal_zone|obstacle" maps/svg_maps/<map>.svg`

2. Run semantic inspection helper
   - Single map:
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps/<map>.svg --show-routes`
   - Batch:
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps --pattern "classic_*.svg" --strict warning`

3. Export machine-readable report when needed
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps --json output/validation/svg_inspection.json`

4. Follow-up verification
   - `uv run python scripts/validation/verify_maps.py --scope all --mode local`

5. Fix and lock behavior
   - Prefer explicit `*_spawn_zone*` and `*_goal_zone*` on canonical maps.
   - If route-only mode is intentional, keep route labels consistent and add/update tests.
   - Add tests in `tests/maps/` or `tests/test_svg_classic_maps_format.py`.

## Notes

- Reusable API lives in `robot_sf.maps.verification.svg_inspection`.
- The parser is regex-based for path coordinates; commands like `H/V` and curves
  can produce mismatches between editor rendering and runtime geometry.
- Route-only mode is supported, but canonical benchmark maps should prefer
  explicit zones for reproducibility.
