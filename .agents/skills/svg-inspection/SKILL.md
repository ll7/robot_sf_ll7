---
name: svg-inspection
description: "Inspect and debug SVG maps for parser-facing issues using reusable Robot SF helpers."
---

# SVG Inspection

## Purpose

Diagnose map parse/runtime mismatches in `maps/svg_maps/` and produce a short, reproducible finding.

## Workflow

1. Inspect labels quickly:
   - `rg -n "ped_route|robot_route|spawn_zone|goal_zone|obstacle" maps/svg_maps/<map>.svg`
2. Run semantic inspection:
   - `uv run python scripts/validation/svg_inspect.py <map>.svg --show-routes`
   - batch mode: `uv run python scripts/validation/svg_inspect.py maps/svg_maps --pattern "classic_*.svg" --strict warning`
3. Generate machine report if needed:
   - `uv run python scripts/validation/svg_inspect.py maps/svg_maps --json output/validation/svg_inspection.json`
4. Cross-check with map verification helper if route/zone behavior is suspicious.

## Guardrails

- Canonical maps should prefer explicit `spawn_zone` and `goal_zone` labels; avoid route-only assumptions.
- Preserve reproducibility: route-only mode only when intentional and documented.
- Keep parser/runtime caveats explicit in the issue or follow-up.

## Output

- Findings (warnings/errors), map(s) checked, helper output path.
- Whether additional follow-up tests or label edits are required.
- Recommendation for safe remediation.
