# Issue 532 JSON-to-SVG Map Migration

## Scope

Issue [#532](https://github.com/ll7/robot_sf_ll7/issues/532) tracks the safe migration from
legacy JSON-authored map definitions to SVG-authored maps without breaking existing JSON support.

## What Changed

- Added a deterministic map-folder loader in `robot_sf/nav/map_config.py` that:
  - prefers `.svg` files when both `.svg` and `.json` exist for the same base name,
  - falls back to legacy JSON if SVG parsing/validation fails,
  - loads map names in sorted order so the pool order is stable.
- Added the one-off migration helper `scripts/dev/json_to_svg_map.py`.
- Migrated `robot_sf/maps/uni_campus_big.json` to `robot_sf/maps/uni_campus_big.svg`.
- Tightened scenario authoring guidance to SVG-first wording in `docs/scenario_spec_checklist.md`.

## Validation

- Semantic parity check:
  - `uv run python -m pytest tests/test_map_migration.py`
- Repository readiness gate:
  - `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

## Notes

- JSON support remains available for legacy inputs, but new and migrated maps should use SVG.
- The SVG migration was validated against the legacy JSON map definition rather than only by
  visual inspection.
