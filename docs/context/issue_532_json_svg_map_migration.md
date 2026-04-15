# Issue 532 SVG Map Cutover

## Scope

Issue [#532](https://github.com/ll7/robot_sf_ll7/issues/532) tracks the cutover from legacy
JSON-authored map definitions to SVG-authored maps.

## What Changed

- Added a deterministic map-folder loader in `robot_sf/nav/map_config.py` that:
  - loads SVG files only,
  - loads map names in sorted order so the pool order is stable,
  - fails closed when a folder contains no SVG maps.
- Added the one-off migration helper `scripts/dev/json_to_svg_map.py`.
- Migrated `robot_sf/maps/uni_campus_big.json` to `robot_sf/maps/uni_campus_big.svg` and removed
  the legacy JSON asset from the repository.
- Tightened scenario authoring guidance to SVG-first wording in `docs/scenario_spec_checklist.md`.

## Validation

- Semantic parity check:
  - `uv run python -m pytest tests/test_map_migration.py`
- Repository readiness gate:
  - `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

## Notes

- The repository no longer ships or loads JSON-based map assets.
- The SVG migration was validated against the legacy JSON map definition before the cutover, then
  the legacy asset and fallback paths were removed.
