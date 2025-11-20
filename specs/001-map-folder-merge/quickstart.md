# Quickstart: Adding a New Map (Post-Merge Hierarchy)

## Goal
Add a new map (SVG layout + JSON metadata) in under 5 minutes using the canonical structure and registry abstraction.

## Prerequisites
- Feature branch with merged canonical hierarchy: `maps/svg_maps/` and `maps/metadata/`
- SVG editor or source asset ready (e.g., Inkscape output)
- Metadata parameters (zones, dimensions) prepared

## Steps
1. Copy SVG into `maps/svg_maps/` named `<map_id>.svg` (use lowercase snake_case for `<map_id>`).
2. Create JSON metadata file `maps/metadata/<map_id>.json` with at least:
   ```json
   {
     "id": "<map_id>",
     "zones": [],
     "version": "v1"
   }
   ```
3. Run registry audit:
   ```bash
   uv run python -c "from robot_sf.maps.registry import build_registry; build_registry()"
   ```
   - Confirms new ID appears; no duplicates.
4. Validate ID programmatically (example):
   ```bash
   uv run python -c "from robot_sf.maps.registry import validate_map_id; validate_map_id('<map_id>'); print('Valid')"
   ```
5. Use the map in an environment:
   ```python
   from robot_sf.gym_env.environment_factory import make_robot_env
   from robot_sf.gym_env.unified_config import RobotSimulationConfig

   config = RobotSimulationConfig(map_pool=['<map_id>'])
   env = make_robot_env(config=config)
   env.reset()
   ```
6. Update documentation if introducing novel zones or semantics.
7. Add an entry to CHANGELOG.md if map ID is new and publicly relevant.

## Validation Checklist
- [ ] SVG present under `maps/svg_maps/`
- [ ] JSON present under `maps/metadata/` with matching `id`
- [ ] Registry lists new ID
- [ ] Environment resets successfully
- [ ] Audit returns zero stray files

## Troubleshooting
| Issue | Cause | Remedy |
|-------|-------|--------|
| Unknown map ID error | ID mismatch between JSON and filename | Align `id` field with file base name |
| Duplicate ID detected | Existing asset with same ID | Choose unique ID; update references |
| Registry missing new ID | Cache not refreshed | Restart process / call `build_registry(force=True)` (future flag) |

## Time Benchmark
Typical addition time observed: ~3 minutes (excluding SVG authoring).

## Next Steps
For advanced semantics (zones, obstacles), consult `docs/SVG_MAP_EDITOR.md` and extend JSON structure; ensure tests reflect new behaviors if public.
