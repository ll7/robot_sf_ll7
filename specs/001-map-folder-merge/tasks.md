# Tasks: Map Folder Merge

Feature: Map Folder Merge (`001-map-folder-merge`)
Spec: `specs/001-map-folder-merge/spec.md`
Plan: `specs/001-map-folder-merge/plan.md`

## Dependency Graph (User Stories)
```
US1 (P1) -> US2 (P2) -> US3 (P3)
```
Rationale: Consolidation (US1) must finish before contributor workflow (US2); error messaging & consumer selection (US3) depend on registry established in US1 and quick addition workflow from US2.

## Parallel Execution Examples
- Registry implementation (`robot_sf/maps/registry.py`) can run in parallel with audit script creation (`robot_sf/maps/audit.py`) after snapshot tasks.
- Documentation updates (`docs/SVG_MAP_EDITOR.md`, `CHANGELOG.md`) can proceed while tests are written (`tests/test_maps_registry.py`).
- Performance measurement task can run in parallel with hard-coded reference removal after migration.

## MVP Scope
MVP = Completion of US1 (consolidation + audit + registry) yielding zero stray files and backward-compatible map ID resolution.

## Phase 1: Setup

Purpose: Establish baseline, snapshot current state for rollback and regression detection.

Independent Test Criteria: After Phase 1, we can list existing map IDs and confirm snapshot artifacts exist.

Tasks:
- [ ] T001 Capture pre-migration map SVG list to `output/tmp/maps_pre_migration_svg.txt`
- [ ] T002 Capture pre-migration map JSON list to `output/tmp/maps_pre_migration_json.txt`
- [ ] T003 Record pre-migration map ID count via script `scripts/validation/snapshot_map_ids.py`
- [ ] T004 [P] Create placeholder directory `maps/metadata/` (canonical JSON target)
- [ ] T005 Add housekeeping issue note for duplicate numeric spec prefixes in `CHANGELOG.md`

## Phase 2: Foundational

Purpose: Implement core abstractions and safety tooling prior to migration.

Independent Test Criteria: Registry lists all existing IDs; audit reports stray legacy directory before migration; validation helper raises ValueError for fake ID.

Tasks:
- [ ] T006 Implement `robot_sf/maps/registry.py` with `build_registry`, `list_ids`, `get`, `validate_map_id`
- [ ] T007 [P] Implement caching in `robot_sf/maps/registry.py` (module-level _CACHE)
- [ ] T008 Implement audit tool `robot_sf/maps/audit.py` scanning for stray SVG/JSON outside `maps/svg_maps` & `maps/metadata`
- [ ] T009 [P] Add hard-coded path/ID grep in audit (`search_patterns = ["robot_sf/maps", ".svg\"", "uni_campus_big"]`)
- [ ] T010 Create test file `tests/test_maps_registry.py` (scaffold without assertions yet)
- [ ] T011 [P] Add pre-migration registry smoke script `scripts/validation/registry_smoke.py`
- [ ] T012 Add invalid ID test in `tests/test_maps_registry.py` expecting ValueError listing known IDs
- [ ] T013 [P] Add list IDs test ensuring snapshot count matches pre-migration artifacts
- [ ] T014 Add audit stray files test in `tests/test_maps_registry.py`

## Phase 3: User Story US1 (P1) – Consolidate duplicate folders; audit zero stray files

Goal: Achieve single canonical hierarchy. Remove legacy duplicates.
Independent Test Criteria: Audit returns zero stray assets; ID count unchanged; environment factory still loads maps by ID.

Tasks:
- [ ] T015 [US1] Migrate JSON metadata files into `maps/metadata/` using script `scripts/migration/move_map_metadata.py`
- [ ] T016 [P] [US1] Migrate any SVGs located under `robot_sf/maps/` to `maps/svg_maps/` (if present)
- [ ] T017 [US1] Update environment factory internal map loading to use `registry.get(id)` instead of direct paths in `robot_sf/gym_env/environment_factory.py`
- [ ] T018 [P] [US1] Remove legacy map asset directory `robot_sf/maps/` (assets only; preserve code)
- [ ] T019 [US1] Run audit tool post-migration; write report to `output/tmp/maps_audit_post.json`
- [ ] T020 [US1] Add assertion-based test verifying zero stray files after migration in `tests/test_maps_registry.py`
- [ ] T021 [P] [US1] Add CHANGELOG entry documenting consolidation and backward compatibility guarantee
- [ ] T022 [US1] Search & replace any hard-coded map path literals in `robot_sf/gym_env` modules
- [ ] T023 [P] [US1] Add performance measurement script `scripts/benchmark/map_init_perf.py` (<5% delta check)

## Phase 4: User Story US2 (P2) – Contributor adds new map quickly

Goal: Ensure workflow for new map addition within 5 minutes.
Independent Test Criteria: Quickstart steps execute successfully; new map ID appears; invalid steps produce clear errors.

Tasks:
- [ ] T024 [US2] Enhance `quickstart.md` with example JSON containing zones & dimensions
- [ ] T025 [P] [US2] Add validation helper docstring clarifying required JSON keys in `robot_sf/maps/registry.py`
- [ ] T026 [US2] Add map addition workflow test `tests/test_map_addition_workflow.py` (temp fixture map)
- [ ] T027 [P] [US2] Add script `scripts/validation/add_map_example.py` demonstrating programmatic addition & registry rebuild
- [ ] T028 [US2] Update `docs/SVG_MAP_EDITOR.md` with new canonical hierarchy references
- [ ] T029 [P] [US2] Update `docs/README.md` index linking quickstart and registry section
- [ ] T030 [US2] Add CHANGELOG entry for new contributor workflow documentation

## Phase 5: User Story US3 (P3) – Environment consumer selects map by ID with clear error on invalid

Goal: Provide consumer-facing selection robustness and error clarity.
Independent Test Criteria: Selecting valid ID yields successful reset; invalid ID raises ValueError enumerating available IDs.

Tasks:
- [ ] T031 [US3] Integrate `validate_map_id` into environment creation in `robot_sf/gym_env/environment_factory.py` before map load
- [ ] T032 [P] [US3] Add test `tests/test_map_invalid_id_error.py` asserting error message contains available IDs
- [ ] T033 [US3] Add test `tests/test_env_map_selection.py` performing env reset with sample ID
- [ ] T034 [P] [US3] Add negative test for partially missing metadata (simulate removal) expecting abort
- [ ] T035 [US3] Document error usage pattern in `docs/ENVIRONMENT.md` (map selection section)

## Phase 6: Polish & Cross-Cutting

Goal: Final refinements, performance, documentation completeness.
Independent Test Criteria: All tests green; performance script shows <5% init delta; docs updated; no stray prints.

Tasks:
- [ ] T036 Run full test suite + coverage `uv run pytest` and record coverage delta
- [ ] T037 [P] Replace any remaining `print` statements in map-related code with Loguru logger
- [ ] T038 Add optional checksum support placeholder in `registry.py` (commented or feature flag) without breaking API
- [ ] T039 [P] Add doc section in `docs/dev_guide.md` summarizing map registry design
- [ ] T040 Final audit rerun; save results `output/tmp/maps_audit_final.json`
- [ ] T041 [P] Open housekeeping issue for duplicate numeric spec prefixes (link to plan complexity tracking)
- [ ] T042 Confirm performance measurement script outputs pass; record result in `output/benchmarks/map_init_perf.json`
- [ ] T043 Final CHANGELOG consolidation for feature release entry

## Implementation Strategy
1. MVP: Complete Phases 1–3 (US1) to stabilize canonical hierarchy and registry.
2. Incremental: Deliver US2 docs/tests next; ensure contributor workflow proven.
3. Final: Implement US3 error surfacing; polish and performance validation.

## Independent Test Criteria Summary
- US1: Audit zero stray files, ID count unchanged, factory loads maps.
- US2: New map addition appears in registry; workflow test passes under time threshold (manual timing optional).
- US3: Valid ID resets env; invalid ID error lists all IDs; missing metadata abort test passes.

## Task Counts
- Total Tasks: 43
- US1 Tasks: 9 (T015–T023)
- US2 Tasks: 7 (T024–T030)
- US3 Tasks: 5 (T031–T035)
- Parallelizable Tasks (marked [P]): 18

## Format Validation
All tasks follow `- [ ] T### [P] [USx] Description with file path` format (where applicable). Setup & foundational phases omit story labels by rule; user story phases include `[US#]`; parallel tasks contain `[P]` marker.

