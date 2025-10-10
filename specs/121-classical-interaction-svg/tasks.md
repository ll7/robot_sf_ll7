# Tasks: Classical Interaction SVG Scenario Pack

**Feature Dir**: `specs/121-classical-interaction-svg/`
**Related Config**: `configs/scenarios/classic_interactions.yaml`
**Maps**: `maps/svg_maps/classic_*.svg`

## Guiding Principles
- Follow existing benchmark architecture; no schema or core code changes unless required for parser robustness.
- Tests-first where new behavior (validation & parser resilience) was introduced.
- Keep tasks atomic and reference exact file paths.
- Mark tasks that are already completed (retroactive documentation) but keep numbering for traceability.

Legend: `[P]` = Can execute in parallel (different files & no dependency), `(done)` = already completed.

## Phase 1: Setup & Planning
- [x] T001 Create implementation plan document in `specs/121-classical-interaction-svg/plan.md` (done)
- [x] T002 Verify specification `specs/121-classical-interaction-svg/spec.md` covers FR-001..FR-010 (done)

## Phase 2: Asset & Config Authoring (Tests reference these)
- [x] T003 [P] Create SVG map `maps/svg_maps/classic_crossing.svg` (done)
- [x] T004 [P] Create SVG map `maps/svg_maps/classic_head_on_corridor.svg` (done)
- [x] T005 [P] Create SVG map `maps/svg_maps/classic_overtaking.svg` (done)
- [x] T006 [P] Create SVG map `maps/svg_maps/classic_bottleneck.svg` (done)
- [x] T007 [P] Create SVG map `maps/svg_maps/classic_doorway.svg` (done)
- [x] T008 [P] Create SVG map `maps/svg_maps/classic_merging.svg` (done)
- [x] T009 [P] Create SVG map `maps/svg_maps/classic_t_intersection.svg` (done)
- [x] T010 [P] Create SVG map `maps/svg_maps/classic_group_crossing.svg` (done)
- [x] T011 Author scenario matrix `configs/scenarios/classic_interactions.yaml` (done)

## Phase 3: Tests First (Validation & Edge Handling)
- [x] T012 [P] Add scenario matrix validation test `tests/test_classic_interactions_matrix.py` (done, asserts map existence + density + groups rules)
- [x] T013 Add/adjust CLI logging flag tests (pre-existing) to ensure parser supports global flags (done, reused `tests/unit/test_cli_logging_flags.py`)

## Phase 4: Core Implementation Adjustments
- [x] T014 Implement parser intermixed/global flag support in `robot_sf/benchmark/cli.py` (done)
- [x] T015 Refine parser to reorder post-subcommand global flags in `robot_sf/benchmark/cli.py` (done)

## Phase 5: Documentation & Indexing
- [x] T016 Link scenario pack in `docs/README.md` (done)
- [ ] T017 Create optional archetype overview README `specs/121-classical-interaction-svg/README.md` (pending)
- [ ] T018 Add future extension notes (advanced densities, directional bias) in `specs/121-classical-interaction-svg/README.md` (pending, depends on T017)

## Phase 6: Polish & Enhancements
- [ ] T019 [P] Generate scenario thumbnails via CLI (`robot_sf_bench plot-scenarios --matrix configs/scenarios/classic_interactions.yaml --out-dir docs/figures/classic_interactions --pdf --montage`) and store outputs under `docs/figures/classic_interactions/`
- [ ] T020 [P] Add thumbnail montage reference line in `docs/README.md` (depends on T019)
- [ ] T021 [P] Add smoke script `scripts/smoke_classic_interactions.py` performing: load matrix, run one seed per archetype, print summary
- [ ] T022 [P] Add README usage snippet to `examples/README.md` referencing classic scenario matrix
- [ ] T023 Add test for thumbnails generation command (lightweight) `tests/test_plot_classic_interactions.py` ensuring command runs (skip if missing DISPLAY vars)
- [ ] T024 Performance sanity check script extension (optional) to measure steps/sec per archetype (append to `scripts/validation/performance_smoke_test.py` or new script)
- [ ] T025 Add follow-up TODO list inside `specs/121-classical-interaction-svg/README.md` capturing potential map refinements (spawn spacing validation)

## Phase 7: Quality Gates
- [ ] T026 Run lint & format (Ruff) after all pending tasks
- [ ] T027 Run full test suite (`uv run pytest tests`) ensuring new tests and thumbnails test pass
- [ ] T028 Update `CHANGELOG.md` with entry: "Add classical interaction scenario pack and parser flexible global flag ordering"

## Dependencies & Ordering
- T001 → prerequisite for generation script (plan required)
- T003–T010 can run in parallel ([P]) once spec confirmed (T002)
- T011 depends on T003–T010 assets existing
- T012 depends on T011 & assets
- T014 depends on T012 (tests define behavior) & T013 baseline
- T015 refines T014 (sequence sensitive)
- T016 after core adjustments (ensures docs reflect final state)
- T017 before T018
- T019 before T020
- T019 before T023 (thumbnail generation test needs images)
- T021 independent (after assets & matrix) so after T011
- T024 optional after baseline validation (T012) for perf insight
- T026–T028 finalization after all implementation tasks

## Parallel Execution Examples
```
# Example: Run all map creation tasks concurrently (already done):
T003,T004,T005,T006,T007,T008,T009,T010

# Example: Generate thumbnails & smoke script concurrently:
T019,T021  (ensure matrix + assets present)

# Example: Polish documentation tasks concurrently:
T020,T022,T025  (after thumbnails + README base)
```

## Validation Checklist
- All FR-001..FR-010 represented: yes (see mapping below)
- Tests precede parser implementation changes: yes (T013 before T014)
- Parallel tasks only touch distinct files: yes
- Asset tasks isolated: yes

### FR Mapping
- FR-001..FR-003 → T003–T010
- FR-004..FR-006 → T011 + T012
- FR-007 → T017/T018 (pending)
- FR-008 → T012
- FR-009 → No schema tasks (implicit) + parser non-invasive (T014/T015)
- FR-010 → Seeding in matrix (T011) validated in test (T012)

## Notes
- Thumbnails (T019/T023) are optional but increase discoverability.
- Future improvement: Add automated geometric validation (spawn/goal clearance) → potential T0XX later.
