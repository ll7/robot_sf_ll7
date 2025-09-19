# Plan: Classical Interaction SVG Scenario Pack

Purpose: Provide implementation pathway and validation steps for classical human-robot interaction archetype maps and benchmark scenarios (feature 121).

## Scope
- Assets: 8 SVG archetype maps (crossing, head_on_corridor, overtaking, bottleneck, doorway, merging, t_intersection, group_crossing).
- Scenario Matrix: `configs/scenarios/classic_interactions.yaml` with density variants and group parameter for group crossing.
- Tests: YAML structure + map existence + CLI logging flag robustness (already integrated), potential thumbnails generation script (future).

## Goals (Mapped to FRs)
- FR-001..FR-003: Provide canonical archetype SVGs.
- FR-004..FR-006: Scenario variants (density, flow, groups) with deterministic seeds.
- FR-007: Validation test ensures structural integrity.
- FR-008: Documentation link in `docs/README.md`.
- FR-009: Parser resilience for global flags (implemented).
- FR-010: Extensibility hooks (thumbnails / future metrics) tracked via follow-up tasks.

## Assumptions
- No schema changes required; reuse existing scenario loader.
- Density values: {0.02, 0.05, 0.08} remain within safe simulation bounds.
- Group parameter interpreted by existing pedestrian grouping logic.

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Overcrowded high-density maps cause instability | Medium | Limit high density to suitable archetypes (exclude overtaking high) |
| CLI argument order regression | Low | Add intermixed parsing shim & test | 
| Future extension needs map thumbnails | Low | Provide follow-up task placeholder |

## Work Segments
1. Asset Creation (done)
2. Scenario Matrix Authoring (done)
3. Validation Test (done)
4. CLI Parser Adjustment (done)
5. Documentation Link (done)
6. Task Generation (current)
7. Optional Enhancements (thumbnails, README, advanced variants)

## Validation
- `pytest tests/test_classic_interactions_matrix.py` passes.
- Full test suite passes except pre-existing warnings (non-blocking).
- Manual visual inspection of SVG geometry (developer action).

## Out of Scope
- Advanced pedestrian behavior modeling changes.
- Automatic figure generation pipeline integration.

## Next
Generate `tasks.md` per repository process for traceability and potential future automation agents.
