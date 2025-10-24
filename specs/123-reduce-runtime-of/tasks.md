# Tasks: Accelerate Reproducibility Integration Test

**Input**: Design documents from `/specs/123-reduce-runtime-of/`
**Prerequisites**: `plan.md` (required), `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

## Conventions
- Modify only `tests/benchmark_full/test_integration_reproducibility.py` unless explicitly noted.
- Keep workload minimal: 1 scenario, 2 seeds, workers=1, smoke flags.
- No new external dependencies.

## Phase 3.1: Setup & Context
- [X] T001 Confirm presence of spec, plan, research, data-model, contract, quickstart docs.
- [ ] T002 Add placeholder measurement comment at top of `tests/benchmark_full/test_integration_reproducibility.py` (no logic change yet).

## Phase 3.2: Tests First (Refactor to TDD Enhancements)
### Helper & Structural Assertions (same file sequential)
- [ ] T010 Introduce `_run_minimal_benchmark(output_dir_base, seeds)` helper returning EpisodeSequence (episode_ids list + scenario_matrix_hash) in `tests/benchmark_full/test_integration_reproducibility.py`.
- [ ] T011 Add test assertion for identical `episode_ids` across two helper invocations (existing logic replaced / minimized).
- [ ] T012 Add assertion for identical `scenario_matrix_hash` (parse manifest JSON from each run).
- [ ] T013 Add assertion that no duplicate `episode_ids` appear in either run.

### Performance Guard & Environment Constraints
- [ ] T014 Add timing wrapper capturing total two-run duration; soft assert <2s local (warn or skip fail unless `STRICT_REPRO_TEST=1`).
- [ ] T015 Force configuration: seeds length=2, workers=1, disable videos/plots/bootstrap, smoke mode; document rationale.
- [ ] T016 Add explicit skip or note for multi-worker determinism (future TODO) with comment referencing spec path.

## Phase 3.3: Core Refactor Implementation
- [ ] T020 Replace any prior large scenario matrix or multiple seeds with minimized config (1 scenario, 2 seeds) in test body.
- [ ] T021 Ensure bootstrap disabled (set samples=0 if parameter or adapt config flags) in orchestrator call.
- [ ] T022 Ensure plots/videos not generated (set flags or assert skip statuses if accessible).
- [ ] T023 Ensure minimal horizon if configurable without breaking episode generation (add comment if not adjustable here).

## Phase 3.4: Integration & Stability
- [ ] T030 Add logging/debug prints replaced with pytest `-s` friendly minimal message only on failure (avoid noise).
- [ ] T031 Validate deterministic ordering unaffected by temp directory differences (use distinct dirs; compare lists).
- [ ] T032 Add negative-control helper (not executed by default) illustrating mismatch scenario (commented or skipped) for future expansion.

## Phase 3.5: Polish & Quality Gates
- [ ] T040 Insert header comment summarizing optimization decisions with links to spec & research docs.
- [ ] T041 Run Ruff + type check + pytest; adjust test naming if necessary for speed markers (`@pytest.mark.fast`).
- [ ] T042 Add soft timing exceedance warning message with actionable guidance.
- [ ] T043 Update documentation if needed (e.g., mention accelerated test in `docs/benchmark_full_classic.md` performance notes section). *(Optional)*
- [ ] T044 Final pass: remove any obsolete code paths in the test (legacy multi-seed loops) while keeping historical rationale in comments.

## Phase 3.6: Finalization
- [ ] T050 Record before/after timing result in PR description (manual step).
- [ ] T051 Mark all tasks [X] and ensure spec & plan remain accurate (update if deviations occurred).
- [ ] T052 Prepare PR summary referencing `specs/123-reduce-runtime-of/` artifacts & runtime delta.

## Dependencies & Ordering Summary
- T010→T011→T012→T013 sequential (same helper context).
- T014–T016 depend on helper existing.
- T020–T023 depend on earlier assertions to ensure logic stable before config minimization.
- Polish tasks (T040+) only after functional & performance guards integrated.

## Parallel Execution Examples
Minimal parallelism due to single-file focus; only documentation updates (T043) could run in parallel with T042 but kept sequential for simplicity.

## Validation Checklist
- [ ] Reproducibility invariants (IDs, hash) asserted.
- [ ] No duplicates asserted.
- [ ] Runtime guard implemented.
- [ ] Heavy artifacts disabled.
- [ ] Clear rationale comments present.

## Notes
- Do not import private internals beyond existing public orchestrator utilities.
- If runtime still >2s after changes, document reasons (env variability) rather than over-complicating test.
