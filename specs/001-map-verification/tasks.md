# Tasks — Map Verification Workflow

## Dependency Overview
- US1 (Single-command audit) forms the MVP and must land before downstream tasks.
- US2 (CI gate) depends on US1 tooling but not on US3.
- US3 (Machine-readable manifest) depends on US1 data structures and can run in parallel with US2 once runner outputs rule results.
- Polish phase executes after all user stories to align docs and smoke guidance.

## Implementation Strategy
Deliver an MVP focused on US1: ship the verification module, CLI entry point, and baseline tests. Once the command works locally, layer in CI automation (US2) and structured manifest outputs (US3). Keep each increment independently runnable (`uv run python scripts/validation/verify_maps.py ...`). Use thin vertical slices: add rule engines and logging before expanding to CI/doc polish.

## Parallel Execution Examples
- **US1**: While T006 builds the rule set, T008 can implement CLI flag parsing because both rely only on foundational modules.
- **US2**: T011 (CI workflow hook) can proceed in parallel with T012 (VS Code task integration) once the CLI contract is stable.
- **US3**: T014 (manifest schema writer) and T015 (CLI `--output` plumbing) can run concurrently after VerificationResult serialization exists.

---

## Phase 1 – Setup
Goal: establish project scaffolding for the verifier.

- [ ] T001 Create `robot_sf/maps/verification/__init__.py` with module docstring and export stubs.
- [ ] T002 Add CLI stub `scripts/validation/verify_maps.py` that loads Loguru and parses `argparse` placeholders.
- [ ] T003 Create test module skeleton `tests/maps/test_map_verifier.py` with pytest fixture placeholders.

## Phase 2 – Foundational Infrastructure
Goal: shared utilities required by every user story.

- [ ] T004 Implement map inventory loader in `robot_sf/maps/verification/map_inventory.py` to enumerate SVGs respecting `ci_enabled` flags.
- [ ] T005 [P] Add scope filter helpers (`all`, `ci`, `changed`) in `robot_sf/maps/verification/scope_resolver.py` consuming git status when needed.
- [ ] T006 Create verification context dataclass + artifact routing helpers in `robot_sf/maps/verification/context.py` leveraging `robot_sf.common.artifact_paths`.

## Phase 3 – User Story 1 (Single-command map audit)
Goal: maintainer can run one command to audit all maps.

Independent Test Criteria: `uv run python scripts/validation/verify_maps.py --scope all --mode local` completes with exit code 0 when repo maps are healthy and prints actionable errors when not.

- [ ] T007 [US1] Implement rule definitions in `robot_sf/maps/verification/rules.py` (geometry checks, metadata presence, spawn coverage).
- [ ] T008 [P] [US1] Build environment instantiation runner in `robot_sf/maps/verification/runner.py` that selects robot vs pedestrian factories per tags and captures timing.
- [ ] T009 [US1] Wire CLI options (`--scope`, `--mode`, `--fix`, `--seed`) in `scripts/validation/verify_maps.py` and call into runner.
- [ ] T010 [US1] Add Loguru-based diagnostics + perf warnings to `robot_sf/maps/verification/logging.py` and integrate with CLI output.
- [ ] T011 [US1] Add unit/integration tests in `tests/maps/test_map_verifier.py` covering pass/fail cases, fix mode, and timing warnings using synthetic SVG fixtures.

## Phase 4 – User Story 2 (CI gate for map regressions)
Goal: CI jobs fail with actionable logs when verification fails.

Independent Test Criteria: Running the CI task (or `uv run python scripts/validation/verify_maps.py --scope ci --mode ci`) on a branch with a broken SVG exits non-zero and prints remediation hints; green branch stays under perf budget.

- [ ] T012 [US2] Add `--mode ci` behaviors in `scripts/validation/verify_maps.py` to enforce perf/timeouts and zero-exit only on full pass.
- [ ] T013 [US2] Integrate verifier into `.github/workflows/ci.yml` (or dedicated script) ensuring artifacts upload and respecting `ROBOT_SF_PERF_*` env vars.
- [ ] T014 [P] [US2] Add VS Code task entry (`.vscode/tasks.json`) mirroring the CI command for local smoke usage.
- [ ] T015 [US2] Extend tests in `tests/maps/test_map_verifier.py` to simulate CI mode (mock perf thresholds, assert exit codes/log text).

## Phase 5 – User Story 3 (Machine-readable verification manifest)
Goal: write structured JSON manifest for tooling/dashboards.

Independent Test Criteria: After running verifier with `--output output/validation/map_verification.json`, the file contains one record per map with status, rule IDs, duration, and can be parsed by downstream scripts to list failing maps.

- [ ] T016 [US3] Implement manifest writer in `robot_sf/maps/verification/manifest.py` serializing `VerificationRunSummary` per data-model spec.
- [ ] T017 [P] [US3] Add CLI plumbing for `--output` path overrides and ensure artifact directories created via `robot_sf.common.artifact_paths`.
- [ ] T018 [US3] Document manifest schema and usage in `docs/SVG_MAP_EDITOR.md` + `docs/README.md` entries referencing quickstart.
- [ ] T019 [US3] Add regression test in `tests/maps/test_map_verifier.py` asserting manifest contents and schema compliance.

## Phase 6 – Polish & Cross-Cutting
Goal: finalize docs, examples, and quickstart alignment.

- [ ] T020 Update `quickstart.md` instructions plus `examples` README snippets referencing the new verifier command.
- [ ] T021 Add smoke command to `docs/SVG_MAP_EDITOR.md` and ensure CHANGELOG entry under Unreleased.
- [ ] T022 Capture sample manifest artifact under `output/validation/` for docs/tests and add clean-up to `.gitignore` if needed.

## Story Dependency Graph
```
US1 -> US2
US1 -> US3
US2, US3 -> Polish
```

## Task Counts
- Total tasks: 22
- US1 tasks: 5
- US2 tasks: 4
- US3 tasks: 4

## Independent Test Criteria (per Story)
- **US1**: Local `--scope all --mode local` command passes with clean repo and shows targeted errors otherwise.
- **US2**: CI/`--mode ci` run fails on broken maps and respects perf/time thresholds.
- **US3**: `--output` manifest captures per-map status and downstream scripts can filter `status == "fail"` reliably.

## MVP Recommendation
Ship Phase 1–3 (through US1) as the MVP to unlock maintainer workflows before tackling CI enforcement and structured manifests.
