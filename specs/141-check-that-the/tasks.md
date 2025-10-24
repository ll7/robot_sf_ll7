# Tasks: Verify Feature Extractor Training Flow

**Input**: Design documents from `/Users/lennart/git/robot_sf_ll7/specs/141-check-that-the/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Phase 3.1: Setup
- [X] T001 Create `docs/dev/issues/verify-feature-extractor-training-flow/` with `README.md` summarizing the spec linkage and `todo.md` seeded from plan milestones.
- [X] T002 Ensure repository ignores timestamped outputs by adding (or verifying) `tmp/multi_extractor_training/` entry in `.gitignore` with a short comment.

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
- [X] T003 [P] Add JSON contract test `tests/contract/test_multi_extractor_summary_json.py` validating `summary.json` against `contracts/training_summary.schema.json` using `jsonschema`.
- [X] T004 [P] Add Markdown contract test `tests/contract/test_multi_extractor_summary_markdown.py` asserting required sections from `contracts/summary_markdown.md` exist in generated output.
- [X] T005 [P] Add integration test `tests/integration/test_multi_extractor_single_thread.py` that runs `scripts/multi_extractor_training.py` in single-thread mode (with env guards) and asserts timestamped directories plus success statuses.
- [X] T006 [P] Add integration test `tests/integration/test_multi_extractor_vectorized.py` that exercises a GPU/vectorized config, skipping gracefully when CUDA is unavailable, and verifies summary artifacts exist.
- [X] T007 [P] Add unit test `tests/unit/test_hardware_profile_capture.py` covering the forthcoming hardware probe helper (platform, arch, GPU metadata, worker count).

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [X] T008 Implement dataclasses for `ExtractorConfigurationProfile`, `ExtractorRunRecord`, `HardwareProfile`, and `TrainingRunSummary` in `robot_sf/training/multi_extractor_models.py` with docstrings per Principle XI.
- [X] T009 Create timestamped path helper `robot_sf/training/multi_extractor_paths.py` that builds `./tmp/multi_extractor_training/<timestamp>-<run-id>/` and per-extractor subfolders.
- [X] T010 Implement hardware probe utility `robot_sf/training/hardware_probe.py` capturing platform, architecture, GPU info, CUDA version, Python runtime, and worker count via Loguru.
- [X] T011 Implement summary writer module `robot_sf/training/multi_extractor_summary.py` that emits `summary.json` (matching schema) and `summary.md` (matching Markdown contract).
- [X] T012 Refactor `scripts/multi_extractor_training.py` to default to single-thread spawn on macOS, leverage `robot_sf/gym_env/environment_factory`, and record run metadata using new helpers.
- [X] T013 Extend the script to populate per-extractor results, collect metrics/artifacts, and invoke the summary writer for JSON and Markdown outputs.
- [X] T014 Add/adjust scenario configs (`configs/scenarios/multi_extractor_default.yaml`, `configs/scenarios/multi_extractor_gpu.yaml`) to expose worker-mode toggles and device options referenced in quickstart tests.
- [X] T015 Ensure logging and error handling in the script use Loguru warnings/errors for skips/failures, aligning with Principle XII.

## Phase 3.4: Integration
- [X] T016 Wire summary generation into any existing automation (e.g., `examples/demo_aggregate.py` if needed) and ensure resume logic respects existing JSONL manifests.
- [X] T017 Create reusable helper re-exports or registration hooks (if necessary) so other scripts can import the timestamp/path/summary utilities without duplication.

## Phase 3.5: Polish
- [X] T018 [P] Update documentation (`docs/README.md`, `docs/dev/issues/verify-feature-extractor-training-flow/README.md`, and quickstart snippets) with new workflow references and summary artifacts.
- [X] T019 [P] Add changelog entry to `CHANGELOG.md` summarizing script improvements and new outputs.
- [X] T020 [P] Update `docs/dev/issues/verify-feature-extractor-training-flow/todo.md` with completed items from this tasks list as progress is made.

## Phase 3.6: Validation & Quality Gates
- [X] T021 Run contract and unit suites: `uv run pytest tests/contract/test_multi_extractor_summary_json.py tests/contract/test_multi_extractor_summary_markdown.py tests/unit/test_hardware_profile_capture.py` and confirm green.
- [X] T022 Execute integration tests: `uv run pytest tests/integration/test_multi_extractor_single_thread.py tests/integration/test_multi_extractor_vectorized.py` (with appropriate env flags) and ensure skips/failures match expectations.
- [X] T023 Perform manual quickstart validation by following `quickstart.md` (both single-thread and GPU configs) and archive resulting summaries for review.
- [ ] T024 Run full quality gates (`uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest tests`) before requesting review.

## Dependencies
- T001 → T018 (docs folder must exist before updates)
- T003-T007 must be completed before T008-T015 (tests written before implementation)
- T008 blocks T012 and T013 (script refactor relies on dataclasses)
- T009 blocks T012 (script needs path helper)
- T010 blocks T012 and T013 (hardware data used in summaries)
- T011 blocks T013 (script needs writer)
- T012 blocks T016-T023 (implementation must exist before integration and validation)
- T014 blocks T006 and T022 (GPU config required for tests)
- T013 blocks T021 (summary files required for contract tests to pass)

## Parallel Execution Example
Run the initial contract/unit test authoring tasks together:
```
Task.run("T003")
Task.run("T004")
Task.run("T005")
Task.run("T006")
Task.run("T007")
```

## Notes
- Tasks marked [P] modify independent files and can proceed in parallel once dependencies clear.
- Maintain TDD discipline: execute tasks in Phase 3.2 before implementing Phase 3.3.
- Each task should conclude with commits documenting changes and test results.
