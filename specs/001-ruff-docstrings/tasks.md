---

description: "Task list for enforcing Ruff docstring rules"
---

# Tasks: Ruff Docstring Enforcement

**Input**: Design documents from `/specs/001-ruff-docstrings/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Run lint (`uv run ruff check`) and regression suite (`uv run pytest tests`) after each phase to keep repo green.

**Organization**: Tasks are grouped by user story to enable independent implementation and verification.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Task can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3)
- Every task lists an exact file or directory path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish baseline visibility into docstring gaps before changing configuration.

- [ ] T001 Capture baseline Ruff docstring violations by running `uv run ruff check --select D,D417,D419,D102,D201` and saving the log to `output/issues/docstrings_baseline.txt`.
- [ ] T002 [P] Append a repository-wide Python path inventory (robot_sf/, fast-pysf/, scripts/, tests/, examples/) to `specs/001-ruff-docstrings/research.md` for future traceability.
- [ ] T003 [P] Create `output/issues/docstrings_todo.jsonl` and seed it with the modules that currently lack docstrings so remediation can be tracked.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Configure the core lint infrastructure that every user story relies on.

- [ ] T004 Update `pyproject.toml` `[tool.ruff]` settings to enable rules D100-D107, D417, D419, D102, and D201 across the entire repository.
- [ ] T005 Define `per-file-ignores` in `pyproject.toml` for generated or third-party directories (e.g., `fast-pysf/pysocialforce/_version.py`) so intentional exclusions are tracked.
- [ ] T006 Document the DocstringRuleSet include/exclude patterns inside `specs/001-ruff-docstrings/data-model.md` so future contributors know which paths are governed.

**Checkpoint**: Ruff configuration is authoritative; user stories can now leverage it.

---

## Phase 3: User Story 1 – Maintainer enforces docstrings at commit time (Priority: P1) 🎯 MVP

**Goal**: Ensure CI and local workflows fail immediately when a public API lacks compliant docstrings.

**Independent Test**: Remove a docstring from `robot_sf/gym_env/environment_factory.py`, run `uv run ruff check`, and confirm CI/local lint blocks the change with a D1xx/D4xx rule reference.

### Implementation

- [ ] T007 [US1] Update `.github/workflows/ci.yml` so the lint job runs `uv run ruff check` with the new docstring rule set and fails the workflow on violations.
- [ ] T008 [US1] Add a small guard script `scripts/tools/validate_docstring_rules.py` that shells out to Ruff and exits non-zero when any docstring rule fails (used by CI and local scripts).
- [ ] T009 [US1] Add "Docstring enforcement" guidance to `docs/dev_guide.md`, covering required sections (summary, Args, Returns, Raises) and pointing to the guard script.
- [ ] T010 [US1] Update `README.md` to mention that PRs must satisfy Ruff docstring checks before review.
- [ ] T011 [US1] Create `tests/tools/test_docstring_rule_config.py` that loads `pyproject.toml` and asserts the configured rule list matches the required set.
- [ ] T012 [US1] Wire the new guard script into `.github/workflows/ci.yml` and `.specify/scripts/bash/check-prerequisites.sh` (lint section) so maintainers get consistent enforcement in both automation and manual checks.

**Checkpoint**: Maintainers cannot merge code lacking compliant docstrings.

---

## Phase 4: User Story 2 – Contributor fixes legacy files efficiently (Priority: P2)

**Goal**: Provide actionable tooling so contributors can remediate docstring gaps per file without manual triage.

**Independent Test**: Run the new docstring report command against `robot_sf/benchmark/` and confirm the output groups violations by file with rule identifiers and suggested fixes.

### Implementation

- [ ] T013 [US2] Implement `scripts/tools/docstring_report.py` that runs `uv run ruff check --output-format json`, groups results by file, and writes a sortable summary to `output/issues/docstrings_summary.json`.
- [ ] T014 [P] [US2] Extend `specs/001-ruff-docstrings/quickstart.md` with a walkthrough for running `scripts/tools/docstring_report.py` and interpreting grouped output.
- [ ] T015 [US2] Add a "Docstring Report" VS Code task (and optional Make target) inside `.vscode/tasks.json` so contributors can launch the grouping script with one command.
- [ ] T016 [US2] Publish a sample remediation checklist (top offenders, rule hints) to `specs/001-ruff-docstrings/research.md` using data from `output/issues/docstrings_summary.json`.
- [ ] T017 [US2] Document the contributor workflow in `docs/dev_guide.md`, focusing on how to use grouped reports to clean modules without touching unrelated files.

**Checkpoint**: Contributors have low-friction tooling to understand and fix docstring issues.

---

## Phase 5: User Story 3 – Documentation consumers trust generated references (Priority: P3)

**Goal**: Upgrade docstrings on key public APIs so generated documentation is accurate and complete.

**Independent Test**: Regenerate API docs (via `pdoc robot_sf`) and verify exported modules include summaries, Args, Returns, and Raises sections without placeholders.

### Implementation

- [ ] T018 [US3] Add full docstrings for factory helpers in `robot_sf/gym_env/environment_factory.py`, covering parameters, return environments, and side effects.
- [ ] T019 [P] [US3] Document aggregation utilities in `robot_sf/benchmark/aggregate.py`, emphasizing metric grouping and bootstrap semantics.
- [ ] T020 [P] [US3] Expand docstrings in `robot_sf/research/extractor_report.py` so statistical helpers describe inputs (dataframes/paths) and outputs (figures/JSON).
- [ ] T021 [P] [US3] Annotate `scripts/training/train_ppo_with_pretrained_policy.py` functions (config loading, training loop, artifact writes) with docstrings suitable for CLI documentation.
- [ ] T022 [P] [US3] Document command helpers in `scripts/tools/compare_training_runs.py`, including `_resolve_manifest_path` and report builders.
- [ ] T023 [P] [US3] Update `examples/advanced/16_imitation_learning_pipeline.py` with docstrings summarizing each pipeline stage so generated docs explain the workflow.
- [ ] T024 [US3] Add a lightweight doc-generation script `scripts/tools/build_api_docs.py` that runs `pdoc` and writes HTML/PDF output to `output/docs/api/` for smoke verification.

**Checkpoint**: Generated documentation reflects the enforced docstring quality.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup, documentation, and validation after all user stories land.

- [ ] T025 Update `CHANGELOG.md` with a summary of repo-wide docstring enforcement and tooling improvements.
- [ ] T026 [P] Re-run `uv run ruff check` and `uv run pytest tests` (repo root) and attach the passing logs to `output/issues/docstrings_final.log`.
- [ ] T027 [P] Remove temporary baseline artifacts (`output/issues/docstrings_baseline.txt`, `output/issues/docstrings_todo.jsonl`) once the final report is archived.
- [ ] T028 Publish a short how-to section in `docs/README.md` pointing readers to the new quickstart and tooling for docstrings.

---

## Dependencies & Execution Order

### Phase Dependencies

1. **Phase 1 → Phase 2**: Baseline data (T001–T003) is needed before editing Ruff configuration.
2. **Phase 2 → User Stories**: Docstring rules (T004–T006) must exist before US1–US3 start.
3. **User Stories**: US1 (P1) delivers the MVP gate; US2 and US3 can proceed in parallel once US1’s CI guard (T007–T012) is merged.
4. **Polish** runs only after desired user stories finish.

### User Story Dependencies

- **US1**: Depends on Phases 1–2 only.
- **US2**: Depends on US1’s guard script so contributors trust lint output.
- **US3**: Depends on US1 (for enforcement) but not on US2; it can run in parallel with US2 once US1 lands.

### Parallel Opportunities

- Setup tasks T002 and T003 can run alongside log capture.
- Foundational tasks T004–T006 edit different files and can be parallelized carefully.
- In US3, tasks T019–T023 touch distinct modules and can proceed simultaneously.

---

## Parallel Example: User Story 3

```bash
# Developer A (factories)
Edit robot_sf/gym_env/environment_factory.py (T018)

# Developer B (benchmark aggregation)
Edit robot_sf/benchmark/aggregate.py (T019)

# Developer C (training scripts)
Edit scripts/training/train_ppo_with_pretrained_policy.py (T021)
```

After each contribution, run `scripts/tools/build_api_docs.py` (T024) to confirm generated docs include the new summaries.

---

## Implementation Strategy

### MVP First (User Story 1)
1. Complete Phase 1 (baseline) and Phase 2 (Ruff configuration).
2. Deliver US1 (T007–T012) so CI blocks missing docstrings.
3. Validate by intentionally removing a docstring and confirming lint failure.

### Incremental Delivery
1. With US1 merged, pursue US2 (report tooling) and US3 (docstring upgrades) in parallel.
2. Each completed story should pass its independent test before moving on.

### Parallel Team Strategy
- Assign one developer to maintainers’ CI work (US1), one to contributor tooling (US2), and another to docstring remediation/doc generation (US3).
- Use the grouped report (`output/issues/docstrings_summary.json`) to avoid touching the same files simultaneously.

---

## Summary Metrics

- **Total Tasks**: 28
- **Per Story**: US1 (6), US2 (5), US3 (7)
- **Parallel Opportunities**: Highlighted in Setup, Foundational, and US3 phases.
- **Independent Tests**:
  - **US1**: `uv run ruff check` fails when docstrings missing.
  - **US2**: `scripts/tools/docstring_report.py` outputs grouped violations per file.
  - **US3**: `scripts/tools/build_api_docs.py` produces docs with complete sections.
- **MVP Scope**: Complete through Phase 3 (US1) to guarantee enforcement before broader remediation.

**tasks.md generated for feature `/specs/001-ruff-docstrings/`.**
````}