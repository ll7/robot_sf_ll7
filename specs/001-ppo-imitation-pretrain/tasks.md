# Tasks: Accelerate PPO Training with Expert Trajectories

**Input**: Design documents from `/specs/001-ppo-imitation-pretrain/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include smoke/integration checks aligned with Constitution Principle IX.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare dependencies and configuration scaffolding used by all stories.

- [x] T001 Update Python dependency block with `imitation` in pyproject.toml
- [x] T002 Record `imitation` version in installed_packages.txt
- [x] T003 [P] Create pipeline overview in configs/training/ppo_imitation/README.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared manifests, configuration dataclasses, and artifact paths required for every story.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Create artifact dataclasses for expert policies, trajectory datasets, and training runs in robot_sf/common/artifacts.py
- [x] T005 Expose new artifact dataclasses via robot_sf/common/__init__.py
- [x] T006 Add helper functions for expert manifests and trajectory storage in robot_sf/common/artifact_paths.py
- [x] T007 Introduce imitation training config dataclasses in robot_sf/training/imitation_config.py
- [x] T008 Add manifest writer helpers for imitation workflows in robot_sf/benchmark/imitation_manifest.py

**Checkpoint**: Foundation ready – manifest + config infrastructure available for all stories.

---

## Phase 3: User Story 1 - Establish Expert Policy Benchmark (Priority: P1) 🎯 MVP

**Goal**: Deliver a reproducible expert PPO training workflow with evaluation reports and manifests.

**Independent Test**: Run `uv run python scripts/training/train_expert_ppo.py --config configs/training/ppo_imitation/expert_ppo.yaml` and verify that a manifest plus evaluation metrics meeting convergence thresholds is written to `output/`.

### Implementation for User Story 1

- [x] T009 [US1] Add integration smoke test for expert training workflow in tests/integration/test_train_expert_ppo.py
- [x] T010 [US1] Define expert PPO configuration defaults in configs/training/ppo_imitation/expert_ppo.yaml
- [x] T011 [US1] Implement expert PPO training script with evaluation callbacks in scripts/training/train_expert_ppo.py
- [x] T012 [US1] Emit expert policy manifest and metrics using imitation helpers in robot_sf/benchmark/imitation_manifest.py
- [x] T013 [US1] Register expert training entry point in scripts/training/__init__.py for discoverability

**Checkpoint**: Expert policy benchmark pipeline functional with reproducible artefacts and smoke test coverage.

---

## Phase 4: User Story 2 - Curate Expert Trajectory Library (Priority: P2)

**Goal**: Capture, validate, and catalogue expert trajectories with playback support.

**Independent Test**: Run `uv run python scripts/training/collect_expert_trajectories.py --policy-id <approved_id> --episodes 200`, then `uv run python scripts/validation/validate_trajectory_dataset.py --dataset <dataset_path>`, and confirm manifests show `qualityStatus=validated`.

### Implementation for User Story 2

- [x] T014 [US2] Add trajectory dataset validation test in tests/integration/test_expert_trajectory_dataset.py
- [ ] T015 [US2] Implement trajectory recorder with metadata capture in scripts/training/collect_expert_trajectories.py
- [x] T016 [US2] Build trajectory dataset validator utilities in robot_sf/benchmark/validation/trajectory_dataset.py
- [ ] T017 [US2] Create playback and inspection tool in scripts/validation/playback_trajectory.py
- [ ] T018 [US2] Persist scenario coverage and integrity results in robot_sf/benchmark/imitation_manifest.py
**Checkpoint**: Validated trajectory datasets with playback tooling and automated integrity checks.

---

## Phase 5: User Story 3 - Accelerate PPO via Pre-Training (Priority: P3)

**Goal**: Warm-start PPO agents from expert trajectories and compare against baseline training.

**Independent Test**: Execute `uv run python scripts/training/pretrain_from_expert.py --dataset <dataset_id> --config configs/training/ppo_imitation/bc_pretrain.yaml` followed by `uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_imitation/ppo_finetune.yaml`, then generate a comparison report via `uv run python scripts/tools/compare_training_runs.py --group <run_group_id>` showing ≤70% timestep budget vs. baseline.

### Implementation for User Story 3

- [ ] T019 [US3] Add pretraining pipeline integration test in tests/integration/test_ppo_pretraining_pipeline.py
- [ ] T020 [US3] Implement behavioural cloning pretraining routine in scripts/training/pretrain_from_expert.py
- [ ] T021 [US3] Implement PPO fine-tuning runner consuming pre-trained weights in scripts/training/train_ppo_with_pretrained_policy.py
- [ ] T022 [US3] Generate comparative metrics CLI in scripts/tools/compare_training_runs.py
- [ ] T023 [US3] Compute sample-efficiency deltas and bootstrap summaries in robot_sf/benchmark/summary.py

**Checkpoint**: Pre-trained PPO workflow delivers measurable sample-efficiency gains with comparative reporting.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, changelog, and quality improvements spanning all stories.

- [ ] T024 Document imitation pipeline usage in docs/dev_guide.md
- [ ] T025 Link new quickstart and scripts from docs/README.md
- [ ] T026 Record feature summary in CHANGELOG.md
- [ ] T027 Align quickstart instructions with final scripts in specs/001-ppo-imitation-pretrain/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)** → prerequisite for Foundational tasks
- **Foundational (Phase 2)** → blocks all user stories until manifest/config infrastructure exists
- **User Story Phases (3–5)** → each can start after Phase 2; prefer priority order (P1 → P2 → P3) but can proceed in parallel with coordination
- **Polish (Phase 6)** → begins once targeted user stories reach checkpoints

### User Story Dependencies

1. **US1 (P1)**: Depends on Phases 1–2 only
2. **US2 (P2)**: Depends on Phases 1–2 and completion of US1 manifest helpers (T012)
3. **US3 (P3)**: Depends on Phases 1–2, plus availability of validated datasets from US2 (T015–T018)

### Within Each User Story

- Create or update tests before implementing core logic
- Configure manifests and metadata before emitting artefacts
- Scripts rely on config/dataclass infrastructure delivered in Phase 2

## Parallel Opportunities

- Phase 1 tasks T001–T003 can run concurrently after coordinating dependency file edits
- Within Phase 2, T004–T007 touch distinct modules and can be split among contributors once file ownership is agreed; T008 should wait for T004–T007
- After Phase 2, US1 and US2 can progress in parallel with shared awareness of `robot_sf/benchmark/imitation_manifest.py`
- Documentation tasks in Phase 6 (T024–T027) can run in parallel once respective user stories are finished

## Parallel Example: User Story 2

```bash
# Validation-first workflow
- Implement tests/integration/test_expert_trajectory_dataset.py (T014)
- Build validator utilities in robot_sf/benchmark/validation/trajectory_dataset.py (T016)

# Recorder vs. playback split
- Develop scripts/training/collect_expert_trajectories.py (T015)
- Build scripts/validation/playback_trajectory.py (T017)
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational)
2. Deliver Phase 3 tasks (US1) to produce an approved expert policy benchmark
3. Validate smoke test T009 and ensure manifest outputs satisfy SC-001

### Incremental Delivery

1. Extend to US2 once expert benchmark is stable; validate dataset tooling before moving to pre-training
2. Add US3 to realise sample-efficiency gains and publish comparison reports
3. Close with Phase 6 documentation and changelog updates

### Parallel Team Strategy

- Developer A: Focus on manifests and expert training (US1)
- Developer B: Own trajectory capture and validation (US2)
- Developer C: Implement pre-training and comparison tooling (US3)
- Coordinate on shared modules (`robot_sf/benchmark/imitation_manifest.py`, configs) to avoid conflicts
