# Implementation Plan: Accelerate PPO Training with Expert Trajectories

**Branch**: `001-ppo-imitation-pretrain` | **Date**: 2025-11-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-ppo-imitation-pretrain/spec.md`

**Note**: This plan follows the `/speckit.plan` workflow and spans research through high-level design. Downstream task breakdown happens in `/speckit.tasks`.

## Summary

Deliver a reproducible pipeline that (1) trains and validates a PPO expert policy, (2) records and vets expert trajectory datasets, and (3) leverages those datasets to pre-train and fine-tune new PPO agents for faster convergence. The solution extends existing Robot SF tooling, emphasising deterministic seeds, governed artefact storage, and benchmark-quality reporting so stakeholders can quantify sample-efficiency gains over the current PPO-from-scratch baseline.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (uv-managed virtual environment)  
**Primary Dependencies**: Stable-Baselines3, imitation (HumanCompatibleAI), Gymnasium, Loguru, NumPy, PyTorch  
**Storage**: Local filesystem artefact tree under `output/` (JSONL episodes, NPZ trajectory files, model checkpoints)  
**Testing**: pytest suite with headless validations; additional smoke scripts in `scripts/validation/`  
**Target Platform**: Linux/macOS headless execution (CI and local workstations)  
**Project Type**: Research platform with scripted pipelines (library-centric)  
**Performance Goals**: Pre-trained PPO must reach benchmark success in ≤70% of the timesteps required by baseline PPO and complete end-to-end runs within 18 hours on reference hardware  
**Constraints**: Curated trajectory datasets capped at 25 GB each with enforced archival rotation after three iterations to protect storage and CI throughput  
**Scale/Scope**: Research-scale experiments covering ≥3 seeds per scenario and ≥200 expert episodes per dataset

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducible Social Navigation Core)**: Plan must capture seeds, configs, and artefact locations for all workflows. Research Decision 1 & 4 supply reproducibility targets; design will formalise run manifests. *Status: On track*
- **Principle II (Factory-Based Environment Abstraction)**: All training, evaluation, and playback scripts must consume environments via `make_*` factories, never instantiating raw classes. *Status: In scope — to be restated in design*
- **Principle III & VI (Benchmark & Metrics, Statistical Rigor)**: Expert validation and comparative reports must emit canonical JSONL episodes and aggregated metrics with documented CIs. Research Decision 3 confirms metric suite and CI expectations. *Status: On track*
- **Principle IV (Unified Configuration & Seeds)**: Workflows must rely on unified config objects and propagate seeds. Research Decision 4 prescribes additive config fields and documented config files. *Status: On track*
- **Principle VIII (Documentation as API Surface)**: Any new scripts or config knobs require documentation updates and central index links. Quickstart outline prepared; implementation must add doc links. *Status: On track*
- **Principle IX (Test Coverage for Public Behavior)**: New public behavior (expert workflow, dataset validator, pre-training runner) necessitates smoke/integration tests. Coverage expectations will be captured in Phase 2 tasks. *Status: Pending detail*
- **Principle XII (Preferred Logging & Observability)**: New library helpers must use Loguru, avoiding raw prints. Enforce in implementation guidelines. *Status: On track*

Gate Verdict: **Pass with Follow-ups** — research resolved quantitative targets and config approach; documentation and test coverage plans remain to be detailed during task breakdown.

## Project Structure

### Documentation (this feature)

```text
specs/001-ppo-imitation-pretrain/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
robot_sf/
├── benchmark/
├── common/
├── gym_env/
├── training/
└── sim/

scripts/
├── training/
├── validation/
└── tools/

tests/
├── integration/
├── smoke/
└── fast_pysf/

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
