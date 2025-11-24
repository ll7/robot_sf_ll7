# Implementation Plan: Automated Research Reporting for Imitation Learning

**Branch**: `270-imitation-report` | **Date**: 2025-11-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/270-imitation-report/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create an automated research reporting system for the imitation learning pipeline that transforms raw multi-seed training runs into publication-ready artifacts. The system orchestrates baseline and pre-trained policy experiments across multiple seeds, aggregates metrics (sample efficiency, success/collision rates), performs statistical analysis (paired t-tests, effect sizes, bootstrap CIs), generates standardized figures (PDF+PNG), and exports structured reports (Markdown + optional LaTeX) with complete reproducibility metadata. Hypothesis evaluation determines whether pre-training achieves ≥40% reduction in PPO timesteps. Supports ablation studies over BC epochs and dataset sizes, integrates with existing run tracker telemetry, and maintains robust degradation for partial failures.

## Technical Context

**Language/Version**: Python 3.11 (existing uv-managed environment)  
**Primary Dependencies**: 
- scipy (paired t-test, effect size computation)
- matplotlib (figure generation with LaTeX export capability)
- pandas (CSV export, data manipulation)
- pyyaml (config parsing)
- loguru (logging facade per Constitution XII)
- Existing: StableBaselines3, imitation, robot_sf modules, run tracker/telemetry

**Storage**: File-based (JSONL episode records, JSON manifests, YAML configs, NPZ trajectories, ZIP model checkpoints)  
**Testing**: pytest (integration tests for report generation, unit tests for metric aggregation)  
**Target Platform**: Linux/macOS (headless execution required for CI, PDF generation via matplotlib backend)  
**Project Type**: Single Python project extending existing robot_sf library  
**Performance Goals**: Report generation < 120 seconds for 3-seed run (per SC-001)  
**Constraints**: 
- Must integrate with existing benchmark schema (episode.schema.v1.json)
- Must respect canonical artifact root (output/)
- Must align with training_summary.schema.json extensions
- Must support headless execution (MPLBACKEND=Agg)

**Scale/Scope**: 
- Process 3-10 seeds per condition
- Handle ablation matrices up to ~20 variants
- Generate 5-10 core figures per report
- Support datasets 100-1000 episodes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducible Research Core
✅ **PASS** - Feature delivers deterministic report generation from versioned inputs (tracker manifests, episode JSONL, configs, seeds). Re-running with identical inputs reproduces figures and statistical outputs within documented tolerances.

### Principle III: Benchmark & Metrics First
✅ **PASS** - Consumes canonical JSONL episode records; extends benchmark output with aggregated metrics, CIs, and hypothesis evaluations. All outputs traceable to episode-level data.

### Principle IV: Unified Configuration
✅ **PASS** - Research report generation accepts config objects or file paths; parameters (seeds, threshold, ablation matrix) are explicit and versioned.

### Principle VI: Metrics Transparency
✅ **PASS** - Reports include mean, median, p95, bootstrap CIs with sample counts and confidence levels. Component metrics (success rate, collision rate, timesteps) reported alongside composite (sample efficiency improvement).

### Principle VII: Backward Compatibility
✅ **PASS** - New report artifacts do not alter existing episode schema or environment contracts. Adds new JSON schema for report metadata (extension of training_summary.schema.json) with versioned fields.

### Principle VIII: Documentation as API Surface
✅ **PASS** - Will add docs/research_reporting.md to central index; quickstart.md covers usage patterns; contracts/ directory documents JSON schemas.

### Principle IX: Test Coverage
✅ **PASS** - Implementation plan includes smoke tests (report generation from minimal manifests) and integration tests (full pipeline with multi-seed data).

### Principle X: Scope Discipline
✅ **PASS** - Feature remains focused on social navigation imitation learning reporting; no scope creep into unrelated ML domains.

### Principle XI: Library Reuse & Helper Documentation
✅ **PASS** - New helpers (metric aggregation, figure generation, template rendering) will live in `robot_sf/research/` with docstrings covering purpose, decision rules, side effects.

### Principle XII: Preferred Logging
✅ **PASS** - All library code will use Loguru; CLI scripts may print summaries but delegate core logic to logged library functions.

**Gate Status**: ✅ ALL GATES PASSED - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/270-imitation-report/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output - technical decisions
├── data-model.md        # Phase 1 output - entities and schemas
├── quickstart.md        # Phase 1 output - usage guide
├── contracts/           # Phase 1 output - JSON schemas
│   ├── report_metadata.schema.json
│   ├── aggregated_metrics.schema.json
│   └── hypothesis_result.schema.json
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
robot_sf/
├── research/               # NEW - research reporting module
│   ├── __init__.py
│   ├── aggregation.py      # Multi-seed metric aggregation with bootstrap CIs
│   ├── statistics.py       # Paired t-test, effect size, hypothesis evaluation
│   ├── figures.py          # Figure generation (learning curves, distributions, etc.)
│   ├── report_template.py  # Markdown/LaTeX report rendering
│   ├── metadata.py         # Reproducibility metadata collection
│   └── orchestrator.py     # End-to-end report generation coordinator
├── benchmark/
│   └── schemas/
│       └── report_metadata.schema.v1.json  # NEW schema
└── telemetry/              # EXISTING - run tracker integration point

scripts/
├── research/               # NEW - research workflow scripts
│   ├── generate_report.py  # CLI for report generation
│   └── compare_ablations.py # Ablation matrix analysis
└── tools/
    └── validate_report.py  # Report artifact validation

tests/
├── research/               # NEW - research module tests
│   ├── test_aggregation.py
│   ├── test_statistics.py
│   ├── test_figures.py
│   └── test_integration_report.py
└── fixtures/
    └── minimal_manifests/  # Test data for report generation

output/
└── research_reports/       # NEW - report artifact destination
    └── <timestamp>_<experiment_name>/
        ├── report.md
        ├── report.tex       # Optional LaTeX export
        ├── figures/         # PDF + PNG exports
        ├── data/            # JSON + CSV raw metrics
        ├── configs/         # Captured config files
        └── metadata.json    # Reproducibility manifest
```

**Structure Decision**: Extends existing single-project Python structure. New `robot_sf/research/` module contains reusable library components per Constitution XI. Scripts orchestrate library functions. Tests follow existing pytest patterns. Output directory adheres to canonical artifact root policy (Constitution principle on artifact management).

## Complexity Tracking

> **No Constitution violations - section not applicable**

All aspects align with Constitution principles. No complexity justification required.

---

## Phase 1: Design & Contracts (COMPLETED)

**Date**: 2025-11-21  
**Deliverables**:
- ✅ `data-model.md` - 8 entity definitions with validation rules
- ✅ `contracts/report_metadata.schema.json` - Metadata schema
- ✅ `contracts/aggregated_metrics.schema.json` - Metrics schema  
- ✅ `contracts/hypothesis_result.schema.json` - Hypothesis schema
- ✅ `quickstart.md` - Usage guide with CLI examples and workflows
- ✅ Agent context updated via `update-agent-context.sh`

**Constitution Re-Check**: All gates remain ✅ PASSED post-design.

**Next Phase**: Execute `/speckit.tasks` to generate task breakdown (tasks.md).

---

## Phase 2: Task Breakdown (COMPLETED)

**Date**: 2025-11-21  
**Deliverable**: ✅ `tasks.md` - Detailed implementation task list

**Summary**:
- **Total Tasks**: 86 organized into 7 phases
- **MVP Scope**: 32 tasks (Setup + Foundational + User Story 1)
- **Parallel Opportunities**: 28 tasks (33% parallelizable)
- **User Stories**: 4 prioritized stories (P1-P4) with independent test criteria
- **Task Distribution**:
  - Phase 1 (Setup): 7 tasks
  - Phase 2 (Foundational): 7 tasks
  - Phase 3 (US1 - P1): 17 tasks (MVP ✅)
  - Phase 4 (US2 - P2): 11 tasks
  - Phase 5 (US3 - P3): 16 tasks
  - Phase 6 (US4 - P4): 14 tasks
  - Phase 7 (Polish): 13 tasks

**Key Decisions**:
- All tasks follow strict checklist format: `- [ ] [ID] [P?] [Story?] Description with path`
- User story phases enable independent implementation and testing
- MVP delivers core value: automated report generation from pipeline runs
- Incremental delivery via 7 sprints with clear success metrics per sprint

**Next Steps**: Begin implementation following tasks.md execution order (Setup → Foundational → US1 MVP).
