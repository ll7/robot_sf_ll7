# Feature Specification: Preserve Algorithm Separation in Benchmark Aggregation

**Feature Branch**: `142-aggregation-mixes-algorithms`  
**Created**: 2025-10-06  
**Status**: Draft  
**Input**: User description: "Aggregation mixes algorithms together. The benchmark aggregates with group_by=\"scenario_params.algo\" (scripts/run_social_navigation_benchmark.py:146), but run_full_benchmark only records the algorithm at the top level (\"algo\") and never injects it into scenario_params (robot_sf/benchmark/full_classic/orchestrator.py:200-214). Because scenario_params.algo is missing, compute_aggregates_with_ci falls back to scenario_id (robot_sf/benchmark/aggregate.py:143-162), so episodes from SF, PPO, and Random in the same scenario get merged into one bucket. That defeats the whole purpose of comparing baselines—the reported stats are per-scenario averages across all algorithms instead of per-algorithm metrics. You’ll need to either change the group key to \"algo\" or ensure the episodes carry scenario_params[\"algo\"] before writing JSONL."

---

## Clarifications

### Session 2025-10-06
- Q: How should we resolve the mismatch between top-level `algo` and missing `scenario_params["algo"]` when preparing episode records and running aggregation? → A: Capture both: inject `scenario_params["algo"]` and treat top-level `algo` as the authoritative fallback during aggregation.
- Q: What should happen if some algorithms never produce episode records in a benchmark run? → A: Continue aggregation but emit a prominent warning and flag missing algorithms in the summary.

## User Scenarios & Testing

### Primary User Story
A benchmarking analyst needs to compare Social Force, PPO, and Random baselines side by side. They export benchmark results and expect every aggregate table and chart to reflect metrics per algorithm without cross-contamination.

### Acceptance Scenarios
1. **Given** the analyst runs the full classic benchmark, **When** the aggregation job completes, **Then** each algorithm’s metrics appear in distinct groups with no mixing between baselines.
2. **Given** an episode JSONL file that lacks the per-algorithm grouping field, **When** aggregation is triggered, **Then** the system halts with a clear validation error instead of silently merging results.

### Edge Cases
- What happens when only a subset of algorithms finish successfully? Aggregation should continue, emit a prominent warning, and separate available algorithms while explicitly flagging the missing ones in summary output.
- How does the system handle legacy JSONL files recorded before this change? Legacy files MUST trigger the fail-fast validation error to instruct the analyst to regenerate results; automatic upgrades are out of scope for this change.

---
### Functional Requirements
- **FR-001**: Benchmark episode records MUST include a stable per-algorithm identifier in both the top-level `algo` field and mirrored under `scenario_params["algo"]` so legacy grouping paths remain valid.
- **FR-002**: Aggregation routines MUST default to grouping by the per-algorithm identifier so that metrics are reported per baseline.
- **FR-003**: Reports and summaries MUST highlight algorithm names for every metric table to reinforce separation.
- **FR-004**: The system MUST fail fast with an actionable error when required algorithm metadata is missing during aggregation.
- **FR-005**: Documentation for the benchmark workflow MUST explain how algorithm grouping is enforced and how to troubleshoot missing identifiers.
- **FR-006**: Aggregation MUST proceed when expected algorithms are absent but MUST emit a prominent warning and annotate summaries to list missing algorithms.

### Key Entities
- **Episode Record**: Represents a single benchmark run output; must carry both scenario parameters and an explicit algorithm grouping tag.
- **Aggregation Report**: Summarizes metrics per algorithm and scenario, making mismatched or missing group keys visible to the analyst.

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

