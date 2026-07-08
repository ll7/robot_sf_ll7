# Research Log – Preserve Algorithm Separation in Benchmark Aggregation

**Spec**: [/specs/142-aggregation-mixes-algorithms/spec.md](spec.md)  
**Date**: 2025-10-06  
**Participants**: GitHub Copilot automated planning agent

## Context Recap
- Benchmark runner currently groups episodes by `scenario_params.algo`, but `run_full_benchmark` only records the algorithm name at the top level (`record["algo"]`).
- Aggregation utilities (`compute_aggregates_with_ci`) already support dotted paths with a fallback to `scenario_id`; missing algorithm metadata silently merges records.
- Clarifications require (a) mirroring algorithm metadata under `scenario_params` and (b) continuing execution when some algorithms are missing while surfacing prominent warnings.

## Decisions & Findings

### D1 – Mirror algorithm metadata at write time
- **Decision**: Enrich every episode manifest emitted by the classic orchestrator so `scenario_params["algo"]` equals the top-level `algo`, while retaining the original `algo` field.
- **Rationale**: Keeps existing aggregation configuration unchanged, satisfies schema consumers expecting the dotted path, and avoids altering downstream scripts that already read `scenario_params.algo`.
- **Alternatives Considered**:
  - *Switch `group_by` default to `algo`*: Risky; would break custom configs that assume nested metadata and complicate resume manifests that store scenario parameters as nested dicts.
  - *Post-process episodes during aggregation*: Increases runtime coupling and prevents fail-fast validation when metadata is missing at the source.

### D2 – Aggregation fallback hierarchy
- **Decision**: Teach `compute_aggregates`/`compute_aggregates_with_ci` to fall back to the top-level `algo` when `scenario_params.algo` is absent, before resorting to `scenario_id` or `unknown`.
- **Rationale**: Provides robust handling for legacy JSONL files while still surfacing validation errors for missing metadata in new runs, aligning with Principle VII (backward compatibility).
- **Alternatives Considered**:
  - *Strict error on missing nested key*: Would force analysts to regenerate all prior results immediately and violates backward-compatible expectations.
  - *Silently keep current fallback to `scenario_id`*: Continues the bug where algorithms mix, violating Principle III.

### D3 – Missing algorithm warning behaviour
- **Decision**: When aggregation detects that configured algorithms are absent from the episode set, log a Loguru `warning` and annotate aggregate summaries with a `missing_algorithms` list while continuing execution.
- **Rationale**: Matches clarification requirement (warn but proceed), supports automated pipelines that expect JSON output, and highlights gaps for analysts without blocking progress.
- **Alternatives Considered**:
  - *Hard failure*: Conflicts with clarified acceptance criteria and could halt runs for transient issues (e.g., single baseline failing mid-run).
  - *Silent ignore*: Reintroduces hidden data quality problems and violates Principle VI (transparency).

### D4 – Validation and documentation updates
- **Decision**: Extend existing validation scripts/tests to assert algorithm-specific grouping and warning behaviour, and document troubleshooting steps in benchmark workflow docs plus `CHANGELOG.md` entry.
- **Rationale**: Keeps regression coverage aligned with Constitution Principle IX (tests) and Principle VIII (documentation as API surface).
- **Alternatives Considered**: None — documentation/tests are mandatory per constitution.

## Outstanding Follow-ups
- None identified; clarifications resolved all high-impact uncertainties.

## References
- `robot_sf/benchmark/full_classic/orchestrator.py`
- `robot_sf/benchmark/aggregate.py`
- `scripts/run_social_navigation_benchmark.py`
