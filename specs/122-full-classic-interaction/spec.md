# Feature Specification: Full Classic Interaction Benchmark

**Feature Branch**: `122-full-classic-interaction`  
**Created**: 2025-09-19  
**Status**: Draft  
**Input**: User description: "Full classic interaction benchmark: run all classic_interactions.yaml scenarios at statistically powered sample sizes; produce JSONL episodes, bootstrap aggregated metrics with confidence intervals, effect sizes across archetype+density, comprehensive plots (distributions, trajectories, KDE density, force field snapshots, Pareto/SNQI trade-offs), and annotated representative videos (robot path, pedestrian flows, collision/stall events). Provide single end-to-end reproducible script classic_benchmark_full.py with resume + parallel workers, statistical sufficiency report (CI widths < target thresholds), and documentation + smoke test mode."

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a benchmarking researcher, I want a single reproducible command that runs every classic interaction scenario (defined in the classic_interactions matrix), aggregates statistically reliable performance and safety metrics, and produces human‑interpretable visual/Video artifacts so I can compare robot behavior across archetypes and densities with quantified uncertainty.

### Acceptance Scenarios
1. **Given** the scenario matrix and a configured output directory, **When** the benchmark is launched with default parameters, **Then** all scenarios (every name in the matrix) are expanded into episodes and a JSONL of episodes plus an aggregated metrics summary with confidence intervals is produced.
2. **Given** a previous partial run and resume enabled, **When** the benchmark is re‑executed, **Then** only missing episodes are computed and the final aggregates reflect the union of all required episodes.
3. **Given** the completed benchmark artifacts, **When** a stakeholder opens the results folder, **Then** they find standardized plots (distribution, trajectory overlays, density/KDE, Pareto/SNQI) and at least one annotated video per archetype.
4. **Given** a request to validate statistical sufficiency, **When** the generated statistical report is reviewed, **Then** each primary metric’s 95% CI half‑width is below the configured threshold (e.g., collision rate ±2 percentage points, time‑to‑goal ±5%).
5. **Given** smoke mode is invoked, **When** the benchmark runs, **Then** a drastically reduced subset executes (<10% of full episodes) producing structurally identical artifact layout for CI verification.

### Edge Cases
- Scenario with zero valid episodes due to early systemic failure → Benchmark MUST flag failure clearly in summary.
- Extremely low collision counts (zero events) → CIs MUST use an appropriate method (binomial) to avoid misleading zero-width intervals.
- Resume with modified scenario definitions → System MUST detect drift and warn or invalidate prior partial results.
- Missing video generation dependencies → Benchmark MUST still succeed for metrics/plots and mark videos as skipped.
- User sets overly small sample target → Statistical sufficiency report MUST highlight unmet precision goals.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: The benchmark MUST execute every scenario defined in the classic interaction scenario matrix without omission.
- **FR-002**: The system MUST expand scenario seeds into individual episode jobs and persist each as a JSONL line conforming to the episode schema.
- **FR-003**: The system MUST support resume (skip previously recorded episode ids) by default.
- **FR-004**: The system MUST compute aggregate metrics (collision rate, time‑to‑goal, path efficiency, success rate, average speed, SNQI) grouped by archetype and density.
- **FR-005**: The system MUST provide bootstrap confidence intervals for each metric with configurable samples and confidence level.
- **FR-006**: The system MUST calculate pairwise or relative effect sizes (e.g., difference in means or rate differences) between densities within an archetype.
- **FR-007**: The benchmark MUST output standardized plot artifacts (at minimum: metric distributions, KDE density maps of positions, trajectory overlays, Pareto/SNQI trade‑off chart, force field or interaction heatmap snapshots when data available).
- **FR-008**: The system MUST generate at least one annotated video per archetype showing robot trajectory, pedestrian flows, and events (collisions, stalls, goal reach) highlighted.
- **FR-009**: The benchmark MUST produce a statistical sufficiency report enumerating metrics, sample sizes, CI half‑widths, and pass/fail against configured precision thresholds.
- **FR-010**: The system MUST provide a smoke mode that completes in <2 minutes on a development machine while preserving artifact structure.
- **FR-011**: All artifacts MUST be organized under a single timestamped results directory with subfolders /episodes, /aggregates, /plots, /videos, /reports.
- **FR-012**: The system MUST allow configuration of workers (parallelism) and horizon without editing source.
- **FR-013**: The system MUST log a manifest of episodes enabling deterministic reproducibility.
- **FR-014**: The system MUST surface a clear summary (JSON) listing total episodes planned, executed, skipped (resume), failed.
- **FR-015**: Failure of non-critical subsystems (video generation) MUST not abort metric computation; failures MUST be recorded.
- **FR-016**: The system MUST validate scenario matrix integrity before execution (map existence, parameter bounds).
- **FR-017**: The system MUST allow optional SNQI weight/baseline injection for recomputation of SNQI.
- **FR-018**: The system MUST version/record config (scenario matrix hash, code git hash) in the report bundle.
- **FR-019**: The system MUST provide a single entry-point script invoked via standard project tooling.
- **FR-020**: The system MUST support deterministic seeding master control for reproducibility.

### Non-Functional / Quality Requirements
- **NFR-001**: Full benchmark SHOULD complete within an operational time budget acceptable for nightly runs (target: <4 hours on reference hardware) [NEEDS CLARIFICATION: define reference hardware].
- **NFR-002**: Smoke mode MUST complete <2 minutes (95th percentile) on developer laptop.
- **NFR-003**: Metric computation SHOULD scale near-linearly up to a modest worker count (e.g., 4–8 workers) [NEEDS CLARIFICATION: exact scaling acceptance].
- **NFR-004**: CI half‑width thresholds defaults (example): collision rate ±0.02, success rate ±0.03, time‑to‑goal mean ±5%, path efficiency ±5%, SNQI ±0.05 absolute — configurable.
- **NFR-005**: All artifact generation MUST be deterministic given fixed seed (excluding inherent bootstrap randomness unless seeded).
- **NFR-006**: Plots and videos MUST be human‑readable without additional processing.

### Assumptions
- Scenario matrix is stable and authoritative for classic interactions.
- Episode schema remains backward compatible during implementation timeframe.
- Video generation dependencies are available or can be substituted with a fallback.

### Risks / Open Questions
- [NEEDS CLARIFICATION: Minimum episodes per scenario to achieve target CI widths]
- [NEEDS CLARIFICATION: Exact effect size definition for each metric]
- [NEEDS CLARIFICATION: Required video annotation elements (text overlays vs icons)]
- Potential long runtime if densities high and horizon large.
- Bootstrap sample size trade‑off vs runtime.

### Key Entities
- **Scenario**: Logical configuration unit (archetype, density, flow, groups, seeds).
- **Episode Record**: Single simulation execution annotated with metrics.
- **Aggregate Summary**: Grouped statistical metrics + CIs per (archetype, density).
- **Statistical Report**: Precision evaluation listing CI widths vs thresholds.
- **Plot Artifact**: Visual representation of metric distributions or spatial behavior.
- **Video Artifact**: Annotated temporal visualization of representative episode.

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous (except marked items)
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarifications removal)

