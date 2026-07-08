# Feature Specification: Automated Research Reporting for Imitation Learning

**Feature Branch**: `270-imitation-report`  
**Created**: 2025-11-21  
**Status**: Draft  
**Input**: User description: "Create automated research reporting for imitation learning pipeline: multi-seed experiment orchestration, metrics aggregation (sample efficiency, success rate, collision rate), statistical analysis (bootstrap CIs, effect size, p-values), automated figure generation, markdown + LaTeX report export, reproducibility metadata (git hash, configs, hardware), ablation support (dataset size, BC epochs), hypothesis evaluation, success criteria tracking."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Generate End-to-End Research Report (Priority: P1)

The research engineer runs the imitation learning pipeline (expert training, trajectory collection, BC pre‑training, PPO fine‑tuning) and automatically receives a structured research report (Markdown + optional LaTeX export) containing aggregated metrics, figures, statistical comparisons, hypothesis evaluation, and reproducibility metadata.

**Why this priority**: Core value proposition—turn raw training outputs into publishable, decision‑ready artifacts without manual collation.

**Independent Test**: Execute a single full pipeline run with multi‑seed settings; verify that a report directory is produced containing `report.md`, figures, data exports, and metadata file matching required keys.

**Acceptance Scenarios**:

1. **Given** a successful pipeline run with seeds completed, **When** the reporting module is invoked automatically at completion, **Then** a `output/research_reports/<timestamp>/report.md` file exists with required sections (Setup, Results, Statistical Analysis, Conclusions, Reproducibility).
2. **Given** a pipeline run where one intermediate step (e.g., comparison) is skipped, **When** report generation executes, **Then** the report still renders with a clearly marked skipped section and no crash.

---

### User Story 2 - Multi-Seed Metrics Aggregation (Priority: P2)

The system orchestrates multiple random seeds per experimental condition (baseline vs pre‑trained) and aggregates per‑seed metrics into summary statistics with bootstrap confidence intervals.

**Why this priority**: Ensures statistical robustness and reproducibility; foundational for valid comparisons.

**Independent Test**: Run an experiment specifying N seeds; confirm aggregated output includes mean, median, p95, and CI bounds for each metric (success rate, collision rate, timesteps to convergence).

**Acceptance Scenarios**:

1. **Given** configured seeds `[42,43,44]`, **When** the pipeline completes, **Then** aggregation output lists metrics with per‑seed raw values and aggregated statistics including CI ranges.
2. **Given** a missing result for one seed due to failure, **When** aggregation runs, **Then** the report marks the seed as failed and excludes it from summary while logging completeness ratio.

---

### User Story 3 - Statistical Analysis & Figures (Priority: P3)

The researcher triggers (or auto‑receives) generation of standardized figures (learning curves, sample efficiency bar chart, success/collision distribution, effect size summary) and statistical tests (effect size, p‑value, bootstrap intervals) to support claims.

**Why this priority**: Visual and statistical summaries accelerate interpretation and publication readiness.

**Independent Test**: Run figure generator on existing aggregated metrics; verify presence of required plot files (`fig-learning-curve.pdf`, `fig-sample-efficiency.pdf`, etc.) and a stats table in the report showing test type, p‑value, effect size.

**Acceptance Scenarios**:

1. **Given** baseline and pre‑trained metric sets, **When** analysis runs, **Then** the report includes an effect size (Cohen's d) and p‑value for sample efficiency metric.
2. **Given** insufficient samples (<2 seeds), **When** statistical module runs, **Then** it gracefully skips inferential tests with an explanatory note.

---

### User Story 4 - Ablation & Hypothesis Evaluation (Priority: P4)

Researcher defines ablation parameters (BC epochs, dataset size) to evaluate hypothesis that pre‑training reduces required PPO timesteps by a target percentage.

**Why this priority**: Supports deeper investigation and parameter sensitivity central to research goals.

**Independent Test**: Execute ablation matrix run; verify report includes table comparing variants and indicates which meet hypothesis threshold.

**Acceptance Scenarios**:
1. **Given** three dataset sizes, **When** ablation completes, **Then** the report lists each size with achieved improvement and marks those exceeding threshold.
2. **Given** missing results for one configuration, **When** report renders, **Then** that row shows status "incomplete" without blocking generation.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Expert policy absent when `--skip-expert` used → Report marks baseline as "unavailable" and omits comparative statistics.
- Partial failure of a seed run → Aggregation excludes failed seed, logs completeness ratio and caveat in statistical section.
- Trajectory dataset below minimum size → BC step warns; report flags reduced validity of pre‑training claims.
- Hardware variance across runs → Report records hardware profile per run; mixed hardware triggers disclaimer.
- Corrupted trajectory file → Validation step quarantines file; report lists dataset integrity failure and halts BC metrics only.
- Missing comparison script → Report omits comparative section gracefully.
- Telemetry sampling disabled → Report still includes static metrics; telemetry section absent without error.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST orchestrate multi‑seed imitation learning experiments (baseline & pre‑trained) from a single command.
- **FR-002**: System MUST aggregate per‑seed metrics (success rate, collision rate, timesteps to convergence) into summary statistics (mean, median, p95) with bootstrap confidence intervals.
- **FR-003**: System MUST compute sample efficiency improvement (% reduction in PPO timesteps to reach convergence) between baseline and pre‑trained policies.
- **FR-004**: System MUST generate standardized figures (learning curve, sample efficiency bar, metric distributions, improvement summary) in vector PDF and PNG formats.
- **FR-005**: System MUST produce a Markdown research report containing sections: Abstract, Experimental Setup, Results, Statistical Analysis, Conclusions, Reproducibility.
- **FR-006**: System MUST export optional LaTeX version of the report and figures suitable for paper integration.
- **FR-007**: System MUST record reproducibility metadata (git commit hash, branch, dirty state, package versions, hardware profile, seeds, configs used).
- **FR-008**: System MUST support ablation matrices over BC epochs and dataset size, aggregating each variant's performance.
- **FR-009**: System MUST evaluate predefined hypothesis: "Pre‑training reduces required PPO timesteps by ≥ 40%" and mark pass/fail per condition.
- **FR-010**: System MUST validate trajectory dataset integrity (expected episode count, action/value ranges) prior to BC training.
- **FR-011**: System MUST gracefully handle missing or failed seeds and exclude them from inferential statistics while documenting omissions.
- **FR-012**: System MUST compute statistical tests for key metrics (sample efficiency, success rate) using a paired t‑test by default, reporting p‑value and effect size (Cohen's d); falls back to bootstrap CIs only if t‑test assumptions invalid (e.g., <2 paired samples).
- **FR-013**: System MUST enforce a minimum dataset size of 200 episodes for BC pre‑training; runs below threshold are flagged in the report with a validity disclaimer.
- **FR-014**: System MUST emit machine‑readable JSON summary aligning with existing training summary schema extensions.
- **FR-015**: System MUST complete report generation even if comparison script absent, omitting only dependent sections.
- **FR-016**: System MUST expose CLI arguments for experiment name, number of seeds, hypothesis threshold, ablation parameters, and output directory override.
- **FR-017**: System MUST finish report generation within specified time budget (< 120 seconds for standard 3‑seed run).
- **FR-018**: System MUST produce a completeness score (0–100%) summarizing successful steps vs attempted steps.
- **FR-019**: System MUST include explicit disclaimer if mixed hardware profiles detected across seeds.
- **FR-020**: System MUST provide a summary table of all ablation variants with improvement %, pass/fail, and CI.
- **FR-021**: System MUST structure report output directory with subdirectories: `figures/`, `data/`, `configs/`, containing respective artifacts for reproducibility.
- **FR-022**: System MUST consume run tracker manifests (when available) to extract step timing, telemetry samples, and recommendations for inclusion in report.
- **FR-023**: System MUST auto-populate report Abstract section with hypothesis statement, primary findings (pass/fail threshold), and quantified improvement ranges.
- **FR-024**: System MUST export raw metrics data in both JSON and CSV formats for external analysis tools.
- **FR-025**: System MUST include auto-generated figure captions matching publication standards (descriptive title, axis labels explained, sample size noted).

### Key Entities *(include if feature involves data)*

- **ExperimentRun**: Logical grouping of baseline & pre‑trained multi‑seed executions; attributes: run_id, seeds[], timestamps, hardware_profile[], configs[].
- **MetricRecord**: Per‑seed/per‑variant result; attributes: seed, policy_type, success_rate, collision_rate, timesteps_to_convergence.
- **AggregatedMetrics**: Derived statistics per condition; attributes: metric_name, mean, median, p95, ci_low, ci_high, effect_size.
- **AblationConfig**: Parameter slice definition; attributes: dataset_size, bc_epochs, variant_id.
- **HypothesisDefinition**: Statement & threshold; attributes: description, metric, threshold_value, decision.
- **ReportArtifact**: Generated asset; attributes: path, type (figure, markdown, latex), generated_at.
- **ReproducibilityMetadata**: Provenance; attributes: git_hash, dirty_flag, packages, hardware, seeds, configs.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Report generation completes in < 120 seconds for a 3‑seed baseline + pre‑trained run.
- **SC-002**: Sample efficiency improvement correctly calculated with ±1% tolerance vs manual computation.
- **SC-003**: 100% of required report sections present when corresponding data available (Abstract, Setup, Results, Stats, Conclusions, Reproducibility).
- **SC-004**: ≥ 95% reproducibility metadata fields populated (git hash, hardware, seeds, configs, package versions).
- **SC-005**: Figures set includes at least 5 core plots (learning curve, sample efficiency, success distribution, collision distribution, improvement summary) in both PDF and PNG.
- **SC-006**: Statistical test selection logic yields p‑values and effect sizes for metrics when ≥2 seeds; skips gracefully with explanatory note otherwise.
- **SC-007**: Hypothesis evaluation table lists all variants with correct pass/fail labeling aligned to chosen threshold.
- **SC-008**: Ablation matrix coverage ≥ 90% for configured variants (missing variants explicitly listed).
- **SC-009**: Completeness score matches (completed_steps / attempted_steps) × 100% within integer rounding.
- **SC-010**: No critical failure (uncaught exception) occurs in presence of any single step failure (robust degradation).
- **SC-011**: Report output directory structure includes all required subdirectories (figures/, data/, configs/) with non-empty contents.
- **SC-012**: When tracker manifests available, report includes telemetry section with CPU/memory metrics and recommendations count.
- **SC-013**: Figure captions include sample size notation (e.g., "n=3 seeds") for all aggregate plots.
- **SC-014**: CSV export validates against standard parsers (pandas.read_csv) without errors.

## Assumptions

- Baseline & pre‑trained policies use identical environment configuration except for initialization.
- Convergence defined via existing success/collision criteria in training scripts.
- Bootstrap samples default to 1000 unless overridden.
- Minimum of 2 seeds required for inferential statistics.
- Hardware differences limited to CPU model; GPU acceleration not assumed.
- Dataset integrity check includes episode count ±0 tolerance and action range sanity.

## Open Questions

None – all clarification markers resolved.

## Risks

- Inaccurate convergence detection could skew sample efficiency.
- Large bootstrap sample sizes may exceed time budget.
- Ablation matrix explosion if uncontrolled parameter ranges.

## Out of Scope

- Generalization to non‑imitation learning algorithms.
- Real‑time dashboard UI.
- Automatic paper drafting beyond LaTeX asset exports.
