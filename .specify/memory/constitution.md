<!--
Sync Impact Report
Previous Version: 1.0.0 -> New Version: 1.1.0 (MINOR bump)
Rationale: Added new Principle XI (Internal Maintainability & Private Helper Documentation) – material expansion of guidance; clarified Principle VIII wording (documentation scope) without semantic contract change.
Modified Principles:
 - VIII. Documentation as an API Surface (clarified internal vs public docs statement)
Added Sections:
 - XI. Internal Maintainability & Private Helper Documentation
Removed Sections: None
Templates Updated:
 - .specify/templates/plan-template.md (footer version reference) ✅
Pending Template Updates: None
Deferred TODOs: None
All placeholder tokens resolved; no bracketed ALL_CAPS identifiers remaining.
-->

# Robot SF Project Constitution

This Constitution specifies WHAT the Robot SF repository delivers: a cohesive, reproducible research and engineering platform for social navigation of a robot among pedestrians. It intentionally avoids HOW (implementation details) and focuses on scope, boundaries, artifacts, contracts, and non‑negotiable principles. Internal engineering rigor (naming, docstrings, helper clarity) is encoded where it impacts long‑term reproducibility or the stability of public contracts.

## Core Principles

### I. Reproducible Social Navigation Research Core
The repository must provide a self‑contained, scriptable simulation and benchmarking substrate for robot–pedestrian social navigation. All primary artifacts (environments, configurations, benchmark outputs, episode logs, figures, metrics tables, weight files) must be derivable deterministically from versioned inputs (code + configs + seeds + model artifacts). Re-running documented commands must reproduce published metrics and figures (subject only to stochastic variance explicitly bounded by seeds and bootstrap CIs).

### II. Factory‑Based Environment Abstraction
All robot and pedestrian simulation capabilities are exposed through a small set of factory functions that yield Gym/Gymnasium‑compatible environments with consistent observation/action/reward interface surface. Direct instantiation of low‑level environment classes is out of scope for users. Extensibility of new environment modalities (e.g., image, pedestrian‑adversary variants) occurs by adding new factory entry points and unified config schema fields without breaking existing ones.

### III. Benchmark & Metrics First
The platform must embed a canonical benchmark runner that produces structured, append‑only JSONL episode records, enabling standardized aggregation of social navigation quality metrics (including SNQI and component metrics such as collisions, comfort exposure, near misses). Every environment feature and baseline must be justifiable in terms of benchmarkability and metric traceability. Partial or opaque outputs are disallowed; each episode record must contain sufficient scenario metadata to support grouping, resuming, and aggregation.

### IV. Unified Configuration & Deterministic Seeds
All tunable simulation, environment, and scenario parameters exist in a unified, typed configuration layer (not ad‑hoc kwargs). Config objects define explicit defaults; overrides are captured in versioned config files or scripted parameter sets. Seeds must propagate so that environment creation + model initialization + benchmark execution are reproducible.

### V. Minimal, Documented Baselines
The repository must ship reference baseline planners (e.g., SocialForce variants, random policy) serving as yardsticks for RL policies. Baselines are required to: (1) run headless, (2) integrate with the benchmark runner, (3) emit identical metric schema, (4) be documented with purpose, assumptions, and limitations. Novel baselines outside social navigation scope are excluded.

### VI. Metrics Transparency & Statistical Rigor
Aggregations must report descriptive statistics (mean, median, p95) and optionally bootstrap confidence intervals with explicit confidence levels, sample counts, and seeds. No single scalar “score” is promoted without its component metrics. Any weighting scheme (e.g., SNQI weights) must be tracked as a versioned artifact and include provenance.

### VII. Backward Compatibility & Evolution Gates
Public environment factory signatures, configuration field names, and episode/metric JSON schema form the stable contract. Breaking changes require a versioned schema bump, migration notes, and dual‑read (old + new) capability or deprecation path. Silent breaking changes are prohibited.

### VIII. Documentation as an API Surface
Core guides (development guide, environment overview, benchmark docs, SNQI tooling) define expected usage. Every new public surface (environment factory variant, metric, baseline, figure generator, configuration field) MUST have a discoverable entry in the central docs index and a concise README (or clearly linked section) describing WHAT it provides and WHAT assumptions it encodes. Clarifications that do not alter semantics (typo fixes, examples, internal helper docstrings) may be added without version bumps, but any change that modifies a public contract or user‑visible invariant requires governance review.

### IX. Test Coverage for Public Behavior
Any change to public environment behavior, benchmark schema, or metrics computation must be accompanied by at least a smoke test (reset/step loop), and if logic‑bearing, an assertion‑based unit/integration test. Omission requires an explicit, temporary TODO with rationale and follow‑up tracking.

### X. Scope Discipline
Out of scope: general robotics control stacks, unrelated perception models, arbitrary RL algorithm zoo, generic data science utilities, unversioned experiment notebooks. The repo remains focused strictly on social navigation simulation, evaluation, and analysis.

### XI. Internal Maintainability & Private Helper Documentation
Private (underscore‑prefixed or local/inner) helpers that embody non‑obvious branching, fallback semantics, resource management, or performance trade‑offs MUST include a concise docstring or inline comment summarizing: (1) purpose, (2) key decision rules / edge cases, (3) side effects (e.g., filesystem writes, environment access). This requirement ensures future refactors preserve reproducibility and benchmarking invariants. Purely trivial pass‑through wrappers (e.g., one‑line value adapters) are exempt. When complexity is reduced (e.g., helper decomposition, signature introspection replacing branching), newly introduced inner helpers inherit the same documentation requirement at creation time—retroactive addition counts as maintenance (PATCH) unless accompanied by principle changes (MINOR).

## Domain Scope & Deliverables

1. Simulation Environments: Robot navigation (state and image modalities) and pedestrian interaction variants (including adversarial pedestrian spawn). Must expose consistent Gymnasium API (reset/step/seed/spec) through factory functions.
2. Pedestrian Dynamics Integration: Fast‑pysf SocialForce physics submodule providing pedestrian motion and interaction forces; accessed indirectly via a wrapper; considered an external but version-pinned dependency.
3. Benchmark Runner: Batch and single episode execution producing JSONL lines with scenario parameters, seed, per‑episode metrics, and outcome flags; supports parallel workers and resume semantics.
4. Metrics Suite: Social navigation metrics (collisions, near misses, comfort exposure, SNQI composite, and others enumerated in docs) with deterministic computation definitions and optional bootstrap resampling for CIs.
5. Baseline Planners: Minimal, documented reference strategies integrated with the benchmark runner producing comparable metrics.
6. Configuration Schema: Unified config objects (robot, image, pedestrian variants) plus file‑based scenario definitions enabling reproducible experiment sets.
7. Analysis & Figure Generation: Scripts that transform benchmark outputs into distribution plots, force field visualizations, tables (Markdown + LaTeX), and Pareto/frontier style summaries; each figure reproducible via versioned script.
8. Model Artifacts: Versioned example RL policies located in a controlled directory to enable immediate environment evaluation; not a general model registry.
9. Documentation Corpus: Central index (`docs/README.md`), development guide, refactoring/migration notes, benchmark and SNQI tooling guides, figure naming scheme, and scenario thumbnails.
10. Validation Scripts: Shell or Python scripts that exercise minimal environment creation, model inference, and a complete simulation run (pass/fail semantics) after code changes.

Non‑Deliverables (explicit exclusions): Cloud deployment templates, hardware drivers, unrelated perception datasets, production robotics stacks, multi-robot coordination, web dashboards.

## Contracts & Invariants

Environment Contract:
- Observation/Action shapes and semantic meaning remain stable for each factory variant unless versioned.
- Step outputs must always include reward, terminated, truncated, and info with documented keys (e.g., collision flags, progress stats).

Benchmark Output Contract:
- Each JSONL line represents one completed (or explicitly aborted) episode.
- Mandatory fields: episode_id, seed, scenario_id, scenario_params, metrics (namespaced), timing, status.
- No partial lines; corruption or truncation must be detectable (e.g., JSON parse failure).

Metrics Contract:
- Metric names are lowercase snake_case and stable.
- Derived composite metrics (e.g., SNQI) document component weights and version tag.
- Confidence intervals appear only when bootstrap_samples > 0 and include [low, high].

Configuration Contract:
- Default config values documented; absence of a value implies documented default, never implicit runtime mutation.
- Unknown config fields are rejected (no silent ignore) in strict modes.

Resume & Determinism Contract:
- Re-running a benchmark with the same scenario set, seeds, and output path must not duplicate existing episodes (idempotent with resume enabled).

## Quality & Performance Targets

Functional Targets:
- Environment initialization < 1s on reference hardware.
- Benchmark runner processes parallel episodes without schema divergence.
- Figure scripts produce vector PDF outputs and optional PNG exports.

Correctness Targets:
- All public factories and baseline entry points have at least one smoke test.
- Metrics computations yield stable results within documented numerical tolerances.

Performance Targets (soft bounds):
- Simulation loop baseline throughput approx. 20–25 steps/sec for standard scenario configuration.

Reliability Targets:
- Headless execution supported (no display) for CI across all demos essential for validation.
- Submodule presence mandatory: missing submodule fails fast with clear error (not undefined behavior).

## Development Workflow & Compliance Gates (WHAT must exist)

Artifacts Required Before Merge:
1. Updated documentation entries for any new public surface (factory, metric, baseline, figure type).
2. Tests covering new or changed behaviors (smoke + logic assertions).
3. Version bump or migration note when contracts change.
4. Validation scripts pass (basic env, model prediction, full simulation).
5. Benchmark schema unchanged or explicitly revised with changelog entry.

Quality Gates (Presence Requirements):
- Lint report clean (style conformance required as a repository invariant).
- Type analysis performed with zero unreviewed errors.
- Test suite executes successfully in headless mode where appropriate.

Documentation State Requirements:
- Central docs index references any newly added guide.
- Figures generated by new scripts stored under tracked directory with reproducible script committed.

Traceability Requirements:
- Each new metric/baseline includes rationale section (problem addressed, expected use).
- Config changes link to a documented assumption or design note.

## Governance

This Constitution supersedes ad‑hoc practices. Amendments require: (1) written proposal in `docs/dev/issues/<topic>/design.md`, (2) explicit enumeration of affected contracts (env, config, metrics, benchmark schema), (3) migration guidance or deprecation plan, (4) version/date update below. Pull Requests must assert compliance by referencing relevant sections. Any introduction of out‑of‑scope functionality must include justification aligning with Core Principles I–X or be rejected.

**Version**: 1.1.0 | **Ratified**: 2025-09-19 | **Last Amended**: 2025-09-22