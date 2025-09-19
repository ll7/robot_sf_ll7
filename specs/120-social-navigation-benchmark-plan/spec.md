# Feature Specification: Social Navigation Benchmark Platform Foundations

## Feature Name
Social Navigation Benchmark (Scenarios + Metrics + Baselines + Aggregation + Figures)

## Problem Statement (WHAT)
Researchers and practitioners lack a reproducible, force-field–aware benchmark to evaluate robot social navigation policies in dynamic pedestrian environments. Existing evaluations are fragmented: metrics are inconsistent, scenario coverage is opaque, and composite indices are either proprietary or under-specified. We need a single, versioned benchmark substrate producing deterministic, schema‑validated episode data and standardized derived artifacts (tables, figures, composite scores) to enable fair comparison and longitudinal experimentation.

## Goals (Explicit Outcomes)
1. Provide ≥12 canonical scenario definitions spanning density, flow pattern, obstacle complexity, and group behavior classes.
2. Emit per‑episode JSONL records (one line = one episode) adhering to a published schema (episode identity, scenario params, metrics, timing, status, provenance hashes).
3. Supply a core metric suite (success, time, path efficiency, collisions, near misses, interpersonal distances, force metrics, comfort exposure, smoothness, energy proxy, gradient norm) plus composite SNQI.
4. Offer baseline planners (SocialForce, pre‑trained PPO policy, Random) integrated with a unified `step(obs)->action` interface.
5. Support deterministic seeding and resume behavior (no duplication) across multi‑process runs.
6. Enable aggregation producing descriptive stats (mean/median/p95) and optional bootstrap confidence intervals per metric and algorithm.
7. Generate publication‑ready artifacts: distribution plots, Pareto fronts, baseline comparison tables, scenario thumbnails, and force-field visualizations—each reproducible from scripted commands.
8. Provide SNQI weight optimization and ablation tooling with persisted weight provenance.
9. Achieve documented success criteria (see below) to qualify for a paper / dataset+benchmark submission.

## Non-Goals
- Real-world sensor fusion beyond map + pedestrian simulation.
- General multi-robot coordination scenarios.
- Live dashboard / web UI.
- ORCA / advanced reciprocal velocity methods in first release (deferred evaluation).

## Success Criteria (Acceptance)
See imported criteria from benchmark TODO (mirrored here for atomicity):
- Scenario coverage: ≥12 core scenarios.
- Metric discriminative power: baselines differ on ≥2 metrics in ≥60% scenarios.
- Reproducibility: identical aggregate metrics (within float tolerance) across 3 independent seeded batches.
- Stability: CoV <10% for success & comfort exposure across ≥5 seeds.
- SNQI sensitivity: removing any major component changes >1 rank for ≥50% baselines.
- Artifact completeness: repository includes schema, lockfile, scripts, regeneration instructions.
- CI: lint + unit tests + smoke benchmark (≤5 min) green.

## Primary Stakeholders
- Internal research team (navigation algorithms).
- External researchers / reviewers evaluating reproducibility and fairness.
- Tooling / analysis contributors generating derived figures.

## Functional Requirements (WHAT only)
FR1. Provide CLI & programmatic batch execution that accepts scenario matrix definitions and writes JSONL episodes (append‑safe, resumable).
FR2. Each episode record MUST include: unique episode_id, scenario_id, scenario_params, seed, algo_id, metrics (dict), status, timings, git hash, config hash, schema version.
FR3. Provide aggregation utilities producing group‑by summaries and optional bootstrap CIs.
FR4. Provide SNQI weight file (versioned) and recomputation command writing new weight artifacts with provenance (input stats, timestamp, seed).
FR5. Provide at least three baseline planners conforming to a minimal planner interface.
FR6. Provide figure generation orchestrator producing Pareto plots, distributions, force-field figures, scenario thumbnails, and baseline tables into versioned output directories.
FR7. Provide scenario montage generation and thumbnail assets stored under `docs/figures/`.
FR8. Provide deterministic seeding utilities covering numpy, torch, random.
FR9. Provide manifest‑based resume optimization (sidecar referencing identities to skip existing episodes).
FR10. Provide schema validation command for scenario matrices and episode outputs.
FR11. Provide metrics specification document enumerating definitions & units.
FR12. Provide quickstart guide demonstrating end‑to‑end run (scenarios→episodes→aggregate→figures).

## Non-Functional Requirements
NFR1. Reproducibility: Running documented commands yields same aggregated metrics (tolerance ≤1e-6 relative for floating aggregates, except stochastic metrics within bootstrap CI overlap).
NFR2. Performance: Baseline simulation throughput target 20–25 steps/sec standard scenario on reference machine (soft bound).
NFR3. Extensibility: Adding a new metric or baseline must not require changes to existing episode schema fields beyond additive metrics entries.
NFR4. Traceability: All derived artifacts embed git hash + input file stems.
NFR5. Determinism: Episode identity stable across runs given same seed and scenario parameters.
NFR6. Documentation: Central docs index updated with new benchmark section and SNQI tooling references.

## Constraints
- Must not break pre-existing environment factory APIs.
- Episode schema evolution requires version bump & migration note.
- Force-field visualizations rely solely on stored episode & scenario data (no hidden state).
- Fast-pysf submodule must remain unmodified except for scoped updates documented separately.

## Open Questions (initial)
1. Final ORCA inclusion decision? (Deferred) [](../../docs/dev/issues/social-navigation-benchmark/adding_orca.md)
2. Formal mathematical specification of SNQI weighting normalization baseline? (Partially documented; refine in metrics spec.)

## Risks
R1. Metrics not discriminative enough → Expand or adjust thresholds.
R2. Scenario coverage insufficient → Add corridor/crossing collision stress scenario.
R3. Training time for RL baseline large → Reuse existing checkpoints.
R4. ORCA licensing delays → Exclude from initial release.

## Initial Timeline (Relative)
Week 1: Lock scenario matrix & finalize metrics spec.
Week 2: Baseline batch runs + aggregation + SNQI weight recompute.
Week 3: Figures + documentation polish + CI enhancements.
Week 4: Paper outline & ablations.

## Out-of-Scope Clarifications
No GPU optimization work, no GUI redesign, no multi-agent RL training pipeline overhaul.

---
Schema Version (initial): `episode.schema.v1`
