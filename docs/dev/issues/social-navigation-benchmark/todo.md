# Social Navigation Benchmark — Master TODO

Guiding objective: Release a reproducible, force-field–aware social navigation benchmark (scenarios, metrics, composite index, baselines, scripts) suitable for a short paper / dataset+benchmark track submission.

Update etiquette:
- Mark progress with `[x]`.
- When partially done, append a short status note (e.g. `(impl 70%)`).
- Add date tags for completed milestones (YYYY-MM-DD).
- Keep scope creep isolated in the Stretch section.

---
## 0. Scoping & Planning
- [ ] Define benchmark scope statement (≤150 words) and success criteria
- [ ] Choose target venues & submission deadlines (ICRA / IROS / CoRL / NeurIPS D&B)
- [ ] Decide license & artifact dissemination strategy (GitHub + Zenodo DOI)
- [ ] Finalize baseline algorithm list (min: SF default, RL policy, Random, Optional: ORCA/RVO)

## 1. Scenario & Dataset Specification
- [ ] Enumerate scenario dimensions (density, flow pattern, obstacles, groups)
- [ ] Draft scenario matrix (table of N core scenarios, each with parameter ranges)
- [ ] Implement deterministic scenario generator (seeded)
- [ ] Implement map variants (simple hall, crossing, bottleneck, obstacle maze)
- [ ] Add grouping / crowd heterogeneity flags to scenario config schema
- [ ] Validate scenario diversity via quick summary script (histograms of min distances, avg speeds)
- [ ] (Optional) Import small real trajectory stats (ETH/UCY aggregate) for parameter calibration

## 2. Metric Definition & Implementation
- [ ] Finalize metric list & definitions (formal doc)
	- [ ] Success rate
	- [ ] Time-to-goal / normalized path efficiency
	- [ ] Collision count / near-miss count (distance < threshold)
	- [ ] Min / mean interpersonal distance distribution
	- [ ] Force magnitude quantiles (per ped & aggregated)
	- [ ] Force exceedance events (above comfort threshold)
	- [ ] Comfort exposure time (% of steps above threshold)
	- [ ] Path smoothness (jerk / curvature stats)
	- [ ] Robot energy proxy (sum |accel|)
	- [ ] (Optional) Divergence / gradient norm of force field along path
- [ ] Implement metric computation module (`metrics/`)
- [ ] Add unit tests for each metric (edge cases: empty crowd, single ped, high density)
- [ ] Define JSON schema for per-episode metric output

## 3. Composite Index (SNQI)
- [ ] Draft formula (weighted normalized metrics)
- [ ] Implement normalization strategy (min-max or percentile baseline)
- [ ] Provide script to recompute weights / sensitivity analysis
- [ ] Ablation: measure discriminative power with and without each component

## 4. Configuration & CLI Harness
- [ ] YAML/JSON schema for benchmark suite definition (list of scenario specs + repetitions)
- [ ] Implement CLI: `robot_sf_bench run --suite core --algo baseline_sf --out results/` 
- [ ] Subcommand: `robot_sf_bench list-scenarios`
- [ ] Subcommand: `robot_sf_bench validate-config`
- [ ] Global flags: seed, parallel workers, progress bar, resume
- [ ] Logging: structured (JSONL) plus human-readable summary

## 5. Baseline Algorithm Integrations
- [ ] Standard Social Force planner wrapper
- [ ] Existing RL policy loader (PPO model) adapter
- [ ] Random / naive reactive baseline
- [ ] (Optional) ORCA integration (licensing check)
- [ ] Unified interface (step(obs) -> action) with timeouts / safety clamp
- [ ] Per-baseline config file (hyperparameters, seeds)

## 6. Evaluation Pipeline & Aggregation
- [ ] Aggregation script: merges JSONL episodes -> metrics CSV + SNQI per algo
- [ ] Confidence interval computation (bootstrap or across seeds)
- [ ] Seed variance analysis script
- [ ] Automatic ranking table generator
- [ ] Failure case extractor (episodes with collisions / low comfort)
- [ ] Caching layer to avoid recomputing unchanged episodes

## 7. Visualization & Reporting Assets
- [ ] Force field heatmap + vector overlays example figure
- [ ] Distribution plots (distance, force magnitude, comfort exposure)
- [ ] Pareto fronts (Time vs. Comfort, Collisions vs. SNQI)
- [ ] Scenario montage thumbnails
- [ ] Baseline comparison table auto-generation (Markdown)
- [ ] Scripts to regenerate all paper figures from raw logs

## 8. Reproducibility & Packaging
- [ ] Deterministic seeding audit (numpy, torch, random)
- [ ] Version stamping (git hash + config hash in outputs)
- [ ] Lock dependencies (uv lock confirmed)
- [ ] Provide `environment.md` / exact reproduction instructions
- [ ] Pre-commit hook to validate metric schema changes
- [ ] Lightweight CI pipeline (lint + unit tests + sample benchmark smoke test)
- [ ] Zenodo deposition draft (metadata prepared)

## 9. Paper Writing Tasks
- [ ] Outline (sections & bullet points)
- [ ] Related work pass (benchmarks in social navigation, force-based metrics)
- [ ] Method section draft (scenarios, metrics, SNQI)
- [ ] Experimental setup description
- [ ] Results & discussion with ablations
- [ ] Threats to validity / limitations
- [ ] Artifact & reproducibility statement
- [ ] Final figure polish
- [ ] Submission formatting & checklist

## 10. Validation & Ablations
- [ ] Metric sanity tests (manually crafted edge episodes)
- [ ] Sensitivity: vary density → observe monotonic comfort change
- [ ] SNQI weight sensitivity plot
- [ ] Remove each metric from SNQI → Δ ranking analysis
- [ ] Cross-seed stability (≥5 seeds / baseline)
- [ ] Runtime profiling (episodes/sec per baseline)

## 11. Stretch / Optional Enhancements
- [ ] Force divergence & curl metrics (interpretability)
- [ ] Residual learned predictor baseline (SF + small MLP)
- [ ] Bayesian parameter calibration for SF (posterior summary only)
- [ ] Risk-aware planner variant (chance-constrained) baseline
- [ ] Minimal web dashboard (interactive metric explorer)
- [ ] User study design draft (comfort perception) — if time

## 12. Risk Log
- [ ] Scenario diversity insufficient → mitigation: expand parameter grid
- [ ] Metrics not discriminative → mitigation: add gradient/jerk or comfort exposure variants
- [ ] Time constraints on RL training → mitigation: reuse existing trained checkpoints
- [ ] ORCA licensing complexity → mitigation: exclude or reimplement simplified reciprocal avoidance
- [ ] Compute resource limits → mitigation: batch evaluation + caching layer

## 13. Immediate Next Actions (Pick 3 to start)
- [x] Draft scenario dimension list (see `scenario_dimensions.md`) (2025-09-02)
- [x] Write metric formal definitions doc stub (see `metrics_spec.md`) (2025-09-02)
- [x] Implement JSON schema draft for episode output (see `episode_schema.json`) (2025-09-02)

---
Last updated: 2025-09-02

