# SNQI Weight Optimization & Recompute – Design Document

## 1. Context
The Social Navigation Quality Index (SNQI) provides a composite score aggregating safety-, efficiency-, and comfort-related metrics for robot navigation episodes. The current PR introduces tooling for recomputation of weight strategies, optimization (grid + differential evolution), sensitivity analysis, and normalization comparisons. Previously, duplicated implementations and lack of schema/metadata impeded reproducibility and maintenance.

## 2. Goals
- Single canonical implementation of SNQI computation & normalization.
- Deterministic, reproducible weight optimization and recompute workflows.
- Extensible output schema with provenance & runtime metadata.
- Modular scripts refactorable into a unified CLI later.
- Tests covering parity, determinism, strategy correctness, robustness, and CLI outputs.

## 3. Non‑Goals (Current Scope)
- Implementing advanced multi-objective optimizers (e.g., NSGA-II) – deferred.
- Providing bootstrap confidence intervals or stability CIs – deferred.
- Real-time dashboard or visualization interface.
- Full benchmark CLI integration (planned follow-up).

## 4. Constraints & Assumptions
- Python ecosystem: NumPy, SciPy available; avoid adding heavy deps by default.
- Runtime target: < 60s for small (<= 50 episode) optimization with grid resolution 5.
- Episode JSONL lines may include malformed entries; these should be skipped gracefully.
- Some metrics may be missing per episode; scoring should degrade gracefully.
- Normalization baselines supplied externally (median/p95 currently).

## 5. Data / File Contracts
### 5.1 episodes.jsonl (input)
Each line: JSON object with minimal structure:
```
{
  "scenario_id": "string",            # required
  "metrics": {
    "success": 0|1,                    # optional but influences success term
    "time_normalized": float,          # normalized time (lower better)
    "collisions": int >=0,
    "near_misses": int >=0,
    "comfort_penalty": float >=0,
    "force_exceed_events": int >=0,
    "jerk_mean": float >=0
  },
  "scenario_params": { ... optional ... }
}
```
Malformed lines: skipped; count may be exposed in future UX improvement.

### 5.2 baseline_stats.json (input)
```
{
  "collisions": {"med": float, "p95": float},
  "near_misses": {"med": float, "p95": float},
  "force_exceed_events": {"med": float, "p95": float},
  "jerk_mean": {"med": float, "p95": float}
}
```
Values used for median/p95 scaling. Missing keys cause score terms to fallback (currently neutral weighting) or raise in future tightening.

### 5.3 optimization output (JSON)
Root object contains:
- `_metadata`: schema_version, generated_at (UTC ISO), git_commit, seed, start_time, end_time, runtime_seconds, provenance (files, invocation, method_requested)
- `grid_search`, `differential_evolution` (optional depending on method) each with: weights, objective_value, ranking_stability, convergence_info
- `recommended`: chosen method summary (weights, objective_value, ranking_stability, method_used)
- `sensitivity_analysis` (optional)
- `summary`: method, weights, runtime_seconds, start_time, end_time, available_methods

### 5.4 recompute output (JSON)
- `_metadata`: as above (strategy_requested, compare flags in provenance)
- `strategy_result` or `strategy_comparison`
- `recommended_weights`
- `normalization_comparison` (optional)
- `summary`: method, weights, compare flags, runtime_seconds, start_time, end_time

## 6. SNQI Computation (Canonical)
Implemented in `robot_sf/benchmark/snqi/compute.py`:
- Takes raw metrics + weight dict + baseline stats.
- Normalizes selected metrics using median/p95: `norm = clamp((value - med) / (p95 - med), 0, 1)` with safe division (epsilon if p95≈med).
- Aggregates weighted sum: higher success & lower penalties -> higher SNQI.
- Finiteness enforced before serialization.

## 7. Weight Strategies (Recompute)
- default, balanced, safety_focused, efficiency_focused, pareto.
- Pareto strategy: heuristic sampling of weight vectors; frontier filtered by discriminative power (std) vs stability heuristic.
- Placeholder `pareto_efficiency` slated for removal or replacement with a true dominance score.

## 8. Optimization Objective
Objective (minimized) = negative weighted combination of ranking stability (0.6) and discriminative power (0.4). Stability heuristic currently uses variance/score patterns; future refinement may adopt bootstrap rank correlation averages.

## 9. Reproducibility & Seeding
- `--seed` seeds NumPy; passed into SciPy differential evolution.
- Pareto sampling & random components deterministic under same seed.
- Metadata records seed + git commit.
- Remaining nondeterminism: potential SciPy parallel RNG differences (document). Future: add note if thread pool engaged.

## 10. Error Modes & Fallbacks
| Condition | Current Handling | Planned Improvement |
|-----------|------------------|---------------------|
| Missing episode metrics | Treat absent metrics as neutral (skip term) | Log warning & optional `--fail-on-missing-metric` |
| Malformed JSONL line | Skip silently (warn logged) | Count + summary report |
| Empty episodes file | Exit(1) with error log | Same |
| Missing baseline metric | KeyError propagates | Pre-validate & friendly message |
| NaN / inf in metrics | Detected by finiteness assertion | Provide field path in error |
| Excessive grid size | Long runtime | Add proactive guard |

## 11. Performance Characteristics
- Grid search complexity: O(R^d). R=5, d=7 => 78,125 evals; borderline but tolerable with fast scoring. Guard recommended.
- Differential evolution typical runtime scales with popsize*maxiter.
- Sensitivity analysis linear in N_episodes * N_weights * 2 directions.
- Normalization comparison adds O(S*N_episodes) where S = alt strategies.

## 12. Testing Strategy
Implemented:
- Parity vs canonical compute
- Schema + finiteness validation
- Deterministic optimization & Pareto sampling
- Strategy structure & weight ranges
- Missing optional metrics resilience
- Malformed JSONL skipping
- Normalization comparison correlations
- CLI invocation tests (optimization & recompute)
Planned:
- Sensitivity monotonic correlation test
- Snapshot/schema regression (jsonschema or structural assertion harness)

## 13. Future Enhancements
- Unified CLI (`robot_sf_bench snqi ...`)
- Bootstrap-based stability metric
- Replace heuristic stability/dominance scoring for Pareto
- Constrain or normalize weight vector (simplex projection option)
- Weight file external schema validation for reuse
- Progress indicators via `tqdm`
- Episode sampling & early stopping

## 14. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Heuristic stability not robust | Misleading weight selection | Introduce bootstrap rank correlation variant |
| Large grids freeze session | Developer time waste | Grid size guard + warning |
| Schema drift unnoticed | Downstream breakage | Snapshot/schema regression tests |
| Silent metric omissions | Biased scores | Add aggregated warning + fail toggle |
| Pareto heuristic unstable | Inconsistent weights | Seed + eventually true MOEA |

## 15. Open Questions
- Should weight outputs be versioned & stored under `model/weights/`?
- How strict should baseline schema be (hard fail vs soft warn)?
- Adopt jsonschema dependency or keep lightweight validator?

## 16. Appendix: Proposed Grid Guard
If `grid_resolution ** n_weights > 200000`: log warning and require `--force-grid` or reduce resolution automatically (document behavior).

## 17. Appendix: Potential Stability Redefinition
```
For B bootstrap resamples:
  For each resample: compute ranking vector r_b
Stability = average pairwise Spearman(r_b, r_{b'}) over all b<b'
```
Expensive; can approximate with subset of pairs.

---
Generated: Initial draft (to be iterated as implementation progresses)# SNQI Weight Recomputation & Sensitivity Analysis

> Status: In Progress (core refactor & parity test completed)
> Related PR: #175  
> Related Issue: #174

## 1. Context
The Social Navigation Quality Index (SNQI) aggregates multiple navigation performance and safety metrics (success, time, collisions, near misses, comfort / force exposure, jerk) into a single scalar score used to compare algorithms. The initial implementation (PR #169) provided the formula but lacked:
- A principled process for (re)computing weights
- Sensitivity / robustness evaluation
- Optimization tooling and reproducible workflows

PR #175 introduces standalone scripts for weight recomputation, optimization, and sensitivity analysis. This design doc formalizes the architecture, contracts, and planned improvements required to meet the repository's Definition of Done.

## 2. Goals
- Provide reproducible tooling to derive and analyze SNQI weights
- Support multiple strategies (preset, Pareto-like sampling, optimization, sensitivity sweeps)
- Ensure stability (ranking robustness) and discriminative power (score variance) are measurable and tunable
- Enable integration with the existing benchmark CLI and data pipeline
- Make outputs machine-consumable with stable JSON schema and provenance metadata

## 3. Non-Goals
- Redesign of the underlying raw episode metric collection pipeline
- Replacement of SNQI with a fundamentally different composite metric
- Full multi-objective optimization research (e.g., full NSGA-II implementation) in this iteration
- Real-time / on-policy adaptive weight tuning

## 4. Constraints & Assumptions
- Episode data provided as JSONL with a `metrics` object and optional `scenario_params.algo`
- Baseline normalization uses median/p95; alternative normalizations may be explored but median/p95 remains canonical for now
- Scripts must run headless (visualizations optional) and avoid hard dependencies on plotting libraries
- Determinism required when seed provided
- Performance acceptable if typical workflows complete < ~5 minutes on ~1–5k episodes

## 5. Current Pain Points / Gaps
(From gap analysis — updated after Phase 1 refactor)
- (Resolved) Duplicated SNQI calculation across scripts → consolidated in `robot_sf/benchmark/snqi/compute.py`.
- (Partially resolved) Lack of design doc → skeleton + updated status; still need full expansion (Sections 8–14 elaboration & rationale expansions pending).
- (Open) Ad hoc formulas for stability / Pareto selection (heuristic still in use; bootstrap stability planned).
- (Resolved) Seeding / reproducibility controls: `--seed` flag + unified `_metadata` (schema_version, generated_at, git_commit, seed, provenance) added to all scripts (remaining: formally seed SciPy DE & test determinism).
- (Open) Benchmark CLI integration.
- (Resolved) Validation that scripts compute canonical SNQI: parity test `tests/test_snqi_parity.py` added.

## 6. Options & Trade-offs
### 6.1 Weight Derivation Approaches
| Approach | Pros | Cons |
|----------|------|------|
| Preset (hand-tuned) | Simple, transparent | Subjective, may not generalize |
| Grid search | Exhaustive (small grids) | Exponential growth, slow for many dims |
| Differential evolution | Handles continuous space | Stochastic, requires seeding & bounds |
| Random sampling + Pareto filtering | Simple multi-criteria surface scan | Coverage quality depends on sample size |
| True multi-objective GA (NSGA-II) | Principled Pareto front | Added complexity & dependency |

Decision (initial): Keep presets + (small) grid + differential evolution + lightweight Pareto sampling; revisit NSGA-II if needed.

### 6.2 Stability Metric
| Metric | Pros | Cons |
|--------|------|------|
| Current heuristic (1/(1+|σ-0.5|)) | Cheap, no pairwise ops | Weak theoretical grounding |
| Spearman across bootstrap samples | Statistically meaningful | More computation |
| Kendall_tau across episodes grouped by algo | Robust to noise | Slower for large N |
Decision: Introduce bootstrap Spearman as preferred stability metric (configurable); keep heuristic as fallback.

### 6.3 Normalization Strategy
Will keep median/p95 canonical; allow alt strategies behind flag with correlation reporting—not changing core benchmark yet.

## 7. Chosen Architecture (Implemented & Planned)
```
robot_sf/benchmark/snqi/
  __init__.py
  compute.py        # canonical compute_snqi(metrics, weights, baseline) (implemented)
  normalization.py  # helpers for median/p95 & alternatives
  weighting.py      # strategy generators + validation
  optimization.py   # grid, evolutionary, Pareto sampling APIs
  sensitivity.py    # sweep, pairwise, ablation, normalization impact

scripts/ (CLI wrappers calling above modules)
```
CLI additions (planned – not yet implemented):
```
robot_sf_bench snqi recompute ...
robot_sf_bench snqi optimize ...
robot_sf_bench snqi analyze ...
```

## 8. Data & API Contracts
### 8.1 Input Episode Record (JSONL line)
Required minimal keys:
```
{
  "episode_id": str,
  "metrics": {
    "success": bool|0/1,
    "time_to_goal_norm": float,
    "collisions": int,
    "near_misses": int,
    "comfort_exposure": float,
    "force_exceed_events": int,
    "jerk_mean": float
  },
  "scenario_params": {"algo": str}?  // optional but improves grouping
}
```
### 8.2 Baseline Stats JSON
```
{
  "collisions": {"med": float, "p95": float},
  ...
}
```
### 8.3 Weights JSON
```
{"w_success": float, "w_time": float, ..., "w_jerk": float}
```
### 8.4 Output Schema (Implemented `schema_version: 1`)
All scripts emit a top-level `_metadata` block encapsulating reproducibility & provenance. Analytical sections differ per script.

Canonical metadata block:
```
"_metadata": {
  "schema_version": 1,
  "generated_at": "2025-09-14T12:34:56.123456+00:00",
  "git_commit": "<short_hash>",
  "seed": 42,
  "provenance": {
    "episodes_file": "episodes.jsonl",
    "baseline_file": "baseline_stats.json",
    "weights_file": "weights.json",            # sensitivity only
    "strategy_requested": "pareto",             # recompute only
    "method_requested": "both",                 # optimization only
    "compare_strategies": true,
    "compare_normalization": false,
    "sweep_points": 20,
    "pairwise_points": 15,
    "skip_visualizations": false,
    "invocation": "python scripts/... <args>"
  }
}
```
Examples:
```
// optimization
{ "grid_search": {...}, "differential_evolution": {...}, "recommended": {...}, "_metadata": { ... } }

// recompute
{ "strategy_result": {...}, "recommended_weights": {...}, "_metadata": { ... } }

// sensitivity (results)
{ "weight_sweep": {...}, "pairwise": {...}, "ablation": {...}, "normalization": {...}, "_metadata": { ... } }
// sensitivity (summary)
{ "analysis_summary": {...}, "_metadata": { ... } }
```
Stability rules (v1): `_metadata` and its child field names are reserved; adding new analytic sections is non-breaking. Schema bumps required for any breaking change to `_metadata` structure or semantics.

## 9. Algorithms & Formulas
### 9.1 SNQI
```
SNQI = w_success * success
       - w_time * time_norm
       - w_collisions * coll_norm
       - w_near * near_norm
       - w_comfort * comfort_exposure
       - w_force_exceed * force_exceed_norm
       - w_jerk * jerk_norm
```
Normalization: `(v - med) / (p95 - med)` clamped to [0,1].

### 9.2 Proposed Stability (Bootstrap)
1. Sample B bootstrap subsets of episodes (with grouping stratified by algo if available)
2. Compute per-bootstrap ranking of algorithms (mean SNQI per algo)
3. Compute average pairwise Spearman across bootstraps → stability score.

### 9.3 Discriminative Power
Option A: Std dev of per-algorithm mean scores.  
Option B: ANOVA F-statistic scaled to [0,1]. (TBD—decide in implementation section.)

### 9.4 Multi-objective Combination
Either:
```
Objective = α * stability + (1-α) * discriminative_power
```
with α configurable (default 0.6).

## 10. Testing Strategy
- Parity test ensures canonical function stability: `tests/test_snqi_parity.py` (DONE)
- Unit tests for compute normalization edge cases (p95==med, missing metric) (TODO)
- Deterministic tests with seed (evolution & Pareto sampling) (TODO)
- Schema validation using `jsonschema` once schema_version added (TODO)
- Sensitivity monotonic sanity: Increasing `w_collisions` should not increase average score when collision counts > 0 (TODO)
- Bootstrap stability reproducibility test (TODO after implementing bootstrap)
- CLI integration tests (subprocess) ensuring exit 0 + JSON validity (TODO post CLI subcommands)

## 11. Performance Considerations
| Component | Baseline Target |
|-----------|-----------------|
| Grid search (5^7 worst-case) | Guard/warn & adaptively prune |
| Differential evolution | < 90s typical (500–1000 episodes) |
| Bootstrap stability (B=30) | Configurable; degrade gracefully for small datasets |
| Pairwise sweeps | Skip or reduce automatically if dataset large |

Mitigations:
- Episode subsampling flag (`--sample N`)
- Early stop if objective stagnates X iterations
- Parallelizable phases (future enhancement)

## 12. Error Handling & Fallbacks
- Missing metric → treat as neutral (0) plus warning counter
- Division by zero in normalization → denom=1 with logged warning once
- Insufficient distinct algorithms for stability → fallback heuristic stability
- Evolution failure (`success=False`) → fallback to best known weights

## 13. Security / Safety
- No external network calls
- All inputs treated as untrusted text; JSON parsing with error isolation
- Large file size guard (optional future enhancement)

## 14. Migration / Rollout Plan
Phase 1: Introduce shared module + modify scripts to import it (backward compatible)  (COMPLETED)  
Phase 1b: Add parity regression test (COMPLETED)  
Phase 2: Add broader tests + formal JSON schema artifact + expand design doc (IN PROGRESS)  
Phase 3: Add CLI integration (`robot_sf_bench snqi`)  
Phase 4: Introduce bootstrap stability metric & deprecate direct script usage (soft warning)  
Phase 5: Optional advanced methodology (ANOVA, NSGA-II)  

## 15. Open Questions
- Should weights be normalized (sum=const)?
- Keep clamping of normalized metrics or allow >1 to penalize extreme outliers more?
- Should comfort vs force_exceed be merged or remain distinct?
- How to version recommended weights artifacts? Embed checksum of input dataset?
- Include per-metric contribution breakdown in output?

## 16. Future Work
- NSGA-II proper Pareto front
- Bayesian optimization for weights
- Interactive notebook visualization widgets
- Confidence intervals on rankings (bootstrap distribution)
- Weight regularization (L1/L2) to discourage extreme values

## 17. Acceptance Criteria (Design Phase)
- Shared module replaces duplication (DONE)
- Parity test ensures canonical score stability (DONE)
- Test suite covers ≥85% of new logic paths (non-viz) (PARTIAL – more tests pending)
- Deterministic optimization under fixed seed (PARTIAL – NumPy seeding applied; ensure SciPy DE reproducibility) 
- Documented JSON schema with `schema_version` (PARTIAL – `_metadata` implemented; jsonschema file pending)
- Reproducibility metadata (`_metadata` with seed/git commit/provenance) (DONE)
- CI passes with new tests & type hints (ONGOING; current additions green)
- Updated root README and feature docs (PENDING)

---
*This is a living document; sections will be refined as implementation proceeds.*
