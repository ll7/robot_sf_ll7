# SNQI Weight Recomputation & Sensitivity Analysis

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
- (Open) Seeding / reproducibility controls (CLI `--seed` not yet added).
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
### 8.4 Output Schema (Proposed `schema_version: 1`)
Core fields (example for optimization):
```
{
  "schema_version": 1,
  "generated_at": "2025-09-14T12:34:56Z",
  "git_commit": "<hash>",
  "seed": 42,
  "method": "differential_evolution",
  "weights": {...},
  "objectives": {"stability": 0.83, "discriminative_power": 0.41},
  "metrics": {"score_variance": 0.12, "ranking_corr_bootstrap_mean": 0.79},
  "provenance": {"episodes_file": "episodes.jsonl", "baseline_file": "baseline.json"}
}
```

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
Phase 2: Add broader tests + schema + expand design doc (IN PROGRESS – next)  
Phase 3: Add CLI integration (`robot_sf_bench snqi`)  
Phase 4: Introduce seed, provenance & bootstrap stability; deprecate direct script usage (soft warning)  
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
- Deterministic optimization under fixed seed (PENDING)
- Documented JSON schema with `schema_version` (PENDING)
- CI passes with new tests & type hints (ONGOING; current additions green)
- Updated root README and feature docs (PENDING)

---
*This is a living document; sections will be refined as implementation proceeds.*
