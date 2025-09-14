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
# SNQI Weight Optimization & Recomputation – Design Document

## 1. Context
The Social Navigation Quality Index (SNQI) aggregates task success and social safety / comfort metrics into a single scalar score for benchmarking navigation policies. Earlier iterations duplicated logic, lacked schema guarantees, and provided limited provenance. The refactor introduces canonical computation, weight optimization (grid + differential evolution), recomputation with strategy comparison, normalization sensitivity, and external weight validation.

Related PR: #175  
Checklist: `README.md` in this directory.

## 2. Goals
- Canonical, reusable SNQI computation & normalization.
- Deterministic (seeded) optimization & strategy evaluation with reproducible metadata.
- Extensible JSON schema (versioned) + finiteness & structural validation.
- Modular scripts ready for future unified CLI (`robot_sf_bench snqi ...`).
- Comprehensive tests (parity, determinism, strategies, normalization comparison, CLI, external weights validation).

## 3. Non‑Goals (Initial Scope)
- Full multi‑objective algorithms (NSGA-II) – deferred.
- Bootstrap confidence intervals / robust stability – deferred.
- Real-time visualization UI.
- Immediate integration into global benchmark CLI (planned follow-up).

## 4. Constraints & Assumptions
- Lightweight deps: NumPy (+ SciPy for differential evolution) only.
- Positive weights semantics maintained; penalties subtract from success term.
- Episode metrics may be partially missing; tool must not crash.
- Baseline stats (median/p95) externally supplied; absent baseline metric => neutral effect (0 normalized) currently.
- Execution target: <60s for moderate grid on small dataset (~50 episodes).

## 5. Data & File Contracts
### 5.1 Episodes JSONL
Line‑delimited JSON objects: `{"scenario_id": str, "metrics": {...}, "scenario_params": {... optional ...}}`  
Notable fields (optional unless noted): success, time_to_goal_norm, collisions, near_misses, comfort_exposure, force_exceed_events, jerk_mean.  
Malformed lines are skipped (warning logged; aggregate count future enhancement).

### 5.2 Baseline Stats JSON
Mapping: metric -> {"med": float, "p95": float}. Used in median/p95 clamp normalization. Missing key => metric contributes 0.

### 5.3 External / Initial Weights JSON
All `WEIGHT_NAMES` present with finite positive floats. Extraneous keys ignored with warning. Validated via `validate_weights_mapping`.

### 5.4 Optimization Output Schema (Kind: optimization)
```
{
  "recommended": {"weights": {...}, "objective_value": float, "ranking_stability": float, "method_used": str},
  "grid_search"?: {"weights": {...}, "objective_value": float, "ranking_stability": float, "convergence_info": {...}},
  "differential_evolution"?: {...},
  "sensitivity_analysis"?: { weight: {"score_sensitivity": float, ...}},
  "initial_weights"?: { ... },
  "_metadata": {"schema_version":1, "generated_at": iso, "git_commit": str, "seed": int|null, "provenance": {...}},
  "summary": {"method": str, "objective_value": float, "ranking_stability": float, "weights": {...}, "has_sensitivity": bool, "runtime_seconds": float, ... }
}
```

### 5.5 Recompute Output Schema (Kind: recompute)
```
{
  "recommended_weights": {...},
  "strategy_result"?: {"strategy": str, "weights": {...}, "statistics": {...}},
  "strategy_comparison"?: { strategy: {"weights": {...}, "statistics": {...}} },
  "strategy_correlations"?: {"a_vs_b": float},
  "recommended_strategy"?: str,
  "external_weights"?: {"weights": {...}, "statistics": {...}, "correlation_with_recommended": float},
  "normalization_comparison"?: { baseline_variant: {"mean_score": float, "correlation_with_base": float } },
  "_metadata": {...},
  "summary": {"method": str, "weights": {...}, "runtime_seconds": float, ...}
}
```

### 5.6 Weight Names
`[w_success, w_time, w_collisions, w_near, w_comfort, w_force_exceed, w_jerk]`

## 6. Canonical SNQI Computation
Located in `robot_sf/benchmark/snqi/compute.py`. Normalization: `(value - med) / (p95 - med)` guarded & clamped to [0,1]. Score: success positive; all other terms negative contributions scaled by weights.

## 7. Strategies (Recompute)
- default, balanced, safety_focused, efficiency_focused, pareto.
- Pareto: random sampling (bounded 600) -> non‑dominated filtering over `(std, stability_proxy)` -> top 10 by discriminative power. Deterministic with seed.
Placeholder fields removed (no synthetic `pareto_efficiency`).

## 8. Optimization Objective
`maximize 0.6 * stability + 0.4 * discriminative_power` implemented as negated minimization.  
Stability: Spearman between algorithm groups if ≥2 groups else variance‑derived fallback.  
Discriminative power: variance (normalized) of SNQI scores.

## 9. External Weight Validation
`weights_validation.py` enforces completeness, finiteness, positivity, warns on extraneous keys and unusually large values (>10).

## 10. Reproducibility & Seeding
`--seed` -> NumPy RNG & SciPy DE seed argument. Pareto & sampled grid reuse deterministic RNG. Metadata records seed + git commit. Residual nondeterminism: internal SciPy parallel evaluation order.

## 11. Error Modes & Fallbacks
| Condition | Current Behavior | Planned Improvement |
|-----------|------------------|---------------------|
| Missing weight key | Abort validation | — |
| Non-numeric / non-finite weight | Abort | — |
| Extraneous weight key | Warn ignore | Add to metadata (future) |
| Malformed JSONL line | Warn skip | Aggregate count in summary |
| Empty episodes | Abort (exit 1) | Distinct exit code |
| Missing baseline metric | Silent neutral (0) | Warn / optional fail |
| NaN/Inf produced | Finiteness assertion error | Field path already provided |
| Grid explosion | Adaptive shrink + sample | Expose stats in summary |

## 12. Performance & Scaling
| Component | Complexity | Mitigation |
|----------|-----------|------------|
| Grid search | O(R^d) | Adaptive shrink + sampling when > threshold |
| Diff. evolution | Iter * Pop | Bounded `--maxiter` |
| Pareto sampling | O(Samples) | Cap 600; early stop possible future |
| Sensitivity | O(Episodes * Weights) | Lightweight scalar operations |

## 13. Testing Summary
Implemented: parity, schema+finite, deterministic Pareto, strategies structure, normalization comparison, CLI (opt + recompute), missing optional metrics, external weight validation. Pending: sensitivity monotonic test, schema snapshot regression.

## 14. Methodology Enhancements (Planned)
- Bootstrap stability (average pairwise Spearman of resamples).
- True multi-objective frontier via NSGA-II.
- Weight simplex constraint mode.
- Objective component breakdown in output.
- Episode weighting support.
- Early stopping for evolution.
- Confidence intervals around SNQI.

## 15. Open Questions
- Should validated recommended weights be version-controlled under `model/`?
  - Yes, to ensure reproducibility and traceability of weight configurations.
- Adopt machine-readable JSON schema artifact or maintain lightweight validator only?
  - Yes, to provide formal validation and facilitate integration with other tools.
- How to parameterize acceptable weight ranges (e.g., dynamic warnings)?
  - Try dynamic warnings based on dataset statistics; allow user overrides.

## 16. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Heuristic stability biased | Suboptimal weights | Replace with bootstrap metric |
| Sampling variance in Pareto | Frontier instability | Seed + consider MOEA |
| Schema drift | Consumer breakage | Add snapshot/schema tests |
| Silent baseline gaps | Skewed scores | Pre-validation & warnings |

## 17. Future Observability
- Phase timing (parse/opt/finalize) with `--verbose`.
- Count & report skipped JSONL lines.
- Emit objective breakdown (stability vs discriminative_power).

## 18. Security & Safety
Pure JSON input, no code execution. Validation prevents NaN/Inf propagation. External weights cannot inject code.

## 19. Compatibility Strategy
`schema_version` increments on breaking shape changes. Extraneous weight keys ignored to allow forward addition. Consumer code should check version.

## 19.1 Schema Artifact & Validation
A formal JSON Schema for script outputs (optimization + recompute + shared summary metadata) is provided at:

`docs/snqi-weight-tools/snqi_output.schema.json`

Scope (v1):
- Validates presence/shape of `_metadata` and `summary` blocks.
- Defines `WeightsObject` (pattern `^w_`) with numeric values; rejects unknown extra weight keys by default (explicit choice to surface typos early).
- Provides reusable `MethodBlock` and `RecommendedBlock` definitions (objective + stability + weights + optional convergence).
- Supports optional analytical sections (`grid_search`, `differential_evolution`, `sensitivity_analysis`, `strategy_comparison`, `normalization_comparison`, etc.).

Non‑goals (v1):
- Enforcing numeric ranges (e.g., weights >0) – handled by runtime validator `validate_weights_mapping`.
- Enumerating strategy names (kept open for forward addition without schema bump).
- Strict datetime format validation – treated as opaque ISO8601 strings.

Backward Compatibility Rules:
| Change Type | Requires `schema_version` bump? | Notes |
|-------------|----------------------------------|-------|
| Add new optional top-level analytical section | No | Consumers must ignore unknown keys |
| Add new field inside existing analytic block (optional) | No | Mark optional; document in changelog |
| Rename or remove existing required field | Yes | Bump version and update validator/tests |
| Tighten validation (e.g., forbid previously allowed pattern) | Yes | Consider deprecation cycle |
| Add new required field to `_metadata` | Yes | Unless defaultable & auto-filled |
| Add new weight name | Yes* | *If required in all outputs; otherwise keep optional and document |

Validation Path:
1. Lightweight internal structural validation (`robot_sf.benchmark.snqi.schema.validate_snqi`).
2. (Planned) Optional `jsonschema` test asserting conformance to `snqi_output.schema.json` (snapshot + drift detection).

Rationale for Lightweight Runtime Validation:
- Keep benchmark scripts dependency‑light (avoid hard dependency on `jsonschema`).
- Reserve full schema enforcement for tests / CI (ensures dev ergonomics + performance).

Planned Enhancements:
- Add schema snapshot regression test (task: Schema snapshot test).
- Emit objective component breakdown fields (`stability_component`, `discriminative_component`) – non‑breaking optional additions.
- Introduce formal enum for strategy names once stabilized (`strategy_names` array in `_metadata.provenance`).

Consumers SHOULD:
- Check `_metadata.schema_version` before parsing.
- Gracefully ignore unknown top-level keys.
- Fail fast if required fields missing.

Consumers SHOULD NOT:
- Depend on incidental ordering of object keys.
- Assume all analytical sections are present (presence driven by CLI flags).

Versioning Procedure:
1. Draft change with proposed bump rationale.
2. Update `EXPECTED_SCHEMA_VERSION` in `schema.py`.
3. Modify schema file & design doc compatibility table.
4. Add migration notes + update tests & user guide.


## 20. Appendix
### 20.1 Grid Search Guard Pseudocode
```
while grid_res**n_weights > max_combos and grid_res > 2:
    grid_res -= 1
if grid_res**n_weights still > max_combos:
    sample = rng.choice(grid_points, size=(max_combos, n_weights))
```
### 20.2 Pareto Sampling Pseudocode
```
for i in range(N):
  w ~ U(0.1,3.0)^d
  scores = snqi(w)
  disc = std(scores)
  stab = 1/(1+|disc-0.5|)
  keep (w, disc, stab)
filter non-dominated -> sort by disc desc -> top 10
```
### 20.3 Potential Bootstrap Stability
```
for b in 1..B:
  sample episodes with replacement
  compute ranking vector r_b
stability = mean_{b<b'} Spearman(r_b, r_{b'})
```

## 21. Summary
Current implementation establishes a reproducible, validated baseline for SNQI weight exploration. Next priority: formalize bootstrap stability & richer error/UX reporting, then unify into a subcommand CLI.

---
Generated: Full design draft (supersedes prior partial draft).
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
