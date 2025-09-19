# Phase 0 Research — Social Navigation Benchmark

Date: 2025-09-19  
Branch: 120-social-navigation-benchmark-plan

## Resolved Topics

### 1. SNQI Normalization Strategy
Decision: Use baseline-median / baseline-p95 pair per metric for normalization (median for central tendency, p95 for tail/penalty shaping).  
Rationale: Median robust to outliers; p95 captures extreme undesirable events (collisions, high force).  
Alternatives: (a) mean/std z-score (sensitive to skew); (b) min-max across current run (non-stable across expansions).  
Open Follow-up: Embed baseline_stats_hash in SNQI weight artifact.

### 2. Bootstrap Defaults
Decision: default bootstrap_samples=1000, confidence=0.95, seed optional (user can set).  
Rationale: 1000 balances stability vs runtime (<2s for typical dataset).  
Alternatives: 10k samples (improved stability, higher cost) rejected for initial baseline.

### 3. Episode Identity Hash Fields
Decision: identity hash = SHA256(concat(sorted_keys(scenario_params JSON canonicalized) + seed + algo_id + scenario_id + repetition_index)).  
Rationale: Excludes runtime metrics/timings to remain stable pre-computation; ensures uniqueness across repetitions.  
Alternatives: include map hash (redundant—encoded inside scenario_params).  
Open: Document canonical JSON serialization (utf-8, no whitespace, sorted keys).

### 4. Collision Distance Threshold
Decision: Use distance < 0.35m for collision; near-miss threshold at 0.60m.  
Rationale: Aligns with pedestrian comfort literature ranges and existing internal metrics code assumptions.  
Alternatives: 0.30m (too strict; undercounts in crowded densities).  
Follow-up: Cite literature references (TODO: add to metrics spec).

### 5. Force Comfort Threshold
Decision: Use norm(force) > F_threshold where F_threshold = 2.0 (unitless model-specific).  
Rationale: Empirically discriminative for current fast-pysf calibration; triggers ~5–15% exposure for mid-density baseline.  
Alternatives: Adaptive percentile threshold (adds complexity & run coupling).  
Follow-up: Revisit after parameter calibration study.

### 6. Pareto Frontier Metric Pairs
Decision: (a) time_to_goal vs comfort_exposure, (b) collisions vs snqi.  
Rationale: Illustrate speed–comfort trade-off and safety–overall quality trade-off.  
Alternatives: force_exceedance vs near_misses (more granular, less intuitive to first-time readers).

### 7. Resume Manifest Invalidation
Decision: Manifest invalidated if file size differs OR mtime newer than manifest timestamp OR stored episodes_count != rescanned lines OR schema_version mismatch.  
Rationale: Low overhead safety checks; avoids full hash each run.  
Alternatives: full file SHA256 each run (higher cost for large files) reserved for debug mode.

### 8. SNQI Weight Provenance Fields
Decision: {weights_version, created_at, git_sha, baseline_stats_path, baseline_stats_hash, normalization_strategy, bootstrap_params(if any), components:list, weights:dict}.  
Rationale: Sufficient to reproduce weighting derivation.  
Alternatives: embed full baseline stats snapshot (redundant; file already tracked) -> rejected.

### 9. Bootstrap CI Display Format
Decision: Add keys mean_ci, median_ci, p95_ci with [low, high].  
Rationale: Simple, uniform, consistent with existing doc examples.  
Alternatives: nested objects (more verbose) or low/high separate columns (table inflation).

### 10. Scenario Diversity Minimum Set
Decision: Core set includes: low_density_straight, medium_density_straight, high_density_straight, bidirectional_hall, crossing_flow, bottleneck, obstacle_maze_sparse, obstacle_maze_dense, group_flow, adversarial_ped, corridor_narrow, mixed_density_wave.  
Rationale: Covers density gradient, directional complexity, obstacle-driven path planning, social grouping, and stress test (narrow corridor).  
Alternatives: splitting crossing vs intersection layout separately (redundant early).  
Follow-up: Add real-data calibrated variant later.

## Remaining Open Items
- Literature references for force & distance thresholds (ACTION: add citations in metrics_spec.md).
- Potential ORCA baseline inclusion (deferred decision gate before paper submission).

## Decisions Summary Table
| Topic | Decision | Status |
|-------|----------|--------|
| SNQI normalization | median/p95 baseline | Locked |
| Bootstrap defaults | 1000 / 0.95 | Locked |
| Episode identity | SHA256 over core scenario+seed+algo fields | Locked |
| Collision threshold | 0.35m | Locked |
| Near-miss threshold | 0.60m | Locked |
| Force comfort threshold | 2.0 | Locked |
| Pareto pairs | (time, comfort), (collisions, snqi) | Locked |
| Resume invalidation | size+mtime+count+schema | Locked |
| SNQI provenance fields | enumerated set | Locked |
| Core scenario list | 12 named scenarios | Locked |

## Rationale Themes
- Prefer stable, additive contracts to avoid migration churn.
- Emphasize interpretability (median/p95) over opaque normalization.
- Ensure each decision traces back to reproducibility, discrimination, or clarity.

## Risk Notes
- If discriminative power test fails, candidate additions: dynamic density ramp scenario, clustered groups variant.
- If bootstrap runtime becomes a bottleneck for large N, reduce samples adaptively (document).

---
End of Phase 0 Research.
