# Adversarial Manifest Quality Metrics

Canonical metric vocabulary for generated adversarial scenario manifests.
These metrics are **diagnostic generator-quality signals only** — they must not be
treated as planner benchmark evidence or adversarial robustness claims.

## Evidence Boundary

```yaml
evidence_tier: analysis_only
claim_boundary: diagnostic generator-quality signals only; not planner benchmark evidence
```

## Metric Groups

### Pre-Execution Metrics (computed from manifests alone, no planner iteration)

| Metric | Numerator | Denominator | Units | Required Inputs | Status |
|---|---|---|---|---|---|
| `validity_rate` | `valid` manifests | total manifests | fraction | manifest status field | implemented |
| `degeneracy_rate` | `degenerate` manifests | total manifests | fraction | manifest status field | implemented |
| `novelty_from_seed` | unique control hashes | hashable manifests | fraction | control field comparison | implemented |
| `perturbation_distance` | mean L2 distance from reference | perturbed manifests | meters | reference manifest controls | implemented |
| `runner_compatibility_rate` | compatible manifests (total - parse_failures - degenerate) | total manifests | fraction | manifest parse + status + degenerate fields | implemented |
| `invalid_rate` | `invalid` manifests | total manifests | fraction | manifest status field | implemented |
| `duplicate_rate` | duplicate control hashes | hashable manifests | fraction | control field comparison | implemented |

### Planner-Response Metrics (require smoke-level planner iteration)

| Metric | Numerator | Denominator | Units | Required Inputs | Status |
|---|---|---|---|---|---|
| `failure_yield` | failure episodes | total episodes | fraction | planner smoke summary | implemented |
| `near_miss_yield` | near-miss episodes | total episodes | fraction | planner smoke summary | implemented |
| `low_progress_yield` | low-progress episodes (timeout + displacement < 0.25 m) | total episodes | fraction | planner smoke summary with termination + displacement | implemented |

### Naturalistic Prior Metrics

| Metric | Numerator | Denominator | Units | Required Inputs | Status |
|---|---|---|---|---|---|
| `naturalistic_prior_pass_rate` | passed prior checks | available checks | fraction | naturalistic prior annotations | implemented |
| `naturalistic_prior_fail_rate` | failed prior checks | available checks | fraction | naturalistic prior annotations | implemented |

### Not Yet Implemented (requires future data pipeline)

These metrics are proposed but depend on runner/planner execution infrastructure
not yet available in compact summary form:

- `fallback_or_degraded_rate`: fraction of planner runs that fell back or degraded
- Additional planner-response diagnostics: requires structured runner compatibility data

## Pre-Execution vs Planner-Response Separation

All metrics in the **Pre-Execution** group can be computed from manifest files alone
without running any planner. Metrics in the **Planner-Response** group require
smoke-level planner iteration results.

## Fail-Closed Exclusion

Per the repository's fail-closed benchmark policy (issue #691):
- Fallback or degraded planner runs must be excluded from planner-response metric denominators
  or explicitly caveated
- The `degeneracy_rate` and `runner_compatibility_rate` pre-execution metrics already
  track runner-fitness signals before planner iteration begins

## Source Code

- Implementation: `robot_sf/adversarial/manifest_quality.py`
- CLI entry point: `python -m robot_sf.adversarial.manifest_quality`
- Related issues: #2524 (manifest generation), #2599 (first applied smoke), #2920 (certification)
