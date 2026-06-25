# Issue #3574 — Per-archetype metrics + mean-matched heterogeneity effect (increment)

**Status:** diagnostic / analysis. **Evidence grade:** idea-level analysis primitives.

## What this is

`robot_sf/benchmark/heterogeneous_population_metrics.py` provides the pure analysis primitives #3574
needs before any heterogeneity effect can be claimed. The #3206 smoke could not isolate heterogeneity
from a population-mean shift, nor compute per-archetype metrics. This adds both, mirroring the
accepted decision/analysis-layer pattern in this run (#3484, #3558, #3557, #3573).

## Primitives (`heterogeneous_population_metrics.v1`)

- `cvar(values, alpha, higher_is_safer)` — mean of the worst `alpha` tail (lowest values when higher
  is safer, e.g. clearance; highest when lower is safer, e.g. exposure).
- `per_archetype_metrics(observations, higher_is_safer, cvar_alpha)` — per-archetype `mean`,
  `worst_stratum`, and `cvar`, plus the worst archetype by mean, so the worst-served archetype is
  visible rather than averaged away.
- `mean_matched_heterogeneity_effect(homogeneous_mean, heterogeneous_mean, homogeneous_is_mean_matched)`
  — the heterogeneity effect, flagged `isolated` only when the homogeneous arm uses the population
  mean (`theta_i = E[theta]`); otherwise `confounded_by_mean_shift`.

## Scope boundary

Pure and side-effect free. Logging per-pedestrian control traces in the sim (the missing input infra),
and running the mean-matched paired ablation / response-law mixture sweep, need code+runs and are the
deliberate deferred follow-ups.

## Tests

`tests/benchmark/test_heterogeneous_population_metrics.py` (9 tests): CVaR tail direction + alpha
validation, per-archetype aggregation for higher- and lower-is-safer metrics, empty rejection, and
the isolated-vs-confounded mean-matched effect.

## Related

- Follows #3206 (heterogeneous archetype axis). Sibling analysis layers: #3573, #3558, #3557.
