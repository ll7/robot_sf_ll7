# Issue #3574 - Per-Archetype Metrics And Mean-Matched Heterogeneity Effect

**Status:** diagnostic logging prerequisite.
**Evidence grade:** analysis primitives plus local trace logging tests; no ablation evidence.

## Plain-Language Summary

This issue now has the analysis helpers and the first logging input needed to compare pedestrian
archetypes, but it still has no mean-matched ablation runs and no heterogeneous-population effect
claim.

## Analysis Primitives (`heterogeneous_population_metrics.v1`)

`robot_sf/benchmark/heterogeneous_population_metrics.py` provides pure analysis primitives that
Issue #3574 needs before any heterogeneity effect can be claimed. Issue #3206 smoke evidence could
not isolate heterogeneity from a population-mean shift and could not compute per-archetype metrics.

- `cvar(values, alpha, higher_is_safer)`: Conditional Value at Risk (CVaR), the mean of the worst
  `alpha` tail. Higher-is-safer metrics use the lowest values; lower-is-safer metrics use the
  highest values.
- `per_archetype_metrics(observations, higher_is_safer, cvar_alpha)`: per-archetype `mean`,
  `worst_stratum`, `cvar`, and worst archetype by mean, so the worst-served archetype is not hidden
  by averaging.
- `mean_matched_heterogeneity_effect(homogeneous_mean, heterogeneous_mean,
  homogeneous_is_mean_matched)`: effect delta, flagged `isolated` only when the homogeneous arm uses
  the population mean (`theta_i = E[theta]`); otherwise `confounded_by_mean_shift`.
- `realized_distribution_audit(control_trace, metric_specs)`: configured-target vs. realized
  per-step distributions (overall and per-archetype) from one control trace, covering DoD item 5
  ("log configured vs. realized speed/clearance/yielding distributions"). Each `RealizedDistributionSpec`
  names a realized per-step field (e.g. `speed_m_s`) and an optional per-pedestrian configured label
  (e.g. `desired_speed_factor`). The realized side reports the distribution the #3206 smoke could not
  compute; the configured side is drawn one-per-pedestrian from trace labels. A numeric
  configured→realized mean shift is reported only when the caller sets `configured_is_comparable`
  (both sides same-unit), so a unitless factor vs. a metric observation is never reported as a real
  shift. Missing or non-finite fields fail closed as blockers; `summarize_distribution` supplies the
  `n/mean/std/min/max/p05/p50/p95` summary and rejects empty or non-finite samples.

## Per-Pedestrian Control Trace (`pedestrian-control-trace.v1`)

`run_map_episode(..., record_simulation_step_trace=True)` attaches
`algorithm_metadata.pedestrian_control_trace` when a scenario has explicit per-pedestrian archetype
metadata under `single_pedestrians`.

The trace records:

- `dt`, `pedestrian_count`, and `step_count`;
- one `pedestrians[]` entry per simulator pedestrian with `id`, `simulator_index`, and `archetype`;
- per-step `x_m`, `y_m`, `vx_m_s`, `vy_m_s`, and `speed_m_s`;
- optional per-step `force_x`, `force_y`, and `force_norm` when map-runner force recording is
  enabled.

The recorder fails closed on non-finite positions, forces, or derived speeds. It also rejects a
labeled control trace without per-pedestrian archetype metadata, because that payload cannot feed
the per-archetype harness honestly.

## Scope Boundary

This remains logging and pure analysis support only. It does not run mean-matched paired ablations,
does not submit Slurm jobs, and does not claim heterogeneous-population effects.

## Tests

- `tests/benchmark/test_heterogeneous_population_metrics.py`: Conditional Value at Risk tail
  direction and alpha validation, plus per-archetype aggregation for higher- and lower-is-safer
  metrics.
- `tests/benchmark/test_heterogeneous_population_distribution_audit.py`: realized-distribution audit
  summaries, per-archetype breakdown, comparable-shift computation, and fail-closed handling of
  missing realized/configured fields, empty traces, and duplicate spec names.
- `tests/benchmark/test_pedestrian_control_trace.py`: emitted control-trace shape, archetype
  detection, and fail-closed non-finite or missing-archetype handling.
- `tests/benchmark/test_map_runner_utils.py`: episode metadata integration for the map-runner trace
  seam.
