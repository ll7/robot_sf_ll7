# Benchmark Auditor campaign scan

The Benchmark Auditor (BA-01) campaign scan is a bounded, offline pass over
already-recorded campaign results. It indexes every expected episode as
`readable`, `missing`, `duplicate`, `invalid`, or `unsupported`, then records
one strict BA-03 `Signal` attempt for every selected detector and inventory
entry. It never starts a simulator, replays an episode, invokes a shell or
network service, or creates a confirmed finding.

## Python route

```python
from robot_sf.analysis_workbench.audit_scan import scan_campaign

report = scan_campaign(
    "campaign.json",
    root="recorded-campaign-root",
    config={"expected_episode_ids": ["episode-1", "episode-2"]},
)
print(report.counts["coverage"])
print(report.counts["detectors"])
```

The source path is resolved under the admitted `root`; source references may
bind the observed SHA-256 digest. Source bytes are treated as untrusted input
and are not rewritten. JSONL rows that fail to decode remain visible as
`invalid` inventory entries, so a corrupt line cannot silently become a
coverage success.

## Registry and signals

`default_registry()` emits a versioned registry with required and optional
capabilities, cohort definition, parameters, units, and method provenance for
the fifteen deterministic detector families. Two robust statistical detectors
(`cohort_multivariate_outlier` and `trajectory_shape_outlier`) are advisory and
included by default; pass `include_advisory=False` or `--no-advisory` to omit
them. Registry and method versions are part of the cache key.

Each detector returns `flagged`, `clear`, `unavailable`, or `error`. A signal
contains its method/version, source identity when available, measured values,
thresholds, an optional simulation-time interval, reason code, and explicit
missingness. A flagged signal is a candidate review priority, not a confirmed
finding. Advisory scores are uncalibrated priorities, never probabilities.
The goal-adjacent timeout predicate follows `goal_adjacent_timeout.v1`; it
explicitly does not claim geometrical impossibility. In particular, a nearby
wall, force response, or observed limit cycle is diagnostic context rather
than a reachability proof.

## Scalar and systematic sensitivity (engine v1.3)

Physical scalar measurements cannot be nested objects, booleans, strings or
non-finite numbers. `extreme_measurements` reports
`error / nonfinite_or_malformed_measurement` for these values. Null remains
explicit missingness. These named structured metric records use their documented
numeric field schemas: `deadlock_stall`, `distributional_disruption`, `force_quantiles`,
`force_sample_stats`, `metric_values`, `signal_metrics_evidence`,
`social_compliance`, and `social_mini_game`. These records come from
the benchmark metric producers. Recorded physical scalar fields reject booleans,
strings, lists, Mappings, non-finite values, and impossible negative magnitudes
(including every force quantile). Count fields must be nonnegative integers;
fraction fields respect their documented bounds. Null/missing fields remain
missing. Status strings and arbitrary extension metadata are not inferred to be
measurements. Every named record must be a mapping or null, including flattened
row aliases. Shadowed named copies are also validated at their original paths.
`metric_values` accepts only the eight scalar names in
`paired_effect_metric_contract.REQUIRED_METRIC_NAMES`; unknown measurement keys are errors. `signal_metrics_evidence` declares only the string fields
`state` and `exclusion_reason`. Explicit `extension`, `metadata`, and `interpretation`
subtrees remain metadata, never measurements. Force quantiles admit only `q50`,
`q90`, and `q95`, must be nondecreasing across recorded values, and use the force
magnitude ceiling. `finite_samples` cannot exceed `raw_samples`. Negative force
magnitudes are flagged without absolute-value normalization.
Named terminal-outcome and validity booleans are accepted; a
boolean physical measurement is malformed. The source-admission layer still
rejects unexpected non-finite raw rows; documented sentinel normalization is
unchanged.

All cohort detectors require complete row admission. The scan separately indexes
rejected observations (invalid, unsupported, duplicate, or missing) before forming
admitted cohorts. Any target or control admission loss returns
`unavailable / cohort_admission_incomplete`, with `cohort_dropped_rows`,
`cohort_dropped_counts` per planner, `planner_admitted_sizes`, and
`planner_recorded_sizes`. Duplicate observations are counted individually.
A rejected observation with a missing or corrupt compatibility key is unassignable.
An unassignable rejected observation appears under `unassigned` and conservatively
gates every cell because its relevance cannot be ruled out. Assigned losses in
other compatibility cells do not contaminate complete cells. Direct detector calls
also check raw cohort admission; the scan supplies its authoritative counts through
the optional `detect(..., cohort_dropped_counts=...)` argument.

Within-outcome outliers are accompanied by independent default channels:

- `outcome_incidence` reports timeout, collision, other failure and total adverse
  rates per planner and exact scenario within a campaign. Its denominator includes
  all terminal outcomes, including successes; missing outcomes make the affected
  planner denominator unavailable. Equal-weight cross-planner median rates and
  excess rates are disclosed. An adverse episode remains a flagged candidate
  above the default absolute failure-rate floor of zero, even with one planner.
  This intentionally preserves every recorded failure subgroup. An excess above
  0.1 flags the planner's cell, including its successes. These are descriptive
  triage signals, not significance tests or proof that a failure is unexpected.
- `planner_cohort_shift` compares unconditioned per-planner scalar-feature medians
  against other planners' medians in the same scenario. It needs at least four
  valid episodes per planner/feature and one external planner control. Each
  feature requires **100% finite numeric scalar coverage** within its planner's
  recorded cell. `feature_sample_sizes` and `feature_missing_counts` disclose
  valid and missing/malformed counts for every planner/feature, alongside full
  `planner_sizes`. `feature_excluded_planners` names incomplete or too-small
  cells omitted from each comparison. A target feature with incomplete coverage
  is unavailable; a complete feature needs an admitted external control.
  With no admitted feature comparisons the signal is unavailable, never clear.
  Other complete features may still produce a signal with partial missingness.
  It catches
  uniform per-planner shifts without requiring unavailable initial-state digests
  or equating planner-specific config hashes. Outcomes, intended planner behavior
  and different configurations can explain the differences; no causal inference
  follows. With no external control it stays unavailable.
- `horizon_consistency` compares `steps` (or `episode_steps`) with recorded
  `run_horizon` (or `horizon_steps` or top-level `horizon`), the simulator's
  `max_episode_steps`, and
  `effective_budget_steps` when recorded, for **every outcome**, including
  unrecorded outcomes. Each `limit_comparisons` entry reports clear, flagged, or
  unavailable; missing fields appear in signal missingness without suppressing
  checks against available maxima. Termination reason and terminal outcome are
  also checked when available.
  A timeout must have steps exactly equal to recorded `effective_budget_steps`,
  or to the smallest available runner/simulator maximum when no budget is recorded.
  `timeout_budget_steps` discloses that comparison. The budget must equal
  `min(runner, simulator)` when both are present and cannot exceed either available
  maximum. Every limit declaration is validated; conflicting top-level and
  `scenario_params` aliases flag rather than silently selecting one. All original
  paths and values appear in `recorded_limit_values`. Steps beyond a maximum or
  an explicit reason contradicting the outcome also flag. A #9999 scheduled row
  with `horizon = scenario_params.run_horizon = effective_budget_steps = steps`
  and an equal or larger simulator maximum remains clear. The simulator's
  `scenario_params.simulation_config.max_episode_steps` remains separate: a
  smaller simulator limit flags the timeout's runner/simulator mismatch and is
  retained as an explanation when steps equal that limit. Early successes and
  collisions within every recorded maximum are consistent. Comparisons without
  steps or a recorded maximum stay unavailable
  and malformed counts are errors. This requires no trace and makes no goal-zone
  or deadlock diagnosis.

All robust scalar/trajectory comparisons use
`max(MAD, 1e-5 feature units, 1e-6 * abs(median))`, with the standard factor
1.4826. The floors suppress micro-unit serialization perturbations near zero
and one-part-per-million relative noise at large centers. They impose no
feature-specific physical tolerance or shared 26% acceptance band; for large
centers, a floor-dominated z threshold of 3.5 corresponds to about 0.000519%
relative deviation. Floors remain configurable and are carried in method
parameters; scores remain uncalibrated diagnostic priorities.

## Separate denominators

The report keeps these groups separate:

- `counts.coverage`: expected/indexed/readable/missing/duplicate/invalid/
  unsupported source rows and unexpected observations;
- `counts.detectors`: scheduled attempts, evaluable (`flagged` + `clear`),
  unavailable/error, candidate signals, and confirmed findings (always zero
  for this component);
- `counts.review`: human, agent, interval, full-episode, and finding-member
  coverage, planner/group/outcome strata, and ordinary-control gaps derived
  from records already present in the source;
- `counts.diagnostic`: recorded evidence, requested enrichment, and
  diagnostic executions. BA-01 only creates selective request records with
  `execution: "not_requested_here"`.

Re-running an identical scan yields the same logical report and cache key.
Changing source bytes, detector selection, detector registry/method version,
or configuration invalidates the cache. When an `AuditStore` is supplied,
the campaign audit and signals are committed through BA-03's existing
idempotent transaction boundary.

## CLI

```bash
python -m robot_sf.analysis_workbench.audit_scan \
  --input campaign.json \
  --base recorded-campaign-root \
  --output audit-output/report.json
```

The command emits machine-readable JSON and uses exit code `2` for malformed,
unsafe, unavailable, or colliding inputs/outputs. Output files are created
exclusively; an existing report is never replaced. Use `--detector ID` more
than once for selective detector execution, pass a JSON configuration file to
`--config` for explicit overrides, and use `--descriptor` for the offline
component contract.
