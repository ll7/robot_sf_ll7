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

## Scalar and systematic sensitivity (engine v1.2)

Physical scalar measurements cannot be nested objects, booleans, strings or
non-finite numbers. `extreme_measurements` reports
`error / nonfinite_or_malformed_measurement` for these values. Null remains
explicit missingness. Only these named structured metric records bypass scalar
validation: `deadlock_stall`, `distributional_disruption`, `force_quantiles`,
`force_sample_stats`, `metric_values`, `signal_metrics_evidence`,
`social_compliance`, and `social_mini_game`. These records come from
`robot_sf/benchmark/metrics.py`; structured admission does not excuse non-finite
contents. Named terminal-outcome and validity booleans are also accepted; a
boolean physical measurement is malformed. The source-admission layer still
rejects unexpected non-finite raw rows; documented sentinel normalization is
unchanged.

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
  recorded episodes per planner and one external planner control. It catches
  uniform per-planner shifts without requiring unavailable initial-state digests
  or equating planner-specific config hashes. Outcomes, intended planner behavior
  and different configurations can explain the differences; no causal inference
  follows. With no external control it stays unavailable.
- `horizon_consistency` compares `steps` (or `episode_steps`) with recorded
  `run_horizon` (or `horizon_steps`), termination reason and terminal outcome.
  A timeout before the runner horizon, steps beyond it, or an explicit reason
  contradicting the outcome flags. The simulator's
  `scenario_params.simulation_config.max_episode_steps` remains separate: a
  smaller simulator limit flags the timeout's runner/simulator mismatch and is
  retained as an explanation when steps equal that limit. Early successes and
  collisions do not violate a maximum horizon. Missing contracts stay unavailable
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
