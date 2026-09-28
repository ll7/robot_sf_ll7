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
the twelve deterministic detector families. Two robust statistical detectors
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
