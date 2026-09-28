# Evidence-linked diagnostic report (issue #7387)

First typed diagnostic leaf: thin adapters over existing planner traces plus a
deterministic evidence-linked report. Observations and symptoms only —
`hypotheses` stays empty because a causal claim needs a separately controlled
test (out of scope). Reuses `mechanism_trace.v1` rows, the failure-diagnosis
taxonomy vocabularies, and the immutable artifact format. Creates no competing
diagnosis framework.

## Two-planner inventory

Query it in code: `robot_sf.benchmark.diagnostic_report.TWO_PLANNER_FIELD_INVENTORY`.

- `orca_residual` (owner `robot_sf/benchmark/mechanism_trace.py::emit_orca_residual_row`,
  emitter `scripts/tools/emit_orca_residual_mechanism_trace.py`): observation
  version, candidate table, cost terms/scales, commanded control, and saturation
  flag are planner-visible (`supported`); executed control is `simulator_only`;
  collision outcome is `post_hoc` augmentation and never triggers a finding.
- `nominal_social_force` (owner `robot_sf/sim/fast_pysf_wrapper.py::diagnostics`):
  only commanded/executed control are supported; observation version, candidate
  table, and cost scales are `unsupported` by design — the reactive force kernel
  computes no candidate costs, so no table is invented. Missing fields are a
  supported result and never invalidate unrelated findings.

## Adapters

- `adapt_orca_residual_row(mechanism_row, ...)`: reshapes owned
  `mechanism_trace.v1` fields; timing/candidate/cost fields stay missing unless
  the caller supplies them from planner-visible sources.
- `adapt_nominal_social_force_diagnostics(sf_diagnostics, ...)`: maps the
  reactive kernel diagnostics; candidate/cost/version fields stay missing.

## Rules (deterministic, versioned)

- `obs_age_mismatch.v1`: measured `consumed_observation_version <
  observation_version` reports stale consumption.
- `cost_scale_inconsistency.v1`: rescaled candidate totals change the winner
  versus the recorded order — a demonstrable order change, not a bare scale
  difference.
- `command_execution_mismatch.v1`: commanded-vs-executed gap above tolerance
  with a recorded saturation flag.

Every finding binds artifact digest, source row, step/time, rule version, and
permitted evidence status. Corrupt provenance, malformed timestamps/units, and
non-finite values fail closed with deterministic reasons. Text fields are data,
never executed; source inputs are never mutated. Independent input-row order
does not change canonical report content; temporal order is preserved.

## Reproducible example

```python
from robot_sf.benchmark.diagnostic_report import (
    adapt_orca_residual_row,
    diagnose_diagnostic_rows,
)

row = adapt_orca_residual_row(
    {"activation_step": 0, "selected_command": [1.0, 0.0]},
    source_digest="sha256:example",
    source_row="mechanism_trace.v1.example.json:row:0",
    timing={
        "sim_time_s": 0.0,
        "observation_source_time_s": 0.0,
        "planner_available_time_s": 0.0,
    },
)
report = diagnose_diagnostic_rows([row])
assert report["schema_version"] == "diagnostic_report.v1"
print(report["report_digest"], len(report["observations"]), len(report["abstentions"]))
```

## Caveats

Observations cite measured trace values only. Collision outcomes never trigger
mechanisms. Unsupported narratives are rejected or qualified, never promoted.
