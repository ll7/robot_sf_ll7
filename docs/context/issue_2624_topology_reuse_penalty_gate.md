# Issue #2624 Topology Reuse-Penalty Paired Diagnostic Gate 2026-06-11

Issue: [#2624](https://github.com/ll7/robot_sf_ll7/issues/2624)
Status: diagnostic result; not benchmark evidence.

## Claim Boundary

This note records the paired diagnostic gate selected by
[issue_2621_topology_revision_hypothesis.md](issue_2621_topology_revision_hypothesis.md) for the
reuse-penalty candidate implemented in
[issue_2540_topology_reuse_penalty_diagnostic.md](issue_2540_topology_reuse_penalty_diagnostic.md).
It does not promote `topology_guided_hybrid_rule_v0_reuse_penalty` as a benchmark improvement.

## Run Contract

Both runs used the canonical double-bottleneck slice:

| Field | Value |
| --- | --- |
| Scenario | `classic_realworld_double_bottleneck_high` |
| Stage | `full_matrix` |
| Seed | `111` |
| Horizon | `160` |
| Max/min hypotheses | `3` / `2` |
| Candidate | `topology_guided_hybrid_rule_v0_reuse_penalty` |
| Comparator | `topology_guided_hybrid_rule_v0` |

Raw diagnostic outputs remain local under `output/diagnostics/issue2624_*`. The tracked compact
evidence is
[summary.json](evidence/issue_2624_topology_reuse_penalty_gate/summary.json).

## Result

```yaml
topology_reuse_penalty_paired_diagnostic:
  revision_name: primary_route_reuse_penalty_under_near_parity_alternatives
  candidate: topology_guided_hybrid_rule_v0_reuse_penalty
  comparator: topology_guided_hybrid_rule_v0
  diagnostic_status: diagnostic_complete
  required_fields_present: true
  non_primary_topology_command_influence_delta: 0
  route_progress_delta_m: -0.9514718767541766
  terminal_outcome_delta: unchanged_horizon_exhausted
  hypothesis_switch_count_delta: 0
  classification: revise
  claim_boundary: diagnostic_only
```

The reuse penalty activated, but it did not clear the #2621 acceptance gate. Non-primary
topology-command influence was preserved but not increased (`7` steps in both runs), max
route-progress delta regressed from `+0.1681 m` in the comparator to `-0.7833 m` in the candidate,
and both runs ended `horizon_exhausted` without success. The classification is therefore `revise`,
not `continue`.

## Comparison

| Metric | Candidate | Comparator | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Reuse-penalty applied steps | 9 | 0 | +9 | Mechanism activated on this slice. |
| Eligible near-parity alternative steps | 55 | 56 | -1 | Eligibility was similar across the pair. |
| Non-primary topology-command steps | 7 | 7 | 0 | Influence was preserved, not increased. |
| Primary topology-command steps | 24 | 26 | -2 | Primary influence fell slightly. |
| Topology-command steps | 31 | 33 | -2 | Candidate used topology commands slightly less often. |
| Max route-progress delta (m) | -0.7833 | 0.1681 | -0.9515 | Candidate regressed against the comparator. |
| Hypothesis switches | 3 | 3 | 0 | No switch-volatility regression. |
| Terminal outcome | horizon_exhausted | horizon_exhausted | unchanged | No terminal improvement. |

The planner-level selected-hypothesis counts did shift away from the primary route: candidate
`primary_route` selections fell from `56` to `49`, and non-primary selections increased from `40`
to `47`. That movement is diagnostic evidence that the penalty can change route selection, but it
does not translate into stronger command influence, route progress, or terminal behavior on this
gate.

## Required Field Status

The compact evidence preserves the required summary fields: diagnostic status, topology status
counts, topology-command influence counts, reuse-penalty aggregates, route-progress delta,
hypothesis switch count, terminal outcome, and selected-hypothesis count summaries. The trace also
contains step-level `reuse_penalty_applied`, `reuse_penalty_reason`,
`recent_primary_selection_count`, and `eligible_near_parity_alternative_exists` fields. The raw
step traces are intentionally not tracked.

The issue template names `route_selector_selected_hypothesis_counts` and
`selected_row_near_parity_gate_reasons`; the current diagnostic output exposes equivalent selected
hypothesis counts under `planner_summary.topology_guided.selected_hypothesis_counts` and
near-parity reasons at step level rather than as those exact top-level summary keys. This is enough
to classify the gate, but it leaves a report-shape cleanup opportunity for a future diagnostics
quality issue.

## Validation

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0_reuse_penalty \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2624_reuse_penalty
```

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2624_baseline_comparator
```

Both commands completed with `diagnostic_status: diagnostic_complete`. The only runtime warning
observed was the existing `uni_campus_big.svg` invalid polygon warning from SVG parsing; this did
not stop the diagnostic.
