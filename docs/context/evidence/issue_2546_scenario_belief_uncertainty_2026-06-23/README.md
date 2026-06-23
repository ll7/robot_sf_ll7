# Issue #2546 ScenarioBelief Uncertainty Diagnostic

- **claim_boundary**: `diagnostic_only`
- **evidence_tier**: `stress`
- **paper_grade**: `false`
- **not benchmark evidence**: this is a bounded stress probe, not a benchmark,
  safety, perception, trained-policy, or planner-performance result.
- **date**: 2026-06-23
- **commit SHA (run HEAD)**: `fedee58afe8c5834f4e6ccb1179cdc00a8354606`
- **seed**: `2546`

## Command

```bash
uv run python scripts/analysis/run_scenario_belief_uncertainty_diagnostic_issue_2546.py \
  --seed 2546 --print-summary
```

Artifacts (durable, git-ignored): `output/issue_2546_belief_uncertainty/`.
Promoted JSON report here: `diagnostic_report.json`.

## What this answers

One bounded question on a tiny fixed crossing scenario (one corridor pedestrian
the planner reacts to, one off-corridor pedestrian): does ScenarioBelief
uncertainty change policy-observation semantics, the `stream_gap` planner
decision, or safety-relevant failure predicates — and where does uncertainty
consumption fail closed?

Five belief conditions are compared against the oracle baseline:

| condition | perturbation | planner linear command | mechanism |
| --- | --- | --- | --- |
| `oracle` | none (zero cov, full visibility, existence 1.0, single class) | 0.0 (wait) | corridor agent present → planner waits |
| `visibility_limited` | corridor agent OCCLUDED | 0.95 (commit) | agent leaves visibility projection → corridor looks clear |
| `covariance_inflated` | position variance 4.0 | 0.95 (commit) | `stream_gap` uncertainty gate drops the agent (variance > 1.0) |
| `class_probability` | class spread to {ped 0.3, cyclist 0.4, unknown 0.3} | 0.95 (commit) | gate drops the agent (class prob < 0.5) |
| `existence_degraded` | existence 0.2 | 0.95 (commit) | gate drops the agent (existence < 0.5) |

## Result (difference found, not null)

All four uncertain conditions flip the consuming `stream_gap` planner from
**wait** (linear `0.0`) to **commit** (linear `0.95`) versus oracle, and change
the `is_waiting` / `is_committing` / `linear_speed` failure predicates. Three of
the four also trip the planner's opt-in uncertainty gate (`dropped_count == 1`);
the `visibility_limited` case removes the agent earlier, at the visibility
projection step, so the gate sees nothing to drop. This is a genuine,
mechanistically explained behavior difference — not a null result.

The unsupported planner key (`predictive_planner_v2`) **fails closed** in every
condition with reason `unsupported_uncertainty_planner` and never consumes the
uncertainty sidecar, as required.

## Follow-up decision: **continue**

The representation carries decision-relevant uncertainty that reaches a
consuming planner and shifts its decision, while unsupported consumption fails
closed. The next bounded step is to wire a runtime ScenarioBelief producer into
planner selection and rerun as an end-to-end stress run. This remains
diagnostic-only.

## Limitations

- Synthetic fixed scenario set; not real sensor data.
- No perception model and no trained policy.
- Planner differences are diagnostic decision-shifts, not validated safety or
  performance improvements; the WAIT→COMMIT flip is *expected* when uncertain
  agents are dropped and must not be read as a safety gain.
- Failure predicates derive from the single-step planner command and the
  uncertainty gate, not from an episode-level safety evaluation.
- `stream_gap` is the only uncertainty-consuming planner; all other planner keys
  fail closed by design.
