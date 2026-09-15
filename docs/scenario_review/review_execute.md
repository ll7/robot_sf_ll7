# SREV-22 review execute

Bounded scenario-experiment executor (`srev22-review-execute`, version
`1.0.0`). It consumes the SREV-01 contracts (`component-request.v1` /
`component-result.v1` / `experiment-recipe.v1`) and produces isolated CPU
control/treatment outputs, measured activation traces, an attempt ledger, and
preserved receipts — usable standalone on fixture inputs without waiting for
the whole workbench.

## Evidence boundary

Diagnostic tooling only. Fixture executions exercise the real simulator path
with a stateless goal-directed holonomic policy (`simple_policy` family) and
record survived / falsified / inconclusive pair verdicts via the canonical
counterfactual-pair evaluator. This is not campaign or evidence-admission
authority: no benchmark claim, planner/simulator behavior change, training,
or publication is authorized. Single episodes and fixture patterns establish
no population or causal conclusion.

## Usage

```bash
uv run python -m robot_sf.analysis_workbench.review_execute \
  --input tests/fixtures/scenario_review/review_execute/request.json \
  --config tests/fixtures/scenario_review/review_execute/config.json \
  --output output/scenario_review/srev-22-smoke
```

`run(request)` executes the request; the CLI merges `--config` over the
request config, overrides the output directory with `--output`, and exits `0`
only when the result status is `complete`. Add `--resume` to continue from the
attempt ledger in an existing output directory (consumed budget is retained;
terminal attempts are not retried).

The component descriptor (`descriptor()`) declares the `bounded-execution`
required capability and the `execute-report.v1` / `attempt-ledger.v1` /
`activation-trace.v1` / `preservation-manifest.v1` output types. Request a
descriptor through the SREV-01 inspect surface or import
`robot_sf.analysis_workbench.review_execute` directly; central
discovery/entry-point registration stays with SREV-29.

## Inputs

The request config is a closed allowlist (validated, never arbitrary code):
`planner` (only `simple_policy`), `seed`, `horizon_steps` (1..600),
`robot_speed_m_s`, `max_candidates`, `max_executions`, `wall_timeout_s`,
`per_execution_timeout_s`, tolerances, optional `required_component_version`,
`intervention_parameters` per intervention ID, and the inline `recipe`.

The recipe carries the hypothesis, `source_identity` (requires a non-empty
`scenario_id`), finite candidate interventions ordered deterministically by
(priority, stable ID), `control_conditions`, one driving measurement with an
`increase`/`decrease` expectation, budget, stop rules, and a
`preservation_destination` recorded as the retrieval destination.

Supported intervention factors are `single_pedestrian_speed_offset`
(executed) and `single_pedestrian_start_delay_offset`. Supported measurements
are `min_robot_ped_distance_m`, `ped_mean_speed_m_s`, `robot_goal_reached`,
and `ped_motion_onset_step`.

## Outputs and unavailable reasons

A `complete` result references all four artifacts; `partial`, `failed`,
`cancelled`, and `unavailable` results carry diagnostics and provenance only.
Each execution runs in an owned child process (one concurrent local CPU
process): a per-execution timeout or cancel terminates the child, and wall /
execution budgets stop the drive with `partial` (some candidates terminal) or
`cancelled` (a child was killed).

Stable reason codes include `invalid_config`, `corrupt_recipe`,
`invalid_source_identity`, `output_collision`, `missing capabilities`,
`incompatible_version`, `unsupported_factor`, `unsupported_measurement`,
`control_fidelity_failure` (blocks treatment interpretation),
`execution_budget_exhausted`, `wall_timeout`, and
`per_execution_timeout`.

`single_pedestrian_start_delay_offset` candidates resolve to `unavailable`
with `intervention_not_executable`: the canonical single-pedestrian
start-delay release path holds pedestrian `max_speeds` at zero, so the
delayed pedestrian never moves on the fixture path, and this component
refuses to synthesize motion. Simulator behavior is owned outside this leaf;
see the portfolio follow-up for the release-path fix.

Control/treatment pairs share the scenario seed and are checked to differ
only in the intervened factor; activation (control motion present,
treatment-versus-control speed change beyond tolerance) is measured from
executed trajectories, never from requested config. Deterministic reruns
agree on verdicts, metrics, and trace bytes; ledgers additionally record
wall timing outside the logical digest.
