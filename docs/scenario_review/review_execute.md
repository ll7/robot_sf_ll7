# SREV-22 review execute

Bounded scenario-experiment executor (`srev22-review-execute`, version
`1.0.0`). It consumes the SREV-01 contracts (`component-request.v1` /
`component-result.v1` / `experiment-recipe.v1`) and produces isolated CPU
control/treatment outputs, measured activation traces, an attempt ledger, and
preserved receipts — usable standalone on fixture inputs without waiting for
the whole workbench.

## Evidence boundary

Diagnostic tooling only. An accepted recipe must explicitly mark its
`source_identity` as `evidence_boundary: diagnostic_only`,
`scientific_claim_allowed: false`, and
`dependent_family_status: standalone_fixture_only`. The fixture path also
requires `scenario_id: srev22-tiny-crossing` and the exact immutable
`source_ref` `{artifact_id: recipe-srev22-smoke, uri: recipe.json,
format: experiment-recipe.v1}`; that object must match the request's sole
source reference before a child is started. Fixture executions use the real
simulator path with a stateless goal-directed holonomic policy
(`simple_policy` family) and record survived / falsified / inconclusive pair
verdicts via the canonical counterfactual-pair evaluator. This is not campaign
or evidence-admission authority: no benchmark claim, planner/simulator
behavior change, training, or publication is authorized. Single episodes and
fixture patterns establish no population or causal conclusion.

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
attempt ledger in an existing output directory. Resume binds the request,
recipe, immutable config, and prior budget identity; a resumed config may
extend a prior budget but may never reduce it. Terminal candidates are not
retried, and a prior timeout/cancellation is settled as a failed candidate
instead of being silently re-executed.

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
Nested intervention parameters are also closed to finite scalar deltas. The
only child targets are the owned simulator episode and the test-only sleep
target; no import, command, script, or callable is accepted from JSON.

The recipe carries the hypothesis, `source_identity` (the supported fixture
scenario and exact source reference, plus the explicit diagnostic-only and
standalone-family markers), finite candidate interventions ordered
deterministically by (priority, stable ID), `control_conditions`, one driving measurement with an
`increase`/`decrease` expectation, budget, stop rules, and a
`preservation_destination` recorded as a retrieval URI rather than a local
write target.

The hard safety envelope is at most 3 candidates, 6 episode executions, a
600-second wall budget, and a 120-second per-execution timeout. The recipe
must carry all four supported stop rules: exhausted candidates, execution
budget exhausted, wall timeout, and control-fidelity failure blocking
treatment. Config values may narrow the recipe budget but may not exceed it.
Before a candidate starts, the executor reserves the complete control/treatment
pair; its wall preflight requires one full `per_execution_timeout_s` for each
reserved execution, so it never starts a pair that cannot fit in the remaining
execution or wall budget. Each execution boundary updates
`attempt-ledger.json`, including partial and failed outcomes.

Supported intervention factors are `single_pedestrian_speed_offset`
(executed) and `single_pedestrian_start_delay_offset`. Supported measurements
are `min_robot_ped_distance_m`, `ped_mean_speed_m_s`, `robot_goal_reached`,
and `ped_motion_onset_step`.

## Outputs and unavailable reasons

A `complete` result references all four artifacts; `partial`, `failed`,
`cancelled`, and `unavailable` results carry diagnostics and provenance only.
The attempt ledger remains the resumable partial record. A budget or wall
timeout produces `partial` when execution state exists; a user interruption
produces `cancelled`. A per-execution timeout terminates its owned child and
settles the drive as `partial`; it is never mislabeled as user cancellation.
Validation, output, or infrastructure failures are `failed`, while a recipe
whose selected candidates are all unsupported is `unavailable`. No incomplete
result is advertised with complete artifact references.

Each execution runs in one owned child process. The parent terminates and
reaps that child at every timeout, interruption, and normal return boundary.
The output directory must be a new relative directory beneath the caller's
base, or an existing non-symlink directory containing a valid attempt ledger
when `--resume` is used. Component artifacts are written atomically with
symlink checks and strict JSON; their manifest records request, recipe, full
effective-config, source, commit, and artifact digests. Paths with traversal,
absolute components, output symlinks, or unsafe preservation destinations are
rejected.

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
wall timing outside the logical digest. The `simple_policy` fixture path is
the only dependent planner family exercised here. This component does not
register sibling families, change benchmark coverage, or authorize any
scientific claim.
