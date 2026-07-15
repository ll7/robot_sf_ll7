# Issue #5579: MPC tuning-budget sensitivity

> Status: current diagnostic packet, 2026-07-14. No benchmark or paper-facing claim is
> established by this note.

This note makes the tuning-budget validity check for the two prediction-aware model-predictive
arms reproducible. The retained configuration history shows when each arm's configuration was
changed, but the repository does not retain a trial ledger or wall-clock accounting. The table
therefore records `unknown` runs and hours rather than inferring effort from commit counts:
[tuning effort table](evidence/issue_5579_mpc_tuning_budget_sensitivity_2026-07-14/tuning_effort_table.csv).

## Bounded analysis contract

The config-first packet is
[`configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml`](../../configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml).
It compares the two h600 prediction-MPC target arms:

* `prediction_mpc` from `configs/algos/prediction_mpc_cv.yaml`;
* `prediction_mpc_cbf` from `configs/algos/prediction_mpc_cv_cbf_collision_cone.yaml`.

Each target receives the same 20-point bounded grid subset over the three predeclared parameters
`max_linear_speed`, `horizon_steps`, and `pedestrian_safety_margin`. The incumbent point is
included once; the other points are one-factor, pairwise, and selected three-factor corners at
the declared levels. Four retained h600 hybrid candidate configs are run unchanged as the
incumbent band. The fixed paired scope is three named scenarios (`classic_bottleneck_medium`,
`classic_cross_trap_high`, and `francis2023_intersection_wait`) with seeds 111, 112, and 113.
The resulting bound is 44 arm/config rows and 396 episode rows.

The runner is
[`scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py`](../../scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py).
Its default `--check` path validates the packet without executing episodes. A bounded local run,
when explicitly requested, writes raw rows only under `output/` and derives compact JSON,
Markdown, and CSV reports. The analyzer lives in
[`robot_sf/benchmark/mpc_tuning_sensitivity.py`](../../robot_sf/benchmark/mpc_tuning_sensitivity.py)
and rejects missing paired rows, duplicate/unexpected keys, malformed typed outcomes, and
missing availability provenance.

## Read and claim boundary

The primary diagnostic outcome is route completion with no collision, using the canonical
`outcome.route_complete` and `outcome.collision_event` fields. The preregistered read compares
each target arm's best-found success rate with the full four-arm hybrid band:

* both best target rates below every incumbent rate: structural reading strengthens on this tested
  slice;
* both best target rates at or above every incumbent rate: budget-bound reading is supported on
  this tested slice;
* otherwise: mixed or inconclusive.

Only rows with explicit native/adapter/mixed execution, native/adapter readiness, available
status, and `benchmark_success=true` are eligible. Fallback, degraded, failed, partial, and
unavailable rows remain visible but cannot become success evidence; any such row blocks the
pre-registered read. This is a fixed-slice diagnostic, not a benchmark ranking, structural
superiority result, or paper/dissertation claim. It does not change metric semantics, roster
status, or paper-facing configuration.

## Validation and remaining action

The packet was validated on the local CPU host with:

```text
uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py --check
uv run pytest tests/benchmark/test_mpc_tuning_sensitivity_issue_5579.py -q
uv run ruff check robot_sf/benchmark/mpc_tuning_sensitivity.py scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py tests/benchmark/test_mpc_tuning_sensitivity_issue_5579.py
```

The bounded sensitivity campaign itself is intentionally not run in this PR. No Slurm/GPU
submission is authorized, and no full benchmark campaign or paper/dissertation edit is in scope.
If a maintainer requests the empirical slice, run the runner without `--check`, review its
fail-closed availability status, and promote only the compact derived report into the evidence
directory; do not promote raw episode JSONL.

Motivation and late validity guidance: [GitHub issue #5579](https://github.com/ll7/robot_sf_ll7/issues/5579).
