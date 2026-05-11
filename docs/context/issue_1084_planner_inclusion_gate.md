# Issue #1084 Planner Inclusion Gate

Issue: [#1084](https://github.com/ll7/robot_sf_ll7/issues/1084)

Status date: 2026-05-09

## Goal

Add a repeatable, mechanical inclusion check for planners being considered for promotion from
experimental or testing-only status into a promoted benchmark set. The gate is a review artifact,
not an automatic status mutator and not a leaderboard.

## Added Surface

Command:

```bash
uv run robot_sf_bench planner-inclusion-check \
  --algo orca \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --output-dir output/planner_inclusion/orca
```

Implementation:

- `robot_sf/benchmark/planner_inclusion.py`
- `robot_sf/benchmark/cli.py`
- `tests/benchmark/test_planner_inclusion.py`

The command writes:

- `<algo>_episodes.jsonl`
- `<algo>_inclusion_report.json`

The report schema version is `planner-inclusion-check.v1`.

## Gate Criteria

The default thresholds are intentionally minimal:

- every scheduled episode must produce a schema-valid record,
- aggregate metrics must be finite,
- runtime must stay within `--max-runtime-sec` (default `60`),
- at least `--min-episodes` records must be written,
- aggregate `success` mean must be at least `--min-success-rate` (default `0.5`),
- aggregate `collisions` mean must be at most `--max-collision-rate` (default `0.0`).

The decision is `pass` only when every check passes. Otherwise it is `revise`, and the report
contains explicit failure reasons.

## Validation Evidence

Focused tests:

```bash
uv run pytest tests/benchmark/test_planner_inclusion.py -q
```

Real passing planner case:

```bash
uv run robot_sf_bench --log-level WARNING planner-inclusion-check \
  --algo orca \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --output-dir output/benchmarks/issue_1084/inclusion_orca \
  --workers 1 \
  --max-runtime-sec 60
```

Observed: `decision=pass`, `episodes=3`, `success_rate=1.0`, `collision_rate=0.0`.

Real failing planner case:

```bash
uv run robot_sf_bench --log-level WARNING planner-inclusion-check \
  --algo goal \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --output-dir output/benchmarks/issue_1084/inclusion_goal \
  --workers 1 \
  --max-runtime-sec 60
```

Observed: `decision=revise`, with the failure reason
`minimum_success_rate: success rate is missing or below threshold`.

## Boundaries

This gate does not automatically edit planner readiness metadata, campaign configs, issue labels, or
Project fields. Promotion still requires a human-reviewed PR that cites the report and explains why
the planner should move into a promoted set.

The command currently checks one planner on one configured reference slice. Broader paper-facing or
full-matrix evidence remains a separate proof obligation.
