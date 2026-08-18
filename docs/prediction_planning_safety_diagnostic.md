# Prediction, Planning, and Runtime-Safety Diagnostic

The `prediction_planning_safety.v1` contract keeps three questions separate: was the
pedestrian forecast covered, did the nominal planner produce a safe-margin decision, and did
runtime verification or contingency handling run? This is a fixture-only implementation and
research-diagnostic surface for [issue #7317](https://github.com/ll7/robot_sf_ll7/issues/7317).

## Run the deterministic fixture

```bash
uv run python scripts/validation/run_prediction_planning_safety_diagnostic.py \
  --output output/prediction_planning_safety/issue_7317_fixture.json
```

The report contains three paired same-seed fixture cases:

- good prediction with poor nominal planning;
- poor prediction with a safe fallback/contingency event;
- unavailable runtime verification.

It also records disjoint fit, calibration, and evaluation trace identities; horizon-specific
empirical coverage; hard-floor checks; runtime event counts; outcome fields that are unavailable;
and the canonical chance-constrained MPC configuration owner. The report is deterministic for a
fixed seed and validates against
[`prediction_planning_safety.schema.v1.json`](../robot_sf/benchmark/schemas/prediction_planning_safety.schema.v1.json).

## Interpretation boundary

The fixture report is smoke/diagnostic evidence. Its empirical coverage is split-specific and
does not establish a per-encounter, deployment, or real-world safety guarantee. The paired lane
rows do not establish navigation benefit or collision reduction. A held-out navigation campaign
must first satisfy the approval and preregistration boundary in [issue #6647](https://github.com/ll7/robot_sf_ll7/issues/6647).

The implementation reuses
[`split_conformal_radius`](../robot_sf/benchmark/uncertainty_safety.py) and records the existing
chance-constrained MPC builder as the planner integration owner. It does not silently reimplement
either primitive or execute a campaign.
