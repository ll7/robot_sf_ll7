# Issue #1904 Scenario Perturbation Criticality Pilot

Issue: [#1904](https://github.com/ll7/robot_sf_ll7/issues/1904)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Prerequisite: [PR #1903](https://github.com/ll7/robot_sf_ll7/pull/1903)

## Goal

Run the first tiny paired planner pilot over the #1858 scenario perturbation manifest after #1903
made eligible no-op and route-offset variants materializable as local scenario-matrix rows.

This is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbation_pilot/issue_1904_seed_sensitive_pilot_v1 \
  --pilot-output-dir output/scenario_perturbation_pilot/issue_1904_seed_sensitive_pilot_v1/results \
  --seed-limit 1 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --evidence-summary docs/context/evidence/issue_1904_scenario_perturbation_pilot_2026-05-31/summary.json
```

## Result

- Materialized variants: 6 included, 0 excluded.
- Planners: `goal`, `orca`.
- Pair rows: 6 completed pairs, 0 excluded pairs.
- Mean deltas over completed pairs:
  - success delta: `0.0000`
  - collision delta: `0.0000`
  - timeout delta: `0.0000`
  - min-distance delta: `+0.0037 m`

The first route-offset pilot is neutral at this scale. It proves the paired execution and compact
aggregation path, but it does not show route perturbation criticality for success, collision, or
timeout outcomes under the two cheap planners and one seed per source scenario.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1904_scenario_perturbation_pilot_2026-05-31/summary.json)

Ignored local outputs:

- materialized matrix and route overrides under the local pilot output directory
- raw planner episode JSONL
- local `summary.json` / `summary.md` generated beside the raw outputs

Fallback, degraded, invalid, missing, and failed rows are classified separately by the pilot
script and excluded from completed-pair means. This first run had no such rows after correcting the
pairing logic to compare only seeds observed for each no-op/perturbed source pair.

## Routing

Continue #1610 only with a larger, still bounded pilot if the next question needs sensitivity
power. The next slice should add either more seeds or a stronger local planner candidate before
spending effort on broader criticality statistics.
