# Issue #7602: matched forecast preparation

This packet records preparation evidence for a small, provenance-complete
oracle-versus-ego forecasting design. It is diagnostic-only: it does not train
a model, run a forecast benchmark campaign, integrate a planner, change robot
observation semantics, make a scientific claim, or support real-world
forecasting.

The current canonical `simulation_trace_export.v1` inputs expose oracle
pedestrian state, robot state, and planner metadata. They do not expose a
canonical tracked-agent/ego observation field. The packet therefore emits the
paired `ego_observation` rows as explicitly `not_available` with robot context
only. It never copies oracle pedestrian state into an ego row and fails closed
if a future source starts declaring an unowned tracked-agent field.

## Selected source sample

The deterministic sample contains three scenario families and three planner
identities:

| Source | Family | Planner | Cutoff |
| --- | --- | --- | --- |
| `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/bottleneck_motion_rich_fixture.json` | bottleneck | `hybrid_rule_v0_minimal` | frame 5 |
| `docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json` | crossing proxy | `orca` | frame 2 |
| `docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json` | head-on corridor | `ammv_social_force` | frame 5 |

Every selected cutoff emits one oracle row and one ego row for the same
source-lineage ID, cutoff/target frame and time, actor, and horizon. The
row-level ledger records field owner, unit, time role, robot availability, and
future-target status. Supervision targets are future-labelled fields and are
never part of an ego input.

## Split and diagnostic policy

Groups are `scenario_family:scenario_id:seed:episode_id` and are assigned
deterministically to `train`, `validation`, and `test` in sorted group order.
The validator rejects group leakage and rejects an exact near-duplicate
trajectory fingerprint in more than one split. The fingerprint rounds trace
positions and velocities to two decimals; a future training design would need
a separate RMS-aligned near-duplicate review.

Runtime and memory entries for stationary, constant velocity, constant
acceleration, Kalman, and Social Force are analytic
`preparation_estimate_not_measured` records. They are not runtime measurements
and do not authorize a campaign. The dependency comparison records reuse/no-add
decisions for NumPy, optional SciPy, the local MIT `fast-pysf` implementation,
the Apache-2.0 `pyrvo2` companion, and an unadopted external Social Force
package. No dependency or planner integration is added by this issue.

The trace-backed false-reassurance case uses the crossing-proxy source: a
stationary prediction has `ADE=0` and `FDE=0`, while robot-pedestrian clearance
is below the packet's 0.8 m diagnostic reference. This is an analytic
counterexample showing why average displacement error (ADE) and final displacement
error (FDE) alone are insufficient for interaction
diagnostics, not a safety or performance claim.

## Public-safe files and verification

- `forecast_preparation_packet.json` contains the paired rows, leakage ledger,
  source lineage, split assignments, baseline estimates, dependency comparison,
  false-reassurance case, and SHA-256 coverage metadata.
- `checksums.sha256` covers the packet, selected inputs, owner module, checker,
  focused tests, README, and cited dependency/license evidence.
- `robot_sf/benchmark/forecast/forecast_preparation.py` is the canonical owner;
  `scripts/validation/check_issue_7602_forecast_preparation.py` is the
  issue-scoped builder/checker.

The exact side-effect-free check command is:

```text
scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/check_issue_7602_forecast_preparation.py --check --packet docs/context/evidence/issue_7399_forecast_preparation/forecast_preparation_packet.json
```

Passing this command proves only that the design packet is internally
consistent, leakage-audited, provenance-bound, and hash-covered. It is not
forecasting-performance evidence. The ego stratum remains unavailable until a
canonical observation adapter and its separate observation-contract review
exist.
