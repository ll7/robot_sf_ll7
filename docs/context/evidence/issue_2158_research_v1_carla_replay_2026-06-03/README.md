# Issue #2158 Research-V1 CARLA Replay Diagnostic Pack

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2158>

This pack reruns the existing CARLA replay diagnostics on three tracked prior
CARLA evidence bundles. It is transfer-boundary diagnostic evidence only. It is
not CARLA parity proof, simulator equivalence proof, safety certification, or
benchmark-success evidence.

## Generation Commands

```bash
uv run python scripts/carla_bridge/diagnose_replay_semantics.py \
  --robot-sf docs/context/evidence/issue_1467_carla_replay_metrics_2026-05-24/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1467_carla_replay_metrics_2026-05-24/live_replay_native_metric_probe.json \
  --output-dir docs/context/evidence/issue_2158_research_v1_carla_replay_2026-06-03/native_metric_probe

uv run python scripts/carla_bridge/diagnose_replay_semantics.py \
  --robot-sf docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/live_replay_native_probe.json \
  --output-dir docs/context/evidence/issue_2158_research_v1_carla_replay_2026-06-03/native_spawn_probe

uv run python scripts/carla_bridge/diagnose_replay_semantics.py \
  --robot-sf docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/t0/pr_promoted_planner_smoke.json \
  --carla docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/live_replay_pr_promoted.json \
  --output-dir docs/context/evidence/issue_2158_research_v1_carla_replay_2026-06-03/pr_promoted_fixture
```

## Result

- Cases: `native_metric_probe`, `native_spawn_probe`, `pr_promoted_fixture`.
- Aggregate row statuses: `available=12`, `degraded=10`, `not_available=35`,
  `unsupported=6`.
- Unsupported semantics in every case: `sensor_perception_replay` and
  `broad_simulator_equivalence`.
- The `pr_promoted_fixture` case is `degraded`; the other two cases are
  `available` diagnostic reports, with many missing comparability surfaces.

## Interpretation

This pack moves `research-v1.amv.transfer_boundary` from blocked/no-current-pack
to diagnostic. It does not move the claim to candidate or paper-ready evidence:
map metadata, static geometry support, timing synchronization, pedestrian replay,
and broad simulator equivalence remain unavailable, degraded, or unsupported
depending on the case.

## Files

- `research_v1_carla_replay_summary.json`: aggregate case summary and source
  paths.
- `<case>/carla_replay_diagnostics.json`: machine-readable diagnostics.
- `<case>/carla_replay_diagnostics.md`: human-readable diagnostics.
- `<case>/carla_capability_matrix.csv`: capability rows.
- `<case>/unsupported_semantics.csv`: unsupported semantic rows.
- `SHA256SUMS`: checksums for promoted compact artifacts.
