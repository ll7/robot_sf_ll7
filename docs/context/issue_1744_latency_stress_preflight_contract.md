# Issue #1744 Latency Stress Preflight Contract

Date: 2026-05-30

Related issues: #1744, #1629, #1556

## Scope

Issue #1744 adds a preflight/config contract for learned-policy latency stress without claiming a
completed latency benchmark campaign. The contract separates:

- observation delay, via `observation_delay_steps`,
- action/actuation delay, via `action_delay_steps`,
- planner update hold, via `planner_update_mode` and `planner_update_period_steps`,
- wall-clock inference reliability, via `inference_timeout_ms`.

The profile is intentionally synthetic-only and preflight/provenance-only. It records what a
latency stress row is meant to exercise, but does not by itself implement delayed observations,
planner throttling, or timeout enforcement inside the controller loop.

## Implemented Contract

The new `latency_stress_profile` YAML block is parsed and validated by the camera-ready campaign
loader and map runner. Preflight artifacts, campaign manifests, map-runner summaries, episode
identity payloads, and episode timing/metadata can carry the profile.

At `dt=0.1`, the #1556 diagnostic anchor now expresses:

- `observation_delay_steps: 1` = 100 ms,
- `action_delay_steps: 1` = 100 ms,
- `planner_update_period_steps: 2` = 200 ms hold-last planner updates,
- `inference_timeout_ms: 200.0` as an inference reliability threshold, not simulated control delay.

Rows with `action_delay_steps > 0` require differential-drive-only scenarios. Non-DD rows fail
closed during map-runner preflight with `status=skipped` and `compatibility_status=incompatible`.

## Non-Success Boundary

The profile requires these statuses to remain non-success outcomes:

- `fallback`
- `degraded`
- `timeout`
- `not_available`
- `failed`

Fallback, degraded, timeout, and unavailable rows are therefore metadata about limitations or
exclusions, not benchmark-strengthening evidence.

## Validation

Targeted validation for this change should include:

```bash
uv run pytest -q tests/benchmark/test_latency_stress.py \
  tests/benchmark/test_issue_1556_amv_actuation_stress_slice.py \
  tests/benchmark/test_camera_ready_campaign.py::test_load_campaign_config_rejects_malformed_latency_stress_profile \
  tests/benchmark/test_camera_ready_campaign.py::test_load_campaign_config_rejects_invalid_latency_stress_scope \
  tests/benchmark/test_camera_ready_campaign.py::test_prepare_campaign_preflight_resolves_synthetic_actuation_slice_metadata \
  tests/benchmark/test_camera_ready_campaign.py::test_run_campaign_writes_synthetic_actuation_artifacts \
  tests/benchmark/test_map_runner_utils.py::test_run_map_batch_fail_closed_when_latency_action_delay_is_not_diff_drive \
  tests/benchmark/test_map_runner_resume_identity.py::test_scenario_identity_includes_latency_stress_profile
```

The canonical preflight smoke for the anchor config is:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml \
  --mode preflight \
  --label issue1744-latency-preflight \
  --output-root output/benchmarks/issue1744 \
  --log-level WARNING
```
