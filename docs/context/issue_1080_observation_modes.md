# Issue 1080 Observation-Mode Contracts

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1080>

## Scope

This change makes planner observation assumptions first-class benchmark metadata and adds a
fail-closed observation-mode override for map-runner and camera-ready campaign paths.

The override is a contract check, not a new observation builder. Existing environment observation
construction remains unchanged; unsupported planner/mode combinations fail before benchmark
episodes are written. This keeps parity experiments explicit without claiming that modality alone
explains planner differences.

## Modes

- `goal_state`: robot state plus route goal only.
- `socnav_state`: structured robot, goal, and pedestrian state.
- `headed_socnav_state`: social-navigation state with headed robot fields.
- `sensor_fusion_state`: configured sensor-fusion stack used by learned policies.
- `lidar_human_state` / `gst_human_state`: upstream learned-wrapper contracts.

The initial two-mode demonstration path is
`configs/benchmarks/observation_mode_goal_parity_smoke.yaml`, which runs the built-in `goal`
planner under both `goal_state` and `socnav_state` on a fixed planner-sanity scenario/seed.

## Validation Notes

Validated on 2026-05-09:

- `uv run pytest tests/benchmark/test_algorithm_metadata_contract.py tests/test_algorithm_metadata.py
  tests/benchmark/test_map_runner_utils.py tests/benchmark/test_map_runner_resume_identity.py
  tests/benchmark/test_map_runner_preflight_profiles.py tests/benchmark/test_camera_ready_campaign.py
  tests/contract/test_episode_schema.py -q` passed (`179 passed`).
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` passed (`3275 passed`, `18 skipped`,
  `3 warnings`). The changed-files coverage gate reported warning-only coverage below the
  aspirational goal for broad benchmark modules; no readiness failure was raised.
- `uv run python scripts/tools/run_camera_ready_benchmark.py --config
  configs/benchmarks/observation_mode_goal_parity_smoke.yaml --mode preflight --output-root
  output/benchmarks/issue_1080 --campaign-id observation_mode_goal_parity_preflight`
  generated validate/preview and matrix-summary artifacts with both goal-planner observation modes.
- `uv run python scripts/tools/run_camera_ready_benchmark.py --config
  configs/benchmarks/observation_mode_goal_parity_smoke.yaml --mode run --output-root
  output/benchmarks/issue_1080 --campaign-id observation_mode_goal_parity_run
  --skip-publication-bundle` completed successfully with two runs and two episodes.
- Episode JSONL inspection confirmed top-level `observation_mode`,
  `scenario_params.observation_mode`, and
  `algorithm_metadata.observation_spec.active_mode` were `goal_state` for one run and
  `socnav_state` for the other.
- `uv run robot_sf_bench run --matrix configs/scenarios/planner_sanity_matrix_v1.yaml --out
  output/benchmarks/issue_1080/direct_goal_socnav/episodes.jsonl --schema
  robot_sf/benchmark/schemas/episode.schema.v1.json --algo goal --benchmark-profile baseline-safe
  --observation-mode socnav_state --horizon 5 --workers 1 --no-video --no-resume` completed
  successfully with `total_jobs=3`, `written=3`, and `failed=0`.
- `uv run robot_sf_bench run --matrix configs/scenarios/planner_sanity_matrix_v1.yaml --out
  output/benchmarks/issue_1080/unsupported_override/episodes.jsonl --schema
  robot_sf/benchmark/schemas/episode.schema.v1.json --algo orca --benchmark-profile
  baseline-safe --socnav-missing-prereq-policy fallback --observation-mode goal_state --horizon 1
  --workers 1 --no-video --no-resume` exited with code `2`, confirming unsupported overrides
  fail closed through the CLI path.

The representative run emitted degenerate-baseline SNQI warnings because the smoke config has only
two goal-planner episodes. Those warnings do not affect the observation-mode contract proof.
