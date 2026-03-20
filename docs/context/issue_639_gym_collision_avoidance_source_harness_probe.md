# Issue 639 gym-collision-avoidance Source Harness Probe

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#605` gym-collision-avoidance reference assessment
- `robot_sf_ll7#632` Python-RVO2 prototype
- `robot_sf_ll7#639` Prototype fail-fast gym-collision-avoidance source-harness reproduction

## Goal

Determine whether one upstream `gym-collision-avoidance` learned-policy path can still run in its
own harness here before any Robot SF wrapper work is attempted.

This issue is intentionally fail-fast:

- attempt the upstream source entrypoint directly,
- attempt a bundled learned-policy import directly,
- record the first real blocker,
- and only recommend wrapper work if the upstream source harness actually runs.

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/gym_collision_avoidance_source_harness_probe/report.json`
- Markdown report:
  `output/benchmarks/external/gym_collision_avoidance_source_harness_probe/report.md`

Generated with:

```bash
uv run python scripts/tools/probe_gym_collision_avoidance_source_harness.py \
  --repo-root output/repos/gym-collision-avoidance \
  --output-json output/benchmarks/external/gym_collision_avoidance_source_harness_probe/report.json \
  --output-md output/benchmarks/external/gym_collision_avoidance_source_harness_probe/report.md
```

## Commands attempted

The probe attempts three upstream-facing commands from the checked-out repository root:

1. upstream example entrypoint

```bash
uv run python output/repos/gym-collision-avoidance/gym_collision_avoidance/experiments/src/example.py
```

2. bundled learned-policy import and checkpoint initialization

```bash
uv run python -c "from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy; policy = GA3CCADRLPolicy(); policy.initialize_network(); print('ga3c_ready')"
```

3. upstream example test collection

```bash
uv run python -m pytest -q output/repos/gym-collision-avoidance/gym_collision_avoidance/tests/test_collision_avoidance.py -k test_example_script
```

## Current result

Verdict: `source harness blocked`

Primary blocker:

- missing python dependency: `gym`

Observed failure stages:

- `upstream_example`
  - fails on `import gym`
- `learned_policy_import`
  - fails before the GA3C-CADRL policy can initialize because
    `gym_collision_avoidance/__init__.py` imports `gym.envs.registration`
- `pytest_example_collection`
  - fails during test collection for the same package-import reason

Interpretation:

- the block is currently at the package/runtime layer, not at planner semantics,
- so this issue does **not** yet prove that GA3C-CADRL or CADRL can execute reproducibly here,
- and any Robot SF wrapper work would currently be premature.

## Extracted source contract

Even though runtime reproduction is blocked, the upstream source still exposes a clear contract for
the learned-policy path.

Observed from the checked-out source:

- example default policies:
  - `['learning', 'GA3C_CADRL']`
- observation states in policy-facing dict:
  - `is_learning`
  - `num_other_agents`
  - `dist_to_goal`
  - `heading_ego_frame`
  - `pref_speed`
  - `radius`
  - `other_agents_states`
- states excluded from the learned policy:
  - `is_learning`
- observation encoding:
  - flatten the dict observation in `Config.STATES_IN_OBS` order
  - skip `Config.STATES_NOT_USED_IN_POLICY`
- default timestep:
  - `0.2` seconds
- default max other agents observed:
  - `3`
- learned policy family attempted:
  - `GA3C_CADRL`
- learned action contract:
  - `speed_delta_heading`
- discrete learned action count:
  - `11`
- checkpoint family:
  - `GA3C_CADRL/checkpoints/IROS18/network_01900000`
- kinematics:
  - `unicycle_like_speed_plus_delta_heading`

Why this matters:

- this family is still materially closer to Robot SF kinematics than holonomic crowd-navigation
  repos,
- but the source harness has to run first or the wrapper would only be guessing at parity.

## Recommendation

Recommendation: `wrapper not yet justified`

Next step if this family stays interesting:

1. reproduce the upstream harness in an isolated side environment that deliberately restores the
   legacy `gym` dependency,
2. rerun the same probe there without changing planner logic,
3. only open a Robot SF wrapper/parity issue if the source harness becomes reproducible.

Not recommended now:

- benchmark-side adapter work,
- silent `gym` to `gymnasium` patching inside Robot SF,
- family-level claims for CADRL or GA3C-CADRL based on the current local runtime.
