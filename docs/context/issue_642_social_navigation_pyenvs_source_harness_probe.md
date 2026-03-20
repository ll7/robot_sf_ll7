# Issue 642 Social-Navigation-PyEnvs Source Harness Probe

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#629` Planner zoo research prompt
- `robot_sf_ll7#639` gym-collision-avoidance source-harness probe
- `robot_sf_ll7#642` Prototype fail-fast Social-Navigation-PyEnvs source-harness reproduction

## Goal

Determine whether the upstream `Social-Navigation-PyEnvs` repository can run enough of its own
source harness here to justify planner-wrapper work in `robot_sf_ll7`.

This issue is intentionally proof-first:

- validate that the upstream package imports,
- validate that the upstream simulator core boots,
- validate at least one upstream planner path (`orca`) in the source simulator,
- record the first real blocker for full env creation,
- and only then decide whether wrapper work is worth pursuing.

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/social_navigation_pyenvs_source_harness_probe/report.json`
- Markdown report:
  `output/benchmarks/external/social_navigation_pyenvs_source_harness_probe/report.md`
- wrapper JSON report:
  `output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.json`
- wrapper Markdown report:
  `output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.md`

Generated with:

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_source_harness.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --output-json output/benchmarks/external/social_navigation_pyenvs_source_harness_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_source_harness_probe/report.md
```

Wrapper proof generated with:

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_orca_wrapper.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --output-json output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_orca_wrapper_probe/report.md
```

## Commands attempted

The probe runs these upstream-facing checks from the checked-out repository root:

1. package import

```bash
uv run python -c "import gymnasium, social_gym; print('import_ok')"
```

2. upstream Gymnasium env creation in the current runtime

```bash
uv run python -c "import social_gym, gymnasium as gym; env = gym.make('SocialGym-v0'); print(type(env).__name__)"
```

3. simulator-core boot with a narrow extra dependency injection

```bash
uv run --with socialforce python -c "from social_gym.social_nav_sim import SocialNavSim; ..."
```

4. upstream Gymnasium env creation with `socialforce` injected

```bash
uv run --with socialforce python -c "import social_gym, gymnasium as gym; env = gym.make('SocialGym-v0'); print(type(env).__name__)"
```

5. non-trainable policy registry import

```bash
uv run --with socialforce python -c "from crowd_nav.policy_no_train.policy_factory import policy_factory; print(sorted(policy_factory.keys()))"
```

6. robot-side ORCA motion-model activation in the upstream simulator

```bash
uv run --with socialforce python -c "from social_gym.social_nav_sim import SocialNavSim; ... sim.set_robot_policy(policy_name='orca', runge_kutta=False); ..."
```

7. pinned-requirements resolution probe

```bash
uv run --with socialforce --with-requirements requirements.txt python -c "print('requirements_ok')"
```

8. shimmed non-trainable ORCA reset-and-step proof

```bash
uv run --with socialforce python -c "import configparser; import numpy as np; np.NaN = np.nan; ..."
```

## Current result

Verdict: `source harness partially reproducible`

Observed positive signals:

- upstream package import succeeds in the current runtime,
- upstream simulator core boots once `socialforce` is injected ephemerally,
- upstream non-trainable policy registry imports successfully,
- upstream robot-side `orca` motion-model activation succeeds without patching local source.
- a narrow local compatibility shim is enough to reset and step the upstream non-trainable `orca`
  path once.
- a thin wrapper around upstream `crowd_nav.policy_no_train.orca.ORCA` can drive a real Robot SF
  step loop.

Primary blocker for full env creation:

- `upstream NumPy 2 incompatibility: np.NaN`

Additional blocker for strict pinned-runtime reproduction:

- pinned requirements are incompatible with the current Python 3.13 runtime because the upstream
  `torch==2.1.1` pin has no `cp313` wheels.

Interpretation:

- this repo is materially closer to runnable in our stack than `gym-collision-avoidance`,
- the problem is no longer generic feasibility,
- the problem is a bounded compatibility layer:
  - missing `socialforce`,
  - `np.NaN` under NumPy 2,
  - and legacy learned-stack pins for a fully faithful reproduction.
- for non-trainable planners, those compatibility boundaries are now concrete enough to justify a
  prototype wrapper path.

## Extracted source contract

Observed directly from the checked-out source:

- declared `gymnasium` version:
  - `0.29.1`
- declared `numpy` version:
  - `1.26.1`
- declared `torch` version:
  - `2.1.1`
- packaged learned weights:
  - not found in the checked-out repository
- non-trainable policy families available:
  - `bp`
  - `ssp`
  - `orca`
  - `socialforce`
  - `sfm_*`
  - `hsfm_*`
- robot actuation:
  - `differential_drive`
- robot action handling:
  - accepts both holonomic (`ActionXY`) and nonholonomic (`ActionRot`) policy outputs
- upstream ORCA semantics:
  - goal-vector preferred velocity inside the upstream simulator
- known runtime bug locations under NumPy 2:
  - `social_gym/src/motion_model_manager.py:264`
  - `social_gym/src/motion_model_manager.py:271`

Why this matters:

- `Social-Navigation-PyEnvs` is a better immediate candidate than the legacy CADRL-family repos for
  main-stack experimentation,
- and its differential-drive framing makes it more relevant to Robot SF than a purely holonomic
  crowd-sim stack.

## Recommendation

Recommendation: `prototype only`

What is justified now:

1. treat this repo as the next promising external candidate,
2. keep wrapper work narrow and benchmark-visible,
3. treat non-trainable planner integration as a real prototype path rather than another assessment,
4. decide explicitly between:
   - a minimal local compatibility path (`socialforce` + NumPy 2 patch boundary), or
   - a side environment for stricter upstream pin fidelity.

Important caveat:

- the live Robot SF wrapper probe currently executes on the default `RobotSimulationConfig`
  SocNav observation path, whose initial `goal.current` snapshot can be degenerate (`goal ==
  robot position`), so that specific live report is a loop-integrity proof rather than a
  performance-quality proof.

What is not justified yet:

- learned-policy integration claims,
- family-level benchmark claims for the learned CrowdNav lineage inside this repo,
- silent patching of upstream behavior while presenting the result as source-faithful.

Pragmatic next step:

1. decide whether to pursue a non-trainable planner wrapper first (likely `orca` or one SFM/HSFM
   family entry), or
2. create a side-environment learned-path reproduction issue if the learned stack is the real
   target.
