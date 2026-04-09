# Goal

- Track the current state of issue 790 SAC work after fixing the benchmark action contract and
  adding a benchmark-style evaluation path.
- Record what the latest evidence says about why benchmark performance is still poor, and what to
  try next.

# Boundaries

- In scope:
  - SAC train/eval contract changes already implemented in this branch.
  - Observed training and benchmark outcomes from the latest `sac_gate_socnav_struct_v1` retrain.
  - Concrete follow-up hypotheses for improving benchmark transfer.
- Out of scope:
  - Claiming SAC is benchmark-ready.
  - Treating training-set success as benchmark evidence.
  - Paper-facing conclusions.

# Evidence

- Training script: `scripts/training/train_sac_sb3.py`
- Benchmark adapter: `robot_sf/baselines/sac.py`
- Benchmark runner integration: `robot_sf/benchmark/map_runner.py`
- SAC eval entry point: `scripts/validation/evaluate_sac.py`
- Observation contract reference: `docs/dev/observation_contract.md`
- Training config: `configs/training/sac/gate_socnav_struct.yaml`
- Baseline eval config: `configs/baselines/sac_gate_socnav_struct.yaml`
- Focused tests:
  - `tests/training/test_train_sac_sb3_config.py`
  - `tests/baselines/test_sac_planner.py`
  - `tests/test_map_runner_sac.py`
  - `tests/validation/test_evaluate_sac_smoke.py`

# What Changed

- `map_runner` now supports a SAC-native action path via `_planner_native_env_action`, so SAC delta
  outputs can go directly to `env.step(...)` without the benchmark’s absolute-to-delta conversion.
- SAC training and inference now both support `relative_obs`, which subtracts `robot_position` from
  `goal_current`, `goal_next`, and `pedestrians_positions` for SocNav-structured dict observations.
- `scripts/validation/evaluate_sac.py` exists and mirrors the benchmark-style eval flow for SAC.

# Observation Contract Clarification

The repository contract for `ObservationMode.SOCNAV_STRUCT` is a mixed-frame contract, not an
all-ego-frame contract.

Per `docs/dev/observation_contract.md` and `robot_sf/sensor/socnav_observation.py`:

- `robot.position`, `goal.current`, `goal.next`, and `robot.velocity_xy` are world-frame.
- `pedestrians.positions` are world-frame.
- `pedestrians.velocities` are rotated into the robot ego frame.
- flattened SB3 keys such as `robot_position`, `goal_current`, and `pedestrians_positions` preserve
  those same semantics.

This matters for SAC because any coordinate normalization or ego-frame preprocessing added in the
trainer or baseline adapter is an explicit model-side transform layered on top of the repository
observation contract. It is not currently part of the base `socnav_struct` producer contract.

# Validation

Targeted implementation proof:

```bash
uv run python -m pytest \
  tests/training/test_train_sac_sb3_config.py \
  tests/test_map_runner_sac.py \
  tests/baselines/test_sac_planner.py \
  tests/validation/test_evaluate_sac_smoke.py \
  -x -q
```

Observed result:

- `28 passed`

Latest gate retrain:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct.yaml
```

Observed result:

- checkpoint saved to `output/models/sac/sac_gate_socnav_struct_v1.zip`
- training finished in about 617 seconds on CPU
- rollout success reached `0.98` by `46,480` timesteps

Benchmark-style eval:

```bash
uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --workers 1 \
  --output-dir /tmp/sac_eval_v1_rel
```

Observed result:

- `success_rate: 0.0`
- `mean_min_distance: 12.9503`
- `mean_avg_speed: 0.1469`
- failure taxonomy: `29 max_steps`, `1 collision`

Longer-horizon check:

```bash
uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --workers 1 \
  --horizon 250 \
  --output-dir /tmp/sac_eval_v1_rel_h250
```

Observed result:

- `success_rate: 0.0`
- `mean_min_distance: 9.7735`
- `mean_avg_speed: 0.1785`
- still effectively a timeout problem rather than a horizon-only problem

# Key Observations

1. The original benchmark-action bug is real and is now fixed.
   Before the native action bypass, SAC delta outputs were being treated like absolute targets and
   effectively double-converted.

2. The current blocker is no longer the benchmark action bridge.
   The fresh checkpoint trains successfully on the training distribution but still fails the
   benchmark subset after the plumbing fix.

3. The current benchmark failure mode is mostly under-driving / no-forward-motion.
   Direct tracing on `empty_map_8_directions_east` showed the checkpoint repeatedly outputting
   commands like `v=0` with oscillating `omega`, leaving the robot at the start pose until timeout.

4. Relative translation alone did not fix transfer.
   Replacing absolute world coordinates with robot-relative coordinates improved the contract, but
   it did not produce usable benchmark transfer by itself.

5. The current repo benchmark contract is not already ego-frame for these keys.
   The failed transfer is therefore not caused by the benchmark violating an all-local-coordinate
   contract. Instead, SAC appears to need a stronger model-side invariance transform than the base
   repository observation contract provides.

# Improvement Hypotheses

## 1. Move from world-relative to ego-frame observations

Current `relative_obs` subtracts `robot_position`, but it does not rotate goal or pedestrian
vectors into the robot frame. Since the base repo contract keeps these keys in world coordinates,
that means SAC still has to learn orientation invariance from data unless the model-side transform
adds it explicitly.

Why this likely matters:

- the policy is failing even on simple atomic maps with rotated goal directions,
- the direct trace shows pure rotation commands with zero forward motion,
- training on `classic_interactions.yaml` may not cover heading and map-orientation diversity well
  enough for raw world-frame transfer.

Recommended next change:

- rotate `goal_current`, `goal_next`, and `pedestrians_positions` into the robot ego frame using
  `robot_heading`,
- keep this transform identical in both `scripts/training/train_sac_sb3.py` and
  `robot_sf/baselines/sac.py`.

## 2. Add an explicit anti-stall / anti-spin training signal

The observed benchmark behavior is not collision-seeking; it is safe but inert. That usually means
the reward lets the policy survive with too little pressure to commit to forward progress.

Recommended next experiments:

- increase the living penalty magnitude,
- add a penalty for near-zero linear speed when the goal is still far away,
- add a penalty for sustained angular velocity with near-zero linear speed,
- optionally reward alignment between heading and goal direction if the robot is stalled.

This should be tested conservatively because it can also create reckless goal-seeking.

## 3. Train on a more benchmark-like scenario curriculum

Current gate training uses `configs/scenarios/classic_interactions.yaml`. The benchmark subset that
currently fails contains atomic open-space direction checks and small structured maps that are not
the same distribution.

Recommended next experiments:

- create a SAC gate config that includes atomic empty-map and simple route-following scenarios,
- include `verified_simple_subset_v1` archetype coverage or a training-safe analogue,
- keep `classic_interactions` as part of the curriculum rather than the whole curriculum.

Observed evidence supports this: training success is high on the source distribution, but transfer
to the benchmark subset is still zero.

## 4. Check whether deterministic SAC inference is collapsing a multimodal policy

Benchmark eval currently uses deterministic SAC inference. If the actor learned a diffuse action
distribution where the mode is near-zero linear speed, deterministic rollout can look much worse
than sampled rollout.

Recommended next experiment:

- run a small diagnostic sweep comparing deterministic and stochastic inference on the same subset.

This is not enough for paper-facing benchmark use on its own, but it can distinguish a true policy
failure from a deterministic-mode collapse artifact.

## 5. Consider a minimum forward-speed constraint only if reward/objective fixes fail

The environment currently allows `min_linear_speed=0.0`. That makes spinning in place a feasible
safe policy.

This should not be the first fix, but if ego-frame observations and reward shaping still leave the
policy inert, test a small positive minimum linear speed or a projection that prevents indefinite
zero-speed spinning far from goal.

This is higher risk because it changes the control contract rather than just improving learning.

# GPU Note

- Training is currently on CPU because CUDA initialization fails on this machine.
- The observed warning is:
  - `CUDA unknown error`
- For this gate-sized SAC network, CPU training is still practical.
- GPU would help throughput, but it is not the reason benchmark transfer is currently failing.

# Recommended Next Step

- Implement ego-frame observation rotation on top of the current relative translation contract.
- Retrain `configs/training/sac/gate_socnav_struct.yaml`.
- Re-run `scripts/validation/evaluate_sac.py` on `configs/scenarios/sets/verified_simple_subset_v1.yaml`.
- Only if that still fails, move to explicit anti-stall reward shaping or a benchmark-like training
  curriculum.

# Risks / Follow-ups

- The current checkpoint is still experimental and should not replace paper-facing baselines.
- The eval outputs used here were written under `/tmp`, so this note records the metric values
  explicitly to preserve the evidence trail.
- If ego-frame observations materially improve transfer, a follow-up issue should tighten the SAC
  observation contract and add a regression test around the transformed benchmark trace.
