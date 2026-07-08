# Goal
* Track the current state of issue 790 SAC work after fixing the benchmark action contract and
  adding a benchmark-style evaluation path.
* Record what the latest evidence says about why benchmark performance is still poor, and what to
  try next.

# Boundaries
* In scope:
  + SAC train/eval contract changes already implemented in this branch.
  + Observed training and benchmark outcomes from the latest `sac_gate_socnav_struct_v1` retrain.
  + Concrete follow-up hypotheses for improving benchmark transfer.
* Out of scope:
  + Claiming SAC is benchmark-ready.
  + Treating training-set success as benchmark evidence.
  + Paper-facing conclusions.
# Evidence
* Training script: `scripts/training/train_sac_sb3.py`
* Benchmark adapter: `robot_sf/baselines/sac.py`
* Benchmark runner integration: `robot_sf/benchmark/map_runner.py`
* SAC eval entry point: `scripts/validation/evaluate_sac.py`
* Observation contract reference: `docs/dev/observation_contract.md`
* Training config: `configs/training/sac/gate_socnav_struct.yaml`
* Baseline eval config: `configs/baselines/sac_gate_socnav_struct.yaml`
* Focused tests:
  + `tests/training/test_train_sac_sb3_config.py`
  + `tests/baselines/test_sac_planner.py`
  + `tests/test_map_runner_sac.py`
  + `tests/validation/test_evaluate_sac_smoke.py`
# What Changed
* `map_runner` now supports a SAC-native action path via `_planner_native_env_action`, so SAC delta
  outputs can go directly to `env.step(...)` without the benchmark’s absolute-to-delta conversion.
* SAC training now uses `ScenarioSwitchingEnv`, so training episodes sample across the loaded
  scenario manifest instead of silently locking onto the first scenario entry.
* SAC training and inference now both support SocNav dict observation transforms:
  + `relative`: subtract `robot_position` from `goal_current`,   `goal_next`, and
 `pedestrians_positions`

  + `ego`: apply the same translation and then rotate those vectors into the robot frame using
 `robot_heading`

* `scripts/validation/evaluate_sac.py` exists and mirrors the benchmark-style eval flow for SAC.
# Observation Contract Clarification

The repository contract for `ObservationMode.SOCNAV_STRUCT` is a mixed-frame contract, not an
all-ego-frame contract.

Per `docs/dev/observation_contract.md` and `robot_sf/sensor/socnav_observation.py` :

* `robot.position`,   `goal.current`,   `goal.next`, and `robot.velocity_xy` are world-frame.
* `pedestrians.positions` are world-frame.
* `pedestrians.velocities` are rotated into the robot ego frame.
* flattened SB3 keys such as `robot_position`,  `goal_current`, and `pedestrians_positions` preserve
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

* `33 passed`

Latest gate retrain:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct.yaml
```

Observed result:

* checkpoint saved to `output/models/sac/sac_gate_socnav_struct_v1.zip`
* training finished in about 617 seconds on CPU
* rollout success reached `0.98` by `46,480` timesteps

Benchmark-style eval:

```bash
uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --workers 1 \
  --output-dir output/tmp/sac_eval_v1_rel
```

Observed result:

* `success_rate: 0.0`
* `mean_min_distance: 12.9503`
* `mean_avg_speed: 0.1469`
* failure taxonomy: `29 max_steps`,   `1 collision`

Longer-horizon check:

```bash
uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --workers 1 \
  --horizon 250 \
  --output-dir output/tmp/sac_eval_v1_rel_h250
```

Observed result:

* `success_rate: 0.0`
* `mean_min_distance: 9.7735`
* `mean_avg_speed: 0.1785`
* still effectively a timeout problem rather than a horizon-only problem

Ego-frame SAC ablation:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct_ego.yaml

uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_ego_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --algo-config configs/baselines/sac_gate_socnav_struct_ego.yaml \
  --workers 1 \
  --output-dir output/tmp/sac_eval_ego_v1
```

Observed result:

* checkpoint saved to `output/models/sac/sac_gate_socnav_struct_ego_v1.zip`
* training finished in about 667 seconds on CPU
* rollout success reached `0.99` by `47,891` timesteps
* benchmark subset `success_rate: 0.3667`
* `mean_min_distance: 3.564`
* `mean_avg_speed: 1.920`
* failure taxonomy: `13 collision`,   `6 max_steps`

Per-scenario split:

* `empty_map_8_directions_east`: `3/3`
* `empty_map_8_directions_north`: `3/3`
* `empty_map_8_directions_west`: `2/3`
* `goal_behind_robot`: `3/3`
* `head_on_interaction`: `0/3`
* `line_wall_detour`: `0/3`
* `narrow_passage`: `0/3`
* `overtaking_interaction`: `0/3`
* `single_obstacle_circle`: `0/3`
* `single_ped_crossing_orthogonal`: `0/3`

Safety-focused reward shaping on top of ego-frame:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct_ego_safe_v1.yaml

uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_ego_safe_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --algo-config configs/baselines/sac_gate_socnav_struct_ego_safe.yaml \
  --workers 1 \
  --output-dir output/tmp/sac_eval_ego_safe_v1
```

Observed result:

* checkpoint saved to `output/models/sac/sac_gate_socnav_struct_ego_safe_v1.zip`
* training finished in about 657 seconds on CPU
* rollout success reached `0.98` by `48,580` timesteps
* benchmark subset `success_rate: 0.2667`
* `mean_min_distance: 3.524`
* `mean_avg_speed: 1.618`
* failure taxonomy: `14 collision`,   `8 max_steps`

Per-scenario split:

* `empty_map_8_directions_east`: `0/3`
* `empty_map_8_directions_north`: `3/3`
* `empty_map_8_directions_west`: `3/3`
* `goal_behind_robot`: `2/3`
* `head_on_interaction`: `0/3`
* `line_wall_detour`: `0/3`
* `narrow_passage`: `0/3`
* `overtaking_interaction`: `0/3`
* `single_obstacle_circle`: `0/3`
* `single_ped_crossing_orthogonal`: `0/3`

Corrected multi-scenario ego-frame retrain:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct_ego_multi_v1.yaml

uv run python scripts/validation/evaluate_sac.py \
  --checkpoint output/models/sac/sac_gate_socnav_struct_ego_multi_v1.zip \
  --scenario-matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --algo-config configs/baselines/sac_gate_socnav_struct_ego_multi.yaml \
  --workers 1 \
  --output-dir output/tmp/sac_eval_ego_multi_v1
```

Observed result:

* checkpoint saved to `output/models/sac/sac_gate_socnav_struct_ego_multi_v1.zip`
* training finished in about 684 seconds on CPU
* rollout success stayed low on the full distribution, reaching only `0.10` by `46,715` timesteps
* benchmark subset `success_rate: 0.4667`
* `mean_min_distance: 5.486`
* `mean_avg_speed: 1.787`
* failure taxonomy: `10 max_steps`,   `6 collision`

Per-scenario split:

* `empty_map_8_directions_east`: `3/3`
* `empty_map_8_directions_north`: `3/3`
* `empty_map_8_directions_west`: `3/3`
* `goal_behind_robot`: `0/3`
* `head_on_interaction`: `0/3`
* `line_wall_detour`: `2/3`
* `narrow_passage`: `0/3`
* `overtaking_interaction`: `0/3`
* `single_obstacle_circle`: `3/3`
* `single_ped_crossing_orthogonal`: `0/3`

Weighted verified-subset curriculum run:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/gate_socnav_struct_ego_curriculum.yaml
```

Observed result:

* checkpoint saved to `output/models/sac/sac_gate_socnav_struct_ego_curriculum_v1.zip`
* training completed successfully through `100,000` steps with periodic benchmark eval every `20,000`
* periodic verified-simple success progression:
  + `20k`: `0.3667`
  + `40k`: `0.6000`
  + `60k`: `0.7000`
  + `80k`: `0.7667`
  + `100k`: `0.8000`
* `100k` benchmark subset summary:
  + `mean_min_distance: 3.9188`
  + `mean_avg_speed: 1.9684`
  + failure taxonomy: `6 collision`

Final per-scenario split at `100k` :

* `empty_map_8_directions_east`: `3/3`
* `empty_map_8_directions_north`: `3/3`
* `empty_map_8_directions_west`: `3/3`
* `goal_behind_robot`: `3/3`
* `head_on_interaction`: `3/3`
* `line_wall_detour`: `0/3`
* `narrow_passage`: `0/3`
* `overtaking_interaction`: `3/3`
* `single_obstacle_circle`: `3/3`
* `single_ped_crossing_orthogonal`: `3/3`
# Key Observations
01. The original benchmark-action bug is real and is now fixed.
   Before the native action bypass, SAC delta outputs were being treated like absolute targets and
   effectively double-converted.

02. The current blocker is no longer the benchmark action bridge.
   The fresh checkpoint trains successfully on the training distribution but still fails the
   benchmark subset after the plumbing fix.

03. The current benchmark failure mode is mostly under-driving / no-forward-motion.
   Direct tracing on `empty_map_8_directions_east` showed the checkpoint repeatedly outputting
   commands like `v=0` with oscillating `omega` , leaving the robot at the start pose until timeout.

04. Relative translation alone did not fix transfer.
   Replacing absolute world coordinates with robot-relative coordinates improved the contract, but
   it did not produce usable benchmark transfer by itself.

05. The current repo benchmark contract is not already ego-frame for these keys.
   The failed transfer is therefore not caused by the benchmark violating an all-local-coordinate
   contract. Instead, SAC appears to need a stronger model-side invariance transform than the base
   repository observation contract provides.

06. Ego-frame preprocessing materially improves transfer on open-space directional tasks.
   The ego-frame ablation moved the verified-simple subset from `0.0%` to `36.7%` success and
   solved the empty-map east/north/goal-behind cases. That is strong evidence that heading /
   orientation invariance was a real blocker.

07. The remaining failures have shifted from stalling to unsafe interaction behavior.
   After the ego-frame change, the dominant failure mode is no longer timeout-only. The remaining
   errors are mostly collisions on interaction, detour, and obstacle scenarios.

08. A first safety-shaped reward pass was a regression, not an improvement.
   The `ego_safe_v1` run dropped from `36.7%` to `26.7%` success, lowered average speed, and still
   ended mostly in collisions or timeouts. That makes the current safety-shaped reward weights a
   discard rather than a new baseline candidate.

09. The earlier SAC training evidence was materially distorted by a trainer bug.
   Before the `ScenarioSwitchingEnv` fix, SAC training loaded a scenario manifest but trained only
   on its first entry. That explains why earlier rollout success values were near `1.0` while
   transfer remained poor.

10. The corrected multi-scenario trainer improves benchmark robustness even though training looks
  much harder.
  The multi-scenario ego run improved verified-simple success from `36.7%` to `46.7%` and cut
  collisions from `13` to `6` , but in-training rollout success on the full source distribution
  stayed low. This suggests the old train-time metric was over-optimistic rather than the new
  run being worse at benchmark transfer.

11. A weighted verified-subset curriculum substantially improves issue-790 transfer without
  changing the public benchmark contract.
  The curriculum run reached `80.0%` on `verified_simple_subset_v1` by `100k` steps and solved
  all dynamic interaction cases in that subset except the two static-navigation bottlenecks
`line_wall_detour` and `narrow_passage` .

# Improvement Hypotheses

## 1. Move from world-relative to ego-frame observations

Current `relative_obs` subtracts `robot_position` , but it does not rotate goal or pedestrian
vectors into the robot frame. Since the base repo contract keeps these keys in world coordinates, 
that means SAC still has to learn orientation invariance from data unless the model-side transform
adds it explicitly.

Why this likely matters:

* the policy is failing even on simple atomic maps with rotated goal directions, 
* the direct trace shows pure rotation commands with zero forward motion, 
* training on `classic_interactions.yaml` may not cover heading and map-orientation diversity well
  enough for raw world-frame transfer.

Recommended next change:

* rotate `goal_current`,   `goal_next`, and `pedestrians_positions` into the robot ego frame using
`robot_heading` , 
* keep this transform identical in both `scripts/training/train_sac_sb3.py` and
`robot_sf/baselines/sac.py` .

Status:

* implemented and validated as `gate_socnav_struct_ego.yaml`
* improves directional transfer substantially
* does not yet solve interaction safety

## 2. Do not keep the current safety-shaped reward variant

The first safety-focused reward pass was a useful experiment, but it did not improve benchmark
behavior. It reduced success from `36.7%` to `26.7%` , increased failed episodes from `19` to `22` , 
and did not materially reduce collision-heavy failures.

Interpretation:

* the ego-frame transform addressed a real representation problem, 
* the current reward weights are now over-constraining the controller without teaching the missing
  interaction policy, 
* more penalty magnitude alone is unlikely to be the highest-value next step.

Recommended next stance:

* keep `gate_socnav_struct_ego.yaml` as the current best SAC config, 
* keep `gate_socnav_struct_ego_safe_v1.yaml` only as a recorded negative result, 
* avoid spending another immediate loop on reward-only tuning unless a more targeted hypothesis
  emerges from interaction-specific traces.

## 3. Train on a more benchmark-like scenario curriculum

Current gate training uses `configs/scenarios/classic_interactions.yaml` . The benchmark subset that
currently fails contains atomic open-space direction checks and small structured maps that are not
the same distribution.

Recommended next experiments:

* create a SAC gate config that includes atomic empty-map and simple route-following scenarios, 
* include `verified_simple_subset_v1` archetype coverage or a training-safe analogue, 
* keep `classic_interactions` as part of the curriculum rather than the whole curriculum.

Observed evidence supports this: ego-frame transfer is now good on directional open-space tasks, 
but it remains zero on detour, obstacle, and social-interaction scenarios. That is exactly the kind
of split a curriculum or source-distribution gap would produce.

Updated interpretation after the multi-scenario fix:

* a stronger scenario distribution already helped benchmark transfer, 
* the remaining gaps are now specific capability gaps, not just a generic coordinate mismatch, 
* the next curriculum work should be weighted rather than purely broader so we do not regress on
  cases like `goal_behind_robot` while adding more detour and interaction coverage.

## 4. Check whether deterministic SAC inference is collapsing a multimodal policy

Benchmark eval currently uses deterministic SAC inference. If the actor learned a diffuse action
distribution where the mode is near-zero linear speed, deterministic rollout can look much worse
than sampled rollout.

Recommended next experiment:

* run a small diagnostic sweep comparing deterministic and stochastic inference on the same subset.

This is not enough for paper-facing benchmark use on its own, but it can distinguish a true policy
failure from a deterministic-mode collapse artifact.

## 5. Consider a minimum forward-speed constraint only if reward/objective fixes fail

The environment currently allows `min_linear_speed=0.0` . That makes spinning in place a feasible
safe policy.

This should not be the first fix, but if ego-frame observations and reward shaping still leave the
policy inert, test a small positive minimum linear speed or a projection that prevents indefinite
zero-speed spinning far from goal.

This is higher risk because it changes the control contract rather than just improving learning.

# GPU Note
* Training is currently on CPU because CUDA initialization fails on this machine.
* The observed warning is:
  + `CUDA unknown error`
* For this gate-sized SAC network, CPU training is still practical.
* GPU would help throughput, but it is not the reason benchmark transfer is currently failing.
# Recommended Next Step
* Keep the ego-frame observation transform.
* Treat `gate_socnav_struct_ego_curriculum.yaml` as the current best SAC checkpoint/config.
* Treat `gate_socnav_struct_ego_safe_v1.yaml` as a negative result and do not promote it.
* Keep multi-scenario episode switching enabled for all future SAC runs.
* Build the next SAC training config around the remaining static-navigation blockers instead of the
  already-solved interaction cases.
* Prefer targeted follow-up loops for `line_wall_detour` and `narrow_passage` before switching to
  TD3.
* Re-run `scripts/validation/evaluate_sac.py` on
`configs/scenarios/sets/verified_simple_subset_v1.yaml` before attempting any broader benchmark
  matrix.

# Risks / Follow-ups
* The current checkpoint is still experimental and should not replace paper-facing baselines.
* The eval outputs used here were written under `/tmp`, so this note records the metric values
  explicitly to preserve the evidence trail.
* If ego-frame observations materially improve transfer, a follow-up issue should tighten the SAC
  observation contract and add a regression test around the transformed benchmark trace.
