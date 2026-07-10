# Reward Profiles Reference

[Back to Documentation Index](../README.md) · [Environment Contract](./environment_contract.md)

This is the **single reference table** for Robot SF reward semantics. Reward weights live across
several files (`robot_sf/gym_env/reward.py`, `robot_sf/gym_env/reward_alyassi.py`, the per-profile
YAML under `configs/training/rewards/`), which makes questions like *"what is the collision
penalty?"* easy to misquote — it has several correct answers depending on context. This page
collects them in one place.

> **Status:** documentation/auditability (Phase 1, no dependencies). Values are kept in sync with
> the code by a mechanical drift-detection test
> (`tests/gym_env/test_reward_profile_reference.py`); if you change a weight, update the matching
> `configs/training/rewards/*.yaml` too or that test fails. See [Glossary](../glossary.md) for
> term definitions (social navigation, SNQI, etc.).

## Plain-language summary

A **reward profile** is a named preset that maps environment step metadata (progress, collisions,
near-misses, comfort, jerk, timeouts) to a scalar training reward. Robot SF ships three
**named weighted profiles** built on a shared weighted-terms formula
(`route_completion_v2`, `route_completion_v3`, `social_quality_v1`), plus several **legacy reward
functions** with their own fixed-shape logic (`simple`, `simple_ped`, `punish_action`,
`stationary_collision_ped`, `snqi_step`, `alyassi`). The environment factory
(`make_robot_env` / `make_image_robot_env`) resolves a profile from `reward_name` and defaults to
`route_completion_v2` when none is given.

## The collision penalty, directly

| Context | Collision penalty | Source |
| --- | --- | --- |
| `route_completion_v2` (env-factory **default**) | **−5.0** (any pedestrian/robot/obstacle collision) | `_ROUTE_COMPLETION_V2_WEIGHTS["collision"]` |
| `route_completion_v3` | **−10.0** (any collision) | `_ROUTE_COMPLETION_V3_WEIGHTS["collision"]` |
| `social_quality_v1` | **−6.0** (any collision) | `_SOCIAL_QUALITY_V1_WEIGHTS["collision"]` |
| Legacy `simple` / `punish_action` (robot) | **−5.0** pedestrian/robot, **−2.0** obstacle (separate terms) | `simple_reward` kwargs |
| Legacy `simple_ped` / `stationary_collision_ped` (pedestrian) | **−5.0** pedestrian, **−5.0** obstacle | `simple_ped_reward` kwargs |

So "what is the collision penalty?" has four common answers: −5 (v2/default), −10 (v3), −6
(social_quality_v1), and the legacy split −5/−2. Which one a trained artifact used is found via the
[recipe below](#how-to-find-the-profile-a-checkpoint-was-trained-with).

## Named weighted profiles

These three profiles share one weighted-terms formula (`_reward_with_terms` in
`robot_sf/gym_env/reward.py`). Each term is a bounded feature (e.g. progress in `[−1, 1]`,
collision in `{0, 1}`); the weight is multiplied by the feature and summed. A blank cell means the
profile does not use that term (weight 0 / term absent).

| Term | Feature range | `route_completion_v2` | `route_completion_v3` | `social_quality_v1` |
| --- | --- | ---: | ---: | ---: |
| `progress` | `[−1, 1]` (goal-distance delta) | 2.5 | 2.2 | 1.0 |
| `living` | `1.0` (per-step) | −0.01 | −0.01 | −0.02 |
| `collision` | `{0, 1}` (any collision) | −5.0 | −10.0 | −6.0 |
| `near_miss` | `[0, 1]` | −0.8 | −1.0 | −1.2 |
| `ttc_risk` | `[0, 1]` (inverse TTC / near-miss proxy) | −0.6 | −0.8 | −0.9 |
| `comfort` | `[0, 1]` | −0.4 | −0.5 | −0.9 |
| `smoothness` | `[0, 1]` (jerk normalised by 5) | −0.15 | −0.2 | −0.2 |
| `timeout` | `{0, 1}` (timeout w/o success/collision) | — | −3.0 | — |
| `stagnation` | `[0, 1]` | — | −1.2 | — |
| `terminal_bonus` | `{0, 1}` (route complete, no collision) | 2.0 | 3.0 | 1.5 |

Per-term decomposition is written back into the step metadata as `reward_terms` and `reward_total`
for logging. **Success is route completion only** (waypoint completion does not trigger the terminal
bonus). Collisions invalidate the terminal bonus.

Each profile also has a canonical YAML config under `configs/training/rewards/`
(`route_completion_v2.yaml`, `route_completion_v3.yaml`, `social_quality_v1.yaml`) that records the
same weights; the drift-detection test guarantees they match the Python dicts above.

## Legacy reward functions

These predate the named weighted profiles and use their own fixed reward shapes. Default keyword
arguments are shown; callers can override any of them via `reward_kwargs`.

| Function | `reward_name` aliases | Key default terms |
| --- | --- | --- |
| `simple_reward` (robot) | `simple`, `simple_reward` | step discount −0.1/`max_sim_steps`; ped/robot collision −5; obstacle collision −2; route complete +1 |
| `punish_action_reward` (robot) | `punish_action`, `punish_action_reward` | `simple_reward` terms + action-change penalty −0.1 × ‖Δaction‖ |
| `simple_ped_reward` (pedestrian) | *(none — pedestrian-factory default only)* | step discount −0.1/`max_sim_steps`; −0.001 × distance-to-robot; ped collision −5; obstacle collision −5; robot collision +5; robot route complete −1 |
| `stationary_collision_ped_reward` (pedestrian) | `stationary_collision_ped`, `ped_stationary_collision` | ped reward terms + stationary robot-collision bonus +10 (speed ≤ 0.1), slow bonus +8 (speed ≤ 1.0), else +5 |
| `snqi_step_reward` | `snqi`, `snqi_step`, `snqi_step_reward` | projects step metadata into the canonical SNQI score (default weights below) + optional `terminal_bonus` / `living_penalty` |
| `alyassi_reward` | `alyassi`, `alyassi_reward`, `alyassi_composite` | weighted multi-objective composition (Alyassi et al. 2025 taxonomy); default weights below |

### SNQI step reward default weights

`snqi_step_reward` uses `_DEFAULT_SNQI_REWARD_WEIGHTS` unless overridden:

| Weight | Value |
| --- | ---: |
| `w_success` | 1.0 |
| `w_time` | 0.8 |
| `w_collisions` | 2.0 |
| `w_near` | 1.0 |
| `w_comfort` | 0.5 |
| `w_force_exceed` | 1.5 |
| `w_jerk` | 0.3 |

Collisions force `success = 0` (mirroring benchmark semantics). Note this is a *step-level
projection* of SNQI; some benchmark SNQI terms may be absent from step metadata, so treat it as an
approximation aligned with benchmark SNQI, not the benchmark metric itself.

### Alyassi default weights

`alyassi_reward` uses the frozen `AlyassiRewardWeights` dataclass defaults:

| Weight | Value |
| --- | ---: |
| `w_goal` | 1.0 |
| `w_collision` | 2.0 |
| `w_efficiency` | 0.2 |
| `w_smoothness` | 0.2 |
| `w_social` | 0.8 |
| `w_geometric_collision` | 0.8 |
| `w_human_preference` | 0.4 |
| `w_human_prediction` | 0.5 |
| `w_exploration` | 0.05 |
| `w_task_specific` | 0.2 |
| `w_demo_learning` | 0.2 |
| `w_weight_learning` | 0.0 |

## Environment factory default

`make_robot_env(...)` and `make_image_robot_env(...)` both resolve the reward as follows
(see `robot_sf/gym_env/environment_factory.py`):

1. If `reward_func` is provided, use it directly (no name resolution).
2. Else resolve `reward_name`, defaulting to **`route_completion_v2`** when `None`.
3. If `reward_curriculum` is provided, wrap the resolved profile in a staged curriculum that
   advances after terminal episodes.
4. Else build the named profile via `build_reward_function(reward_name, reward_kwargs)`.

`make_pedestrian_env(...)` does **not** use `reward_name` resolution; when `reward_func` is `None`
it falls back to the canonical `simple_ped_reward` directly.

## How to find the profile a checkpoint was trained with

A trained artifact's reward profile is discoverable three ways, in order of reliability:

1. **Training config (authoritative).** The config YAML records the resolved profile under
   `env_factory_kwargs.reward_name` (and optional `reward_kwargs`). For example
   `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml` sets
   `reward_name: route_completion_v3`. If `reward_name` is absent and no `reward_func` is set, the
   factory default `route_completion_v2` applies. The checkpoint dir also stores a copy as
   `<policy_id>.config.yaml`.
2. **Training run manifest.** `write_training_run_manifest(...)` records a
   `reward_profile=<resolved name>` entry in the manifest `notes`, so the run JSON is
   self-describing even if the config is misplaced. The resolved name distinguishes
   `route_completion_v2 (default)`, an explicit name, a `curriculum[N stages]: <base>` summary, or
   `custom_callable` when a raw `reward_func` was supplied.
3. **Startup log line.** The training entrypoint emits a structured
   `Training startup summary: ... reward_profile=<resolved name> ...` line at startup (see
   `_resolved_reward_name` in `scripts/training/train_ppo.py`).

When a custom `reward_func` was used, only the config carries the actual weights; the manifest and
log will report `custom_callable`.

## Adding or changing a profile

1. Add or edit the weight dict in `robot_sf/gym_env/reward.py` (or `reward_alyassi.py`).
2. Mirror the change in the matching `configs/training/rewards/<profile>.yaml`.
3. Register any new alias in `build_reward_function`.
4. Update the table in this page.
5. `uv run pytest tests/gym_env/test_reward_profile_reference.py` — this fails if the Python dicts,
   the YAML configs, or the factory default drift out of sync.
