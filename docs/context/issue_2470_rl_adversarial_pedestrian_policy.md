# Issue #2470 RL Adversarial Pedestrian Policy Scope (2026-06-06)

Status: scoped proposal and launch-packet direction, not RL training or adversarial-behavior evidence.

Related surfaces:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2470
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Existing PedestrianEnv (ego-ped RL): `robot_sf/gym_env/pedestrian_env.py`
- Existing ped reward: `robot_sf/gym_env/reward.py:162` (`simple_ped_reward`)
- Adversarial scenario search: `robot_sf/adversarial/` (search-based NPC spawn/route perturbation)
- Adversarial generation protocol: `docs/context/issue_1457_adversarial_generation_protocol.md`
- Adversarial failure archive: `docs/context/issue_1237_adversarial_failure_archive.md`
- Adversarial manifest smoke: `docs/context/issue_2562_adversarial_manifest_smoke.md`
- Adversarial manifest quality metrics: `docs/context/issue_2567_adversarial_manifest_quality.md`
- Learned-expansion gate: `docs/context/issue_2568_adversarial_expansion_gate.md`
- NPC ped behavior: `robot_sf/ped_npc/ped_behavior.py` (CrowdedZone, FollowRoute, SinglePedestrian)
- Pedestrian definition: `robot_sf/nav/map_config.py:SinglePedestrianDefinition`
- Adversarial force module: `robot_sf/ped_npc/adversial_ped_force.py`
- Scenario contract docs: `docs/scenario_contracts.md`
- Single-pedestrian knobs: `docs/single_pedestrians.md`
- Learned policy registry: `docs/context/policy_search/learned_policy_registry.md`
- Learned policy adapter interface: `docs/context/issue_1618_learned_policy_adapter_interface.md`
- Benchmark fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Result

Issue #2470 asks for an RL-based adversarial pedestrian policy direction — a learned policy that
controls NPC pedestrian behavior at a strategic level to challenge a robot planner, rather than
the existing search-based adversarial perturbation (`robot_sf/adversarial/`) or the existing
ego-pedestrian RL (`PedestrianEnv`). This proposal defines the adversary action/control space,
reward terms, validity criteria, first deterministic smoke path, stop rule, and compute/data
prerequisites — without training an RL adversary, claiming behavioral realism, or changing
planner rankings.

The existing repository has two distinct adversarial paths:

1. **Search-based NPC perturbation** (`robot_sf/adversarial/`): searches over `CandidateSpec`
   fields (start, goal, spawn_time, speed, delay) to find dangerous configurations, materializes
   them as scenario overrides. No learning, no policy.
2. **Ego-pedestrian RL** (`PedestrianEnv`): trains a unicycle-drive ego pedestrian to actively
   collide with a frozen robot policy. The action space is low-level (acceleration, steering).

This issue defines a **third path**: an RL policy that produces high-level NPC control signals
(pedestrian spawn locations, route choice, start time, speed, pause duration, group behavior)
at scenario-episode granularity. The policy would be trained against a population of frozen
robot planners, learning to generate adversarial pedestrian configurations that exploit planner
weaknesses.

## Adversary Action / Control Space

The RL adversary operates at **scenario-initialization granularity** (actions are taken before
or at low frequency during an episode), not per-simulation-step. The proposed action space is a
`Dict` or `MultiDiscrete` / `Box` composite:

| Action Dimension | Type | Range / Choices | Description |
|---|---|---|---|
| `pedestrian_spawn` | Discrete | 0..N-1 (spawn-zone index) | Which spawn zone or map region to place each NPC pedestrian |
| `route_choice` | Discrete | 0..R-1 (route index) | Which goal/route the NPC follows after spawning |
| `start_time` | Continuous | [0.0, max_start_delay_s] | Seconds after episode start before the NPC appears |
| `speed` | Continuous | [min_speed_mps, max_speed_mps] | NPC walking speed along its route |
| `pause_duration` | Continuous | [0.0, max_pause_s] | Duration the NPC waits at a pause point or before starting |
| `group_behavior` | Discrete | 0..G-1 (behavior-mode index) | Whether NPC acts solo, in a group, with group cohesion/join/leave |

**Total action shape implications**: a single action produces a `SinglePedestrianDefinition`
(or override to an existing one). For K NPCs the action is a concatenation or a per-NPC action
bundle. The initial spike should use a single NPC adversary (`K=1`) and expand only after the
deterministic smoke passes.

**Relationship to existing config fields**: the action dimensions map directly to existing
`SinglePedestrianDefinition` fields (`start`, `goal`, `start_delay_s`, `speed_m_s`, `wait_at`,
`role`, `role_target_id`) and `SimulationSettings.ped_density_by_difficulty`. The policy
output must be convertible to these config types before simulation starts.

**Not included in v0 action space**: pedestrian radius, obstacle-avoidance aggressiveness,
animation style, appearance. These are deferred to a follow-up issue if evidence supports
their relevance.

## Observation Space

The adversary policy observes the environment state **before deciding NPC configurations**.
Proposed observation components:

- **Map topology**: occupancy grid or graph encoding of walkable areas, spawn zones, goal
  regions, obstacle layout
- **Robot start/goal**: robot initial pose and intended route waypoints
- **Planner identity**: one-hot encoding of the frozen planner being tested (adversary must
  learn planner-specific weaknesses)
- **Scenario difficulty**: scalar or categorical encoding of the current scenario profile
- **Previous episode outcome** (if multi-episode training): collision rate, near-miss count,
  route-completion status from the prior episode with the same planner

A minimalist v0 observation could omit map topology and planner identity, relying on a
randomized scenario seed. The deterministic smoke must use the minimalist v0 to bound scope.

## Reward Terms

The adversary reward must incentivize the robot-impeding behaviors the search-based adversary
currently finds through enumeration. The proposed reward decomposition (per simulation step,
accumulated and returned as episode return):

### Main reward terms

| Term | Default Weight | Condition | Notes |
|---|---|---|---|
| `collision` | +1.0 | Robot collides with any NPC pedestrian | Primary adversarial success signal. Existing `simple_ped_reward` gives +5 for ego-ped collision; NPC adversary should reward any robot-NPC collision. |
| `near_miss` | +0.3 | Robot passes within `near_miss_threshold_m` of an NPC | Continuous shaping. Near-miss threshold should match the benchmark definition (currently 0.5m in `_snqi_compute_near_misses`). |
| `low_progress` | +0.1 | Robot forward progress < `min_progress_m` over a window | Incentivizes blocking without requiring collision. Use robot displacement over N steps. |
| `timeout_bonus` | +0.5 | Robot fails to complete route within `max_sim_steps` | The adversary wins if the robot times out. Binary at episode end. |

### Penalty terms

| Term | Default Weight | Condition | Notes |
|---|---|---|---|
| `invalid_spawn` | -10.0 | Generated spawn overlaps with obstacle or off-map | Hard rejection: this configuration must never reach simulation. |
| `degenerate_route` | -10.0 | Route is unreachable from spawn or self-intersecting | Hard rejection: caught before simulation. |
| `zero_speed_stall` | -2.0 | All NPCs have speed=0 or pause=inf for entire episode | Degenerate solution that trivially blocks without realistic behavior. |
| `group_overflow` | -1.0 | Group size exceeds `max_group_size` or groups overlap | Structural constraint violation. |

### Total reward

```
r_total = Σ (weight * term) — applied per episode (sparse, with near_miss/low_progress shaping)
```

The collision term must be verified to dominate near_miss + low_progress when a collision
actually occurs, to prevent the adversary from preferring "stalking" over "striking".

## Validity and Degeneracy Rejection Criteria

An adversarial pedestrian configuration (action) is **valid** only when all of:

1. **Spawn validity**: `pedestrian_spawn` maps to a non-overlapping, walkable spawn zone
   in the current map. Verified by `validate_multi_ped_runtime_plausibility()` (from
   `robot_sf/adversarial/runtime.py`).
2. **Route reachability**: `route_choice` produces a collision-free path from spawn to
   goal using the nav-mesh / waypoint graph. Verified by path planner query (A* on walkable
   graph) before simulation.
3. **Speed range**: `speed` ∈ [min_speed_mps, max_speed_mps] with non-zero minimum.
4. **Temporal consistency**: `start_time + pause_duration <= episode_duration_s`, and
   start_time ≥ 0.
5. **Group consistency**: `group_behavior` does not exceed `max_total_pedestrians` and
   group assignments are internally consistent (no circular references, valid targets).

A configuration is **degenerate** (and must be rejected) if:

- All NPCs have speed < 0.01 m/s (effective stalling).
- Spawn positions are clustered within `degenerate_cluster_radius_m` of each other
  (pathologically overlapping).
- Route length is zero (NPC never moves).
- The configuration is identical to a previously rejected configuration within tolerance
  (repeat-degenerate guard).

Rejected actions are fail-closed: the episode terminates with `reward = invalidity_penalty`
and the action is recorded in a `rejected_actions` log for diagnostics. They must not be
treated as successful adversarial evidence under the benchmark fallback policy
(`docs/context/issue_691_benchmark_fallback_policy.md`).

## First Deterministic Smoke / Spike Path

The recommended first executable spike (that does not require RL training) is:

1. **Define the action-to-config transformer**: a function that converts the proposed action
   space dict into a `SinglePedestrianDefinition` (or `MultiPedCandidateSpec`). Must validate
   all input ranges and fail closed on out-of-bounds values.
2. **Write a deterministic mock policy**: a hardcoded action (one NPC, spawn in front of
   robot, speed=1.5 m/s, no pause) that produces a known dangerous pedestrian config.
3. **Inject the mock config** into `make_pedestrian_env()` (or a new `make_adversarial_env()`
   factory) using the override mechanism from `materialize_multi_ped_single_pedestrian_overrides()`.
4. **Run one episode** with a frozen ORCA robot policy and verify:
   - The NPC spawns at the correct location and time
   - The simulation runs without error
   - The trace output records spawn, route, and collision events
   - The reward computation is non-zero and breaks down per-term
5. **Prove fail-closed rejection**: submit an invalid action (spawn on obstacle, speed=0)
   and verify that `invalidity_penalty` is returned with no simulation step.

The spike proves the action→config pipeline, the reward decomposition, and the rejection
criteria — all without any training, GPU, or learned behavior.

Before any broad RL adversary training, the generated action-to-config batches must also satisfy
the Issue #2568 learned-expansion gate: run through the Issue #2562 manifest-smoke path, summarize
validity, degeneracy, duplicate/novelty, perturbation, and planner-yield signals with the
Issue #2567 quality metrics, and keep the result labeled diagnostic unless a later certified
benchmark issue supplies durable benchmark proof.

## Stop Rule

Work on this issue is complete when all of the following hold:

1. This launch-packet scope note is reviewed and merged.
2. The action-space→config transformer interface is defined (this note or a follow-up YAML
   fixture).
3. The reward decomposition with per-term weights is documented (this note).
4. The validity and degeneracy rejection criteria are documented (this note).
5. The first deterministic smoke path is documented (this note).

Work **stops** at the proposal/launch-packet boundary. No RL adversary training, no
adversarial-behavior realism claim, no benchmark improvement, no planner ranking — those are
explicitly deferred to follow-up issues.

## Data and Compute Prerequisites (Future RL Training)

If RL adversary training is later attempted, the following prerequisites must be satisfied
first:

### Data
- [ ] A set of at least 3 map geometries with annotated spawn zones, walkable surfaces, and
      waypoint graphs
- [ ] At least 2 frozen robot planner policies available through the learned policy adapter
      interface (`docs/context/issue_1618_learned_policy_adapter_interface.md`)
- [ ] A benchmark-config manifest that enumerates the planner/scenario combinations for
      adversary training (adversary trains against multiple planners to avoid overfitting)
- [ ] Replay traces from the deterministic smoke as baseline comparison data
- [ ] An Issue #2567-style quality summary for the manifest batch that will seed or evaluate the
      learned adversary, linked through the Issue #2568 gate.

### Compute (estimate for a v0 proof-of-concept PPO adversary)
- [ ] CPU-only training for v0: feasible for 1-NPC adversary with ~4 GB RAM and ~1 hour per
      1M environment steps (single-threaded simulation)
- [ ] GPU (>= 4 GB VRAM) recommended for faster policy-network inference during training if
      the observation space includes occupancy grids or larger encoders
- [ ] ~20 GB disk for trace checkpoints, rejected-action logs, and evaluation artifacts
- [ ] Training runtime estimate: 4-24 hours for a small MLP adversary (2-3 layers, 64-128
      units) trained for 5-10M steps across 3-5 seeds, single CPU worker
- [ ] Evaluation: 1-2 hours per checkpoint for a 100-episode evaluation sweep across all
      training planners and held-out maps

These are v0 feasibility estimates. Full multi-NPC adversarial RL training may require
10-100x more compute and is out of scope until the v0 smoke passes.

## Claim Boundary

This is proposal and launch-packet evidence only.

It does **not**:
- Prove that an RL adversarial policy can generate useful or realistic adversarial
  pedestrian configurations in Robot SF
- Provide a trained adversarial policy, checkpoint, or inference code
- Replace the existing search-based adversarial module (`robot_sf/adversarial/`), the
  ego-pedestrian RL (`PedestrianEnv`), or the scripted adversarial force
  (`AdversarialPedForce`)
- Claim any benchmark improvement, planner robustness degradation, or adversarial coverage
  metric
- Claim human-behavior realism, behavioral diversity, or generated-pedestrian naturalness
- Change planner rankings or benchmark scenario sets

A reward decomposition records a proposed optimization target; the action-to-config
transformer, simulation smoke, and rejection-criteria spike remain separate gates that must
be completed before any adversary-training claim is made.

The Issue #2568 gate is an additional workflow gate: even after the transformer smoke exists,
broad RL expansion must show useful, non-degenerate manifest-batch behavior under the Issue #2567
metrics before training is treated as ready.

## Validation

```bash
# Validate referenced paths exist
test -f robot_sf/gym_env/pedestrian_env.py && echo "pedestrian_env OK"
test -f robot_sf/gym_env/reward.py && echo "reward.py OK"
test -f robot_sf/adversarial/runtime.py && echo "adversarial runtime OK"
test -f robot_sf/nav/map_config.py && echo "map_config OK"
test -f docs/context/issue_1457_adversarial_generation_protocol.md && echo "protocol OK"
test -f docs/context/issue_1618_learned_policy_adapter_interface.md && echo "adapter OK"
test -f docs/context/issue_691_benchmark_fallback_policy.md && echo "fallback OK"
test -f docs/context/policy_search/learned_policy_registry.md && echo "registry OK"

# Validate YAML evidence fixture
uv run python -c "import yaml; yaml.safe_load(open('docs/context/evidence/issue_2470/adversarial_ped_rl_proposal.yaml'))"

# Consistency check
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

The recommended next issue is an executable action-to-config transformer spike: define the
action space as a YAML schema, write the conversion function, inject a deterministic
hardcoded action into the existing `make_pedestrian_env()` path, run one ORCA episode, and
prove fail-closed invalidity rejection. Stop there before any RL training, realism claim,
or planner comparison.

The issue after that (if the spike passes) is a reward-term instrumentation issue: wire the
proposed reward terms into the PedestrianEnv reward function as a new named profile
(`adversarial_npc_v1`), verify per-term decomposition in trace output, and confirm that
the collision term dominates near_miss + low_progress for collision episodes.
