## Executive Summary

Create this as an **optimization-and-evidence project**, not as a “try more planners” project. The current benchmark already shows the key problem: `ppo` has the best success rate but too many collisions, while `orca` is safer but less successful. In the frozen campaign, `orca` reaches **18.44% success / 3.55% collision**, while `ppo` reaches **24.82% success / 9.93% collision**; the paper explicitly concludes that no planner dominates both safety and task completion.

The project should therefore search for a **Pareto-improving local planning stack**: higher success than `orca`, collision close to `orca`, and better scenario-stratified robustness than the current `ppo`. The strongest near-term direction is not pure RL. It is a **hybrid stack**:

```text
global/local waypoint tracking
+ ORCA / velocity-obstacle safety layer
+ local trajectory sampling or MPC-style scoring
+ scenario-type-aware parameterization
+ optional learned value/risk model
+ hard collision guard
```

The coding agent should operate in gated loops: implement one candidate, run nominal smoke tests, run stress slices, run the full matrix only if promising, generate a short interpretation report, then decide whether to promote, revise, or discard.

## 1. Core Project Definition

### Project name

```text
robot_sf_ll7_local_policy_search
```

### Scientific objective

Find a local navigation policy or planning stack for `robot_sf_ll7` that **strictly improves the current benchmark trade-off** under the existing AMV benchmark contract.

The paper defines the benchmark contract as scenarios, seeds, metrics, output schema, and aggregation protocol, and emphasizes that every reported number must be traceable from scenario matrix to episode JSON to aggregate tables.  The new project should preserve this contract and extend it, not bypass it.

### Practical objective

Produce one or more planner candidates that can be promoted from experimental status to a credible headline comparator.

### Non-goal

Do not chase a single scalar leaderboard score. The paper explicitly treats multi-objective interpretation as necessary because success alone can hide collision risk, discomfort, or unstable motion.

## 2. Definition of “Substantially Better”

Use a hard promotion gate. A candidate is only considered substantially better if it satisfies one of these dominance criteria.

### Tier A: Paper-worthy Pareto improvement

A candidate must satisfy:

```text
success_rate >= 0.30
collision_rate <= 0.05
near_miss_rate <= current_ppo_near_miss_rate
ttg_norm <= current_ppo_ttg_norm + 0.03
```

This means it beats `orca` clearly on success, stays close to `orca` on safety, and avoids becoming a “fast but unsafe” planner.

### Tier B: Strong safety-first improvement

```text
success_rate >= current_orca_success + 0.08
collision_rate <= current_orca_collision + 0.02
```

Using the frozen numbers:

```text
success_rate >= 0.264
collision_rate <= 0.055
```

### Tier C: Strong learning-policy repair

```text
success_rate >= current_ppo_success
collision_rate <= current_ppo_collision / 2
```

Using the frozen numbers:

```text
success_rate >= 0.248
collision_rate <= 0.050
```

### Scenario-stratified gate

The candidate must not hide failure in a subset. The paper reports that the current `ppo` collision exposure concentrates in classic constrained-geometry scenarios, while `orca` remains more consistent across classic and Francis motifs.  Therefore require:

```text
classic_collision_rate <= 0.07
francis_collision_rate <= 0.05
classic_success_rate >= current_orca_classic_success
francis_success_rate >= current_orca_francis_success
```

This gate matters more than the aggregate score.

## 3. Recommended Project Structure

Add a project directory like this:

```text
docs/projects/local_policy_search/
  README.md
  project_contract.md
  candidate_registry.yaml
  experiment_ledger.md
  failure_taxonomy.md
  promotion_gates.md
  agent_runbook.md
  reports/
    YYYY-MM-DD_candidate_name.md

configs/policy_search/
  nominal_sanity_matrix.yaml
  stress_slice_matrix.yaml
  full_matrix_eval.yaml
  seed_schedules.yaml
  scoring_profiles/
    safety_first.yaml
    balanced.yaml
    success_first_guarded.yaml

configs/planners/
  hybrid_orca_sampler.yaml
  mpc_clearance_sampler.yaml
  dwa_social.yaml
  teb_like_elastic_band.yaml
  risk_guarded_ppo.yaml
  planner_selector.yaml

robot_sf/planner/
  hybrid/
    candidate_base.py
    velocity_sampler.py
    trajectory_rollout.py
    social_costs.py
    safety_guard.py
    risk_model.py
    planner_selector.py

tools/policy_search/
  run_candidate.py
  compare_candidate.py
  build_failure_report.py
  plot_pareto_front.py
  promote_candidate.py
```

The important artifact is `candidate_registry.yaml`. The agent should never implement planners informally. Each candidate needs a registry entry.

```yaml
candidates:
  hybrid_orca_sampler_v1:
    status: proposed
    family: hybrid_model_based
    hypothesis: >
      ORCA provides collision avoidance, while trajectory sampling improves goal progress
      in constrained geometry.
    parent_baseline: orca
    expected_gain:
      success_rate: "+0.08 absolute over ORCA"
      collision_rate: "<= ORCA + 0.02"
    risk:
      - overly conservative in bottlenecks
      - oscillatory in narrow passages
    required_tests:
      - nominal_sanity
      - stress_slice
      - full_matrix_if_promising
    promotion_gate: tier_b
    owner_agent_iteration: null
    latest_report: null
```

## 4. Policy Search Strategy

### Phase 0: Freeze evaluation infrastructure

Before implementing candidates, make the benchmark pipeline frictionless.

Required outputs:

```text
one-command candidate evaluation
automatic aggregate comparison
scenario-stratified report
Pareto plot
failure taxonomy
candidate registry update
```

The agent should first implement scripts that answer:

```text
Did the candidate improve success?
Did it increase collisions?
Where did it fail?
Which scenario family caused the regression?
Is the result seed-stable enough to continue?
```

This aligns with the paper’s claim that the value of the benchmark is not a single planner ranking, but stratifiable and verifiable evidence.

### Phase 1: Establish a nominal sanity matrix

The paper identifies a missing nominal-scenario calibration layer: the frozen matrix is stress-test oriented and does not include an easier deployment-like benchmark for estimating nominal planner success.  Add this immediately.

Purpose:

```text
A planner that cannot solve easy shared-space cases should not be evaluated on the full stress matrix.
```

Suggested nominal scenarios:

```text
straight corridor, no pedestrians
straight corridor, sparse same-direction pedestrians
wide crossing, low density
doorway, single pedestrian
parallel traffic, low density
gentle turn around static obstacle
```

Gate:

```text
nominal_success_rate >= 0.80
nominal_collision_rate <= 0.02
```

A planner that fails this gate should be debugged, not benchmarked.

### Phase 2: Build a fast stress slice

Use a reduced version of the full matrix containing the known hard cases:

```text
classic_bottleneck_medium
classic_doorway_medium
classic_group_crossing_medium
classic_t_intersection_medium
francis2023_blind_corner
francis2023_intersection_wait
francis2023_parallel_traffic
francis2023_crowd_navigation
```

Purpose:

```text
Catch failure modes cheaply before running all 47 scenarios × seeds × planners.
```

This slice should include both classic constrained geometry and Francis interaction motifs because the paper shows that aggregate performance can hide scenario-type-specific collision risk.

### Phase 3: Candidate planner families

Implement candidates in this order.

#### Candidate 1: `hybrid_orca_sampler_v1`

This is the highest-probability improvement path.

Architecture:

```text
Input:
  robot state
  goal / waypoint
  nearby agents
  obstacle segments

Generate:
  candidate velocity commands (v, omega)

For each candidate:
  rollout short horizon
  compute goal progress
  compute obstacle clearance
  compute pedestrian clearance
  compute time-to-collision
  compute smoothness penalty
  compute ORCA feasibility / reciprocal collision penalty

Select:
  lowest-cost safe command

Guard:
  if predicted collision within horizon, override with safest braking/yaw command
```

Cost function:

```text
J =
  w_goal       * goal_distance_cost
+ w_heading    * heading_error_cost
+ w_clearance  * obstacle_clearance_cost
+ w_ped        * pedestrian_clearance_cost
+ w_ttc        * time_to_collision_cost
+ w_smooth     * command_change_cost
+ w_progress   * negative_progress_reward
+ w_deadlock   * low_progress_penalty
```

Why this is promising:

```text
ORCA already has the cleanest safety profile.
The sampler can recover progress where ORCA is too conservative.
The safety guard prevents PPO-like collision regressions.
```

#### Candidate 2: `mpc_clearance_sampler_v1`

A more explicit rollout planner.

Use a lattice of controls:

```text
v ∈ {0.0, 0.2, 0.4, 0.6, 0.8}
omega ∈ {-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0}
horizon ∈ 2.0–4.0 s
dt ∈ 0.2 s
```

Evaluate each rollout against:

```text
static collision
dynamic-agent collision
minimum clearance
TTC
goal progress
path deviation
smoothness
```

This planner should become the main interpretable challenger.

#### Candidate 3: `risk_guarded_ppo_v1`

Do not retrain first. Wrap the existing PPO.

Architecture:

```text
ppo proposes action
safety guard simulates action over short horizon
if unsafe:
  replace with best safe sampled action
else:
  execute PPO action
```

Purpose:

```text
Test whether PPO's collision problem can be repaired by action shielding.
```

Promotion gate:

```text
success >= current_ppo_success - 0.03
collision <= 0.05
```

This is attractive because the current PPO has the best success rate but poor safety. The paper explicitly states that PPO’s higher success is paired with substantially higher collision exposure than ORCA.

#### Candidate 4: `scenario_adaptive_orca_v1`

Tune ORCA parameters by scenario class:

```text
classic constrained geometry:
  larger clearance
  lower max speed
  stronger wall avoidance
  stronger deadlock escape

Francis flowing motifs:
  higher progress weight
  smoother lateral passing
  less conservative reciprocal response
```

This should be treated carefully because scenario-aware tuning can overfit. The agent must report whether parameter changes generalize across seeds and scenario families.

#### Candidate 5: `planner_selector_v1`

A meta-policy that selects among safe subplanners:

```text
if narrow geometry:
  use mpc_clearance_sampler
elif dense crowd:
  use orca or social-force-biased sampler
elif open crossing:
  use risk_guarded_ppo or progress-biased sampler
else:
  use hybrid_orca_sampler
```

This should come after individual planners are stable.

#### Candidate 6: `learned_risk_model_v1`

Train a lightweight risk estimator from episode traces:

```text
input:
  local occupancy / agent features / candidate command / horizon rollout features

target:
  collision within horizon
  near miss within horizon
  low progress / deadlock
```

Use it only as an additional cost term, not as the sole planner.

```text
J_total = J_model_based + w_risk * predicted_risk
```

This avoids making the learned component safety-critical without a guard.

## 5. Experiment Funnel

The coding agent should use a fixed funnel.

```text
Stage 1: Static checks
  - config validates
  - planner imports
  - deterministic smoke test
  - command bounds respected

Stage 2: Nominal sanity matrix
  - easy scenarios
  - 3 seeds
  - must reach high success and low collision

Stage 3: Stress slice
  - 8–12 hard scenarios
  - 3 seeds
  - must beat at least one baseline on Pareto tradeoff

Stage 4: Full frozen matrix
  - 47 scenarios
  - seeds 111,112,113
  - compare against ORCA, PPO, goal

Stage 5: Robustness extension
  - add seeds 114,115 or S5 set
  - run only if Stage 4 is promising

Stage 6: Promotion decision
  - promote, revise, or discard
```

The paper notes that single-seed estimates can misrepresent success rates by up to 15 percentage points, so seed-aware evaluation is not optional.

## 6. Progress Tracking

Use a compact status table in `docs/projects/local_policy_search/experiment_ledger.md`.

```markdown
# Local Policy Search Experiment Ledger

| Date | Candidate | Stage | Success | Collision | Near Miss | TTG Norm | Classic Coll. | Francis Coll. | Decision | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-04-29 | orca | baseline | 0.184 | 0.036 | 4.333 | 0.963 | 0.030 | 0.040 | reference | frozen paper |
| 2026-04-29 | ppo | baseline | 0.248 | 0.099 | 3.525 | 0.927 | 0.150 | 0.050 | unsafe success reference | frozen paper |
| TBD | hybrid_orca_sampler_v1 | stress_slice | TBD | TBD | TBD | TBD | TBD | TBD | TBD | reports/... |
```

Each candidate report should be short and standardized.

```markdown
# Candidate Report: hybrid_orca_sampler_v1

## Decision
Promote / Revise / Discard

## Hypothesis
...

## Implementation Summary
...

## Evaluation Scope
- Matrix:
- Seeds:
- Planner config:
- Commit:
- Command:

## Aggregate Results
| Planner | Success | Collision | Near Miss | TTG Norm |
|---|---:|---:|---:|---:|

## Scenario-Stratified Results
| Scenario group | Success | Collision | Near Miss | Main failure |
|---|---:|---:|---:|---|

## Failure Modes
- ...

## Interpretation
...

## Next Action
...
```

## 7. Failure Taxonomy

The agent should classify every failed episode into one dominant failure mode.

```yaml
failure_modes:
  static_collision:
    description: robot hits wall or obstacle
  pedestrian_collision:
    description: robot collides with dynamic agent
  near_miss_intrusive:
    description: no collision, but clearance threshold violated
  deadlock:
    description: low progress for long interval
  oscillation:
    description: repeated command sign changes or path dithering
  timeout_low_progress:
    description: no collision but fails to reach goal
  wrong_waypoint_behavior:
    description: planner follows poor heading or wrong local target
  bottleneck_yield_failure:
    description: planner enters constrained space without resolving priority
  overconservative_stop:
    description: safe but fails due to excessive stopping
```

This is essential because “better planner” is too vague. The agent must learn which failure mode it is reducing.

## 8. Exact Coding-Agent Instruction

You can give the following directly to a coding agent.

```markdown
# Mission: Find a Strong Local Navigation Policy for robot_sf_ll7

You are working in the `robot_sf_ll7` repository. Your task is to create an iterative local-policy search project that finds a planning stack that substantially improves over the current frozen benchmark results.

## Baseline facts

The current paper-facing frozen campaign reports:

- `orca`: success ≈ 0.1844, collision ≈ 0.0355
- `ppo`: success ≈ 0.2482, collision ≈ 0.0993
- `goal`: success ≈ 0.0142, collision ≈ 0.2411

Interpretation:

- `orca` is the safety reference.
- `ppo` is the success reference but has too much collision exposure.
- A good candidate must improve success without inheriting PPO's collision rate.
- Aggregate results are insufficient. Always report classic-vs-Francis and scenario-family splits.

## Objective

Implement an iterative policy-search workflow and use it to develop candidate planners until one candidate satisfies at least one promotion gate:

### Tier A
- success_rate >= 0.30
- collision_rate <= 0.05

### Tier B
- success_rate >= 0.264
- collision_rate <= 0.055

### Tier C
- success_rate >= 0.248
- collision_rate <= 0.050

Additionally:
- classic_collision_rate <= 0.07
- francis_collision_rate <= 0.05

## Required project files

Create:

- `docs/projects/local_policy_search/README.md`
- `docs/projects/local_policy_search/project_contract.md`
- `docs/projects/local_policy_search/candidate_registry.yaml`
- `docs/projects/local_policy_search/experiment_ledger.md`
- `docs/projects/local_policy_search/failure_taxonomy.md`
- `docs/projects/local_policy_search/promotion_gates.md`
- `docs/projects/local_policy_search/agent_runbook.md`

Create or extend tools:

- `tools/policy_search/run_candidate.py`
- `tools/policy_search/compare_candidate.py`
- `tools/policy_search/build_failure_report.py`
- `tools/policy_search/plot_pareto_front.py`
- `tools/policy_search/promote_candidate.py`

## Required evaluation funnel

Every candidate must pass this funnel:

1. Static checks:
   - imports
   - config validation
   - command bound validation
   - deterministic smoke episode

2. Nominal sanity matrix:
   - easy scenarios
   - 3 seeds
   - required: success >= 0.80 and collision <= 0.02

3. Stress slice:
   - hard subset of classic and Francis scenarios
   - 3 seeds
   - report aggregate and scenario-stratified metrics

4. Full frozen matrix:
   - classic_interactions_francis2023
   - seeds 111,112,113
   - compare against `orca`, `ppo`, and `goal`

5. Robustness extension:
   - only for promising candidates
   - add two extra seeds if runtime permits

## Candidate implementation order

Implement candidates in this order:

1. `hybrid_orca_sampler_v1`
   - ORCA-informed velocity sampling
   - short-horizon rollout
   - goal progress cost
   - obstacle clearance cost
   - pedestrian clearance cost
   - TTC cost
   - smoothness cost
   - emergency safety guard

2. `mpc_clearance_sampler_v1`
   - lattice rollout over `(v, omega)`
   - short horizon
   - explicit collision and clearance scoring

3. `risk_guarded_ppo_v1`
   - use PPO action as proposal
   - simulate short horizon
   - override unsafe actions with best safe sampled action

4. `scenario_adaptive_orca_v1`
   - scenario-family-aware ORCA parameters
   - must report overfitting risk

5. `planner_selector_v1`
   - choose among stable subplanners based on local scene features

6. `learned_risk_model_v1`
   - train a lightweight risk estimator from traces
   - use only as an auxiliary cost term, never without a hard guard

## Required output for every iteration

For every candidate and run, produce:

- candidate config
- command used
- git commit
- matrix name
- seed schedule
- aggregate result table
- scenario-stratified result table
- failure taxonomy counts
- short interpretation
- decision: promote, revise, discard

Write the report to:

`docs/projects/local_policy_search/reports/YYYY-MM-DD_<candidate>.md`

Update:

- `candidate_registry.yaml`
- `experiment_ledger.md`

## Decision rules

Do not run the full matrix for candidates that fail nominal sanity.

Do not promote candidates that improve success by increasing collision beyond the gate.

Do not use SNQI as the primary promotion criterion. Use component metrics first.

Do not hide failures in aggregate metrics. Always inspect classic constrained-geometry scenarios separately from Francis motifs.

## First implementation target

Start with `hybrid_orca_sampler_v1`.

The expected improvement mechanism is:

- keep ORCA-like safety behavior,
- add trajectory rollout to improve progress,
- add deadlock escape behavior for bottlenecks,
- use a hard safety guard to avoid PPO-like collision exposure.

End the first iteration with a report that states whether the candidate should be promoted, revised, or discarded.
```

## 9. Technical Design for `hybrid_orca_sampler_v1`

The first serious candidate should be implemented cleanly and kept interpretable.

### Inputs

```python
@dataclass
class LocalPlanningState:
    robot_pose: Pose2D
    robot_velocity: Twist2D
    goal_position: np.ndarray
    agents: list[AgentState]
    obstacles: list[ObstacleSegment]
    dt: float
```

### Outputs

```python
@dataclass
class LocalCommand:
    v: float
    omega: float
    debug: dict
```

### Candidate generation

```python
v_samples = np.linspace(0.0, max_v, n_v)
omega_samples = np.linspace(-max_omega, max_omega, n_omega)

candidate_commands = [(v, omega) for v in v_samples for omega in omega_samples]
```

### Rollout scoring

```python
def score_rollout(rollout, state, previous_command, weights):
    return (
        weights.goal * goal_distance_cost(rollout, state.goal_position)
        + weights.heading * heading_cost(rollout, state.goal_position)
        + weights.obstacle * obstacle_clearance_cost(rollout, state.obstacles)
        + weights.pedestrian * pedestrian_clearance_cost(rollout, state.agents)
        + weights.ttc * ttc_cost(rollout, state.agents)
        + weights.smoothness * smoothness_cost(rollout, previous_command)
        + weights.deadlock * deadlock_cost(rollout)
    )
```

### Hard safety filter

Before scoring, reject rollouts that violate:

```text
min_static_clearance < robot_radius + safety_margin
min_dynamic_clearance < robot_radius + agent_radius + safety_margin
predicted_ttc < ttc_min
```

If all candidates are unsafe:

```text
choose controlled braking / rotate-away command with maximum predicted clearance
```

This fallback must be explicit and logged. Silent fallback behavior makes interpretation impossible.

## 10. Research Logic Behind the Candidate Order

The current benchmark evidence suggests the search should start from safety and add progress, not start from success and try to repair safety.

Reason:

```text
ORCA already provides low collision exposure.
PPO already provides higher success but unsafe collision exposure.
A hybrid planner can use ORCA-like safety constraints while adding progress-optimized trajectory scoring.
```

The paper’s limitation section also identifies exactly the extensions this project should implement: broader planner coverage, paired nominal/stress evaluation, cross-kinematics evaluation, and scenario-stratified reporting.

## 11. What Not To Do

Do not let the agent spend weeks on a monolithic RL training pipeline before the evaluation funnel exists.

Do not accept “higher success” if collision also rises.

Do not compare planners on different seeds or scenario subsets.

Do not tune only on the full matrix without a held-out sanity/stress split.

Do not promote a planner because its mean score looks good while classic constrained scenarios remain unsafe.

Do not use SNQI as the main optimization target. The paper states that SNQI is retained as an implementation-level aid, not a calibrated scientific endpoint.

## 12. Minimal First Milestone

The first useful milestone is not a new planner. It is this:

```text
Given any planner config, run:
  nominal sanity
  stress slice
  full matrix if eligible

Then automatically produce:
  aggregate table
  ORCA/PPO deltas
  classic-vs-Francis split
  failure taxonomy
  promote/revise/discard recommendation
```

Only after this exists should the agent iterate aggressively over planner ideas.

## 13. Recommended GitHub Issues

Create these issues:

```text
# Local Policy Search: project scaffold and promotion gates
# Local Policy Search: nominal sanity matrix
# Local Policy Search: stress slice matrix
# Local Policy Search: candidate registry and experiment ledger
# Local Policy Search: automatic failure taxonomy
# Local Policy Search: hybrid_orca_sampler_v1
# Local Policy Search: mpc_clearance_sampler_v1
# Local Policy Search: risk_guarded_ppo_v1
# Local Policy Search: scenario_adaptive_orca_v1
# Local Policy Search: planner_selector_v1
# Local Policy Search: learned_risk_model_v1
# Local Policy Search: full-matrix promotion report
```

Each issue should require:

```text
implementation
unit/smoke test
benchmark command
result table
interpretation
next decision
```
