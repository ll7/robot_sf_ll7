# Pedestrian Goal-Force Prediction: Research Basis and Implementation Programme

**Status:** proposal / design-only / not benchmark evidence
**Canonical owner:** [issue #8060](https://github.com/ll7/robot_sf_ll7/issues/8060)
**Design pull request:** [PR #8058](https://github.com/ll7/robot_sf_ll7/pull/8058)
**Initial repository snapshot:** `895d396abb47757b0742cab6b2d677dd00eb80ae` on 2026-08-30
**Source-contract revalidation:** current `main` at `f94323e5f8fa478538104cedb8916b8721033573` on 2026-08-31

Plain-language summary: Robot SF can calculate the forces that it gives pedestrians, but a planner-facing observation does not reveal each pedestrian's private destination. The proposed capability reconstructs the part of motion not explained by observable interaction forces, maintains uncertainty over local waypoints and final routes, and resets that belief only when evidence supports a real segment change.

This note is governed by the [context-note workflow](README.md), is listed in the [context index](INDEX.md), and uses the current [observation contract](../dev/observation_contract.md). It continues the bounded metadata work from issue #4164 while preserving the broader-work boundary recorded in the [#4164 closure audit](evidence/issue_4164_closure_audit_2026-07-04.md).

## 1. Decision and claim boundary

Build a stateful, hierarchical `GoalBelief` for every observation-derived pedestrian track:

```text
policy-visible observations
  -> stable track and frame normalization
  -> causal motion estimate
  -> reconstructable non-goal force terms
  -> uncertain residual goal-force estimate
  -> active-waypoint posterior
  -> final-route/destination posterior
  -> arrival and change-point state
  -> existing forecast and planner interfaces
```

The estimator must keep four quantities separate:

1. the goal-directed force that currently explains motion;
2. the active force-generating waypoint or open direction;
3. the longer-term route or final destination;
4. the motion mode, including ordinary tracking, ambiguity, waypoint progress, arrival, stop/restart, track loss, or a genuine route change.

A narrow exactness claim is admissible only when all of the following are true:

- the post-behaviour, pre-force state is known;
- every force stage used by the selected pedestrian model is captured;
- the integration scheme and time interval are known;
- no velocity cap censors the force magnitude;
- no unmodelled controller mutation lies inside the observed transition;
- pedestrian identity is stable;
- the active force-generating goal is fixed during the force evaluation;
- the relevant force parameters are known.

Outside that envelope, the correct result is a calibrated distribution, a censored estimate, or an explicit `unknown` / `unavailable` state. This programme must not claim that simulator traces establish calibrated human intention, that one force direction uniquely identifies an endpoint, or that a prediction improvement implies safer planning without a separate paired closed-loop comparison.

## 2. Current repository truth

### 2.1 Existing goal-posterior plumbing and delivered baseline contracts

`robot_sf/prediction/goal_intention.py` contains a useful heading-likelihood calculation over explicit candidate points. The historical environment adapter constructed one candidate from the true PySocialForce goal columns, which is an oracle identity mapping, not deployable inference.

Delivered progress across early programme waves:
- Issue [#8068](https://github.com/ll7/robot_sf_ll7/issues/8068) delivered the actor-safe one-frame heading posterior baseline (`robot_sf/prediction/goal_intention.py`), rejecting oracle state.
- Issue [#8072](https://github.com/ll7/robot_sf_ll7/issues/8072) delivered the inverse goal-force estimator (`robot_sf/prediction/goal_force_inverse_dynamics.py`).
- Issue [#8073](https://github.com/ll7/robot_sf_ll7/issues/8073) delivered the public goal candidate provider (`robot_sf/prediction/goal_candidate_provider.py`) with compact receipt [`evidence/issue_8073_goal_candidate_provider_smoke_2026-09-01.json`](evidence/issue_8073_goal_candidate_provider_smoke_2026-09-01.json).
- Issue [#8205](https://github.com/ll7/robot_sf_ll7/issues/8205) delivered the tracker goal belief adapter.

Required disposition:

- retain an explicitly named oracle helper for tests and upper-bound analysis;
- add an observation-only constructor that cannot accept simulator state;
- require externally generated candidates plus a first-class `unknown` hypothesis;
- make provenance machine-readable;
- make actor-only adapters reject oracle-source beliefs.

### 2.2 Exact planner-visible observation contract

The default `SOCNAV_STRUCT` observation does not already provide a stable tracked-agent stream.
When `ObservationVisibilitySettings.include_track_ids` and its tracking configuration are
enabled, the channel can expose observation-derived episode-local IDs, but the option is
default-off and the structured payload still does not expose the full tracking state needed by
an estimator.

| Field | Current semantics | Consequence for goal inference |
| --- | --- | --- |
| `pedestrians.positions` | world-frame positions, nearest-first, fixed-size/padded rows | array index is not a persistent identity |
| `pedestrians.velocities` | robot-ego-frame velocities | rotate to world using the same-step robot heading before differentiation |
| `pedestrians.count` | count represented by the current structured payload | does not provide stable IDs or per-row age |
| `robot.position`, `robot.velocity_xy`, goals | world frame | usable after timestamp alignment |
| `robot.heading` | wrapped world heading | required for the pedestrian-velocity transform |
| `sim.timestep` | simulation-step duration | not an observation timestamp |

Stable pedestrian IDs are therefore available only through an explicit observation-derived
tracking option. Observation timestamps, visibility-age masks, association confidence, and a
durable lost-track identity are not part of the default structured payload (and ambiguous rows
remain unavailable to the presentation channel). They are implementation prerequisites, not
fields the estimator may assume exist without enabling and validating the tracking seam.

Benchmark observation levels remain separate provenance contracts:

- `oracle_full_state` is privileged and evaluator-only;
- `tracked_agents_no_noise` and `tracked_agents_with_noise` are tracked-agent benchmark conditions;
- `occluded_partial_state` carries partial-observation assumptions;
- `lidar_2d` is not interchangeable with tracked pedestrian state.

An estimator that needs temporal identity must explicitly consume the opt-in episode-local
`track_id` values or own an equivalent observation-only tracking seam; it must never infer
identity from row positions. The observation-derived tracker must be evaluated separately
against simulator identity.

### 2.3 Force pipeline and the meaning of `last_ped_forces`

The active Robot SF wrapper executes the pedestrian part of a step in this order:

```text
A. snapshot pre-behaviour state
B. behaviour.step() may mutate goal, position, velocity, group or population state
C. snapshot post-behaviour / pre-force state
D. evaluate configured force objects
E. apply residual-adversary processing
F. save last_ped_forces
G. _step_pedestrians applies model-specific additions, replacements or transforms
H. integrate and cap pedestrian velocity
I. apply robot action after pedestrian integration
```

`last_ped_forces` is therefore the force array handed into `_step_pedestrians`, not a universal synonym for the final force used by every supported pedestrian model. The final integration force can still differ through TTC-predictive addition, Zanlungo replacement/combination, anisotropic field-of-view transformation, or a dedicated hybrid Social Force Model integration path.

The oracle contract must distinguish these stages:

```text
f_registry_total
  = sum(exact arrays returned by each configured force object)

f_after_residual
  = exact array returned after residual-adversary processing

f_variant_delta_or_replacement
  = model-specific additive, replacement, transform or dedicated-integrator record

f_final_pre_cap
  = exact translational force supplied to the selected integration rule

v_uncapped
  = integration-scheme-specific velocity before maximum-speed projection

v_applied
  = velocity actually stored after capping
```

Required invariant:

```text
replay(component records, residual operation, variant operation)
  == f_final_pre_cap
```

The invariant must hold for both `Simulator` and `PedSimulator`. Components must be evaluated once; tracing may not call force objects again merely to log them.

Model-variant semantics must be typed rather than flattened into a misleading additive list:

- `identity`: final force equals the wrapper input;
- `additive_delta`: a named delta is added;
- `replacement_total`: the variant supplies a replacement total;
- `transform_total`: a function transforms an existing total;
- `dedicated_integrator`: the model uses additional state/heading dynamics that require their own replay contract.

### 2.4 State actually used by force evaluation

A behaviour controller can advance a waypoint, respawn/reposition a group, hold a pedestrian by zeroing velocity, release a wait, or change group/role state before forces are evaluated. Recording only the pre-behaviour state and post-behaviour goal is insufficient.

The force-time oracle snapshot must include:

- post-behaviour position and velocity for every pedestrian;
- post-behaviour active goal;
- route, waypoint, group and segment identifiers where available;
- robot poses/states read by pedestrian-robot and adversarial force callables;
- the commanded robot action separately, noting that it is applied only after pedestrian integration;
- mutation flags for position, velocity, goal, group, population, hold/wait, respawn and role changes;
- the exact force registry and selected model-variant operation;
- post-integration position and applied velocity.

Inverse-force exactness must start from the post-behaviour, pre-force state. An actor transition that crosses an unobserved reposition, hold-induced velocity reset, respawn, population replacement, or equivalent controller mutation is not an exact finite-difference force row. Such rows must be excluded from the exactness denominator or modelled as a separate exogenous state jump.

### 2.5 Desired-force parameter sources

For the current `DesiredForce` implementation, the relevant sources are:

- preferred speed: `peds.max_speeds[i]` for pedestrian `i`;
- relaxation time: the scalar `DesiredForceConfig.relaxation_time` used by the force object;
- desired-force multiplier: the scalar desired-force `factor` used by the force object;
- direction: the force-time position and active goal.

The state `tau()` column is not the source consumed by `DesiredForce` and must not be labelled as the true relaxation-time parameter for this inversion.

The oracle may record exact `max_speeds`, relaxation time, force factor and cap state. The actor path may use only public configuration and observation-derived priors/estimates. These form separate evaluation arms.

### 2.6 Oracle cap truth versus actor censoring

The oracle trace may expose:

```text
speed_cap_active: bool
maximum_speed_mps
v_uncapped_xy
v_applied_xy
```

The actor estimator must not read that hidden Boolean. It should emit an observation-derived status such as:

```text
censoring_status = clear | possible | unknown
```

- `clear`: the actor can establish that the row is not near a public/estimated cap within tolerance;
- `possible`: observed speed and known/estimated limits make capping plausible;
- `unknown`: the cap or enough state is unavailable.

Force-magnitude scoring must keep oracle-confirmed capped rows separate. Direction may still carry information only when the selected censoring model justifies it.

## 3. Mathematical target

Let the post-behaviour, pre-force state be position `p_t`, velocity `v_t`, active goal `g_t`, preferred speed `s_i = peds.max_speeds[i]`, scalar relaxation time `tau`, desired-force factor `alpha`, and time interval `dt`.

Outside the goal-threshold braking region:

```text
f_goal_t = alpha * (s_i * d_t - v_t) / tau

d_t = (g_t - p_t) / ||g_t - p_t||
```

Inside the goal threshold, the desired-force branch is braking rather than attraction. The exact branch and threshold state belong in the oracle trace.

For an eligible, uncapped transition:

```text
a_observed_t = (v_applied_(t+1) - v_t) / dt

z_goal_t = a_observed_t - sum(actor_reconstructable_non_goal_forces_t)

u_desired_hat_t = v_t + (tau_hat / alpha_hat) * z_goal_t

d_goal_hat_t = u_desired_hat_t / ||u_desired_hat_t||
```

The equality between finite-difference acceleration and force is integration-scheme dependent and valid only under the exact replay conditions above. The implementation must derive the equation from the selected integrator rather than assume one universal update.

### 3.1 Identifiability

Even exact direction does not identify endpoint distance. Every point

```text
g = p_t + lambda * d_goal_hat_t, lambda > 0
```

has the same desired direction until near-goal behaviour becomes informative. Therefore:

- goal-force and direction are primary continuous targets;
- active waypoint and final destination are separate probabilistic targets;
- same-ray candidates must remain ambiguous absent independent evidence;
- open-ray candidates are valid outputs;
- `unknown/not-in-candidate-set` is mandatory;
- endpoint error must not replace direction, route and calibration metrics.

### 3.2 Uncertainty

Actor uncertainty must include, at minimum:

```text
R_goal
  = R_motion_differentiation
  + R_tracking
  + R_reconstructed_force
  + R_parameter
  + R_unobserved_contributors
  + R_model_mismatch
  + R_censoring
```

An unavailable contributor increases uncertainty or makes the estimate unavailable. It is not silently replaced by an exact zero.

## 4. Actor/oracle separation

### 4.1 Actor-admissible inputs

- current and causal historical policy-visible observations;
- observation-derived `track_id` and association uncertainty;
- same-step robot pose/velocity/action information actually available to the policy;
- public map geometry and public candidate semantics;
- public force-model configuration;
- prior actor belief state;
- causal time maintained by the environment/service, not inferred from `sim.timestep` as a timestamp.

### 4.2 Oracle-only inputs

- simulator pedestrian identity;
- pre- and post-behaviour state;
- true active goal, route, waypoint and final destination;
- exact force-object component arrays;
- residual and variant operations;
- exact preferred speed and force parameters;
- cap truth, uncapped velocity and applied velocity;
- behaviour mutation and goal-change labels.

### 4.3 Mandatory leakage canaries

1. Randomize all oracle goals while holding actor observations fixed; actor bytes must remain identical.
2. Remove oracle route, waypoint, identity, force and cap fields; actor inference must still execute.
3. Actor constructors reject generic simulator/PySocialForce objects.
4. Actor feature schemas contain no unused oracle columns or simulator IDs.
5. Candidate generation cannot access assigned pedestrian routes or goals.
6. An actor-only planner/forecast adapter rejects oracle-source beliefs.
7. A future-data timestamp or history entry fails closed.

## 5. Oracle transition contract

A versioned transition row should contain the following logical groups.

### Identity and provenance

```text
schema_version
repository_commit
backend
integration_scheme
scenario_id
episode_id
seed
step_index
dt_seconds
simulator_pedestrian_id
actor_track_id_if_joined
goal_segment_id
config_hash
```

### Snapshot A: pre-behaviour

```text
position_pre_behavior_xy
velocity_pre_behavior_xy
goal_pre_behavior_xy
waypoint_pre_behavior
route_pre_behavior
```

### Snapshot B: post-behaviour / force-time

```text
position_force_time_xy
velocity_force_time_xy
goal_force_time_xy
waypoint_force_time
route_force_time
robot_states_force_time[]
behavior_mutation_flags[]
population_generation
```

### Force stages

```text
force_components[]:
  component_id
  component_type
  source_entity
  operation = base_component
  force_xy

f_registry_total_xy
residual_operation
residual_delta_or_replacement_xy
f_after_residual_xy
variant_id
variant_operation
variant_delta_replacement_or_transform_record
f_final_pre_cap_xy
```

### Integration result

```text
maximum_speed_mps
v_uncapped_xy
speed_cap_active
v_applied_xy
position_after_integration_xy
goal_threshold_active
desired_force_branch
```

### Semantic event

```text
goal_switch_kind = none | waypoint_advance | route_redirect | arrival |
                   respawn | hold | stop | restart | role_update | unknown
label_source = behavior_event | exact_state_transition | inferred
exact_inverse_eligible
ineligibility_reasons[]
```

The trace is evaluator infrastructure. It must not be placed in actor observations, model inputs, or nominal planner metadata.

## 6. Proposed actor `GoalBelief`

The actor-facing contract should expose finite, validated fields and explicit unavailable states:

```text
schema_version
track_id
timestamp_s
source = observation_only
coordinate_frame
history_steps
track_confidence

force_mean_xy
force_covariance_2x2
desired_velocity_mean_xy
desired_direction_mean_xy
direction_concentration
censoring_status

candidate_goals[]:
  candidate_id
  candidate_source
  candidate_role
  position_xy | null
  open_direction_xy | null
  route_signature | null
  active_waypoint_probability
  final_destination_probability

unknown_candidate_probability
active_waypoint_entropy
route_entropy
arrival_probability
change_probability
mode
model_residual_norm
last_reset_step
last_reset_reason
blockers[]
config_hash
```

`source=oracle_upper_bound` belongs to a separate evaluator type or a source value that actor-only adapters reject. Probability mass, including unknown, must normalize. Covariances must be finite, symmetric and positive semidefinite within declared tolerance.

## 7. Observation-derived tracking and histories

Use one internal global Cartesian frame. Convert current pedestrian ego velocities with the same-step robot heading before storing history.

Initial tracker:

1. constant-velocity Kalman prediction;
2. position/velocity covariance-aware gating;
3. deterministic Hungarian assignment;
4. tentative/confirmed/lost/retired lifecycle;
5. bounded occlusion memory and reacquisition;
6. monotonic episode-local track IDs;
7. association confidence propagated into inference.

Each history transition must retain contemporaneous context:

- positions/velocities and covariances;
- same-step robot state;
- neighbour snapshot and masks;
- observation-level provenance;
- actual causal timestamp and `dt`;
- candidate-set digest;
- component-availability state.

Earlier force terms may not be recomputed from the current scene.

Evaluate histories `H in {1, 2, 3, 5, 8, 16}`:

| H | Evidence | Role |
| ---: | --- | --- |
| 1 | current velocity and candidates | heading-only baseline; no force magnitude |
| 2 | one eligible velocity transition | first inverse-force estimate |
| 3 | two transitions | minimum filtered operational candidate |
| 5 | short stable segment | improved covariance and change evidence |
| 8 | medium segment | route confirmation |
| 16 | long segment | destination evidence with change-lag risk |

Each result records exactly which samples it used. After a detected change, long windows are suppressed and reactivate only from post-change observations.

## 8. Candidate and hierarchical route inference

Candidate sources may include public map destination zones, doors/exits, corridor terminals, public navigation-graph terminals/branches, feasible route waypoints, open directional rays, and training-only flow clusters with provenance.

For geometry constrained by obstacles, compare force direction with the local tangent of a feasible path, not automatically with the straight Euclidean ray. Candidate IDs, route signatures, deduplication and pruning must be deterministic.

Maintain separate distributions:

```text
P(active_waypoint | observations)
P(route_or_final_destination | observations)
P(unknown | observations)
```

A waypoint shared by multiple routes may be locally certain while route entropy remains high. Intermediate waypoint progress should update the waypoint state without discarding a still-supported final route.

At `H >= 2`, force evidence is primary. Heading evidence reuses velocity and must not be multiplied at full weight as if independent. Required ablations are heading-only, force-only, force plus a downweighted heading auxiliary, force plus route/path-tangent evidence, and an oracle-component upper bound reported separately.

## 9. Change, arrival and reset

Raw heading change is not a sufficient reset signal. Use the innovation of the current force/candidate model:

```text
r_t = z_goal_t - E[f_goal_t | belief_(t-1)]
NIS_t = r_t^T S_t^-1 r_t
```

Combine it with route-likelihood collapse, alternative-candidate Bayes factor, waypoint proximity, arrival/braking evidence, candidate-set mutations, tracking quality and model-mismatch uncertainty.

Required classes:

```text
no_change
avoidance_or_model_mismatch
waypoint_advance_same_route
branch_commitment
route_redirect_or_destination_change
arrival
nonterminal_stop
restart_new_segment
track_loss
unknown_change
```

Reset actions are separate from classifications:

- `none`: retain all valid state;
- `active_waypoint_advance`: clear only waypoint-local direction history;
- `soft_reset`: flatten toward public priors, inflate covariance, raise unknown mass, suppress stale long windows;
- `hard_reset`: close the old goal segment and clear route-specific statistics after persistent evidence;
- `arrival_freeze`: preserve completed-route evidence while stationary;
- `lost_track_hold`: propagate uncertainty without calling loss an intention change.

One noisy, capped, partially observed or lost frame must not cause an irreversible hard reset.

## 10. Forecast and policy adapters

The repository contains different prediction surfaces that must not be conflated:

- benchmark/coarse-intent forecast logic under the canonical pedestrian-forecast owner;
- fixed-size deterministic learned prediction in `robot_sf/planner/learned_short_horizon_predictor.py`;
- fixed-size multimodal GMM prediction in `robot_sf/planner/learned_gmm_predictor.py`;
- compact planner/policy summaries in `robot_sf/planner/predictive_foresight.py`.

Introduce an explicit `GoalBelief`-to-forecast adapter rather than passing variable candidate lists into every consumer. The first integration path is:

```text
GoalBeliefBatch
  -> GoalBeliefToForecastAdapter
  -> existing forecast protocol
  -> existing fixed-size predictive summaries
```

Existing checkpoints may not receive new feature dimensions silently. A changed learned input schema requires a new checkpoint/normalizer contract. Oracle-source beliefs, fallback forecasts and unavailable beliefs remain separately labelled. Shadow mode must prove no action, observation, reward, RNG or termination side effect before active forecast consumption.

A direct compact RL observation is a later, versioned, default-off experiment. It is not part of the initial forecast integration.

## 11. Rule-to-learning progression

### R0: observation-only heading baseline

Current velocity, external candidates, stationary handling and unknown mass. It establishes the honest `H=1` comparator.

### R1: inverse-force baseline

Two/three-frame acceleration, actor-reconstructable forces, parameter priors, covariance, mutation eligibility and censoring.

### R2: temporal hierarchical baseline

Histories through `H=16`, path-tangent candidates, waypoint/route hierarchy, parameter uncertainty and candidate lifecycle.

### R3: change-aware baseline

Arrival, waypoint advance, redirect, stop/restart, soft/hard reset and history suppression.

### M1: physics-residual learned estimator

Only after R0-R3 and the evaluator are valid, train a small recurrent residual model. Inputs are actor-visible kinematics, reconstructed force terms, covariances, candidate geometry, track quality, masks and time intervals. Outputs correct the rule force, uncertainty, candidate/route probabilities and event probabilities.

The model is residual to the frozen rule baseline. Dataset production must generate actor features before joining oracle labels, group splits by episode/map/scenario/route, fit normalizers on training only, and prove that randomizing future/oracle data leaves actor features unchanged.

Online adaptation remains optional and default-off. It may adjust only bounded calibration/residual parameters from delayed actor-observable one-step errors. It may not train destination heads on their own predictions.

## 12. Verification matrix

Minimum deterministic scenarios:

1. straight fixed goal, no interactions;
2. obstacle avoidance with unchanged goal;
3. pedestrian-pedestrian interaction;
4. pedestrian-robot interaction using the force-time robot pose;
5. combined force components;
6. intermediate waypoint advance;
7. behaviour hold that zeros velocity;
8. respawn/reposition mutation;
9. abrupt route redirect;
10. final arrival and braking;
11. nonterminal stop and same-route continuation;
12. stop and new-direction restart;
13. branch ambiguity;
14. same-ray near/far destination ambiguity;
15. true goal absent from candidates;
16. speed cap inactive/boundary/active;
17. short and long occlusion;
18. nearest-row reorder and crossing tracks;
19. hidden influencing pedestrian;
20. model variant with additive operation;
21. model variant with replacement/transform operation;
22. adversarial or residual unmodelled force;
23. rotated/translated scene;
24. repeated episode reset and deterministic replay.

Each fixture declares:

- actor-visible inputs;
- oracle-only labels;
- exactness eligibility;
- expected mode and reset action;
- expected uncertainty/censoring state;
- component and variant operation;
- deterministic seed and config digest;
- failure condition that would falsify the intended mechanism.

## 13. Metrics and denominator rules

### Force and desired direction

- vector L2 MAE/RMSE (mean/root-mean-square Euclidean error) on oracle-eligible, uncapped rows;
- angular error and cosine similarity;
- magnitude error with censored rows excluded from the exact denominator;
- Gaussian/mixture NLL and credible-region coverage;
- component-reconstruction error and oracle-upper-bound gap.

### Candidates and route

- candidate recall before posterior scoring;
- active-waypoint and route/final-destination NLL, Brier, top-1/top-K;
- same-route waypoint-retention probability;
- unknown precision/recall/AUPRC;
- calibration and sharpness.

### Change/reset

- event precision/recall/F1;
- avoidance-as-change and waypoint-as-route-change rates;
- false resets per pedestrian-minute;
- detection delay;
- arrival/stop/restart confusion;
- posterior recovery time and reset oscillation.

### Tracking and availability

- ID switches, fragmentation and reacquisition;
- nominal, partial, censored, mutation-ineligible, lost, unavailable, fallback, degraded and oracle counts;
- metrics stratified by tracking correctness and observation level.

### Forecast and planning

- open-loop ADE/FDE plus proper probabilistic scores and coverage;
- mechanism activation and forecast-difference rate;
- only in a preregistered paired campaign: collisions, near misses, clearance, route completion, comfort, runtime and SNQI components.

Transition rows are not independent campaign samples. Inferential uncertainty uses episode/seed-level paired units or another preregistered cluster unit.

## 14. Canonical issue graph and module ownership

[Issue #8060](https://github.com/ll7/robot_sf_ll7/issues/8060) is the durable programme owner. Work is dependency-ordered and may stop after any negative gate.

| Wave | Issue | Canonical ownership and deliverable | Gate |
| --- | --- | --- | --- |
| 0 | [#8063](https://github.com/ll7/robot_sf_ll7/issues/8063) | prediction/benchmark contracts; snapshots A-I, identity, frames, leakage canaries | contract |
| 0 | [#8065](https://github.com/ll7/robot_sf_ll7/issues/8065) | `fast-pysf` + `robot_sf/sim`; one-pass components, residual/variant stages, cap truth | A prerequisite |
| 0 | [#8066](https://github.com/ll7/robot_sf_ll7/issues/8066) | `robot_sf/sensor`; observation-derived tracking, timestamps, frame transforms, histories | B prerequisite |
| 1 | [#8068](https://github.com/ll7/robot_sf_ll7/issues/8068) | `robot_sf/prediction/goal_intention.py`; honest external-candidate H1 baseline | comparator |
| 1 | [#8072](https://github.com/ll7/robot_sf_ll7/issues/8072) | prediction inverse dynamics; H2/H3, parameter source, censoring and covariance | A/C smoke |
| 1 | [#8073](https://github.com/ll7/robot_sf_ll7/issues/8073) | prediction/nav candidate provider; path tangents, open rays and unknown | coverage |
| 2 | [#8075](https://github.com/ll7/robot_sf_ll7/issues/8075) | prediction hierarchy; H1-H16, route/waypoint factorization and candidate lifecycle | C |
| 2 | [#8076](https://github.com/ll7/robot_sf_ll7/issues/8076) | prediction change/reset controller; waypoint, redirect, arrival, stop/restart | D |
| 3 | [#8077](https://github.com/ll7/robot_sf_ll7/issues/8077) | benchmark evaluator; proper scores, calibration, trace join and replay validation | A-D confirmation |
| 3 | [#8078](https://github.com/ll7/robot_sf_ll7/issues/8078) | forecast adapter and frozen-input open-loop comparison | forecast admission |
| 4 | [#8079](https://github.com/ll7/robot_sf_ll7/issues/8079) | dataset/manifests/trainer integrity; grouped splits and leakage audits | preflight |
| 4 | [#8081](https://github.com/ll7/robot_sf_ll7/issues/8081) | learned residual campaign and checkpoint selection | E |
| optional | [#8082](https://github.com/ll7/robot_sf_ll7/issues/8082) | bounded online calibration/adaptation | optional stress |
| optional | [#8083](https://github.com/ll7/robot_sf_ll7/issues/8083) | direct fixed-size Gym channel and checkpoint guards | optional contract |
| 5 | [#8084](https://github.com/ll7/robot_sf_ll7/issues/8084) | zero-episode closed-loop preregistration and launch packet | F preflight |
| 6 | [#8085](https://github.com/ll7/robot_sf_ll7/issues/8085) | exact admitted execution, reconciliation, analysis and decision | F result |

Critical path:

```text
#8063
  -> (#8065 || #8066 || pure #8068 work)
  -> (#8072 || #8073)
  -> #8075 -> #8076 -> #8077 -> #8078
  -> optional #8079/#8081/#8082/#8083
  -> #8084 -> #8085
```

Do not launch learning merely because the rule implementation exists. Do not launch the closed-loop campaign unless the best frozen actor-capable rule or learned estimator passes its preceding gate. A negative rule result may terminate the programme before the optional waves.

## 15. Promotion gates

### Gate A: mathematical and oracle correctness

- component/variant replay equals `f_final_pre_cap`;
- integration replay produces `v_uncapped`, cap state and `v_applied`;
- H2 exact inversion passes on eligible nominal and supported-variant fixtures;
- behaviour mutation rows are correctly classified.

### Gate B: observation and leakage integrity

- actor bytes are invariant to randomized oracle fields;
- row reorder/crossing/occlusion tests establish explicit tracking behaviour;
- frame round trips and same-step robot-state alignment pass;
- no actor path accepts a simulator object.

### Gate C: rule-based value

On held-out interaction fixtures, actor-reconstructed force either improves direction/proper score over H1 or is explicitly classified negative. Candidate misspecification is visible through unknown mass, and calibration does not materially regress.

### Gate D: change handling

Waypoint progress, redirect, arrival, stop, restart and loss are separate. False reset and delay thresholds are frozen before held-out confirmation. One noisy, capped or partial frame cannot cause hard reset.

### Gate E: learned value

A learned residual model advances only if held-out proper scores, calibration, route/unknown behaviour, change quality, forecast propagation and runtime satisfy the preregistered rule-baseline comparison. Otherwise retain the rule baseline.

### Gate F: planning value

A frozen estimator passes open-loop forecast admission and shadow no-side-effect proof before active planner consumption. A paired closed-loop campaign must preserve hard safety, executability, comfort and runtime guardrails. Aggregate score improvements cannot override a hard-safety regression.

## 16. Validation and evidence workflow

Every implementation PR must provide:

1. static schema/frame/unit/timestamp/source checks;
2. deterministic unit fixtures with at least one falsification canary;
3. config-first integration through the real owner path;
4. disabled-path compatibility and reset isolation;
5. runtime/memory measurement at declared scene scale;
6. compact, versioned receipts with source/config/schema digests;
7. explicit nominal, oracle, mutation-ineligible, partial, censored, lost, unavailable, fallback and degraded denominators;
8. a stop rule and negative/inconclusive disposition.

Documentation validation for this note:

```bash
BASE_REF=origin/main scripts/dev/check_context_notes.sh

git diff --check origin/main...HEAD

scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/check_docs_proof_consistency.py \
  --base origin/main \
  --check-context-note-freshness \
  --freshness-scope diff

scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/dev/check_docs_evidence_integrity.py --full

scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/tools/check_context_note_freshness.py \
  --index docs/context/INDEX.md \
  --context-dir docs/context \
  --catalog docs/context/catalog.yaml
```

The PR may merge only after a fresh exact-head review supersedes the blocker attached to SHA `147123b39f7ac05a8bc7f0574fa9508d41a2e997`.

## 17. Research references

- Helbing, D. and Molnár, P. (1995), *Social force model for pedestrian dynamics*, Physical Review E, DOI `10.1103/PhysRevE.51.4282`.
- Adams, R. P. and MacKay, D. J. C. (2007), *Bayesian Online Changepoint Detection*, arXiv `0710.3742`.
- Mangalam et al. (2020), *It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction*.
- Ivanovic, B. and Pavone, M. (2020), *Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data*.
- *ForceFormer: Exploring Social Force and Transformer for Pedestrian Trajectory Prediction*, arXiv `2302.07583`.
- Gneiting, T. and Raftery, A. E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, DOI `10.1198/016214506000001437`.

These references motivate model structure and evaluation. They do not transfer empirical claims to Robot SF without the repository-specific gates above.

## 18. Current conclusion

The smallest defensible first implementation is:

```text
#8063 contract
  -> #8065 exact force/transition oracle
  -> #8066 stable observation histories
  -> #8068 honest H1 comparator
  -> #8072 exact and actor-reconstructed H2/H3 baseline
```

The first success criterion is not a neural checkpoint. It is a trace whose force stages and state timing can be replayed exactly, followed by an observation-only estimator whose uncertainty and failure states remain truthful. Candidate hierarchy, resets, learning and closed-loop use are justified only after that foundation passes.
