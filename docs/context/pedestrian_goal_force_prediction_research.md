# Pedestrian Goal-Force Prediction: Research Basis and Implementation Plan

**Status:** proposal / design-only / not benchmark evidence.  
**Repository snapshot reviewed:** `895d396abb47757b0742cab6b2d677dd00eb80ae` on 2026-08-30.  
**Primary scope:** infer each observed pedestrian's current goal-directed force, active waypoint,
and likely final destination from the same causal observation information available to a planner or
reinforcement-learning policy.  
**Out of scope for this note:** implementation, trained checkpoints, benchmark results, planner
performance claims, calibrated human-intention claims, and paper-facing evidence.

Plain-language summary: Robot SF already knows how every configured Social Force component is
computed. The missing quantity in a deployment-like observation is the pedestrian's private goal.
The proposed system subtracts reconstructable interaction forces from observed motion, estimates the
remaining goal-directed force, combines that signal with map and route candidates, and maintains a
stateful probability distribution that can detect waypoint advances and genuine goal changes.

## 1. Research question and claim boundary

### Research question

Can Robot SF estimate, for every tracked pedestrian and at every causal simulation step:

1. the force attributable to the pedestrian's current movement goal;
2. the direction and uncertainty of the active force-generating waypoint;
3. the probability of candidate final destinations or routes;
4. whether an apparent direction change is ordinary collision avoidance, an intermediate-waypoint
   transition, final-goal arrival, stopping/restarting, or a genuine intention change; and
5. whether adding this belief improves pedestrian forecasting and local planning without leaking
   simulator-only truth into the actor observation?

### What a successful result may claim

A narrow exactness claim is possible under an explicitly controlled envelope:

> With noise-free full observations, known model parameters, every enabled non-goal force accounted
> for, no velocity saturation, a stable pedestrian identity, and a fixed active goal, the two-frame
> inverse estimator reconstructs the force used by the Social Force integration step to numerical
> tolerance.

Outside that envelope, the correct output is a calibrated distribution, a censored measurement, or
an explicit `unknown` state. A system that always emits one confident endpoint is incorrect whenever
multiple goals are observationally equivalent.

### Claims this work must not make without later evidence

- simulator traces establish calibrated human intention prediction;
- a goal endpoint is uniquely observable from one velocity or one force direction;
- a learned model is better because training loss decreases;
- fallback, degraded, oracle-assisted, or candidate-leaking execution is nominal success;
- predictor improvements imply safer planning without a same-seed closed-loop comparison;
- a diagnostic or smoke run is paper-grade evidence.

## 2. Current repository state

### 2.1 Existing goal-intention prototype

`robot_sf/prediction/goal_intention.py` provides an interpretable Bayesian posterior over explicit
candidate points. Its likelihood is based on alignment between observed velocity and the direction
to each candidate. The module correctly retains the prior for a nearly stationary pedestrian and
records a blocker instead of producing invalid likelihoods.

The current environment adapter in `robot_sf/gym_env/robot_env.py`, however, obtains the
PySocialForce state columns containing the true pedestrian goal and constructs exactly one candidate
for each pedestrian. The resulting posterior is necessarily one with probability one. This is a
useful planner-metadata and lifecycle prototype, but it is an oracle-backed identity mapping rather
than deployable goal inference.

Required correction:

- retain an explicit oracle constructor for tests and upper-bound evaluation;
- add an observation-only constructor that cannot access goal state columns;
- require externally generated candidate sets plus an `unknown` hypothesis;
- make actor/oracle provenance machine-readable and leakage-testable.

The earlier issue and merged slices around #4164, #4236, and #4274 remain useful historical context:
they established a default-off metadata channel and one bounded planner-consumption path, but they
did not establish calibrated intention inference or broad benchmark value.

### 2.2 Observation surfaces

`robot_sf/sensor/socnav_observation.py` exposes structured pedestrian positions, velocities, count,
robot state, robot goal, map dimensions, and optional predictive features. The observation history
configuration in `robot_sf/gym_env/observation_config.py` defaults to three temporal steps.

Two contracts must be repaired before stateful inference is trustworthy:

1. **Stable identity:** nearest-first or visibility-dependent array ordering is not a persistent
   pedestrian identity. Temporal belief must attach to an observation-derived `track_id`, not an
   array slot.
2. **Explicit coordinate frame:** pedestrian position, velocity, map geometry, robot state, force,
   candidate goal, and output features must declare and use one frame. The internal estimator should
   use global Cartesian coordinates; policy-facing compact features may be transformed later.

### 2.3 Force model surfaces

The vendored `fast-pysf/pysocialforce/forces.py` defines at least:

- `DesiredForce` for goal-directed velocity relaxation;
- `SocialForce` for pedestrian-pedestrian interaction;
- `ObstacleForce` for static geometry;
- group coherence, group repulsion, and group gaze terms when enabled.

`robot_sf/sim/simulator.py` can additionally add pedestrian-robot repulsion, adversarial forces,
residual-adversary behavior, and pedestrian-model variants. It stores `last_ped_forces`, but that is
the total applied pedestrian force, not a stable typed decomposition suitable for inverse-dynamics
supervision.

Therefore, “known forces” must mean every force term enabled for the evaluated configuration. A
residual formed by subtracting only obstacle, pedestrian-pedestrian, and pedestrian-robot terms is
wrong when group, adversarial, or model-variant terms are active.

### 2.4 Step timing

The Robot SF simulator wrapper advances behavior controllers before computing pedestrian force for
the transition. A waypoint change made by a behavior controller therefore affects the force used in
that same transition. Oracle instrumentation must record at least:

1. state and goal before behavior update;
2. goal, route, and waypoint after behavior update;
3. each force component evaluated from that post-behavior state;
4. speed-cap and integration state;
5. resulting position and velocity after integration.

Any training or evaluation trace that labels a transition with the wrong side of this timing
boundary will create systematic one-step errors around waypoint changes.

## 3. Identifiability: what can and cannot be inferred

Let pedestrian `i` at transition `t` have position `p_t`, velocity `v_t`, active goal `g_t`, preferred
speed `s_i`, relaxation time `tau_i`, desired-force factor `alpha_i`, and simulation interval
`delta_t`.

Outside the near-goal braking region, Robot SF's desired force is of the form

```text
f_goal_t = alpha_i * (s_i * d_t - v_t) / tau_i

d_t = (g_t - p_t) / ||g_t - p_t||.
```

Inside the configured goal threshold, the implementation applies a braking force of the form

```text
f_goal_t = -alpha_i * v_t / tau_i.
```

The total force is

```text
f_total_t = f_goal_t + sum_k f_known_t,k + f_unmodelled_t.
```

When the integration is not velocity-capped and observation timing is aligned,

```text
a_observed_t = (v_t+1 - v_t) / delta_t = f_total_t.
```

The observation-side goal-force measurement is therefore

```text
z_goal_t = a_observed_t - sum_k f_known_hat_t,k.
```

The implied desired velocity and direction are

```text
u_hat_t = v_t + tau_i * z_goal_t / alpha_i
s_hat_i = ||u_hat_t||
d_hat_t = u_hat_t / ||u_hat_t||.
```

### 3.1 Goal direction is easier than goal endpoint

Even a perfectly reconstructed direction does not identify one Cartesian endpoint. Every point

```text
g_t = p_t + lambda * d_hat_t, lambda > 0
```

produces the same direction until the pedestrian reaches the near-goal region. Endpoint distance and
route identity require additional information from map topology, destination candidates, route
priors, later observations, or an explicit open-ended/unknown hypothesis.

Consequences:

- the primary continuous estimate is the goal-force distribution;
- direction error must be reported separately from endpoint error;
- route and destination output should remain multimodal at branches;
- exact endpoint claims are invalid when candidates lie on the same observable ray or path tangent;
- `unknown/not-in-candidate-set` is a required hypothesis, not an optional error state.

### 3.2 Velocity saturation creates censored force measurements

`PedState.step()` caps desired velocity. If clipping activates, the observed velocity difference is
not the uncapped total force. The transition is still informative about direction, but force
magnitude is censored.

Required behavior:

- record `speed_saturated` per transition;
- do not score saturated transitions as exact force-reconstruction failures;
- use a censored likelihood or increase magnitude covariance;
- retain direction information only when it remains mathematically justified;
- separately evaluate preferred-speed estimation on unsaturated and saturated transitions.

### 3.3 Hidden and unobservable force contributors

In a realistic observation, a pedestrian or obstacle influencing the target may be outside the
field of view, occluded, lost by tracking, or absent from the selected observation subset. Its force
must not silently become part of a confidently estimated goal force.

Represent uncertainty as

```text
R_goal_t = R_acceleration_t
           + R_known_force_t
           + R_tracking_t
           + R_unmodelled_force_t
           + R_model_mismatch_t.
```

When a contributing term is unavailable, increase `R_unmodelled_force_t` and expose a blocker or
quality flag. Do not substitute zero unless the force law itself proves the contribution is zero.

## 4. Required actor/oracle separation

### 4.1 Actor estimator inputs

The observation-only actor path may consume only:

- policy-visible pedestrian positions, velocities, visibility, and timestamps;
- observation-derived track identity and association confidence;
- policy-visible robot state, action, or planned motion;
- public map geometry and semantic destinations available to the planner;
- public force-model configuration;
- prior actor-side belief state;
- causal historical snapshots no newer than the current decision point.

It must not consume:

- PySocialForce goal columns;
- true route assignment or waypoint index;
- simulator pedestrian identity as a substitute for tracking;
- exact hidden-force contributions unavailable to the policy;
- future positions, velocities, goals, or behavior events.

### 4.2 Oracle evaluator inputs

The oracle sidecar may additionally record:

- simulator pedestrian identity;
- true active goal before and after behavior update;
- true final destination, route identifier, and waypoint index;
- exact preferred speed, relaxation time, and force parameters;
- exact typed force components and total force;
- speed-cap state;
- redirect, waypoint, arrival, respawn, stop, and restart events.

Oracle fields exist only for labels, diagnostics, upper bounds, and falsification. They must not be
serialized into actor observations or model inputs.

### 4.3 Mandatory leakage tests

- randomize all oracle goals while holding actor observations fixed; actor outputs must be
  byte-identical;
- remove oracle route and waypoint fields; actor inference must still execute;
- verify actor feature tensors contain no key or array derived from goal columns;
- compare an oracle-force upper bound against observation-reconstructed force as separate arms;
- fail closed if an actor constructor receives a simulator state object with privileged fields.

## 5. Proposed hierarchical belief

The estimator should not immediately collapse the problem to one endpoint. Maintain, for each
`track_id`, a hierarchical state

```text
P(route_or_destination, active_waypoint, motion_mode, parameters | observations_1:t).
```

### 5.1 Proposed `GoalBeliefV1` fields

```text
schema_version
timestamp
track_id
source = observation_only | oracle_upper_bound
coordinate_frame
history_steps
track_confidence

force_mean_xy
force_covariance_2x2
desired_velocity_mean_xy
desired_direction_mean_xy
direction_concentration
preferred_speed_mean
preferred_speed_variance

candidate_goals[]:
  candidate_id
  candidate_source
  position_xy
  route_signature
  active_waypoint_probability
  final_destination_probability

unknown_candidate_probability
route_entropy
arrival_probability
change_probability

mode:
  initializing
  tracking
  ambiguous
  approaching_waypoint
  waypoint_transition
  intention_change_pending
  stopped
  lost
  unavailable

speed_saturated
model_residual_norm
last_reset_step
last_reset_reason
blockers[]
config_hash
```

### 5.2 Active waypoint versus final destination

A pedestrian following a multi-waypoint route can change the force-generating waypoint while
retaining the same destination. The system must distinguish:

| Event | Active waypoint | Final route/destination | Required response |
| --- | --- | --- | --- |
| normal walking | stable | stable | ordinary Bayesian update |
| collision avoidance | usually stable | stable | no reset when known forces explain motion |
| intermediate waypoint reached | changes | usually stable | advance waypoint sub-belief only |
| route branch selected | changes or concentrates | concentrates | update route posterior |
| genuine redirect | changes | changes | mutate candidates or reset route belief |
| final goal reached | arrival/braking | completed | enter arrival/stopped state |
| starts moving after stop | new segment | initially unknown | initialize a new goal segment |

This hierarchy prevents every intermediate waypoint transition from being mislabeled as a new human
intention.

## 6. Observation-derived tracking and frame normalization

### 6.1 Stable tracking

Use simulator identity only as an evaluation label. The actor path should run the tracking algorithm
intended for deployment.

Initial baseline:

1. constant-velocity Kalman prediction per active track;
2. gated Hungarian assignment using position, velocity, size, and optional appearance/semantic
   signals;
3. track creation and confirmation hysteresis;
4. lost-track memory with covariance growth;
5. explicit track retirement and reacquisition policy;
6. association confidence propagated into goal-belief covariance.

Required metrics:

- identity switches;
- association accuracy against simulator identity;
- track fragmentation;
- time to confirm a new track;
- false reacquisition rate;
- goal error stratified by correct versus incorrect association.

### 6.2 Coordinate frame

Use `global_xy` internally for:

- map geometry;
- route and destination candidates;
- force reconstruction;
- oracle comparison;
- model training labels.

Every observation and output contract must state:

- frame name;
- units;
- timestamp ownership;
- whether velocity is absolute or robot-relative;
- whether a history tensor is oldest-to-newest;
- whether padded rows are valid, missing, or masked.

A direct rotation/translation invariance test should prove that transforming an entire scene and
then transforming the result back preserves the belief within numerical tolerance.

## 7. Rule-based estimator ladder

The rule-based family remains permanently available as an interpretable baseline and runtime
fallback after learned models are introduced.

### R0: one-frame heading posterior

Inputs:

- current position and velocity;
- map/route candidate set;
- stationary threshold;
- candidate prior.

Use the existing heading-alignment likelihood from `goal_intention.py`, corrected so candidates are
not constructed from true goals. This is the valid `H=1` baseline. It cannot perform acceleration
or inverse-force estimation.

### R1: two-frame inverse-force estimator

Inputs:

- two aligned observations;
- one velocity transition;
- reconstructed observable force components;
- preferred-speed and relaxation-time assumptions.

Compute `z_goal_t` and an uncertainty-aware desired direction. This is the first estimator that can
separate observed avoidance from goal-directed drive.

### R2: temporal Bayesian estimator

Add:

- histories of length 3, 5, 8, and 16;
- causal acceleration filtering;
- parameter posteriors for preferred speed and relaxation time;
- sticky candidate transitions;
- route feasibility and path-tangent likelihoods;
- explicit unknown candidate;
- arrival/braking mode;
- saturation and occlusion covariance.

### R3: hierarchical change-aware estimator

Add:

- active-waypoint and final-route hierarchy;
- run-length or change-point state;
- waypoint-only transitions;
- route-level candidate mutation;
- soft and hard reset policies;
- stopping, restart, track loss, and reacquisition semantics.

## 8. Candidate generation

Candidate goals must be generated without consulting the target pedestrian's hidden simulator goal.

### 8.1 Candidate sources

- map destination zones, exits, doors, crossings, points of interest, and corridor endpoints;
- terminal and branch nodes in a public pedestrian navigation graph;
- active-waypoint candidates on the top `K` feasible routes;
- route-flow clusters learned only from training partitions;
- open-space directional particles or rays;
- `unknown/not-in-candidate-set`.

Each candidate must include:

- stable identifier;
- source and provenance;
- coordinate frame;
- final-destination versus waypoint role;
- route/topology signature;
- feasibility state;
- reason when rejected or unavailable.

### 8.2 Path tangent rather than straight-line direction

For a candidate behind an obstacle or around a corner, the immediate force direction should be
compared with the tangent of a feasible path, not necessarily the Euclidean vector to the final
point. Otherwise a correct destination can be rejected because its locally feasible direction is a
detour.

Candidate likelihood should therefore support:

- direct line-of-sight direction;
- navigation-graph path tangent;
- route-corridor direction;
- open ray for unbounded or unknown destinations.

### 8.3 Candidate-set falsification

Evaluation must include episodes where the true goal is intentionally absent. A correct system
should raise `unknown_candidate_probability`, not force probability mass onto the nearest wrong
candidate.

## 9. Force and candidate likelihoods

For candidate `k`, predicted desired force is

```text
mu_goal_t,k = alpha_i * (s_i * d_t,k - v_t) / tau_i.
```

A primary force likelihood is

```text
log L_force_t,k = -0.5 *
  (z_goal_t - mu_goal_t,k)^T * S_t,k^-1 * (z_goal_t - mu_goal_t,k).
```

Optional auxiliary terms:

- route/path-tangent alignment;
- current heading alignment;
- candidate prior and semantic class;
- route feasibility;
- cross-track distance;
- active-waypoint arrival compatibility;
- temporal persistence;
- group destination consistency when group evidence is public.

Avoid double-counting the same velocity evidence. At `H >= 2`, inverse-force likelihood is primary;
raw heading alignment should be a downweighted auxiliary term and an explicit ablation.

A sticky Bayesian update may use

```text
P_t(k) proportional to L_t,k * ((1 - rho_t) * P_t-1(k) + rho_t * P_0(k)),
```

where `rho_t` is the current probability of a segment change. Weak evidence must increase entropy or
unknown probability rather than create a forced winner.

## 10. Observation-history horizons

Observation history and future forecast horizon are separate experimental variables.

| Observed history `H` | Available signal | Intended use |
| ---: | --- | --- |
| 1 | position, current velocity, map context | heading/candidate prior only |
| 2 | one velocity transition | first inverse-force estimate; noise-sensitive |
| 3 | two transitions | recommended minimum operational estimator; matches current default stack |
| 5 | stable local acceleration/turn trend | main filtered estimate candidate |
| 8 | stronger mode and change evidence | route confirmation and calibration |
| 16 | longer route history | destination evidence, but high post-change lag risk |

Implement a multi-window bank over `H in {1, 2, 3, 5, 8, 16}`. Each window returns a mean,
covariance, and validity state. Fuse by reliability:

- `H=1` supplies the immediate candidate/heading prior;
- `H=2` reacts quickly but carries high covariance;
- `H=3` or `H=5` is expected to provide the main estimate;
- longer windows confirm route-level hypotheses.

After a probable change point, suppress long windows immediately and gradually restore them as the
new segment ages. This prevents pre-change history from dominating a genuine redirect.

The estimator may keep more private causal memory than the policy's public stack depth, provided the
memory is built only from past policy-visible observations and the output contract records the
history length used.

## 11. Goal-change detection and reset

A velocity-direction change alone is not a goal-change detector. Known obstacle, pedestrian, or
robot forces can cause a sharp turn while the underlying goal remains unchanged.

### 11.1 Innovation signal

Use the current belief to predict goal force and compute

```text
r_t = z_goal_t - E[f_goal_t | current belief]
NIS_t = r_t^T * S_t^-1 * r_t.
```

Evidence for a change may combine:

1. sustained high normalized innovation;
2. collapse of probability assigned to the current route;
3. a Bayes factor favoring a new candidate;
4. a large shift in residual desired-force direction;
5. braking near a probable waypoint followed by a new direction;
6. movement after a stopped interval;
7. increasing unknown probability because all current candidates fail;
8. a topology-incompatible turn that cannot continue the old route.

### 11.2 Change-point model

Start with two transparent baselines:

- a windowed sequential likelihood-ratio detector with hysteresis;
- Bayesian Online Change-Point Detection over force residual or candidate log likelihood.

The detector should maintain a run-length distribution or equivalent segment age. Thresholds must
be selected on a validation partition against a preregistered false-reset target, not tuned on test
results.

### 11.3 Soft reset

A soft reset should:

- flatten candidate probabilities toward the map prior;
- inflate force, direction, and parameter covariance;
- add or mutate new candidates;
- retain track identity and kinematic state;
- preserve a small probability that the old route remains valid;
- suppress stale long-history estimators.

### 11.4 Hard reset

A hard reset should:

- retire goal-specific sufficient statistics;
- start a new segment identifier;
- preserve only physical track history needed for causal derivative estimation;
- record reason, threshold evidence, and previous belief digest.

Hard reset requires persistent evidence or high posterior change probability. One noisy frame must
never cause an irreversible reset.

### 11.5 Waypoint transition versus route intention change

If the pedestrian is near a high-probability intermediate waypoint and the new direction matches
the next waypoint on the same route, classify `waypoint_transition` and preserve route belief.

If no continuation of the old route explains the new residual direction, classify
`intention_change_pending`; mutate or reset route-level hypotheses only after the configured
confirmation rule.

## 12. Oracle trace and evaluation contract

Add a versioned transition-level sidecar, provisionally
`pedestrian_goal_force_trace.v1`, with actor/oracle separation.

### 12.1 Oracle fields

```text
schema_version
episode_id
step_index
transition_start_time
transition_end_time
simulator_pedestrian_id
actor_track_id

goal_before_behavior_xy
goal_after_behavior_xy
final_destination_xy
route_id
waypoint_index_before
waypoint_index_after
goal_switch_kind

true_desired_force_xy
true_force_components[]
true_total_force_xy
preferred_speed
relaxation_time
desired_force_factor
goal_threshold_active
speed_cap_active

position_before_xy
velocity_before_xy
position_after_xy
velocity_after_xy
```

### 12.2 Actor prediction fields

```text
goal_belief_digest
force_mean_xy
force_covariance
candidate_probabilities
unknown_probability
arrival_probability
change_probability
mode
history_steps
track_confidence
blockers
runtime_us
```

### 12.3 Primary metrics

| Target | Metrics |
| --- | --- |
| goal force | vector mean absolute error, root mean squared error, angular error, magnitude error |
| desired direction | circular error, cosine similarity, direction credible-region coverage |
| active waypoint | true-candidate negative log likelihood, top-k recall, probability mass near truth |
| final destination/route | route negative log likelihood, top-1/top-k accuracy, topology correctness |
| uncertainty | negative log likelihood, Brier score, energy score, coverage, calibration error |
| change detection | precision, recall, F1, false resets per pedestrian-minute, detection delay |
| waypoint classification | waypoint-transition versus route-change confusion matrix |
| tracking | identity switches, fragmentation, association accuracy, confidence calibration |
| forecasting | average/final displacement error, probabilistic score, coverage by horizon |
| planning | collision, near miss, clearance, route completion, time, comfort, SNQI components |
| runtime | update latency per pedestrian, memory, scaling with pedestrian count and candidate count |

Do not collapse active-waypoint, final-destination, route, and force-direction error into one endpoint
number. Those targets have different observability.

### 12.4 Proper probabilistic scoring

A multimodal model must be evaluated with proper scoring rules and calibration, not only best-of-`K`
endpoint distance. Report sharpness only alongside coverage/calibration. Treat a high-confidence
wrong candidate as worse than an honest ambiguous distribution.

## 13. Deterministic test scenarios

The minimum fixture suite should include:

1. **Straight fixed goal without interactions:** two-frame inversion recovers force direction and,
   in the exact envelope, force vector to numerical tolerance.
2. **Static-obstacle avoidance with unchanged goal:** velocity turns; residual goal direction remains
   stable.
3. **Pedestrian-pedestrian avoidance:** known social force is removed without a false reset.
4. **Pedestrian-robot avoidance:** robot repulsion is not interpreted as new intention.
5. **Group-force activation:** every enabled group term appears in decomposition or explicit
   unmodelled covariance.
6. **Intermediate waypoint advance:** active waypoint changes, final route remains unchanged.
7. **Abrupt redirect to another route:** route change is detected with measured delay.
8. **Final arrival and braking:** estimator enters arrival mode rather than predicting a reversed
   goal.
9. **Stop and restart:** a new segment initializes after movement resumes.
10. **Branch ambiguity:** posterior remains multimodal until observations discriminate candidates.
11. **True goal absent from candidate set:** unknown probability dominates.
12. **Velocity-cap transition:** magnitude is censored and covariance increases.
13. **Occlusion and reacquisition:** uncertainty grows and tracking semantics are explicit.
14. **Nearest-slot reordering:** histories remain attached to physical tracks.
15. **Hidden influencing pedestrian:** unmodelled-force covariance grows instead of contaminating
    goal force.
16. **Adversarial/residual force:** mismatch is exposed; the estimator does not claim certainty.
17. **Rotation and translation invariance:** transformed scene produces transformed-equivalent
    belief.
18. **Oracle leakage canary:** randomized oracle goals leave actor output unchanged.
19. **Behavior/force timing canary:** transition label matches the goal selected before force
    evaluation.
20. **Deterministic replay:** same config, seed, observations, and checkpoint produce identical
    beliefs and digests.

## 14. Machine-learning progression

Do not begin with an end-to-end Transformer. The physics-based baselines and data contracts must be
correct first.

### 14.1 Baseline learned models

Evaluate in increasing complexity:

1. regularized linear or ridge residual regressor;
2. gradient-boosted tree baseline over fixed window features;
3. small multilayer perceptron;
4. gated recurrent unit over variable history;
5. graph-gated recurrent unit using neighbouring-agent context;
6. multimodal graph/Transformer model only if simpler models leave a demonstrated gap.

### 14.2 Recommended first production candidate

Use a small physics-residual gated recurrent unit. The physical estimator provides a mean prediction;
the network learns only residual correction, uncertainty, candidate logits, arrival state, and
change probability.

Per-history-step inputs:

- normalized position and velocity;
- causal acceleration mean/covariance;
- typed reconstructed force components;
- residual goal-force measurement;
- robot-relative state and planned action;
- neighbouring-agent summary or graph edges;
- candidate directions, distances, and route embeddings;
- visibility, association, saturation, and validity masks;
- actual time interval.

Outputs:

1. residual correction to goal-force mean;
2. force covariance or mixture parameters;
3. active-waypoint and final-route logits;
4. unknown probability;
5. arrival probability;
6. change probability;
7. optional preferred-speed and relaxation-time posterior.

### 14.3 Training loss

A multi-task objective may combine:

```text
L = lambda_force * NLL(true_goal_force)
  + lambda_direction * (1 - cosine(predicted_direction, true_direction))
  + lambda_goal * cross_entropy(candidate_logits, true_candidate)
  + lambda_unknown * binary_cross_entropy(unknown_probability, goal_missing)
  + lambda_change * binary_cross_entropy(change_probability, true_change)
  + lambda_arrival * binary_cross_entropy(arrival_probability, true_arrival)
  + lambda_dynamics * huber(predicted_next_velocity, true_next_velocity)
  + lambda_calibration * calibration_regularizer.
```

Use class weighting or focal treatment only after reporting the natural event prevalence. Do not
hide rare-change performance behind aggregate accuracy.

### 14.4 One model across history lengths

Train with random causal truncation over `H in {1, 2, 3, 5, 8, 16}` and an explicit history mask.
The `H=1` arm must never receive derivative features requiring older frames. Report performance for
every `H`, not only the average.

### 14.5 Dataset partitioning

Split by grouped identities, never by neighbouring frames:

- map;
- scenario family;
- route family;
- episode seed;
- goal-switch pattern;
- optional pedestrian-model and parameter regime.

Required held-out cases:

- unseen branch geometry;
- unseen goal location;
- goal missing from candidate set;
- waypoint transitions;
- stochastic redirects;
- stopping/restarting;
- occlusion/reacquisition;
- force-parameter perturbation;
- density and preferred-speed shift.

### 14.6 Offline and online fine-tuning

In simulation, offline supervised training may use exact oracle force and goal labels.

In deployment-like online adaptation, the true destination is unavailable. Restrict adaptation to
quantities with causal delayed supervision:

- one-step velocity and position innovation;
- force residual calibration;
- observation/process-noise scale;
- preferred-speed prior;
- candidate temperature;
- a bounded residual adapter.

Do not train the destination head on its own top prediction. Delayed endpoint labels are admissible
only when a complete track is observed, termination represents known arrival rather than occlusion,
and the endpoint belongs to a recognized destination zone. Otherwise the last visible point is only
a proxy.

Operational safeguards:

- frozen base model;
- small bounded adapter;
- replay buffer with provenance and age limits;
- low-rate updates;
- shadow evaluation before activation;
- rollback on calibration or held-out regression;
- drift alarms;
- no joint predictor/policy adaptation in the first causal studies.

## 15. Forecast and policy integration

The preferred first integration is

```text
GoalBelief
  -> existing goal-aware/interaction-aware pedestrian forecast
  -> existing fixed-size predictive observation channel
  -> planner or reinforcement-learning policy.
```

Advantages:

- variable candidate sets remain outside the fixed policy tensor;
- the existing prediction stack provides a natural open-loop comparator;
- the same estimator serves classical and learned planners;
- predictor quality can be established before retraining policies;
- default observations and published benchmark behavior remain unchanged while disabled.

A later compact direct observation may include, per tracked pedestrian:

```text
goal_force_x
goal_force_y
force_uncertainty_parallel
force_uncertainty_perpendicular
goal_direction_sin
goal_direction_cos
route_entropy
unknown_probability
change_probability
arrival_probability
belief_age
valid_mask
```

Any direct channel must be versioned, opt-in, fixed-size, mask-safe, and accompanied by observation
contract migration and policy retraining. Existing checkpoints must fail closed on an incompatible
observation schema.

## 16. Experiment matrix and ablations

A complete staged comparison should cross:

```text
Estimator:
  heading
  inverse_force
  temporal_bayes
  hierarchical_change
  hybrid_machine_learning

Observed history H:
  1, 2, 3, 5, 8, 16

Force information:
  oracle_components_upper_bound
  observation_reconstructed
  partial_with_unmodelled_covariance

Parameters:
  known
  estimated
  perturbed

Observation condition:
  full
  noisy
  occluded
  track_reordered

Goal event:
  fixed
  waypoint_advance
  route_redirect
  arrival
  stop_restart
  goal_missing_from_candidates
```

Critical ablations:

- heading likelihood versus inverse-force likelihood;
- raw total-force residual versus typed component subtraction;
- straight-line candidate direction versus feasible-path tangent;
- no unknown candidate versus explicit unknown;
- full reset versus hierarchical waypoint-only reset;
- fixed threshold versus change-point model;
- one window versus multi-window fusion;
- known preferred speed/relaxation time versus estimated parameters;
- oracle track identity versus observation-derived tracking;
- rule-based mean versus learned residual correction;
- predictor-only open loop versus frozen-predictor closed loop.

## 17. Acceptance gates

### Gate A: mathematical correctness

In a noise-free, uncapped, fixed-goal fixture with exact parameters and complete force decomposition:

- the two-frame estimator reconstructs the true desired-force vector to numerical tolerance;
- direction matches the active goal direction;
- repeated execution is deterministic;
- a deliberately omitted force component causes the expected residual error.

### Gate B: observation and leakage integrity

- actor inference runs with goal columns and route truth removed;
- randomizing oracle goals does not change actor output;
- track and frame contracts pass reorder and transform tests;
- oracle and actor schemas are versioned and physically separated;
- unavailable inputs produce explicit blockers rather than fabricated values.

### Gate C: rule-based value

On held-out episodes:

- inverse force improves force-direction error over velocity-only heading in interaction cases;
- uncertainty coverage is not worse than the baseline;
- known-force subtraction reduces false goal-change detections;
- goal-missing episodes route probability to unknown.

If these conditions fail, stop before learned policy integration and diagnose force decomposition,
tracking, candidate coverage, or model mismatch.

### Gate D: change handling

- waypoint progression and route intention changes are scored separately;
- detection delay and false resets satisfy preregistered limits;
- one noisy frame cannot trigger a hard reset;
- post-change long-window suppression reduces lag without unacceptable instability.

### Gate E: learned-model value

- the learned residual model improves proper probabilistic scores over the best rule baseline;
- improvement holds on held-out maps or scenario families;
- calibration coverage remains acceptable;
- no fallback/degraded row is counted as nominal success;
- runtime remains inside the intended planner-cycle budget.

### Gate F: planning value

With the predictor frozen and paired seeds/configurations:

- mechanism activation is verified;
- forecast differences are measured before policy outcomes;
- collision, near miss, executability, route completion, comfort, and runtime do not regress outside
  preregistered bounds;
- low-confidence or unavailable beliefs use conservative fallback;
- default-disabled benchmark behavior remains unchanged.

Only after Gate F should direct goal-belief policy features or joint retraining be considered.

## 18. Proposed repository work packages

| Order | Work package | Primary deliverable |
| ---: | --- | --- |
| 0 | contracts and timing | versioned actor/oracle schemas, semantics, leakage boundary |
| 1 | typed oracle force instrumentation | exact per-component transition trace and speed-cap labels |
| 2 | tracking and frame normalization | stable observation-derived tracks and global-frame history |
| 3 | inverse-force rule baseline | `H=1/2/3` force and direction estimator with covariance |
| 4 | candidate generation | map/route/open-ray/unknown providers without goal leakage |
| 5 | temporal hierarchical belief | multi-window active-waypoint and final-route posterior |
| 6 | change/reset controller | waypoint, redirect, arrival, stop/restart, soft/hard reset |
| 7 | evaluation harness | proper scores, calibration, tracking, change, runtime reports |
| 8 | forecast integration | goal-conditioned predictor input and open-loop comparisons |
| 9 | dataset and training pipeline | grouped splits, manifests, checkpoint/provenance contracts |
| 10 | hybrid learned estimator | physics-residual recurrent/graph model and ablations |
| 11 | policy observation experiment | opt-in compact channel, schema migration, frozen-model study |
| 12 | paired benchmark campaign | preregistered closed-loop evaluation and evidence disposition |

Each implementation issue should name its predecessor dependencies, canonical files, exact tests,
validation command, falsification case, stop rule, evidence tier, and downstream propagation.

## 19. Verification process for every implementation slice

Every issue and pull request in this programme should include four distinct proof layers.

### 19.1 Static contract proof

- type and shape validation;
- coordinate-frame and timestamp declarations;
- config hash and schema version;
- no silent defaults that change evidence meaning;
- docs and public terminology clarity;
- actor/oracle separation review.

### 19.2 Focused executable proof

- unit tests for formulas and edge states;
- regression test that fails against the previous implementation where practical;
- deterministic seeded fixture;
- direct validation of unavailable, invalid, degraded, and nominal states;
- no fallback/degraded state counted as nominal success.

### 19.3 Integration proof

- one canonical config-first command;
- observation and reset lifecycle through the real environment path;
- same-seed action/observation compatibility when disabled;
- exact provenance for any generated trace, model, or report;
- runtime and memory bounds.

### 19.4 Research validity proof

- target hypothesis and competing explanations;
- discriminating mechanism signal;
- comparator validity;
- held-out partition definition;
- proper uncertainty metrics;
- decision/stop rule;
- explicit result classification: blocked, diagnostic-only, smoke evidence, nominal benchmark
  evidence, or paper-grade.

## 20. Initial research hypotheses

These are proposals to test, not established conclusions:

1. Subtracting observable interaction forces will improve goal-direction inference most during
   obstacle, pedestrian, and robot avoidance, where velocity heading is a biased proxy.
2. Three or five observed frames will provide the best initial reaction/stability trade-off at the
   current simulation timestep.
3. Hierarchical waypoint-only reset will reduce false intention changes relative to resetting the
   full route belief.
4. An explicit unknown candidate will materially improve calibration on unseen or absent goals.
5. A physics-residual recurrent model will improve negative log likelihood and calibration more
   reliably than raw endpoint error.
6. Downstream planner value will depend more on calibrated uncertainty and change detection than on
   a small improvement in top-1 destination accuracy.

Each hypothesis requires a preregistered comparator and a condition that would falsify or narrow it.

## 21. Risks and mitigations

| Risk | Failure mode | Mitigation |
| --- | --- | --- |
| privileged leakage | perfect-looking posterior from true goal or route fields | separate constructors, randomized-goal canary, actor feature audit |
| identity mismatch | one pedestrian inherits another's history | observation-derived tracking, ID-switch metrics, slot-reorder fixture |
| frame mismatch | incorrect force direction or candidate geometry | single internal frame, transform invariance tests |
| missing force terms | avoidance becomes fake goal force | typed force registry, unmodelled covariance, component-ablation tests |
| speed clipping | acceleration no longer equals total force | censored transition state and stratified metrics |
| candidate misspecification | forced wrong endpoint | unknown hypothesis and open rays |
| branch non-identifiability | unjustified top-1 goal | multimodal posterior and proper scores |
| reset oscillation | belief repeatedly forgets and relearns | hysteresis, run-length posterior, refractory policy, soft reset |
| long-window lag | stale pre-change evidence dominates | multi-window bank and post-change suppression |
| simulation overfitting | strong synthetic accuracy, poor transfer | held-out maps/models, parameter perturbation, real-data lane only with provenance |
| online confirmation loop | model trains on its own predictions | restrict online adaptation to causally supervised residual/calibration heads |
| policy confounding | predictor and policy change together | freeze predictor, open-loop first, paired closed-loop second |
| evidence inflation | smoke result becomes performance claim | evidence tier and stop rule in every issue/PR |

## 22. Recommended implementation order

1. Freeze transition timing, actor/oracle schemas, and leakage canaries.
2. Instrument exact typed force components without changing dynamics.
3. Repair stable tracking and frame semantics on the planner-visible observation path.
4. Implement `H=1`, `H=2`, and `H=3` rule baselines in shadow mode.
5. Add candidate generation, unknown hypothesis, and route hierarchy.
6. Add change detection and hierarchical reset, then evaluate deterministic fixtures.
7. Build proper-score and calibration reports before any policy consumption.
8. Integrate the belief with the existing pedestrian forecast stack and run open-loop comparisons.
9. Build the dataset/training pipeline and the small physics-residual model.
10. Freeze the selected estimator and run paired closed-loop planner studies.
11. Consider a direct reinforcement-learning observation channel only after the frozen predictor
    passes the earlier gates.

## 23. References

Primary research references informing the design:

- Helbing, D. and Molnár, P. “Social force model for pedestrian dynamics.” *Physical Review E*,
  51, 4282–4286, 1995. <https://doi.org/10.1103/PhysRevE.51.4282>
- Adams, R. P. and MacKay, D. J. C. “Bayesian Online Changepoint Detection.” 2007.
  <https://arxiv.org/abs/0710.3742>
- Mangalam, K. et al. “It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory
  Prediction.” *ECCV*, 2020. <https://arxiv.org/abs/2004.02025>
- Yao, Y. et al. “BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal
  Estimation.” *IEEE Robotics and Automation Letters*, 2021.
  <https://arxiv.org/abs/2007.14558>
- Salzmann, T. et al. “Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous
  Data.” *ECCV*, 2020. <https://arxiv.org/abs/2001.03093>
- “ForceFormer: Exploring Social Force and Transformer for Pedestrian Trajectory Prediction.” 2023.
  <https://arxiv.org/abs/2302.07583>
- Gneiting, T. and Raftery, A. E. “Strictly Proper Scoring Rules, Prediction, and Estimation.”
  *Journal of the American Statistical Association*, 2007.
  <https://doi.org/10.1198/016214506000001437>

Repository surfaces reviewed for this proposal:

- `robot_sf/prediction/goal_intention.py`
- `robot_sf/gym_env/robot_env.py`
- `robot_sf/gym_env/observation_config.py`
- `robot_sf/sensor/socnav_observation.py`
- `robot_sf/sensor/history_stack.py`
- `robot_sf/sim/simulator.py`
- `robot_sf/ped_npc/ped_behavior.py`
- `robot_sf/planner/learned_short_horizon_predictor.py`
- `robot_sf/planner/learned_short_horizon_trainer.py`
- `robot_sf/planner/learned_gmm_predictor.py`
- `robot_sf/planner/predictive_foresight.py`
- `robot_sf/benchmark/map_runner/map_runner_episode.py`
- `fast-pysf/pysocialforce/forces.py`
- `fast-pysf/pysocialforce/scene.py`
- `fast-pysf/pysocialforce/simulator.py`

## 24. Decision summary

The recommended system is a stateful, hierarchical `GoalBelief` rather than a single predicted
point:

```text
policy-visible observations
  -> stable tracking and frame normalization
  -> causal acceleration estimate
  -> reconstruction of every observable non-goal force
  -> uncertain residual goal-force measurement
  -> map/route/open-ray candidate likelihoods
  -> active-waypoint and final-route posterior
  -> arrival and change-point controller
  -> existing pedestrian forecast and planner/policy interfaces.
```

The first implementation target is not a neural model. It is a leakage-proof oracle trace plus an
exact, falsifiable inverse-force baseline. Learned prediction is justified only after that baseline,
its uncertainty, candidate coverage, and reset semantics are independently verified.
