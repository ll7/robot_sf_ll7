# Robot-SF improvement strategy — 2026-04-30

Core recommendation

The project should not optimize for “one best local policy” first. It should optimize for a layered, falsification-driven local-navigation stack whose failures are attributable to perception, prediction, decision, control, or scenario infeasibility.

The current repository is already more than a simple simulator. It provides a Gymnasium-compatible training environment with Stable-Baselines3-oriented reinforcement-learning support, SocialForce pedestrian simulation, e-scooter kinematics, OSM-based maps, a refactored environment factory, a documented adversarial-pedestrian extension, a Zenodo-backed benchmark artifact, and a sizeable test suite.  ￼ The internal documentation also shows that the project already has a social-navigation benchmark platform, SNQI tooling, baseline runners, bootstrap aggregation, planner-readiness profiles, atomic scenario sets, and scenario-difficulty analysis infrastructure.  ￼  ￼  ￼

The missing piece is therefore not “add another planner.” The missing piece is a policy-improvement loop:

scenario generator
  -> solvability certificate
  -> policy sweep
  -> failure attribution
  -> adversarial search
  -> counterexample replay set
  -> training / planner update
  -> frozen holdout evaluation

1. What to improve for the best local policy

Strategic target

Build a portfolio local policy rather than a monolithic PPO, ORCA, SocialForce, or Dreamer policy. The best near-term architecture is:

global route / topology
  -> scene graph + local free-space representation
  -> pedestrian / occupancy prediction
  -> candidate trajectory generation
  -> risk-aware trajectory scoring
  -> safety shield
  -> robot-specific controller

This matches the broader social-navigation literature. A 2025 review classifies learning-based social-navigation methods into end-to-end, human-position-based, human-attention-based, human-prediction-based, and safety-aware methods, and notes that these methods function as local planners requiring global-planner integration for long-horizon navigation.  ￼ It also emphasizes that social navigation must go beyond obstacle avoidance to human-motion interpretation, social norms, realistic training environments, accurate anticipation of human motion, and evaluation methods that capture real-world complexity.  ￼

Priority order

Layer	Current risk	Improvement
Shared route semantics	Local policies may fail because route handoff is ill-posed rather than because the planner is weak.	Fix first-waypoint / spawn / segment-progress handling before policy claims. This aligns with the known issue #730￼.
Static geometry handling	Invalid SVG obstacle geometry or narrow-passage ambiguity can contaminate planner evaluation.	Add geometry-certification and parser-regression tests before adversarial static-scenario generation. See #837￼.
Perception representation	Flattened occupancy grids or purely structured states lose spatial structure.	Use a two-branch observation model: structured state plus CNN-encoded local occupancy / semantic map. This is consistent with the direction in #789￼.
Scene understanding	Current policies may see pedestrians as independent moving discs rather than agents with interaction context.	Build a scene graph with robot, pedestrians, groups, obstacle segments, bottlenecks, route topology, and right-of-way features.
Prediction	Pedestrian prediction without obstacle context is weak near walls, corners, doors, queues, and bottlenecks.	Add obstacle-conditioned prediction: nearest-obstacle distance, obstacle normal, free-space sectors, then graph + occupancy-grid fusion. This is already aligned with #592￼.
Decision making	Reactive methods solve many geometric cases but fail in deadlocks, timing traps, and bottlenecks.	Use candidate generation from ORCA/HRVO/DWA/TEB/MPPI/PPO proposals, then rank with risk, progress, comfort, and uncertainty.
Control	Policy outputs may be feasible in abstract but not for e-scooter / differential-drive dynamics.	Track acceleration, jerk, braking distance, command projection, curvature, and actuator saturation as first-class metrics.
Training objective	Optimizing a proxy metric can diverge from benchmark quality.	Unify SNQI between training and evaluation; #455￼ should be treated as a prerequisite for fair learned-policy comparison.

2. State of the art and what it implies for Robot-SF

Classical and optimization-based local planning

ORCA remains a strong baseline because it gives reciprocal collision-avoidance guarantees under its assumptions, distributes pairwise avoidance responsibility, and reduces each agent’s action choice to a low-dimensional linear program.  ￼ HRVO extends reciprocal velocity-obstacle reasoning with explicit oscillation reduction for multi-agent navigation.  ￼ DWA remains relevant because it derives the admissible control set directly from robot dynamics, while TEB is relevant because it optimizes local trajectories for execution time, obstacle separation, and kinodynamic constraints.  ￼

For Robot-SF, this means ORCA/HRVO/DWA/TEB should not be treated as outdated baselines. They should be used as trajectory proposal generators and diagnostic references inside a layered stack. The repository’s own planner-family coverage matrix already separates benchmark-ready planners from experimental and conceptually adjacent families; that distinction should remain strict.  ￼

Learning-based social navigation

The SOTA trend is not simply “bigger RL.” It is structured learning with prediction and safety. The 2025 Frontiers review explicitly identifies prediction-based and safety-aware families as central categories.  ￼ Recent methods such as SCOPE emphasize stochastic occupancy prediction conditioned on robot motion, dynamic-object motion, and static geometry, while future-aware social-navigation methods such as Falcon explicitly predict human trajectories and penalize actions that block future human paths.  ￼

For Robot-SF, the practical SOTA-informed direction is:

do not train policy(state) -> action only
train / design policy(scene, route, prediction, uncertainty) -> short trajectory or command

Benchmarking and evaluation

The social-navigation evaluation literature warns that fair evaluation is hard because it includes static geometry, dynamic humans, and human perception of robot behavior. Francis et al. define social navigation in terms of safety, comfort, legibility, politeness, social competency, agent understanding, proactivity, and responsiveness to context.  ￼ SocNavBench addresses the same evaluation problem through photorealistic simulation, curated scenarios grounded in pedestrian data, and a metric suite for interpretable comparison.  ￼ BARN is a useful static-obstacle analogue: it shows that even metric ground navigation remains difficult in tightly constrained spaces, and it evaluates navigation using success, traversal time, and environment difficulty.  ￼

Robot-SF should therefore keep a strict separation between:

training scenarios
development / adversarial scenarios
frozen benchmark scenarios
invalid or unsolvable fixtures

The existing atomic scenario suite and verified-simple subset are a good foundation for this. The internal docs already define full runnable atomic scenarios, a verified-simple subset, validation-only fixtures, and scenario metadata such as purpose, expected behavior, pass criteria, failure modes, and primary capability.  ￼  ￼

3. Proposed “best local policy” architecture

I would implement policy_stack_v1 as a policy portfolio:

Route layer:
  visibility / Theta* / graph route
  spawn-aware waypoint rebasing
  local subgoal selection
Scene layer:
  robot state
  pedestrian tracks
  obstacle polygons / occupancy grid
  bottleneck / corridor / doorway labels
  local route topology
Prediction layer:
  constant-velocity baseline
  SocialForce baseline
  obstacle-conditioned graph predictor
  stochastic occupancy-grid predictor
Candidate layer:
  ORCA / HRVO proposal
  DWA / TEB / MPPI proposal
  learned PPO / Dreamer proposal
  stop / yield / backoff / commit maneuvers
Risk layer:
  collision probability
  TTC
  min-distance quantiles
  comfort exposure
  deadlock probability
  route progress
  uncertainty CVaR
Safety layer:
  hard clearance filter
  braking-distance check
  command feasibility check
  emergency stop / yield policy
Control layer:
  unicycle / bicycle / e-scooter controller
  acceleration and jerk limits
  projection diagnostics

The policy output should be a short trajectory or command sequence, not only an instantaneous (v, ω) action. Instantaneous commands are too weak for bottlenecks, head-on interactions, overtaking, and timing-sensitive pedestrian crossings.

Minimal viable implementation

Start with a non-learning stack:

global path
  + obstacle-aware local subgoal
  + ORCA/HRVO/DWA candidate set
  + TTC / distance / progress scoring
  + braking-distance safety shield
  + unicycle controller

Then add learning only where it is most useful:

1. learned pedestrian / occupancy prediction;
2. learned risk scoring;
3. learned proposal policy;
4. end-to-end policy only as an ablation.

This avoids a common failure mode: training an RL policy to compensate for missing route semantics, missing prediction, or invalid scenario geometry.

4. Adversarial pedestrians

The repository already contains the concept of a pedestrian as an adversarial agent that searches for weak points in the robot policy.  ￼ I would extend that into three levels.

Level A — black-box parameter adversary

Search over scenario parameters:

robot_start: [x, y, theta]
robot_goal: [x, y]
ped_start: [x, y]
ped_goal: [x, y]
ped_spawn_time: t
ped_speed: v
ped_delay: dt
ped_policy_seed: seed

Objective:

maximize:
  robot_failure
  + delay
  + near_miss exposure
  + comfort violation
  + deadlock duration
subject to:
  pedestrian remains physically plausible
  pedestrian has a valid route
  pedestrian does not teleport
  pedestrian does not intentionally collide from an impossible configuration

This is close to adaptive stress testing. AST formulates failure search as a Markov decision process and uses reinforcement learning or MCTS-like methods to find likely failure trajectories in black-box simulators.  ￼

Level B — scripted adversarial pedestrian families

Implement reusable adversarial templates:

Template	Failure mode
ttc_crossing	pedestrian enters path at minimum time-to-collision margin
doorway_blocker	pedestrian occupies the bottleneck exactly when robot commits
head_on_mirror	pedestrian mirrors robot’s avoidance side, inducing oscillation
group_squeeze	two pedestrians create a narrowing dynamic gap
late_stop	pedestrian stops after the robot has committed
overtake_cutoff	pedestrian moves parallel, then cuts across robot path
shadowing	pedestrian remains near the robot’s blind or high-cost side
queue_edge_case	pedestrian appears from behind a static obstacle or group

These should be treated as development stress tests, not frozen benchmark cases until their solvability and plausibility are certified.

Level C — learned multi-agent adversary

Train one or more pedestrians against a frozen robot policy, then alternate:

freeze robot policy
train adversarial pedestrian(s)
collect counterexamples
retrain / tune robot policy
freeze new robot policy
retrain adversary

The adversary reward should not simply reward collisions. It should reward plausible policy weakness:

R_adv =
  + failure_weight * robot_failure
  + delay_weight * robot_delay
  + near_miss_weight * near_miss_exposure
  + deadlock_weight * deadlock_time
  - implausibility_weight * deviation_from_human_motion_model
  - effort_weight * pedestrian_control_effort
  - illegality_weight * invalid_route_or_obstacle_overlap

This prevents the adversary from becoming an unrealistic missile agent.

5. Adaptive scenario definition

Use a scenario grammar rather than only YAML enumeration. Scenic is the right conceptual model here: it is a probabilistic programming language for specifying distributions over scenes and dynamic-agent behaviors, with hard and soft constraints over scenario geometry and behavior.  ￼ Scenic also supports modeling stochastic multi-agent interactions with pedestrians, cyclists, and other traffic participants, and is an official scenario modeling language for CARLA.  ￼

Robot-SF does not need to adopt Scenic immediately. It can implement a lightweight internal equivalent:

scenario_template: crossing_ttc
parameters:
  robot_start: distribution
  robot_goal: distribution
  pedestrian_start: distribution
  pedestrian_goal: distribution
  spawn_time: distribution
constraints:
  - valid_robot_path_exists
  - valid_pedestrian_path_exists
  - min_static_clearance > 0.25
  - initial_collision_free
  - crossing_ttc in [1.0, 4.0]
objectives:
  - maximize near_miss
  - maximize delay
  - minimize pedestrian_implausibility

Then implement:

uv run python scripts/adversarial/search_scenarios.py \
  --policy orca \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --objective worst_case_snqi \
  --out output/adversarial/strategy_2026_04_30/orca_crossing

The output should be a counterexample bundle:

scenario.yaml
certificate.json
episodes.jsonl
trajectory.csv
failure_attribution.json
video.mp4

6. Adversarial static scenario design: hard vs unsolvable

The key problem is not generating harder static maps. It is certifying that they are solvable under the robot model.

I would add robot_sf/scenario_certification/ with the following checks.

Solvability taxonomy

Label	Definition	Benchmark use
invalid	start, goal, route, or obstacle geometry is malformed	validation fixture only
geometrically_infeasible	no collision-free path exists after inflating obstacles by robot radius and safety margin	exclude from benchmark
kinodynamically_infeasible	geometric path exists, but no path satisfies turning, acceleration, braking, or controller limits	exclude or mark separately
dynamically_overconstrained	pedestrians can block all feasible paths indefinitely under allowed behavior	adversarial stress only
knife_edge	oracle can solve only with near-zero clearance or timing margin	stress test, not headline benchmark
hard_but_solvable	an oracle or high-budget planner solves with positive clearance and timing margin	valid benchmark candidate

Certificate fields

Each generated scenario should carry a machine-readable certificate:

{
  "schema": "robot_sf.scenario_cert.v1",
  "scenario_id": "narrow_passage_generated_042",
  "geometry": {
    "inflated_path_exists": true,
    "min_static_clearance_m": 0.32,
    "shortest_path_length_m": 18.7,
    "path_length_ratio": 1.42
  },
  "kinodynamics": {
    "candidate_exists": true,
    "min_turning_radius_ok": true,
    "braking_margin_m_p05": 0.44
  },
  "dynamic_agents": {
    "nominal_oracle_success_rate": 0.96,
    "adversarial_success_rate": 0.71,
    "oracle_min_distance_p05_m": 0.38
  },
  "baselines": {
    "goal": "fail",
    "orca": "pass",
    "social_force": "fail",
    "policy_stack_v1": "pass"
  },
  "difficulty": {
    "consensus_rank": 0.82,
    "seed_cv": 0.18,
    "label": "hard_but_solvable"
  }
}

The existing scenario-difficulty analysis already uses a consensus-outcome ranking and planner residual logic to distinguish “hard for everyone” from planner-specific mismatch. That should become a formal input into the certificate, not just a post-hoc report.  ￼

7. Transfer to CARLA

Robot-SF should transfer insights to CARLA only after the abstractions are simulator-independent. CARLA is appropriate for the next realism layer because it supports autonomous-driving simulation, flexible sensor suites, environmental conditions, open assets, and controlled scenarios of increasing difficulty.  ￼ CARLA also exposes control over static and dynamic actors, pedestrians, sensors, weather, maps, ScenarioRunner, and ROS integration.  ￼

Transfer readiness criteria

Move to CARLA when all six conditions are true:

1. robot_sf has a stable scenario certificate format.
2. The policy consumes simulator-independent observations: robot state, local grid, obstacle geometry, pedestrian tracks, prediction uncertainty, route.
3. The policy outputs physical commands or local trajectories, not simulator-specific action IDs.
4. Metrics are trajectory-based: success, collision, TTC, min distance, comfort, jerk, curvature, intervention rate, SNQI.
5. Counterexamples are stable across seeds and not artifacts of SVG parsing, route handoff, or action projection.
6. The policy passes:
    * verified-simple subset,
    * full atomic suite,
    * classic + Francis 2023 scenario set,
    * adversarial development set,
    * frozen holdout seeds.

Transfer stages

Stage	Goal	Sensor model	Actor model
T0	Export Robot-SF scenarios to neutral JSON	oracle	replay
T1	CARLA 2D oracle parity	ground-truth tracks and map	scripted pedestrians
T2	CARLA kinematic parity	ground truth	controlled robot dynamics
T3	CARLA perception stress	LiDAR / camera / noisy tracks	scripted + Traffic Manager
T4	CARLA policy validation	realistic perception stack	ScenarioRunner / Scenic-generated cases

Do not start with CARLA training. Start with CARLA replay and parity. Otherwise, simulator complexity will hide whether a failure comes from policy logic, perception, dynamics, pedestrian behavior, or scenario translation.

8. Roadmap

Phase 1 — clean benchmark semantics

Implement these first:

1. Fix first-waypoint / spawn handoff: #730￼.
2. Fix or fail-close invalid SVG obstacle conversion: #837￼.
3. Unify SNQI semantics across training and benchmarking: #455￼.
4. Add scenario_cert.v1.
5. Produce a policy-by-scenario failure matrix.

Recommended baseline command:

uv run python scripts/tools/policy_analysis_run.py \
  --scenario configs/scenarios/classic_interactions_francis2023.yaml \
  --policy-sweep \
  --seed-set eval \
  --output output/benchmarks/strategy_2026_04_30/policy_sweep

Phase 2 — adversarial development loop

Add:

1. black-box scenario-parameter search;
2. scripted adversarial pedestrian families;
3. counterexample replay suite;
4. adversarial scorecards;
5. separation of training_adversarial, dev_falsification, and frozen_eval.

Phase 3 — layered policy stack

Implement policy_stack_v1:

1. route rebasing;
2. obstacle-aware local subgoals;
3. ORCA/HRVO/DWA/MPPI candidate generation;
4. obstacle-conditioned pedestrian prediction;
5. TTC / CVaR / comfort scoring;
6. hard safety shield;
7. unicycle / e-scooter controller.

Phase 4 — CARLA bridge

Create:

robot_sf_carla_bridge/
  scenario_export.py
  carla_replay.py
  observation_adapter.py
  policy_adapter.py
  metric_adapter.py

First target: replay 10 certified Robot-SF scenarios in CARLA with oracle observations and scripted pedestrians. Only after parity should sensor-level perception be introduced.

9. Concrete new issues to open

1. feat: scenario certification v1
    Add geometric, kinodynamic, dynamic, and oracle feasibility checks.
2. feat: adversarial scenario search CLI
    Add black-box seed/start/goal/timing search with CMA-ES, Bayesian optimization, or AST-style RL.
3. feat: multi-pedestrian adversarial environment
    Extend the adversarial-pedestrian environment to 1–N pedestrians with plausibility constraints.
4. feat: obstacle-conditioned prediction baseline
    Start with hand-engineered obstacle features, then move to graph + occupancy-grid fusion.
5. feat: policy_stack_v1 portfolio planner
    Combine ORCA/HRVO/DWA/learned proposals with risk scoring and a safety shield.
6. feat: CARLA oracle replay bridge
    Export certified Robot-SF scenarios into CARLA and compare trajectory-level metrics.

Bottom line

The strongest next step is a certified, adversarially tested, layered local-navigation stack. Better perception, scene understanding, prediction, decision making, and control all matter, but they should not be pursued independently. The immediate bottleneck is the lack of a closed loop that can say:

this scenario is valid,
this scenario is solvable,
this policy failed for a known reason,
this adversarial case is plausible,
this improvement transfers beyond Robot-SF.

Once that loop exists, Robot-SF becomes a fast falsification and policy-development platform. CARLA then becomes the higher-fidelity validation layer, not a premature replacement.

References and visited links

* robot_sf_ll7 repository￼ — Author: ll7; Publication: GitHub repository; Publication date: not applicable; Access date: 2026-04-30.

<!-- Relevance: Primary repository inspected for current architecture, adversarial pedestrian support, environment factory, tests, release artifact, and project scope. -->

* docs/README.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Documents current benchmark platform, SNQI, runner, metrics, and analysis capabilities. -->

* docs/benchmark_spec.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Defines current scenario split, seed policy, baseline categories, metrics, and reproducible benchmark commands. -->

* docs/context/issue_596_verified_simple_gate_proposal.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Provides existing atomic-scenario and verified-simple gate structure. -->

* docs/context/issue_596_atomic_scenario_matrix.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Identifies scenario capabilities, target failure modes, and verified-simple membership. -->

* docs/context/issue_692_scenario_difficulty_analysis.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Existing basis for scenario difficulty, consensus ranking, and planner residual analysis. -->

* docs/benchmark_planner_family_coverage.md￼ — Author: ll7; Publication: GitHub repository documentation; Publication date: not visible; Access date: 2026-04-30.

<!-- Relevance: Current planner-family coverage and readiness boundaries. -->

* Issue #730: Fix brittle first-waypoint handoff￼ — Author: ll7; Publication: GitHub issue; Publication date: not visible from connector; Access date: 2026-04-30.

<!-- Relevance: Shared route-handling failure can contaminate planner comparisons. -->

* Issue #789: DreamerV3 multi-modal encoder￼ — Author: ll7; Publication: GitHub issue; Publication date: not visible from connector; Access date: 2026-04-30.

<!-- Relevance: Supports the recommendation to preserve spatial occupancy-grid structure. -->

* Issue #592: Hybrid graph + obstacle-context predictive model￼ — Author: ll7; Publication: GitHub issue; Publication date: not visible from connector; Access date: 2026-04-30.

<!-- Relevance: Directly matches the proposed obstacle-conditioned prediction layer. -->

* Issue #707: Improve ORCA performance on atomic failure scenarios￼ — Author: ll7; Publication: GitHub issue; Publication date: not visible from connector; Access date: 2026-04-30.

<!-- Relevance: Shows known ORCA failure slices and the need for failure attribution. -->

* Issue #768: Benchmark ORCA variants￼ — Author: ll7; Publication: GitHub issue; Publication date: not visible from connector; Access date: 2026-04-30.

<!-- Relevance: Relevant to ORCA-DD / nonholonomic ORCA comparison. -->

* Social robot navigation: a review and benchmarking of learning-based methods￼ — Author: Rashid Alyassi, Cesar Cadena, Robert Riener, Diego Paez-Granados; Publication: Frontiers in Robotics and AI; Publication date: 2025-12-11; Access date: 2026-04-30.

<!-- Relevance: Current SOTA taxonomy for learning-based social-navigation methods. -->

* Principles and Guidelines for Evaluating Social Robot Navigation Algorithms￼ — Author: Anthony Francis et al.; Publication: ACM Transactions on Human-Robot Interaction; Publication date: 2025-02-20; Access date: 2026-04-30.

<!-- Relevance: Evaluation principles for social navigation: safety, comfort, legibility, politeness, social competency, agent understanding, proactivity, and responsiveness to context. -->

* SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation￼ — Author: Abhijat Biswas, Allan Wang, Gustavo Silvera, Aaron Steinfeld, Henny Admoni; Publication: ACM THRI / CMU Robotics Institute page; Publication date: 2021-08; Access date: 2026-04-30.

<!-- Relevance: Benchmark design reference for social-navigation scenarios and metrics. -->

* BARN Challenge￼ — Author: Xuesu Xiao et al.; Publication: ICRA 2022 Challenge page; Publication date: 2022; Access date: 2026-04-30.

<!-- Relevance: Static-obstacle benchmark reference for difficulty-ranked, constrained navigation. -->

* Optimal Reciprocal Collision Avoidance￼ — Author: Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha; Publication: ORCA project page; Publication date: related publications 2008–2011; Access date: 2026-04-30.

<!-- Relevance: Classical reciprocal collision-avoidance baseline and proposal generator. -->

* The Hybrid Reciprocal Velocity Obstacle￼ — Author: Jamie Snape, Jur van den Berg, Stephen J. Guy, Dinesh Manocha; Publication: IEEE Transactions on Robotics; Publication date: 2011-08; Access date: 2026-04-30.

<!-- Relevance: HRVO baseline for oscillation-aware reciprocal avoidance. -->

* The Dynamic Window Approach to Collision Avoidance￼ — Author: Dieter Fox, Wolfram Burgard, Sebastian Thrun; Publication: IEEE Robotics and Automation Magazine; Publication date: 1997-03; Access date: 2026-04-30.

<!-- Relevance: Dynamics-aware local control baseline. -->

* teb_local_planner ROS package￼ — Author: Christoph Rösmann et al.; Publication: ROS package index; Publication date: package page, rolling updates; Access date: 2026-04-30.

<!-- Relevance: Kinodynamic local trajectory-optimization reference. -->

* Adaptive stress testing: finding likely failure events with reinforcement learning￼ — Author: Ritchie Lee et al.; Publication: Journal of Artificial Intelligence Research / MIT Lincoln Laboratory page; Publication date: 2020-12-01; Access date: 2026-04-30.

<!-- Relevance: Methodological basis for adversarial failure search. -->

* Scenic: a language for scenario specification and data generation￼ — Author: Daniel J. Fremont et al.; Publication: Machine Learning; Publication date: 2022-02-02; Access date: 2026-04-30.

<!-- Relevance: Scenario grammar, probabilistic scenario generation, and hard/soft constraints. -->

* The Scenic Programming Language￼ — Author: Scenic project contributors; Publication: Project website; Publication date: site copyright 2019–2025; Access date: 2026-04-30.

<!-- Relevance: Shows Scenic’s multi-agent, spatiotemporal, formal scenario-modeling capabilities and CARLA connection. -->

* CARLA: An Open Urban Driving Simulator￼ — Author: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun; Publication: Proceedings of Machine Learning Research, CoRL 2017; Publication date: 2017; Access date: 2026-04-30.

<!-- Relevance: CARLA transfer target and simulator baseline. -->

* CARLA Simulator￼ — Author: CARLA Team; Publication: Project website; Publication date: current site, latest news through 2025; Access date: 2026-04-30.

<!-- Relevance: Current CARLA capabilities: sensors, dynamic actors, ScenarioRunner, ROS bridge, maps. -->

* SCOPE: Stochastic Cartographic Occupancy Prediction Engine for Uncertainty-Aware Dynamic Navigation￼ — Author: Zhanteng Xie, Philip Dames; Publication: GitHub repository / IEEE Transactions on Robotics 2025 paper; Publication date: 2025; Access date: 2026-04-30.

<!-- Relevance: Modern occupancy-prediction reference for uncertainty-aware dynamic navigation. -->

* Falcon: From Cognition to Precognition￼ — Author: Zeying Gong, Tianshuai Hu, Ronghe Qiu, Junwei Liang; Publication: GitHub repository / ICRA 2025 paper; Publication date: 2025; Access date: 2026-04-30.

<!-- Relevance: Recent future-aware social-navigation example using trajectory prediction and future-path blocking penalties. -->