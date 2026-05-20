# Issue #1357 Tentabot-Style Motion-Primitive Value Policy Assessment

Date: 2026-05-20

## Question

Is a Tentabot-style learned motion-primitive value policy a better Robot SF fit than end-to-end
visual or world-model navigation policies?

## Verdict

`Robot SF reimplementation spike`

Tentabot is a better conceptual fit than end-to-end visual or world-model navigation policies under
the local-planner reuse rubric below because the core decision surface is a bounded set of motion
primitives/tentacles with per-candidate occupancy, closeness, and smoothness scores. That shape is
close to Robot SF's existing local candidate-scoring surfaces (`risk_dwa`, `mppi_social`,
`hybrid_rule_local_planner`, and `policy_stack_v1`) and can emit a direct differential-drive
command.

Do not wrap or copy the upstream implementation now. The source repository has no GitHub license
metadata, `package.xml` still declares `<license>TODO</license>`, and the runnable path is a
ROS Noetic/Gazebo/OpenAI-ROS stack with substantial robot-specific side dependencies. A Robot
SF-native spike should treat Tentabot as a literature/design reference only unless upstream license
and runtime provenance become clear.

## Source Evidence

Upstream sources checked on 2026-05-20 at commit
`6fb94da1e38303e600fb38b6bf1c8d2a9a97e829` unless otherwise noted:

- Paper: <https://arxiv.org/abs/2208.08034>
- Source repo: <https://github.com/RIVeR-Lab/tentabot>
- Repository metadata: default branch `master`, pushed `2025-08-26`, no GitHub license metadata.
- License endpoint: `gh api repos/RIVeR-Lab/tentabot/license` returned `404`.
- Package manifest: `package.xml` contains `<license>TODO</license>`.
- README says the system was tested with Ubuntu 20.04 and ROS Noetic.
- Manual install requires catkin, Gazebo/ROS packages, several RIVeR-Lab `noetic-akmandor`
  branches, `openai_ros`, `stable-baselines3[extra]`, `GitPython`, and `squaternion`.
- Runnable source entrypoints exist, but only inside the ROS stack:
  - README uses `roslaunch tentabot tentabot_framework.launch` for the framework.
  - `launch/tentabot_framework.launch` wires the Gazebo/map utility/Tentabot server flags.
  - `scripts/tentabot_drl/tentabot_drl_training.py` and
    `scripts/tentabot_drl/tentabot_drl_testing.py` are the PPO train/test scripts.
- `config/tentabot_server/config_tentabot_server_turtlebot3.yaml` exposes the relevant contract:
  - `trajectory_gen_type: "kinematic"`
  - `lat_velo_samp_cnt: 5`
  - `ang_velo_samp_cnt: 21`
  - `drl_service_flag`
  - `observation_space_type` values including `Tentabot_FC`, `Tentabot_1DCNN_FC`, and
    `Tentabot_2DCNN_FC`
  - `initial_training_path` pointing to tracked benchmark training data.
- `src/tentabot.cpp` computes candidate values from occupancy, closeness, and smoothness terms in
  `select_best_tentacle()`, then publishes `geometry_msgs/Twist` with
  `linear.x = velocity_control_data[best_tentacle][0]` and
  `angular.z = velocity_control_data[best_tentacle][1]`.
- `srv/rl_step.srv` returns arrays for `occupancy_set`, `navigability_set`, `clearance_set`,
  `clutterness_set`, `closeness_set`, and `smoothness_set`.
- Those train/test scripts run PPO through `openai_ros` environments rather than a standalone
  inference API.

No source-side smoke was attempted because the issue asks for an assessment, the upstream license is
not clear enough for reuse, and the runnable path requires ROS Noetic/Gazebo/OpenAI-ROS side
dependencies that are outside the Robot SF local test environment.

## Local-Planner Reuse Rubric

| Criterion | Tentabot-style scorer | End-to-end visual navigation | World-model navigation |
| --- | --- | --- | --- |
| Robot SF observation fit | Candidate values can be rebuilt from local occupancy, route, pedestrian, and previous-command features. | Requires image/depth/topomap observations that Robot SF does not expose as the default local-planner contract. | Usually requires learned latent dynamics, long-horizon rollout state, and training artifacts outside the local-planner contract. |
| Robot SF action fit | Selects a lattice command that maps directly to bounded `unicycle_vw`. | Often emits waypoints or platform-specific commands that need extra projection. | Often couples action selection to a learned simulator or policy stack rather than a small local command adapter. |
| Diagnostics fit | Per-candidate scores can explain selected/rejected proposals. | Explanations are usually post-hoc unless the source model exposes internal scoring. | Latent rollouts are harder to map to Robot SF's proposal-status diagnostics. |
| Reimplementation burden | Moderate: reuse in-repo candidate generators and train a scorer. | High: add visual/topological observation stack and source parity. | High: train/import world model plus policy and provenance gates. |

## Robot SF Contract Fit

Best fit:

- Treat the method as a learned scorer over Robot SF-generated local candidate commands or short
  rollout trajectories.
- Input should be Robot SF-native and benchmark-fair: route-relative goal features, local static
  occupancy probes, pedestrian-relative state or rollout collision probes, and previous command.
- Output should be either:
  - a score/logit per candidate in a fixed lattice, followed by existing safety filters, or
  - the selected bounded `unicycle_vw` command with diagnostics tying it back to the candidate set.
- Candidate availability should follow the `policy_stack_v1` statuses: `native`, `adapter`,
  `unavailable`, `rejected`, and `degraded`.

Do not use:

- upstream ROS/Gazebo/OctoMap state as hidden privileged input,
- source robot assets, 3D sensors, or real-robot assumptions as Robot SF benchmark inputs,
- upstream training data, weights, or code while licensing remains unclear,
- fallback execution as benchmark success.

## Comparison To Existing Robot SF Surfaces

| Surface | Relationship | Consequence |
| --- | --- | --- |
| `risk_dwa` | Already scores a velocity lattice with hand-authored risk and progress terms. | Tentabot-style work must beat or explain this baseline, not duplicate it. |
| `mppi_social` | Samples/optimizes short action sequences with social costs. | Tentabot-style work is cheaper and more discrete, but less expressive unless the lattice is broad enough. |
| `hybrid_rule_local_planner` | Generates route/subgoal/recovery candidates and scores them with explicit filters. | The natural insertion point is a learned scorer over this candidate family, with existing filters retained. |
| `learned_risk_model_v1` / predictive planners | Learn or consume risk terms over rollout evidence. | A Tentabot-style scorer is adjacent; its unique value would be candidate ranking from compact occupancy-value features. |
| `policy_stack_v1` | Defines proposal availability, risk-score diagnostics, and fail-closed semantics. | Any spike should be implemented as a proposal/scorer with explicit selected/rejected diagnostics. |

This family is not already covered by current Robot SF code. The implemented sampling planners cover
the non-learning candidate-scoring side; Tentabot adds a learned value/ranking component over
candidate occupancy features.

## Recommended Spike

Smallest useful Robot SF-native experiment:

1. Add a config-only experimental candidate, for example `tentabot_value_scorer_v0`, behind
   `allow_testing_algorithms: true`.
2. Reuse an existing fixed lattice from `risk_dwa` or `hybrid_rule_local_planner` so the first
   experiment tests the learned scorer, not a new candidate generator.
3. Build per-candidate features from information already available to current planners:
   route-progress delta, static occupancy/clearance probes, pedestrian minimum distance or TTC,
   smoothness from previous command, and candidate command bounds.
4. Start with a tiny supervised imitation target from the current best safe hand-scored candidate,
   then compare whether the scorer can reproduce candidate ranking on held-out scenarios before any
   RL training.
5. Validate with `scripts/validation/run_policy_search_candidate.py` on smoke, nominal-sanity, and a
   narrow hard-case slice.

Stop condition:

- stop if the scorer cannot beat or match `risk_dwa` / `hybrid_rule_v3_*` on smoke and
  nominal-sanity while preserving collision and near-miss gates,
- stop if the learned component needs non-Robot-SF observations, benchmark-specific reward leakage,
  or source assets,
- stop if diagnostics cannot explain selected, rejected, and unavailable candidates per step.

## Follow-Up Boundaries

- Follow-up spike issue: <https://github.com/ll7/robot_sf_ll7/issues/1387>
- Source-side reproduction is not the first step because the upstream source license is unclear and
  its harness is not close to Robot SF's 2D AMV benchmark contract.
- A Robot SF-native spike should not claim Tentabot parity. The safe claim is "Tentabot-style
  learned motion-primitive value scoring."
- Candidate registry updates should wait until the spike defines a concrete config path and
  validation command.
