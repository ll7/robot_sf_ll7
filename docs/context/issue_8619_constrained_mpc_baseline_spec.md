# Issue #8619 — Constrained-MPC baseline specification (lane 2)

**Status:** proposal / design specification only. This note adds no controller code, config
registration, test, scenario, metric, ranking, benchmark result, or paper-facing claim.
**Issue:** [#8619](https://github.com/ll7/robot_sf_ll7/issues/8619). **Parent:** [#7319](https://github.com/ll7/robot_sf_ll7/issues/7319), with lane 1 delivered by
[#7603](https://github.com/ll7/robot_sf_ll7/issues/7603). The evidence base is the exact
`origin/main` commit `d0e84a27dd39ea505ccc1fd66265bb6c1905db66`.

Plain-language summary: this note specifies a future comparable baseline that uses the existing
Robot SF planner to choose a short control sequence, or **Model Predictive Control (MPC)**, from
predicted pedestrian positions, then passes its first command through the existing
**Control Barrier Function (CBF)** filter and differential-drive command projection. It is a
reproducibility and integration contract, not evidence that the composition is safe, better, or
equivalent to any source method.

## 1. Scope and claim boundary

The child owns only the lane-2 specification:

* inventory the current state/observation, prediction, constraint, and action-generation seams;
* define a future composed baseline, its observation/action/lifecycle contracts, resolved-config
  digest, and deterministic no-simulator smoke;
* record the integration gaps and coordination boundaries needed before implementation.

The following are deliberately absent: a controller implementation, a new planner interface,
external code or dependency, a scenario or episode matrix, a metric or result, tuning or ranking,
and any zero-collision, transfer, safety-certificate, or source-paper-fidelity claim. A fallback,
degraded, solver-failure, or missing-input execution can be recorded only as `unavailable` or
`failed`; it cannot become success evidence.

“Comparable” here means matched observation normalization, action space, kinematic limits, solver
budget, and fully resolved configuration. It does not mean identical behavior, formal safety, or
fidelity to an external implementation.

## 2. Current repository evidence and gaps

The following paths were checked on the stated base commit. They are the current owners; this
issue does not duplicate them.

| Concern | Current owner | What exists now | Explicit gap for this future baseline |
| --- | --- | --- | --- |
| Observation and state extraction | [`robot_sf/sensor/socnav_observation.py`](../../robot_sf/sensor/socnav_observation.py) (`SocNavObservationFusion.next_obs`); [`robot_sf/planner/socnav_occupancy.py`](../../robot_sf/planner/socnav_occupancy.py) (`_socnav_fields`); [`robot_sf/benchmark/map_runner/map_runner_observations.py`](../../robot_sf/benchmark/map_runner/map_runner_observations.py) (`normalize_map_observation`) | Structured or flattened SocNav fields for robot, goal, fixed-capacity pedestrians, map size, and simulation timestep; the planner helper normalizes both encodings. | There is no separate state estimator or uncertainty/covariance contract. The future adapter must consume the declared observation, not hidden simulator state. |
| Deterministic prediction | [`robot_sf/planner/prediction_mpc.py`](../../robot_sf/planner/prediction_mpc.py) (`ConstantVelocityPedestrianPredictor`, `PredictedPedestrianFutures`) | Constant-velocity futures are emitted in world coordinates; SocNav ego-frame velocities are rotated to world coordinates; `count` is applied by this predictor. Unsupported backends fail closed unless fallback is explicitly allowed. | The richer [`robot_sf/nav/predictive_types.py`](../../robot_sf/nav/predictive_types.py) (`ProbabilisticPrediction`) and [`robot_sf/nav/baseline_probabilistic_predictor.py`](../../robot_sf/nav/baseline_probabilistic_predictor.py) are not adapted to `PredictedPedestrianFutures`. [`robot_sf/planner/predictive_model.py`](../../robot_sf/planner/predictive_model.py) is not wired into this MPC adapter. No probability, learned forecast, or multimodal path may be invented here. |
| MPC objective and hard pedestrian constraints | [`robot_sf/planner/prediction_mpc.py`](../../robot_sf/planner/prediction_mpc.py) (`PredictionMPCPlannerAdapter`) over [`robot_sf/planner/nmpc_social.py`](../../robot_sf/planner/nmpc_social.py) (`NMPCSocialPlannerAdapter`) | SLSQP optimizes a bounded unicycle sequence. The prediction adapter can add a time-varying hard pedestrian-clearance constraint at every horizon step, with the existing goal, control, static-obstacle, occupancy, and soft-cost terms. | This is a deterministic CV prediction plus hard-clearance primitive, not a chance-constrained or horizon-CBF implementation. The uncertainty envelope is opt-in and must stay off for this baseline. |
| CBF constraint enforcement | [`robot_sf/planner/cbf_safety_filter.py`](../../robot_sf/planner/cbf_safety_filter.py) (`CollisionConeCbfSafetyFilter`, `CbfSafetyFilterPlannerWrapper`) | The filter projects a nominal unicycle command using current robot/pedestrian positions and velocities. The current common builder applies it after the planner and records `prediction_source: current_state`. | `_obstacles` expands every row in `pedestrians.positions` and does not apply `pedestrians.count`; fixed-capacity padding can therefore reach the CBF unless a count-aware adapter is added. The filter is current-state only and its best-effort fallback is not a formal safety certificate. |
| Composition and command generation | [`robot_sf/benchmark/map_runner_policies/map_runner_policy_common.py`](../../robot_sf/benchmark/map_runner_policies/map_runner_policy_common.py) (`build_adapter_policy`); [`robot_sf/benchmark/policy_builders.py`](../../robot_sf/benchmark/policy_builders.py); [`robot_sf/benchmark/planner_command_contract.py`](../../robot_sf/benchmark/planner_command_contract.py); [`robot_sf/planner/kinematics_model.py`](../../robot_sf/planner/kinematics_model.py) | The common path builds the adapter, optionally applies the CBF filter, then projects the command through the resolved kinematics model. The canonical command is differential-drive unicycle `(v, omega)`. | `prediction_mpc_cbf` is an alias to the prediction-MPC adapter builder; it does not enable CBF by name alone. The nested `cbf_safety_filter` block must be explicit. `PredictionMPCPlannerAdapter` currently has `reset`, `plan`, and diagnostics but no `close`, so it does not yet satisfy the full lifecycle target in the next row. |
| Unified lifecycle | [`robot_sf/planner/protocol.py`](../../robot_sf/planner/protocol.py) (`LocalPlannerProtocol`) | The repository protocol defines `plan`, keyword-only `reset(seed=...)`, `diagnostics`, and idempotent `close`. Existing protocol tests cover native adapters. | #8619 must consume this protocol when the composition is integrated; it must not define a competing MPC protocol. The missing `close` implementation is an integration gap, not silently synthesized evidence. |
| Digest and method-card provenance | [`robot_sf/benchmark/predictive_baseline_contract.py`](../../robot_sf/benchmark/predictive_baseline_contract.py) (`canonical_sha256`, `PlannerMethodCard`) | Existing diagnostic contracts hash canonical JSON and record action/observation contracts, implementation mode, fallback policy, and claim boundary. | This note defines the fields a future receipt must resolve and hash. It does not add a schema, digest value, or method-card registration. |

The current focused evidence surfaces are [`tests/planner/test_prediction_mpc.py`](../../tests/planner/test_prediction_mpc.py), [`tests/planner/test_cbf_safety_filter.py`](../../tests/planner/test_cbf_safety_filter.py), [`tests/benchmark/test_cbf_safety_filter_policy.py`](../../tests/benchmark/test_cbf_safety_filter_policy.py), and [`tests/planner/test_local_planner_protocol.py`](../../tests/planner/test_local_planner_protocol.py). They are implementation evidence for existing primitives, not results for this proposed composition.

## 3. Proposed future composition

The provisional method-card label is `prediction_mpc_cv_cbf_v1`; it is a documentation label, not
a current algorithm registration. The intended flow is:

`SocNav observation -> count-aware normalization -> CV futures + hard MPC -> current-state collision-cone CBF post-filter -> differential-drive projection -> (v, omega)`

1. Normalize the nested or flattened SocNav observation through the existing normalization
   helpers. Apply `pedestrians.count` before both prediction and CBF obstacle construction. Reject
   malformed, non-finite, or out-of-range fields as unavailable.
2. Construct `PredictionMPCPlannerAdapter` from the resolved values of
   [`configs/algos/prediction_mpc_cv.yaml`](../../configs/algos/prediction_mpc_cv.yaml):
   `predictor_backend=constant_velocity`, `allow_predictor_fallback=false`, horizon 6, and
   `rollout_dt=0.25`. Keep the uncertainty envelope disabled (`false`, `0.0`), enable hard
   pedestrian constraints, and keep the hard-arm soft pedestrian weight at `0.0`.
3. Use the existing hard predicted-clearance constraint for the MPC horizon. This is the
   repository's current `PredictionMPCPlannerAdapter` primitive; it is not a new barrier formula
   and it does not claim to enforce a CBF over the horizon.
4. Compose the post-MPC collision-cone filter from
   [`configs/algos/prediction_mpc_cv_cbf_collision_cone.yaml`](../../configs/algos/prediction_mpc_cv_cbf_collision_cone.yaml).
   The nested `cbf_safety_filter.enabled=true` is required. The current filter consumes the
   current state only, so diagnostics must say `prediction_source=current_state`.
5. Apply the existing kinematics projection last. The final command, not the unfiltered nominal
   command, is the action exposed to the caller. A CBF fallback, a solver fallback, or a
   kinematics projection that indicates an unavailable native command must be surfaced in the
   receipt and cannot be relabeled as a successful constrained-MPC step.

The two YAML files are source precedents, not a promise that their implicit defaults are already a
complete future method card. In particular, the future resolved configuration must materialize
the hard-constraint and nested-CBF defaults before hashing. No scenario or campaign configuration
is part of this child.

## 4. Future observation and action contract

### Observation: `socnav_structured_v1` (provisional name)

The adapter must accept the existing nested SocNav shape; the flattened map-runner shape may be
normalized by the existing bridge. Units and frames are part of the contract:

| Field | Shape and meaning | Policy |
| --- | --- | --- |
| `robot.position`, `robot.heading`, `robot.speed`, `robot.velocity_xy`, `robot.angular_velocity`, `robot.radius` | World position in metres; heading in radians; forward speed and planar velocity in metres per second; angular rate in radians per second; non-negative radius in metres. | Use the current observation values; do not substitute ground-truth state. |
| `goal.current`, `goal.next` | World-frame 2-vectors in metres. `goal.next` follows the existing zero/sentinel behavior when no next waypoint exists. | Preserve the current `_extract_state` waypoint selection semantics. |
| `pedestrians.positions` | Fixed-capacity `(N, 2)` world-frame positions in metres. | Only rows `[0:count]` are active. Padded rows are not obstacles and are not predicted. |
| `pedestrians.velocities` | Fixed-capacity `(N, 2)` ego-frame velocities in metres per second. | Rotate active rows to world coordinates exactly once for CV prediction and CBF construction. |
| `pedestrians.count`, `pedestrians.radius`, optional `pedestrians.track_id` | Active-row count, shared radius in metres, and optional identity. | Count must be an integer in `[0, N]`; identity is not used for control. |
| `map.size`, `sim.timestep` | Map extent in metres and simulator timestep in seconds. | Timestep must agree with the resolved rollout contract. |
| `occupancy_grid` and its metadata | Optional top-level grid payload consumed by `OccupancyAwarePlannerMixin`. | A missing payload means no static-map claim in the minimal smoke; malformed supplied payload is unavailable, not free space. |

There is no current separate state-estimation module in this path. A future implementation that
adds tracking, uncertainty, or a learned prediction provider must first publish a separate
versioned contract and coordinate it with #5307; it is not an implicit extension of this note.

### Action and lifecycle: `unicycle_vw` (existing command space)

The future adapter consumes the existing [`LocalPlannerProtocol`](../../robot_sf/planner/protocol.py)
shape and returns one first-step command:

* `plan(observation) -> (v, omega)`, where `v` is forward linear speed in metres per second and
  `omega` is angular rate in radians per second;
* the MPC sequence uses the existing unicycle rollout and exposes only its first command;
* for the reference configuration, `0 <= v <= 0.9` (or the lower local speed cap) and
  `-1.1 <= omega <= 1.1`; final feasibility is checked by the existing differential-drive
  kinematics model;
* `reset(seed=...)` must clear warm-start state and make the next smoke repeat independent;
* `diagnostics()` must identify the method, resolved config digest, prediction source/horizon/
  timestep, active-row count, hard-constraint mode, CBF variant/source/fallback, final action,
  and any unavailable reason;
* `close()` must be present and idempotent before the method is considered protocol-ready.

The nominal MPC command, CBF-filtered command, and final projected command may all be retained as
diagnostic fields, but only the final command is the action. Runtime duration and mutable counters
are volatile and must be excluded or normalized before deterministic diagnostic comparison.

## 5. Resolved configuration and digest contract

The future receipt must compute `config_digest` using the existing `canonical_sha256` convention:
canonical JSON with sorted keys, compact separators `(',', ':')`, and `ensure_ascii=true`, then
SHA-256. Hash the fully resolved configuration, not only the source YAML. The receipt must retain
the source paths and exact implementation commit separately from the digest.

The resolved object must contain these field groups. Values below are the current reference
values where the existing YAML and parser jointly determine them; implicit defaults must be made
explicit in the future resolved object.

| Group | Required digest fields |
| --- | --- |
| Identity and contracts | `spec_version`, `method_id`, `planner_family`, `implementation_mode=adapter`, `benchmark_status=not_available`, `observation_contract`, `action_contract`, `normalization_policy`, `lifecycle_protocol`. |
| Prediction | `predictor_backend=constant_velocity`, `allow_predictor_fallback=false`, `horizon_steps=6`, `rollout_dt=0.25`, `pedestrian_uncertainty_envelope_enabled=false`, `pedestrian_uncertainty_alpha_mps=0.0`, `prediction_frame=world`, `pedestrian_velocity_input_frame=ego`, `active_row_policy=first_count_rows`. |
| MPC limits and objective | `max_linear_speed=0.9`, `max_angular_speed=1.1`, `goal_tolerance=0.25`, `waypoint_switch_distance=0.75`, `path_goal_weight=1.5`, `terminal_goal_weight=5.0`, `heading_weight=0.5`, `control_effort_weight=0.05`, `smoothness_weight=0.2`, `pedestrian_safety_margin=0.35`, `static_obstacle_soft_weight=1.0`, `solver_ftol=0.001`, `solver_max_iterations=40`, `warm_start=true`, and `fallback_to_stop=false`. |
| Constraint mode and effective NMPC mapping | `hard_pedestrian_constraints_enabled=true`, `pedestrian_clearance_weight=0.0`, derived `progress_reward_weight=0.0`, `pedestrian_margin=0.35`, `obstacle_clearance_weight=1.0`, `occupancy_cost_weight=1.0`, plus every effective `NMPCSocialConfig` obstacle, speed-scale, symmetry-bias, collision-kappa, hard-obstacle-guard, and solver field. |
| CBF post-filter | Every field of `CbfSafetyFilterConfig`, including `enabled=true`, `variant=collision_cone`, `alpha=1.0`, `safety_margin=0.15`, `robot_radius=0.3`, `pedestrian_radius=0.3`, `max_linear_speed=null`, `max_angular_speed=null`, `turn_gain=2.0`, `max_projection_passes=3`, `min_clearance_h=1e-6`, `dpcbf_lambda_gain=1.0`, `dpcbf_mu_gain=1.0`, `relative_speed_epsilon=1e-6`, and `dpcbf_grid_samples=161`. |
| Command projection | `robot_kinematics=differential_drive`, `robot_command_mode=unicycle`, resolved linear/angular limits, projection policy, and whether a native command was changed. |
| Input and availability policy | `occupancy_grid_policy`, `missing_input_policy=unavailable`, `solver_fallback_policy=unavailable`, `cbf_fallback_policy=unavailable`, `external_code_policy=forbidden`, and `source_claim_boundary=design_only`. |

The receipt must also record `source_config_paths`, `implementation_commit`, and the canonical
digest algorithm. Those provenance fields must never be inferred from the current checkout at
read time. There is no digest value to report for this docs-only change.

## 6. Deterministic smoke definition

The future implementation must provide one no-simulator, one-observation smoke receipt. It is an
observation fixture, not a simulated situation, scenario definition, campaign, or behavioral
result. The fixture is intentionally fixed-capacity so that count masking is observable:

| Field | Frozen fixture value or rule |
| --- | --- |
| Robot | `position=[1.0, 1.0]`, `heading=[0.0]`, `speed=[0.0]`, `velocity_xy=[0.0, 0.0]`, `angular_velocity=[0.0]`, `radius=[0.25]`. |
| Goal and map | `goal.current=[3.0, 1.0]`, `goal.next=[0.0, 0.0]`, `map.size=[4.0, 4.0]`, `sim.timestep=[0.25]`. |
| Pedestrians | `positions=[[2.0, 0.7], [0.0, 0.0]]`, `velocities=[[0.0, 0.2], [0.0, 0.0]]`, `radius=[0.25]`, `count=[1.0]`. The second row is padding and must not reach prediction or CBF constraints. |
| Static map | No occupancy grid is required for this minimal smoke; the receipt must state `static_map_status=not_claimed`. |
| Repeat identity | Resolve the reference configuration once, use seed `0`, reset before each repeat, and pass byte-equivalent observations to two fresh/restarted planner instances. |

The smoke is `smoke_pass` only when all of the following hold:

1. both repeats use the same resolved `config_digest`, report the CV source, and produce an
   active-future tensor with one active pedestrian and the configured horizon/timestep;
2. the active-row count is one in both the MPC and CBF inputs; no padded row affects a command or
   constraint count;
3. the hard predicted-clearance constraint is present, the collision-cone CBF reports
   `prediction_source=current_state`, and neither solver nor CBF fallback is applied;
4. the final `(v, omega)` is finite and within the resolved command limits after projection;
5. the two final commands and all non-volatile diagnostics are equal under an exact numeric
   comparison policy (`rtol=0`, `atol=1e-12`), with runtime fields omitted or normalized; and
6. reset and idempotent close satisfy the `LocalPlannerProtocol` lifecycle.

If count-aware CBF preprocessing, protocol lifecycle, required dependencies, finite inputs, or
the strict predictor is not available, the receipt must be `unavailable` with a structured reason.
If the implementation runs and violates an assertion, it is `failed`. Neither status is a
benchmark result, and no fallback command may be relabeled `smoke_pass`.

## 7. Coordination and non-duplication

| Existing lane | Its ownership | #8619 boundary |
| --- | --- | --- |
| Parent [#7319](https://github.com/ll7/robot_sf_ll7/issues/7319) / lane 1 [#7603](https://github.com/ll7/robot_sf_ll7/issues/7603) | Lane 1's delivered method-card and predictive-baseline diagnostic precedent. | Reuse the method-card and claim-boundary pattern; do not alter or reimplement lane 1. This child is lane 2 specification only. |
| [#5307](https://github.com/ll7/robot_sf_ll7/issues/5307) | Chance-constrained MPC preregistration and its probabilistic/K-mode prediction decisions. | Use deterministic CV futures and hard predicted clearance only. Do not define probabilities, GMM/K-mode futures, calibration, chance constraints, or a second preregistration. If that lane later supplies a provider contract, integrate through it in a separately reviewed change. |
| [#6487](https://github.com/ll7/robot_sf_ll7/issues/6487) | Unified planner-interface direction, with the current protocol in [`robot_sf/planner/protocol.py`](../../robot_sf/planner/protocol.py). | Consume `LocalPlannerProtocol`, `planner_command_contract`, and common adapter construction. Do not introduce `ConstrainedMPCProtocol`, a second action schema, or duplicate lifecycle/diagnostics plumbing. |
| Existing MPC tuning notes | [`docs/context/issue_5579_mpc_tuning_budget_sensitivity.md`](issue_5579_mpc_tuning_budget_sensitivity.md) and the two reference YAMLs. | Treat current values as a configuration precedent only. No tuning grid, episode rows, metrics, or comparison read is part of this child. |

## 8. License-review and source boundary

The repository already records a stable Gravina et al. source link in the existing
[`docs/predictive_baseline_diagnostic.md`](../predictive_baseline_diagnostic.md). This note carries
that link only as an interface-shape inspiration pointer:
[Gravina et al. source reference](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1812386/full).
The link is not a license grant. This note makes no paper-specific algorithm, parameter-fidelity,
zero-collision, transfer, or safety claim.

Before any future change introduces external source code, a copied snippet, a new dependency, or
paper-specific implementation detail, work must stop for an explicit provenance and license
review. The review must use the repository's
[`docs/context/dependency_license_inventory.md`](dependency_license_inventory.md) and
[`scripts/validation/dependency_license_policy.v1.json`](../../scripts/validation/dependency_license_policy.v1.json)
surfaces, preserve source/notice identity, and record the disposition; those surfaces are a
maintainer policy boundary, not a legal opinion. The implementation permitted by this child is a
clean-room re-derivation from the existing Robot SF interfaces and primitives, with no external
code copied or vendored.

## 9. Implementation gate and conclusion

This note is ready as a design handoff only. A later implementation change must first resolve the
count-aware CBF input, optional occupancy policy, explicit nested-CBF registration, and
`LocalPlannerProtocol.close` gaps; then freeze the resolved digest fields and implement the
single-observation smoke. Until that happens, the method-card status is `not_available` and the
evidence classification remains **design-only**. A passing smoke would establish only wiring and
determinism for the fixture. It would not establish safety, navigation quality, comparison,
benchmark standing, ranking, transfer, or source-paper reproduction.
