# Force-Coupled Potential-Field Diagnostic Comparator Method Card (Issue #8015)

**Status:** experimental / diagnostic-comparator / smoke-only — implementation-integrity and local comparative evidence. Not a benchmark ranking, not a release-roster change, not social compliance certification.
**Issue:** [#8015](https://github.com/ll7/robot_sf_ll7/issues/8015) (parent research [#7882](https://github.com/ll7/robot_sf_ll7/issues/7882), prerequisite implementation [#7889](https://github.com/ll7/robot_sf_ll7/issues/7889)).
**Harness Module:** `robot_sf/benchmark/force_coupled_comparator.py`.
**CLI Entry Point:** `scripts/benchmark/check_force_coupled_comparator.py`.
**Receipt Schema:** `robot_sf/benchmark/schemas/force_coupled_comparator_receipt.v1.json`.
**Contract Tests:** `tests/benchmark/test_force_coupled_comparator.py`.
**Source Reference:** Jing, H., et al. (2026). *Local path planning for autonomous vehicles: a dynamic potential field-guided and force-coupled adaptive pure pursuit approach*. *Scientific Reports* (21 August 2026). DOI: [10.1038/s41598-026-63761-z](https://doi.org/10.1038/s41598-026-63761-z).

Plain-language summary: a deterministic diagnostic comparison harness that evaluates the opt-in `force_coupled_potential_field` local planner against reference baselines (`pure_pursuit_goal`, attractive-dominant ablation, and repulsive-dominant ablation) on canonical analytic scenarios (`analytic_static_obstacle`, `analytic_pedestrian_interaction`, `analytic_symmetric_obstacle`, `analytic_unobstructed`).

## 1. Source Method and Scope Limits

The comparator evaluates a local planner derived from the improved adaptive pure pursuit (i-APP) architecture described by Jing et al. (*Scientific Reports*, 21 August 2026, DOI: [10.1038/s41598-026-63761-z](https://doi.org/10.1038/s41598-026-63761-z)). In the source publication, the method couples:
- velocity-adaptive look-ahead path search;
- dynamic potential-field refinement for obstacle repulsion;
- force-coupled target selection balancing goal progress and obstacle avoidance;
- forward-kinematic integration and continuous steering-rate constraints for Ackermann-steered autonomous road vehicles.

**Scope boundaries in the source:**
- **Autonomous road vehicle scope:** The source publication specifically addresses highway and urban road navigation for passenger vehicles operating within defined lanes, interacting with motorized vehicles, and following vehicular turning kinematics.
- **Vehicle results are not social-navigation evidence:** Source experiments measure road vehicle lane-keeping, high-speed collision avoidance, and vehicle overtaking maneuvers. These results do not model pedestrian crowd dynamics, proxemics, pedestrian compliance, or reciprocal collision avoidance, and must not be treated as social-navigation evidence.

## 2. Implementation Deviations and Transfer Risks

The repository implementation (`robot_sf/planner/force_coupled_potential_field.py`, issue [#7889](https://github.com/ll7/robot_sf_ll7/issues/7889)) adapts core force-coupling and potential-field concepts to the Robot SF unicycle simulation environment, introducing explicit engineering deviations:

1. **Forward-Kinematic Integration:**
   - *Source:* Formulates forward kinematics for Ackermann-steered vehicles with wheelbase, front-wheel steering angle, and continuous curvature bounds.
   - *Deviation:* Adapted to planar differential-drive / unicycle robot kinematics emitting bounded `(linear, angular)` speed commands `(v, \omega)` over discrete control intervals (`control_dt`).
2. **Steering-Rate Handling:**
   - *Source:* Enforces continuous steering-rack angular rates and dynamic lateral tire acceleration limits.
   - *Deviation:* Enforced through hard-predicate rate clipping on linear acceleration (`max_linear_rate`) and angular acceleration (`max_angular_rate`) per discrete control step. Overlap conditions issue explicit rate-limited stop requests (`status: degraded`) rather than dynamic vehicle evasion maneuvers.
3. **Pedestrian-Interaction Separation:**
   - *Source:* Dynamic obstacles in the source paper are other road vehicles obeying vehicular traffic dynamics.
   - *Deviation:* In this harness, pedestrians are observed as separate geometric entities emitting isotropic repulsive forces without vehicle-traffic heuristics. Pedestrian-interaction scenarios (`analytic_pedestrian_interaction`) are strictly executed and reported separately from static-obstacle scenarios (`analytic_static_obstacle`), rather than combined into a single collapsed vehicular safety metric.

**Expected Transfer Risks and Domain Limitations:**
- **No Direct Parameter Transfer:** Potential-field gains, influence cutoffs (`influence_radius_m`), and look-ahead scaling calibrated for high-speed road vehicles cannot be transferred directly to low-speed pedestrian-shared social navigation spaces.
- **Static vs. Dynamic Separation:** Zero-collision performance in static obstacle scenarios provides zero evidence of safe or socially compliant behavior around moving pedestrians. Evaluating them separately prevents false confidence in social navigation capability.
- **Diagnostic Comparator Boundary:** The harness evaluates local implementation integrity, numeric stability, and relative performance against canonical baseline baselines on four fixed synthetic scenarios. It establishes no benchmark ranking, no leaderboard placement, and no release-roster promotion.

## 3. Compared Planner Configurations

| Planner ID | Description | Role |
|---|---|---|
| `force_coupled_potential_field` | Canonical opt-in force-coupled potential-field local planner from #7889 | Candidate under test |
| `pure_pursuit_goal` | Unicycle pure pursuit straight to goal with bounded speeds and rate limits | Non-avoiding reference baseline |
| `ablation_attractive_dominant` | Force-coupled planner with near-zero repulsive weight (`repulsive_weight=0.001`) | Ablation baseline |
| `ablation_repulsive_dominant` | Force-coupled planner with near-zero attractive weight (`attractive_weight=0.001`) | Ablation baseline |

## 4. Canonical Analytic Scenarios

| Scenario ID | Seed | Start $(x,y,\theta)$ | Goal $(x,y)$ | Obstacles | Pedestrians |
|---|---:|---|---|---|---|
| `analytic_static_obstacle` | 1 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[(1.0, 0.5)]` | `[]` |
| `analytic_pedestrian_interaction` | 7 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[]` | `[(1.0, 0.0)]` |
| `analytic_symmetric_obstacle` | 42 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[(2.0, 0.0)]` | `[]` |
| `analytic_unobstructed` | 1 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[]` | `[]` |

## 5. Aggregate Summary Results

| Planner ID | Runs | Success Rate | Collision Rate | Near-Miss Rate | Mean Path (m) | Mean Jerk Metric | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `force_coupled_potential_field` | 4 | 1.0000 | 0.0000 | 0.2500 | 6.0333 | 0.7217 | 0.1396 |
| `pure_pursuit_goal` | 4 | 0.5000 | 0.5000 | 0.0000 | 3.7501 | 5.2554 | 0.0122 |
| `ablation_attractive_dominant` | 4 | 0.5000 | 0.5000 | 0.0000 | 3.7651 | 0.8402 | 0.1400 |
| `ablation_repulsive_dominant` | 4 | 0.5000 | 0.0000 | 0.0000 | 8.9043 | 0.5993 | 0.1376 |

## 6. Evidence Boundary

- Target: implementation integrity and local comparative diagnostic proof on canonical analytic scenarios only.
- Exclusions: no SLURM execution, no large-scale campaign, no simulator dataset promotion, no benchmark leaderboard ranking, no paper-facing claim.
