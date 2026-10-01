<!-- AI-GENERATED (#10074) - NEEDS-REVIEW -->
| site | literal | classification |
|---|---|---|
| configs/algos/guarded_ppo_release_v0_0_8.yaml:43 | `pedestrian_radius_m: 0.4` | planner/guard assumption |
| configs/algos/guarded_ppo_release_v0_0_8.yaml:54 | `guard_pedestrian_radius_m: 0.4` | planner/guard assumption |
| configs/algos/hybrid_rule_v4_clearance_braking.yaml:44 | `pedestrian_radius_default: 0.4` | planner/observation assumption |
| configs/algos/prediction_planner_release_v0_0_8.yaml:45 | `predictive_pedestrian_radius: 0.4` | planner/observation assumption |
| configs/algos/predictive_mppi_release_v0_0_8.yaml:62 | `predictive_pedestrian_radius: 0.4` | planner/observation assumption |
| configs/algos/risk_dwa_release_v0_0_8.yaml:36 | `pedestrian_radius_m: 0.4` | planner/observation assumption |
| configs/algos/social_force_release_v0_0_8.yaml:8 | `social_force_ped_v3_default_ped_radius: 0.4` | planner/observation assumption |
| configs/benchmarks/issue_3574_mean_matched_harness_full_s30.yaml:62 | `radius_m: 0.35` | simulation truth; population body-size setup |
| configs/benchmarks/issue_3574_mean_matched_harness_full_s30_goal.yaml:17 | `cautious: {desired_speed_factor: 0.7, radius_m: 0.35}` | simulation truth; population body-size setup |
| configs/benchmarks/issue_3574_mean_matched_harness_full_s30_orca.yaml:17 | `cautious: {desired_speed_factor: 0.7, radius_m: 0.35}` | simulation truth; population body-size setup |
| configs/benchmarks/issue_3574_mean_matched_harness_full_s30_social_force.yaml:17 | `cautious: {desired_speed_factor: 0.7, radius_m: 0.35}` | simulation truth; population body-size setup |
| configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml:41 | `radius_m: 0.35` | simulation truth; population body-size setup |
| configs/benchmarks/issue_5504_mean_matched_harness_scenario_matrix_smoke.yaml:29 | `radius_m: 0.35` | simulation truth; population body-size setup |
| configs/benchmarks/issue_5829_geometryless_forced_population_smoke.yaml:29 | `radius_m: 0.35` | simulation truth; population body-size setup |
| configs/benchmarks/observation_noise/issue_3300_false_positive_actor_injection_v1.yaml:7 | `pedestrian_false_positive_radius: 0.35` | planner/observation assumption |
| configs/benchmarks/observation_noise/issue_3952_robustness_smoke_v1.yaml:8 | `pedestrian_false_positive_radius: 0.35` | planner/observation assumption |
| configs/benchmarks/observation_noise/robustness_smoke_v1.yaml:11 | `pedestrian_false_positive_radius: 0.35` | planner/observation assumption |
| configs/benchmarks/pedestrian_validation_0_0_9.json:47 | `"force_radius_m": 0.35,` | diagnostic baseline force calibration |
| configs/benchmarks/pedestrian_validation_0_0_9.json:48 | `"baseline_physical_radius_m": 0.4,` | diagnostic physical truth |
| configs/research/fidelity_sensitivity_v1.yaml:138 | `ped_radius: 0.40` | simulation/metric truth; sensitivity setup |
| configs/research/fidelity_sensitivity_v1.yaml:183 | `pedestrian_radius_m: 0.40` | simulation/metric truth; sensitivity setup |
| configs/research/fidelity_sensitivity_v1.yaml:192 | `pedestrian_radius_m: [0.30, 0.40, 0.50]` | simulation/metric truth; sensitivity setup |
| configs/research/issue_3207_fidelity_sensitivity_full_fixed_scope.yaml:112 | `ped_radius: 0.40` | simulation/metric truth; sensitivity setup |
| fast-pysf/pysocialforce/config.py:389 | `agent_radius: float = 0.35` | simulation force-kernel calibration (distinct from physical truth) |
| fast-pysf/pysocialforce/sim_view.py:88 | `ped_radius: float = 0.4` | rendering assumption |
| robot_sf/adversarial/matched_compute.py:83 | `ped_radius: float = 0.4` | simulation truth |
| robot_sf/baselines/ppo.py:382 | `radius = float(obs.agents[0].get("radius", 0.35)) if obs.agents else 0.35` | planner/observation assumption |
| robot_sf/baselines/social_force.py:340 | `agent_radius=0.35,` | planner/observation assumption |
| robot_sf/benchmark/collision/collision_definition_inventory.py:52 | `DEFAULT_PED_RADIUS: float = 0.4` | diagnostic/fixture assumption; inspect caller before treating as truth |
| robot_sf/benchmark/full_classic/orchestrator.py:1029 | `ped_radius=float(getattr(sim_cfg, "ped_radius", 0.4)),` | simulation truth |
| robot_sf/benchmark/last_avoidable_fixtures.py:866 | `physical_collision_radius=0.4,  # true physical radius` | diagnostic combined contact threshold, not a pedestrian body radius |
| robot_sf/benchmark/map_runner/map_runner_episode.py:1650 | `ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:2459 | `ped_radius_val = getattr(config.sim_config, "ped_radius", 0.4)` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:2460 | `ped_radius = float(ped_radius_val if ped_radius_val is not None else 0.4)` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:2833 | `ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:2835 | `ped_radius = 0.4` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:2839 | `ped_radius = 0.4` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_episode.py:4199 | `ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),` | simulation truth |
| robot_sf/benchmark/map_runner/map_runner_observations.py:113 | `ped_radius_source = _default_if_none(pedestrians.get("radius"), [0.35])` | planner/observation assumption |
| robot_sf/benchmark/map_runner/map_runner_observations.py:115 | `ped_radius = float(ped_radius_raw[0]) if ped_radius_raw.size else 0.35` | planner/observation assumption |
| robot_sf/benchmark/map_runner/map_runner_trace.py:270 | `actor_radius_m = _metadata_float(vru, "actor_radius_m", default=0.35)` | planner/observation assumption |
| robot_sf/benchmark/metrics.py:147 | `ped_radius: float = 0.4` | simulation truth |
| robot_sf/benchmark/observation_noise.py:35 | `"pedestrian_false_positive_radius": 0.35,` | planner/observation assumption |
| robot_sf/benchmark/observation_noise.py:423 | `radii = np.full((positions.shape[0],), 0.35, dtype=float)` | planner/observation assumption |
| robot_sf/benchmark/observation_noise.py:455 | `np.asarray([float(spec.get("pedestrian_false_positive_radius", 0.35))]),` | planner/observation assumption |
| robot_sf/benchmark/observation_noise.py:501 | `np.full((positions.shape[0],), 0.35, dtype=float),` | planner/observation assumption; multiline range-occlusion radius |
| robot_sf/benchmark/radius_sensitivity_gate0_audit.py:152 | `METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M = 0.4` | historical audit constant; not live radius control |
| robot_sf/benchmark/radius_sensitivity_gate0_audit.py:154 | `RUNNER_DEFAULT_PED_RADIUS_M = 0.35` | historical audit constant; not live radius control |
| robot_sf/benchmark/runner.py:138 | `DEFAULT_BENCHMARK_PED_RADIUS_M = 0.35` | alternate runner geometry default (simulation/metric truth; legacy mismatch) |
| robot_sf/benchmark/safety/safety_wrapper_runtime.py:371 | `ped_radius = _radius(config, "ped_radius", "pedestrian_radius", default=0.4)` | planner/guard assumption |
| robot_sf/benchmark/simulator_counterfactual_adapter.py:1089 | `getattr(getattr(self.sim, "pysf_sim", None), "peds", None), "agent_radius", 0.4` | simulation truth |
| robot_sf/benchmark/socnavbench_canary.py:354 | `ped_radius=0.4,` | diagnostic/fixture assumption; inspect caller before treating as truth |
| robot_sf/gym_env/robot_env.py:1138 | `ped_radii = np.full(len(ped_positions), 0.35)` | planner/observation assumption |
| robot_sf/gym_env/robot_env.py:1266 | `ped_radii = np.full(len(ped_positions), 0.35)` | planner/observation assumption |
| robot_sf/gym_env/snqi_proxy.py:18 | `_DEFAULT_PED_RADIUS = 0.4` | simulation truth |
| robot_sf/nav/occupancy.py:262 | `ped_radius: float = 0.4` | simulation truth |
| robot_sf/ped_ego/unicycle_drive.py:19 | `radius: float = 0.4  # Collision radius, not relevant for kinematics` | simulation truth |
| robot_sf/ped_npc/ped_population.py:1343 | `ped_radius: float = 0.4,` | simulation truth |
| robot_sf/ped_npc/residual_adversary.py:1196 | `ped_radius: float = 0.4` | simulation truth |
| robot_sf/ped_npc/residual_adversary.py:1601 | `ped_radius: float = 0.4,` | simulation truth |
| robot_sf/ped_npc/residual_search.py:236 | `ped_radius: float = 0.4` | simulation truth |
| robot_sf/ped_npc/residual_search.py:457 | `ped_radius: float = 0.4,` | simulation truth |
| robot_sf/planner/risk_dwa.py:99 | `pedestrian_radius_m: float = 0.4` | planner/observation assumption |
| robot_sf/planner/risk_dwa.py:692 | `pedestrian_radius_m=float(cfg.get("pedestrian_radius_m", 0.4)),` | planner/observation assumption |
| robot_sf/planner/socnav_base.py:462 | `social_force_ped_v3_default_ped_radius: float = field(default=0.4, kw_only=True)` | planner/observation assumption |
| robot_sf/render/sim_view.py:176 | `ego_ped_radius: float = 0.4` | rendering assumption |
| robot_sf/render/sim_view.py:177 | `ped_radius: float = 0.4` | rendering assumption |
| robot_sf/research/emergent_phenomena.py:152 | `agent_radius=0.35,` | simulation force-kernel calibration (distinct from physical truth) |
| robot_sf/scenario_certification/v1.py:565 | `ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))` | simulation truth |
| robot_sf/sim/sim_config.py:318 | `ped_radius: float = 0.4` | simulation truth |
| scripts/perf/raycast_numba_toggle_benchmark.py:55 | `PED_RADIUS = 0.4` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/tools/density_runtime_smoke.py:268 | `ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/tools/policy_analysis_run.py:1772 | `ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/tools/probe_python_rvo2_integration.py:59 | `"radius": np.array([0.4], dtype=np.float32),` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/calibrate_obstacle_force_10061.py:33 | `FOOTPRINT_RADIUS = 0.4` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/pedestrian_validation_10074.py:45 | `"force_radius_m": 0.35,` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/pedestrian_validation_10074.py:46 | `"baseline_physical_radius_m": 0.40,` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/pedestrian_validation_10074.py:323 | `parser.add_argument("--radius", type=float, default=0.4)` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/pedestrian_validation_10074.py:327 | `if args.radius not in config["radii_m"] or (args.mode == "baseline" and args.radius != 0.4):` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/pedestrian_validation_10074.py:349 | `"resolved_force_radius_m": 0.35 if args.mode == "baseline" else args.radius,` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/run_issue_9735_fault_injection.py:66 | `config=SimpleNamespace(ped_radius=0.4),` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/run_scenario_perturbation_trace_response.py:402 | `ped_radius = float(getattr(config.sim_config, "ped_radius", 0.4))` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/run_topology_hypothesis_diagnostics.py:293 | `radius = _first_float(pedestrians.get("radius"), 0.35)` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/run_topology_hypothesis_diagnostics.py:295 | `radius = 0.35` | diagnostic/fixture assumption; inspect caller before treating as truth |
| scripts/validation/validate_forecast_planner_consumer.py:67 | `"radius": np.array([0.35, 0.35, 0.35], dtype=np.float32),` | diagnostic/fixture assumption; inspect caller before treating as truth |
