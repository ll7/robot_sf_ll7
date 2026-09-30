# PEDFIX 0.0.8 diagnostic evidence

Implementation verification and development-seed diagnostics only. This is not release admission, held-out evaluation, a planner ranking, or thesis claim admission. No fallback/degraded row is counted as successful execution.

Base: `32b58d273f3dbdec72a385844cb8c100f8581136`. Final runtime source: `e8af63df613529517472671a891e4e684540ac5c`. Later commits add evidence only. Release input: `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`; roster: `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`.

## Changed contracts

Pedestrian placement, group-size draws, zone goals, route-end respawns, archetypes, response-law assignments, desired-speed sampling and the controlled ego pedestrian use private NumPy generators. `SeedSequence(episode_seed).spawn(4)` derives placement, archetype, response-law and desired-speed streams; explicit per-feature seed overrides remain supported. Robot-exclusion retries use a separate child stream, so extra retries do not consume the unguarded respawn stream. Robot route sampling retains its existing seeded global context; the shared zone helper retains its legacy fallback for robot callers. Seeded resets rebuild pedestrian population and behavior streams, including navigators. Unseeded construction uses private entropy.

`simulation_config.groups` means P(sampled group has more than one member). P(size=1)=1-groups; sizes 2..N divide the remaining mass in the default 0.3 decay ratio. For max size 3 and groups=0.5 the probabilities are [0.5, 5/13, 3/26]. A final population remainder may still be singleton. Values outside [0,1], non-finite values, and groups>0 with max size 1 are rejected. The three group-crossing scenarios already declare 0.5. No scenario files changed.

The 48 release scenarios use these simulation keys: goal_completion_policy, groups, max_episode_steps, max_peds_per_group, ped_density, robot_goal_sampling_policy, route_spawn_distribution, route_spawn_jitter_frac, social_force_kernel_version. Only groups was unread before this change; all nine are now applied. Unknown keys fail before an episode runs.

Only overlapping reset rows are relocated. Required centre distance is robot radius + pedestrian radius + 0.1 m + max(current speed, walking speed cap) × 1 s. Static geometry, other pedestrians and all robot footprints must remain clear. The relocated velocity keeps its speed and points toward its current route goal; a non-closing candidate is preferred. If the route goal is inside a robot footprint, every route heading closes on it; a geometrically clear candidate with the full reaction buffer is used instead of leaving the original overlap unresolved. If no candidate is admissible, the existing unresolved-placement report remains explicit. The wall force is unchanged.

## Reproduction

From the requested source checkout with project dependencies installed:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf uv run python scripts/validation/measure_pedfix_stationary.py
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf uv run python scripts/validation/measure_pedfix_planners.py --mode roster
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf uv run python scripts/validation/measure_pedfix_planners.py --mode gate
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf uv run pytest -n0 tests/ped_npc/test_pedfix_episode_contract.py tests/ped_npc/test_force_population_size_split.py
```

For baseline runs, execute the supplied measurement script from a detached checkout of the base and set PYTHONPATH to that checkout and its fast-pysf directory. Scripts use only dev seeds: 1001..1005 for stalls, 1001/1002 for planner runs. Never run more than two simulation processes. The original audit probes were also preserved unchanged outside the checkout.

## Global NumPy consumption before the fix

Two scenarios (head-on corridor medium and circular crossing) × seeds 1001/1002 × all 14 roster planners. Canonical policy builder and episode harness, horizon 600, dt 0.1, normal termination. Fingerprint includes the entire NumPy RandomState tuple before/after each policy call and construction, including lazy first-step initialization. The wrapper preserves policy runtime attributes. No trace reset, fallback planner, or degraded row was substituted.

| Planner | Episodes | Calls | Changed calls | Construction changed |
|---|---:|---:|---:|---|
| prediction_planner | 4 | 588 | 0 | False |
| goal | 4 | 543 | 0 | False |
| social_force | 4 | 860 | 0 | False |
| orca | 4 | 489 | 0 | False |
| ppo | 4 | 521 | 0 | False |
| socnav_sampling | 4 | 526 | 0 | False |
| sacadrl | 4 | 717 | 0 | False |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 4 | 488 | 0 | False |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 4 | 488 | 0 | False |
| hybrid_rule_v4_fast_progress_static_escape | 4 | 504 | 0 | False |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 4 | 529 | 0 | False |
| guarded_ppo | 4 | 1353 | 0 | False |
| predictive_mppi | 4 | 687 | 0 | False |
| risk_dwa | 4 | 736 | 0 | False |

Observed consumers: none (0 of 9029 policy calls; no construction changed global state). SICNav is outside this roster. The code defect is real, but this measurement does not establish historical 0.0.7 crowd differences: it covers the current 0.0.8 roster/configs and finite observed paths, not old planner revisions or all possible episodes. Different robot actions can still legitimately change pedestrian motion through physical interaction.

## Behavior change gate

Five affected scenarios × seeds 1001/1002 × goal, ORCA, and frozen hybrid_rule_v4_fast_progress_static_escape. Same scenario/seed and planner config, different pedestrian sampling contract. Seeds match; initial crowds are intentionally not byte-identical across versions. No unchanged-outcome promise or statistical performance claim is made.

| Planner | Before success/collision/timeout | After success/collision/timeout |
|---|---|---|
| goal | 8/2/0 | 5/5/0 |
| orca | 10/0/0 | 10/0/0 |
| hybrid_rule_v4_fast_progress_static_escape | 9/1/0 | 10/0/0 |

| Planner | Scenario | Seed | Before (steps) | After (steps) |
|---|---|---:|---|---|
| goal | classic_head_on_corridor_medium | 1001 | collision (147) | collision (164) |
| goal | classic_head_on_corridor_medium | 1002 | collision (154) | success (308) |
| goal | francis2023_circular_crossing | 1001 | success (123) | success (123) |
| goal | francis2023_circular_crossing | 1002 | success (119) | success (119) |
| goal | classic_group_crossing_low | 1001 | success (171) | success (171) |
| goal | classic_group_crossing_low | 1002 | success (161) | success (161) |
| goal | classic_group_crossing_medium | 1001 | success (171) | collision (90) |
| goal | classic_group_crossing_medium | 1002 | success (161) | collision (72) |
| goal | classic_group_crossing_high | 1001 | success (171) | collision (90) |
| goal | classic_group_crossing_high | 1002 | success (161) | collision (72) |
| orca | classic_head_on_corridor_medium | 1001 | success (162) | success (166) |
| orca | classic_head_on_corridor_medium | 1002 | success (163) | success (162) |
| orca | francis2023_circular_crossing | 1001 | success (83) | success (83) |
| orca | francis2023_circular_crossing | 1002 | success (81) | success (81) |
| orca | classic_group_crossing_low | 1001 | success (100) | success (96) |
| orca | classic_group_crossing_low | 1002 | success (88) | success (91) |
| orca | classic_group_crossing_medium | 1001 | success (101) | success (107) |
| orca | classic_group_crossing_medium | 1002 | success (88) | success (88) |
| orca | classic_group_crossing_high | 1001 | success (100) | success (108) |
| orca | classic_group_crossing_high | 1002 | success (92) | success (207) |
| hybrid_rule_v4_fast_progress_static_escape | classic_head_on_corridor_medium | 1001 | success (202) | success (254) |
| hybrid_rule_v4_fast_progress_static_escape | classic_head_on_corridor_medium | 1002 | success (227) | success (209) |
| hybrid_rule_v4_fast_progress_static_escape | francis2023_circular_crossing | 1001 | collision (3) | success (75) |
| hybrid_rule_v4_fast_progress_static_escape | francis2023_circular_crossing | 1002 | success (72) | success (71) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_low | 1001 | success (144) | success (118) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_low | 1002 | success (104) | success (252) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_medium | 1001 | success (144) | success (153) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_medium | 1002 | success (105) | success (150) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_high | 1001 | success (142) | success (160) |
| hybrid_rule_v4_fast_progress_static_escape | classic_group_crossing_high | 1002 | success (125) | success (287) |

The goal baseline loses three successes in this small sample; ORCA stays at ten successes, and the hybrid improves from nine to ten. Review this changed crowd calibration before freezing. These outcomes are implementation diagnostics, not enough to infer a performance ordering.

## Reset contacts

Base audit and new fail-on-base regressions reproduce contact at steps 3 (seed 1001) and 4 (1008), under 1.4 m centre distance. On fixed code, a stationary robot has no contact in its first 20 steps (2 s):

- Seed 1001: minimum centre distance 2.091036 m; relocated rows [].
- Seed 1008: minimum centre distance 1.736550 m; relocated rows [].

A forced-overlap regression independently verifies reaction clearance and goal-aligned, preferentially non-closing velocity, so the proof does not rely solely on new RNG layouts moving the two old problematic pedestrians.

## Wall/obstacle stall disclosure (issue #10017)

240 episodes on each version: all 48 scenarios × seeds 1001..1005. Stationary zero robot action, robot reaction force enabled, original wall law, full declared scenario horizon (400..700 steps, 40..70 s; including 650 steps/65 s); termination intentionally ignored for this pedestrian measurement. A slot is flagged once if speed is <0.1 m/s continuously for >5 s, current-goal distance is >0.2 m (the actual desired-force stopping threshold), and robot centre distance is >3.4 m (1.0 m robot + 0.4 m pedestrian + 2.0 m activation clearance). At dt=0.1, strictly >5 s means at least 51 consecutive eligible samples. Counters reset whenever any condition fails. Each persistent pedestrian row is counted once per episode, including across route respawns; this is slot accounting, not unique-person accounting. Fraction is flagged slots / all spawned slots, pooled across the five seeds; N=0 is N/A.

All flagged slots also have a qualifying stall within 3 m of a wall/obstacle segment. Thus the observed stall and wall-associated fractions coincide here. This operational measurement does not causally separate wall repulsion, missing pedestrian routing, or group interactions; it does not prove the absence of shorter or slower-onset stalls. Goals within the 0.2 m stopping threshold and pedestrians near the robot are excluded. The earlier exploratory 1 m goal cutoff was superseded and is not used in these tables. No wall-law code or thesis prose was changed.

| Scenario | Base blocked/slots | Fixed blocked/slots | Fixed fraction | Fixed blocked counts (1001..1005) |
|---|---:|---:|---:|---|
| classic_bottleneck_low | 0/0 | 0/0 | N/A | 0, 0, 0, 0, 0 |
| classic_bottleneck_medium | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| classic_bottleneck_high | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| classic_realworld_double_bottleneck_high | 30/40 | 30/40 | 75.00% | 6, 6, 6, 6, 6 |
| classic_station_platform_medium | 5/130 | 5/130 | 3.85% | 1, 1, 1, 1, 1 |
| classic_cross_trap_low | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| classic_cross_trap_medium | 0/40 | 0/40 | 0.00% | 0, 0, 0, 0, 0 |
| classic_cross_trap_high | 0/60 | 0/60 | 0.00% | 0, 0, 0, 0, 0 |
| classic_doorway_low | 9/15 | 7/15 | 46.67% | 2, 1, 2, 1, 1 |
| classic_doorway_medium | 6/30 | 6/30 | 20.00% | 1, 2, 1, 1, 1 |
| classic_doorway_high | 14/45 | 8/45 | 17.78% | 2, 2, 1, 1, 2 |
| classic_group_crossing_low | 0/10 | 0/10 | 0.00% | 0, 0, 0, 0, 0 |
| classic_group_crossing_medium | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| classic_group_crossing_high | 0/20 | 0/20 | 0.00% | 0, 0, 0, 0, 0 |
| classic_head_on_corridor_low | 0/10 | 0/10 | 0.00% | 0, 0, 0, 0, 0 |
| classic_head_on_corridor_medium | 0/20 | 0/20 | 0.00% | 0, 0, 0, 0, 0 |
| classic_merging_low | 0/20 | 0/20 | 0.00% | 0, 0, 0, 0, 0 |
| classic_merging_medium | 0/45 | 0/45 | 0.00% | 0, 0, 0, 0, 0 |
| classic_overtaking_low | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| classic_overtaking_medium | 0/30 | 0/30 | 0.00% | 0, 0, 0, 0, 0 |
| classic_t_intersection_low | 0/10 | 0/10 | 0.00% | 0, 0, 0, 0, 0 |
| classic_t_intersection_medium | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| classic_urban_crossing_medium | 0/25 | 0/25 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_frontal_approach | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_pedestrian_obstruction | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_pedestrian_overtaking | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_robot_overtaking | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_down_path | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_intersection_no_gesture | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_blind_corner | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_narrow_hallway | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_narrow_doorway | 5/5 | 5/5 | 100.00% | 1, 1, 1, 1, 1 |
| francis2023_entering_room | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_exiting_room | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_entering_elevator | 5/5 | 5/5 | 100.00% | 1, 1, 1, 1, 1 |
| francis2023_exiting_elevator | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_intersection_wait | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_intersection_proceed | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_following_human | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_leading_human | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_accompanying_peer | 0/5 | 0/5 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_join_group | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_leave_group | 0/15 | 0/15 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_crowd_navigation | 0/45 | 0/45 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_parallel_traffic | 0/40 | 0/40 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_perpendicular_traffic | 0/35 | 0/35 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_circular_crossing | 0/30 | 0/30 | 0.00% | 0, 0, 0, 0, 0 |
| francis2023_robot_crowding | 0/120 | 0/120 | 0.00% | 0, 0, 0, 0, 0 |

## Validation and limitations

- Ten regression cases fail on the base for the intended bug and pass on fixed code: nine in test_pedfix_episode_contract.py plus the runtime crowded-zone goal case in test_force_population_size_split.py. Exact failures: unequal trajectories, ignored group override, unknown-key acceptance, early contact for both seeds, 1.500001 m instead of required 2.15 m relocation, and changed global RNG state in runtime and vendored crowd sampling.
- 174 focused/regression tests passed with pytest -n0 and OMP_NUM_THREADS=1; changed-file lint, format and diff checks passed. The test-value gate is recorded in [test_value.md](test_value.md).
- Factory smoke checks use seed 1001 and one real step: robot, pedestrian, two-robot, and crowd-only factories reproduce initial pedestrian arrays under external NumPy perturbations. No held-out or non-dev environment episodes were run.
- Whole-repository pr_ready_check was not run: its unfiltered test lanes can step environments on held-out/non-dev seeds, conflicting with the task constraints. No full-suite or hosted-CI pass is claimed. Draft review/CI and maintainer acceptance of the changed crowd contract remain outstanding.
- The raw planner episode records and complete command logs are retained outside the worktree; sanitized row-level data, outcome pairs, per-slot stall JSONL and checksums are committed here. No private infrastructure paths or logs are published.

## Clearance-only control with the original crowds

The base plus only the reset-clearance method and geometry helper (see [clearance_only.patch](clearance_only.patch)) preserves the old NumPy population path. Both original problem seeds now have no contact for the first 20 stationary steps. Minimum centre distances are 1001: 2.150001 m, 1008: 2.089636 m. Every relocated row starts at 2.150001 m with route-consistent velocity. This isolates the clearance repair from the RNG migration. Apply the patch to the base and run the two circular-crossing dev-seed regression cases.
