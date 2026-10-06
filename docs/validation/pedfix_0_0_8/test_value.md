# Test-value gate

The bug controls ran against base `32b58d273f3dbdec72a385844cb8c100f8581136`, with the new test files copied into that checkout and its implementation unchanged. Every environment episode uses a dev seed. Commands use `OMP_NUM_THREADS=1`, `PYTHONPATH=.:fast-pysf`, and `pytest -n0`.

| Case | Protected behavior / credible regression | Nearest old coverage and gap | Production test seam |
|---|---|---|---|
| trajectory draw / reseed (two cases) | Identical pedestrian paths for the same seed and robot controls after a route respawn; forgetting rng on respawn would fail | test_respawn_keeps_random_stream_when_legacy_sample_is_clear checks an identical RNG history, not external per-step perturbation | None: actual release env, 400 steps, assert a respawn occurred |
| groups override | Actual sampled population reflects groups=1; removing loader/factory forwarding fails | test_build_robot_config_applies_archetype_simulation_overrides checks another override, not actual group probability | None: actual scenario builder and population |
| unknown key | Misspelled simulation_config key fails before execution; permissive dispatch fails | test_build_robot_config_rejects_unknown_goal_completion_policy covers one known value, not unknown mapping keys | None: actual scenario builder; assert exact unknown-key reason |
| circular crossing (two seeds) | No stationary-robot contact in first 2 s for audit seeds 1001/1008; reverting clearance and RNG restores the bug | test_relocation_moves_only_overlapping_pedestrians proves instantaneous clearance only | None: actual release env |
| forced overlap | At least one second of reaction distance and route-consistent, non-closing velocity; restoring position-only relocation fails | test_relocation_moves_only_overlapping_pedestrians checks 0.1 m margin and does not inspect velocity | None: mutate a real simulator state to a controlled overlap, with independent 2.15 m and vector expectations |
| robot-contained goal | Relocate with the reaction buffer even if all route headings close on the robot; strict non-closing candidate rejection would fail | test_relocation_respects_walls does not put the current goal inside the robot footprint | None: real overlapping row; independent 2.15 m requirement |
| vendored global state | Standalone fast-pysf population and behavior reset do not consume global NumPy; a forgotten RNG argument fails | test_force_population_scatter_spawn_is_seed_deterministic does not exercise standalone vendored population | None: actual population API and complete global-state tuple |
| native crowd goal (replaces mocked call-shape test) | Default runtime crowd behavior owns its generator; fallback to global draws fails | Old test asserted only a sampler mock's positional invocation | None: actual sampling and global-state tuple |

The changed respawn controls in test_spawn_clearance_issue_9725.py protect footprint exclusion, covered-zone fallback, clear-sample stream consumption and reset handling of the live robot pose; extra draws or stale footprint use would fail. Their existing assertions remain intact and now receive seeded private generators, because global seeding no longer controls pedestrian sampling. No production seam was added.

The changed test_kernel_selector_preserves_legacy_simulation_and_environment_hashes protects the frozen historical payload after projecting out the two newly introduced fields; edits to pre-existing defaults still fail its original digest. It is the existing coverage, with the projection needed for the explicit configuration extension; no production seam was added and current runtime hashes still include the new fields.

The group test permits one final singleton remainder, matching the finite-population contract. The reaction test computes expected clearance from physical radii and speed, and expected velocity from the supplied route direction; it does not call the production clearance/velocity helper as its oracle. No tests were deleted.

Base commands and results:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf /path/to/project/.venv/bin/python -m pytest -n0 tests/ped_npc/test_pedfix_episode_contract.py -q --tb=short
# 9 failed: unequal trajectories after first respawn; old [1,1,2] group sizes;
# no unknown-key error; centre distance <1.4 m on seeds 1001/1008;
# relocation 1.500001 m < required 2.15 m; global state changed in vendored population; robot-contained goal relocation lacks reaction clearance.
OMP_NUM_THREADS=1 PYTHONPATH=.:fast-pysf /path/to/project/.venv/bin/python -m pytest -n0 tests/ped_npc/test_force_population_size_split.py::test_native_crowded_zone_goal_sampling_leaves_global_numpy_untouched -q --tb=short
# 1 failed: global NumPy state array differs (624 of 624 state words).
```

The fixed regression command includes those cases, existing population/desired-speed/archetype tests, scenario-loader tests, spawn-clearance tests and configuration tests: **174 passed**. The same explicit bug commands on the fixed source were also rerun: 9 passed in 12.42 s and 1 passed in 0.96 s. An initial vendored test used the wrong state accessor; it was corrected to raw_states, then rerun on both versions. That fixture error is not counted as bug proof.

Existing respawn positive controls now use equal explicitly seeded private generators, preserving their original footprint/fallback and stream-consumption assertions. The simulation-config historical hash test projects out the two newly added fields before checking the original frozen payload digest; current environment hashes include the new fields, which carry the changed crowd contract. These are compatibility-test adaptations, not additional bug controls.

No unproven bug test is claimed as validation. Whole-repository readiness was not run because its unfiltered lanes conflict with the task's held-out and dev-only environment constraints.
