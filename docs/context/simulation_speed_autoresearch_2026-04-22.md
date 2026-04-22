# Simulation Speed Autoresearch - 2026-04-22

## Goal

Improve steady-state Robot SF simulation throughput without changing rollout behavior.

## Experiment Contract

- Speed metric: 1,000 fixed-seed `make_robot_env` steps after 50 warmup steps.
- Accuracy guard: compare final robot/pedestrian state hash, total reward, reset count, and
  pedestrian count.
- Scope: local CPU-only short run on `LeLuMBP24`; no public API or benchmark schema changes.
- Artifact log: `output/ai/autoresearch/simulation_speed/results.tsv`.

## Evidence

Warm-profile evidence showed steady-state time concentrated in `RobotEnv.step`, with obstacle
collision checks over flat NumPy segment arrays contributing about 0.25 seconds over 500 profiled
steps.

The retained change moves the `np.ndarray` segment path in
`robot_sf/nav/occupancy.py::circle_collides_any_lines` into a Numba-compiled loop. The generic
iterable segment path remains unchanged.

## Result

| Variant | Steps/sec | Elapsed | State hash | Reward | Resets |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 360.42 | 2.775 s | `eb2e1ccf6d9e6ec916f7fc4daa6c1a81` | -216.11873176486793 | 13 |
| numba obstacle segments | 451.18 | 2.216 s | `eb2e1ccf6d9e6ec916f7fc4daa6c1a81` | -216.11873176486793 | 13 |
| direct group centroid | 485.39 | 2.060 s | `eb2e1ccf6d9e6ec916f7fc4daa6c1a81` | -216.11873176486793 | 13 |
| numba group repulsive | 566.22 | 1.766 s | `eb2e1ccf6d9e6ec916f7fc4daa6c1a81` | -216.11873176486793 | 13 |
| numba pedestrian collision | 576.71 | 1.734 s | `eb2e1ccf6d9e6ec916f7fc4daa6c1a81` | -216.11873176486793 | 13 |

Observed retained improvement on this short local contract: about 60.0% higher steady-state
steps/sec with identical rollout hash and reward.

## Multi-Map Current-Branch Check

After retaining the speedups, a broader current-branch sweep ran 500 measured steps after 30 warmup
steps, with two repeats per map and fixed seed `12345`. The artifact log is
`output/ai/autoresearch/simulation_speed/multimap_current.tsv`.

| Map | Pedestrians | Obstacle segments | Mean steps/sec | Repeat hashes match |
| --- | ---: | ---: | ---: | --- |
| `planner_sanity_open` | 0 | 20 | 7990.17 | yes |
| `classic_head_on_corridor` | 1 | 20 | 6422.43 | yes |
| `classic_bottleneck_high` | 4 | 36 | 4436.35 | yes |
| `classic_crossing` | 2 | 28 | 6286.04 | yes |
| `uni_campus_big` | 81 | 862 | 593.98 | yes |

This check broadens coverage across empty, sparse, bottleneck/crossing, and dense-campus maps. It
does not replace an A/B benchmark against the pre-optimization commits.

## Follow-Up Experiments

- `direct group centroid`: replaced allocation-heavy `pos_of_many` plus `np.mean` in
  `robot_sf/ped_npc/ped_grouping.py::group_centroid` with direct state-array summation.
- `numba group repulsive`: replaced the `each_diff` allocation path in
  `fast-pysf/pysocialforce/forces.py::GroupRepulsiveForce` with a Numba helper for one group's
  pairwise repulsive force.
- `numba pedestrian collision`: replaced generator-based circle checks for NumPy pedestrian
  position arrays in `robot_sf/nav/occupancy.py::ContinuousOccupancy.is_pedestrian_collision`.

Discarded experiments:

- `numba group coherence`: improved speed but changed the exact rollout state hash, including when
  compiled without `fastmath`.
- `inplace force accumulation`: preserved the state hash but measured slower than the retained
  current best.

## Validation

- `.venv/bin/python -m pytest tests/test_occupancy_additional.py -q`
- `.venv/bin/python -m pytest tests/ped_grouping_test.py fast-pysf/tests/test_forces.py::test_group_rep_force -q`
- `.venv/bin/python -m ruff check robot_sf/nav/occupancy.py tests/test_occupancy_additional.py`
- `.venv/bin/python -m ruff check fast-pysf/pysocialforce/forces.py robot_sf/ped_npc/ped_grouping.py tests/ped_grouping_test.py`
- `.venv/bin/python -m ruff format --check robot_sf/nav/occupancy.py tests/test_occupancy_additional.py`
- `.venv/bin/python -m ruff format --check fast-pysf/pysocialforce/forces.py robot_sf/ped_npc/ped_grouping.py tests/ped_grouping_test.py`
- Multi-map current-branch sweep over `planner_sanity_open`, `classic_head_on_corridor`,
  `classic_bottleneck_high`, `classic_crossing`, and `uni_campus_big`; artifact:
  `output/ai/autoresearch/simulation_speed/multimap_current.tsv`.

## Follow-Up Boundary

These are local throughput checks, not camera-ready benchmark results. Before making broad runtime
claims, run an A/B benchmark against the pre-optimization commits or a dedicated release-style
throughput suite.
