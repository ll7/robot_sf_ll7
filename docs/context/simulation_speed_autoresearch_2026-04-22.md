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

Observed improvement on this short local contract: about 25.2% higher steady-state steps/sec with
identical rollout hash and reward.

## Validation

- `.venv/bin/python -m pytest tests/test_occupancy_additional.py -q`
- `.venv/bin/python -m ruff check robot_sf/nav/occupancy.py tests/test_occupancy_additional.py`
- `.venv/bin/python -m ruff format --check robot_sf/nav/occupancy.py tests/test_occupancy_additional.py`

## Follow-Up Boundary

This is a short local micro-benchmark, not a camera-ready benchmark result. Before making broad
runtime claims, rerun a larger benchmark or environment throughput suite across representative maps
and obstacle densities.
