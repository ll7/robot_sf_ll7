# Issue #1318 TEB Corridor-Deadlock Evaluation

Date: 2026-05-20

Related issue:

- #1318: <https://github.com/ll7/robot_sf_ll7/issues/1318>

Related context and configs:

- `docs/context/issue_1022_route_corridor_design_research.md`
- `configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml`
- `configs/algos/teb_commitment_camera_ready.yaml`
- `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml`
- `docs/context/evidence/issue_1318_teb_corridor_deadlock_2026-05-20/summary.json`

## Goal

Evaluate the in-repo TEB commitment planner on the classic-merging corridor-deadlock seeds
identified by #1022, and compare it with ORCA plus the current hybrid-rule local-planner incumbent.
This note is not a paper-facing benchmark claim; it is a narrow dependency-unblocking slice for
follow-up issues that need a reusable TEB-vs-incumbent baseline.

## Scenario Slice

The tracked slice uses `classic_merging.svg` with a 600-step horizon:

- `classic_merging_low`, `ped_density: 0.02`, seeds `111` and `113`;
- `classic_merging_medium`, `ped_density: 0.05`, seeds `111`, `112`, and `113`.

These seeds come from the #1022 corridor-design research note. The previous #1022 hybrid-rule
regeneration found one low-density success and four timeouts on current main. The #1318 rerun uses
the normal map-runner path instead of the step-diagnostics script so TEB, ORCA, and the hybrid-rule
incumbent are compared through the same episode-record contract.

## Integrity Fix

The first TEB and ORCA attempts failed closed before producing usable comparison data:

- TEB: 5 jobs, 0 written, 5 failed;
- ORCA: 5 jobs, 2 written, 3 failed;
- failure: `outcome.collision_event=true but collision metrics <= 0`.

The root cause was an integrity mismatch between exact environment collision flags and sampled
collision metrics. `RobotEnv` reports obstacle collisions from exact occupancy geometry, while
`wall_collisions()` checks sampled obstacle points. A contact between sampled points can therefore
produce `collision_event=true` with zero sampled wall collisions.

Map-runner now tracks per-step exact collision flags and floors the matching metric count to one
when a typed flagged collision would otherwise be lost. If a supported environment reports only the
top-level `info["collision"]` flag, map-runner floors the total collision metric without inventing a
typed attribution. This preserves the benchmark outcome instead of downgrading a collision, and it
keeps fallback/degraded runs fail-closed.

## Results

All commands used `LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2`, `--benchmark-profile
experimental`, `--horizon 600`, `--workers 1`, `--no-resume`, `--no-video`, and
`--structured-output json`.

| Algo | Scenario | Episodes | Successes | Collisions | Timeouts | Avg steps | Terminations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `teb` | `classic_merging_low` | 2 | 0 | 2 | 0 | 247.5 | `collision=2` |
| `teb` | `classic_merging_medium` | 3 | 0 | 3 | 0 | 249.0 | `collision=3` |
| `orca` | `classic_merging_low` | 2 | 1 | 1 | 0 | 280.5 | `collision=1`, `success=1` |
| `orca` | `classic_merging_medium` | 3 | 1 | 2 | 0 | 307.0 | `collision=2`, `success=1` |
| `hybrid_rule_local_planner` | `classic_merging_low` | 2 | 2 | 0 | 0 | 427.0 | `success=2` |
| `hybrid_rule_local_planner` | `classic_merging_medium` | 3 | 2 | 1 | 0 | 487.7 | `collision=1`, `success=2` |

Seed-level collision attribution:

- TEB collisions are static obstacle collisions on all five selected seeds.
- ORCA collisions are pedestrian collisions on low seed `111` and medium seeds `111` and `113`.
- Hybrid-rule has one pedestrian collision on medium seed `112`.

## Interpretation

The in-repo TEB commitment planner is runnable on the corridor-deadlock slice, but this run does not
support promoting it as an incumbent replacement. It collides with static geometry on every selected
seed, while ORCA is mixed and the current hybrid-rule candidate solves four of five.

The useful #1318 outcome is therefore negative evidence plus a reusable scenario slice. Follow-up
work should use this slice as a quick regression surface for corridor-commitment changes, but should
not claim TEB improves the corridor-deadlock case without a new planner/config change and a rerun.

## Validation

Focused proof:

```bash
uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_map_runner_utils.py
uv run pytest tests/benchmark/test_map_runner_utils.py -k 'collision_wins or floors_exact_obstacle_collision_metrics or teb' -q
```

Benchmark evidence:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench run --matrix configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml --algo teb --algo-config configs/algos/teb_commitment_camera_ready.yaml --benchmark-profile experimental --horizon 600 --out output/benchmarks/issue_1318_teb_corridor_deadlock_teb.jsonl --workers 1 --no-resume --no-video --structured-output json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench run --matrix configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml --algo orca --benchmark-profile experimental --horizon 600 --out output/benchmarks/issue_1318_teb_corridor_deadlock_orca.jsonl --workers 1 --no-resume --no-video --structured-output json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench run --matrix configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml --algo hybrid_rule_local_planner --algo-config configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml --benchmark-profile experimental --horizon 600 --out output/benchmarks/issue_1318_teb_corridor_deadlock_hybrid.jsonl --workers 1 --no-resume --no-video --structured-output json
```

Artifact decision:

- Raw JSONL outputs under `output/benchmarks/` are ignored and reproducible.
- The compact aggregate summary is tracked at
  `docs/context/evidence/issue_1318_teb_corridor_deadlock_2026-05-20/summary.json`.
