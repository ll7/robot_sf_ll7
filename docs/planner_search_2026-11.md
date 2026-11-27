# Two opt-in planners for pedestrians and narrow passages

The velocity-aware sampler and DRA-MPPI are experimental local planners for
the simulated differential-drive robot. Select the experimental benchmark
profile and one of the algorithm configurations below. Each file includes
`allow_testing_algorithms: true`.

| Planner | Algorithm key | Configuration |
| --- | --- | --- |
| Velocity-aware sampler | `velocity_sampler` | [velocity_sampler_dev_r9.yaml](../configs/algos/velocity_sampler_dev_r9.yaml) |
| DRA-MPPI with corridor passing | `dra_mppi` | [dra_mppi_r2.yaml](../configs/algos/dra_mppi_r2.yaml) |
| DRA-MPPI with corridor passing and crossing yield | `dra_mppi` | [dra_mppi_r3.yaml](../configs/algos/dra_mppi_r3.yaml) |

## How the planners choose a command

The **velocity-aware sampler** projects two-stage commands within the
robot's drive limits. It uses tracked pedestrian velocities to estimate
clearance through the turns, and checks the robot's path to a stop against
walls. After a stall it can consider a turn in place. It asks Optimal
Reciprocal Collision Avoidance (ORCA) for a command if it stalls near a
walker without a clear way forward, or if the selected sample predicts
contact before the robot can stop and no clear sample is available. The
robot follows the ORCA command only if the command fits the drive limits and
its braking path clears pedestrians and walls. A contact handoff needs a
fresh prediction at the next decision.

ORCA half-planes bound the robot velocities that avoid a predicted contact. An
optional cost, `velocity_sampler.orca_half_plane_weight`, penalizes the
first segment's velocity by how far it falls outside those half-planes.
The evaluated profile uses a weight of zero. Earlier trials that used
the half-planes to change sample selection lost completions.

**DRA-MPPI**, or dynamic risk-aware model predictive path integral control,
uses the unimodal prediction model from Trevisan et al. It samples sequences
of drive commands and estimates the chance of contact with any pedestrian at
each forecast step. In a corridor with room to pass, it assigns a side lane and
adds samples that steer into it. If none clear both the walls and the 5%
risk threshold, it checks a braking path before issuing a turn to yield.
The crossing profile adds a check near route openings. It keeps a command
that can stop clear and wait, tries the lowest-cost sample that passes the
check if the proposed command fails, or commands a stop if none pass.
[Dynamic Risk-Aware MPPI](https://arxiv.org/abs/2506.21205)

## Paired development results

All planners used the same 30 development seeds (7301–7330) and the same
inputs for each setting and seed. The release grid has 48 settings from
Robot SF 0.0.8. Suite v1 has eight cases in corridors, dense crowds,
doorways and at platforms. Suite v2 has 18 cases and adds situations such as
overtaking and sudden stops. Suite v3 has 15 cases selected to expose
failures of the corridor profile, including crossings and merges.
[Frozen inputs and acquisition details](context/evidence/planner_search_2026-11/manifest.json)

Each count gives completions, episodes with contact and timeouts, in that
order. The differences are candidate minus ORCA rates in percentage points.
The 95% intervals come from 20,000 bootstrap draws of whole seed blocks,
keeping the same cases and planner outcomes together for each drawn seed.

| Input (runs per planner) | Planner | Complete / contact / timeout | Δ completion | Δ contact | Δ timeout | Δ pedestrian contact | Δ obstacle contact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Release (1,440) | ORCA | 1293 / 106 / 41 | — | — | — | — | — |
| Release (1,440) | Velocity-aware sampler | 1384 / 11 / 45 | +6.32 [+4.79, +7.85] | -6.60 [-8.06, -5.28] | +0.28 [-0.42, +0.97] | -6.53 [-7.99, -5.21] | -0.07 [-0.21, +0.00] |
| Release (1,440) | DRA-MPPI corridor | 1371 / 4 / 65 | +5.42 [+3.82, +6.94] | -7.08 [-8.47, -5.69] | +1.67 [+1.04, +2.29] | -7.01 [-8.40, -5.62] | -0.07 [-0.21, +0.00] |
| Release (1,440) | DRA-MPPI crossing | 1370 / 5 / 65 | +5.35 [+3.75, +6.88] | -7.01 [-8.40, -5.62] | +1.67 [+1.04, +2.29] | -7.01 [-8.40, -5.62] | +0.00 [-0.21, +0.21] |
| Suite v1 (240) | ORCA | 179 / 49 / 12 | — | — | — | — | — |
| Suite v1 (240) | Velocity-aware sampler | 149 / 58 / 33 | -12.50 [-17.50, -7.50] | +3.75 [-1.25, +8.75] | +8.75 [+4.58, +12.92] | +11.25 [+6.67, +15.83] | -7.50 [-10.42, -5.00] |
| Suite v1 (240) | DRA-MPPI corridor | 215 / 25 / 0 | +15.00 [+9.17, +20.42] | -10.00 [-15.00, -5.00] | -5.00 [-7.50, -2.50] | -2.50 [-6.67, +2.08] | -7.50 [-10.42, -5.00] |
| Suite v1 (240) | DRA-MPPI crossing | 215 / 25 / 0 | +15.00 [+9.17, +20.42] | -10.00 [-15.00, -5.00] | -5.00 [-7.50, -2.50] | -2.50 [-6.67, +2.08] | -7.50 [-10.42, -5.00] |
| Suite v2 (540) | ORCA | 356 / 175 / 9 | — | — | — | — | — |
| Suite v2 (540) | Velocity-aware sampler | 255 / 223 / 62 | -18.70 [-23.15, -14.44] | +8.89 [+4.26, +13.33] | +9.81 [+7.59, +12.04] | +11.85 [+7.78, +15.74] | -2.96 [-4.26, -1.67] |
| Suite v2 (540) | DRA-MPPI corridor | 472 / 67 / 1 | +21.48 [+17.78, +25.19] | -20.00 [-23.89, -15.93] | -1.48 [-2.59, -0.56] | -17.04 [-20.56, -13.52] | -2.96 [-4.26, -1.67] |
| Suite v2 (540) | DRA-MPPI crossing | 472 / 67 / 1 | +21.48 [+17.78, +25.19] | -20.00 [-23.89, -15.93] | -1.48 [-2.59, -0.56] | -17.04 [-20.56, -13.52] | -2.96 [-4.26, -1.67] |
| Suite v3 (450) | ORCA | 162 / 251 / 37 | — | — | — | — | — |
| Suite v3 (450) | Velocity-aware sampler | 169 / 227 / 54 | +1.56 [-1.11, +4.44] | -5.33 [-9.33, -1.11] | +3.78 [+0.89, +6.44] | -4.44 [-8.44, -0.44] | -0.89 [-2.00, +0.00] |
| Suite v3 (450) | DRA-MPPI corridor | 85 / 345 / 20 | -17.11 [-20.89, -13.56] | +20.89 [+17.11, +24.67] | -3.78 [-6.22, -1.33] | +21.78 [+18.00, +25.56] | -0.89 [-2.00, +0.00] |
| Suite v3 (450) | DRA-MPPI crossing | 124 / 306 / 20 | -8.44 [-12.00, -4.89] | +12.22 [+8.44, +16.00] | -3.78 [-6.22, -1.33] | +13.11 [+9.56, +16.67] | -0.89 [-2.00, +0.00] |

The sampler completed more routes than ORCA on the release grid. ORCA led on
suites v1 and v2. On suite v3, the sampler had fewer contacts. Its completion
difference on that suite has an interval that includes zero.

Both DRA-MPPI profiles completed more routes and had fewer contacts than ORCA on
the release grid and suites v1 and v2. They matched each other on both early
challenge suites. In suite v3, crossing yield gained 39 completions over
corridor passing alone. ORCA still completed 38 more routes than the crossing
profile and had fewer contacts than either DRA-MPPI profile. On the release grid,
the crossing profile had one additional wall contact.

The intervals describe variation across seeds for these fixed settings.
The trials use known maps and tracked pedestrians without observation noise.
In the release simulator, pedestrian wall forces can keep walkers away from
narrow openings. The results apply to these simulations and development
seeds. They cannot estimate physical robot contact rates.
[Release notes on narrow passages](release/0.0.8/release_notes.md#narrow-passages)

## Reproduce the comparison

From the repository root on Linux, install the locked environment with
native RVO2, the library used by ORCA.

```bash
uv sync --locked --python 3.13.14 --extra benchmark --extra viz
export PYTHONPATH="$PWD:$PWD/fast-pysf" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python -m scripts.repro.planner_repro --summary
.venv/bin/python -m scripts.repro.planner_repro --allow-rebuilt-rvo2 --workers 16
```

The summary reads the archived rows and recalculates the grid-level intervals
without running the simulator. The replay checks 1,068 episodes on three
seeds across all 89 settings. It verifies the terminal outcome, both contact
flags and the step count for each episode. Use `--all-seeds` for the full
comparison or `--smoke` for one release setting on the three seeds. You can select
planners with `--arms orca drammpi2 drammpi3 sampler`. The native library's binary
hash can differ after a source build. The rebuild option prints that hash and
requires the replay to match the archived rows.

The archive contains the original challenge manifests as gzip files, with
checksums for their uncompressed bytes. They preserve the source names and
selection history. The replay loads the exact simulation inputs archived for
all 30 development seeds.
[Suite v1](context/evidence/planner_search_2026-11/challenge_suite.v1.json.gz) ·
[Suite v2](context/evidence/planner_search_2026-11/challenge_suite.v2.json.gz) ·
[Suite v3](context/evidence/planner_search_2026-11/challenge_suite.v3.json.gz) ·
[ORCA and DRA-MPPI outcomes](context/evidence/planner_search_2026-11/paired_outcomes.csv.gz) ·
[Sampler outcomes](context/evidence/planner_search_2026-11/sampler_outcomes.csv.gz)
