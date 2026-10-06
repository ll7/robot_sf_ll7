# Scenario authoring successor for 0.1.0

[Back to Scenario Zoo](scenario_zoo/index.md)

Use [the 0.1.0 authoring matrix](../configs/scenarios/classic_interactions_francis2023_authoring_0_1_0_v1.yaml)
for the corrected double bottleneck and station platform. It inherits the 48 scenarios from the
0.0.8 matrix and overrides only those two scenarios. It is a new authoring input, not a published
benchmark release. Released YAML, SVG maps, seed sets, manifests and hash-bound bytes remain unchanged.

The double bottleneck uses the high route density, 0.08 pedestrians per square metre of spawnable
sidewalk area. Its eight markers use opposing obstacle-clear lanes through both openings instead of
wall-face starts and straight paths through blocks. The station retains p1's trajectory and pause;
p3 pauses at (60, 5.5), outside the stair block, instead of the unreachable (60, 8).

## Speed settings

`simulation_config` accepts `ped_speed_tier`, `desired_speed_mean`, `desired_speed_std`, and
`desired_speed_seed`. Tier normalization and parameter derivation use the existing simulation
settings validator after all scenario fields are applied. Explicit mean/std take precedence over
the tier. Numeric values must be finite and non-negative; the seed must be a non-negative integer.
Omitting the fields preserves native speeds. Standard deviation alone retains the existing settings
contract: it needs a mean or tier to select a desired-speed distribution.

```yaml
simulation_config:
  ped_speed_tier: typical
  desired_speed_seed: 1001
```

Unknown simulation keys still raise `ValueError`, including misspellings. No warning exception is
needed: all 48 released scenario configurations build successfully without constructing a simulator.

## Retired width-slice manifest

The historical [v1 manifest](../configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.yaml)
is retired as an execution input. Its seed policy has already been changed on main from
`paper_eval_s30` to `release_eval_0_0_8`, but it still binds the obsolete v1 campaign.
The existing release validator refuses it under D-049's canonical-path rule, before any simulation.
Keep its bytes for historical references. The maintained
[template](../configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml)
binds the [v2 campaign](../configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v2.yaml).
Template materialization and sealed execution belong to the release lane; this authoring change
neither executes that campaign nor modifies its inputs.

## Diagnostic evidence

Run from the repository root:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/probe_scenario_authoring.py --output /tmp/scenario-authoring.json
```

The diagnostic only permits seeds 1001–1030 and runs seeds 1001–1005. It holds the robot still,
steps the actual simulator for each scenario's authored budget, and records pedestrian motion,
obstacle occupancy, pause progress, input digests and trajectory digests. Opening-crossing counts
exclude route-end respawn jumps of one metre or more per step. Low displacement in the final ten
seconds includes actors that have reached a goal; it is not a general failure metric.

The checked-in [evidence](context/evidence/scenario_authoring_0_1_0.json) is diagnostic-only.
It does not establish planner rankings, benchmark eligibility or platform/bottleneck claims for the
unchanged 0.0.8 rows. No pedestrian physics files change.

| Seed | Bottleneck crossings x=20, before → after | x=40, before → after | Inside-obstacle pedestrian steps, before → after | p3 target index, before → after | p3 pause steps, before → after |
| --- | --- | --- | --- | --- | --- |
| 1001 | 2 → 17 | 4 → 21 | 1400 → 0 | 1 → 2 | 0 → 51 |
| 1002 | 2 → 17 | 4 → 20 | 1400 → 0 | 1 → 3 | 0 → 51 |
| 1003 | 2 → 17 | 4 → 24 | 1400 → 0 | 1 → 3 | 0 → 51 |
| 1004 | 2 → 16 | 4 → 15 | 1400 → 0 | 1 → 3 | 0 → 51 |
| 1005 | 2 → 18 | 4 → 25 | 1400 → 0 | 1 → 3 | 0 → 51 |

Every bottleneck seed changes from 8 marker pedestrians to 31 pedestrians (23 route spawns).
At dt=0.1 s the existing wait controller reports 51 steps for a 5 s pause. Every platform run
advances beyond the paused waypoint, although p3 has not reached its final destination by 65 s.
On seed 1001 the successor's native cap mean is 0.717097 m/s (route caps 0.65, marker caps 0.91).
Selecting `typical` yields mean 1.324000, range 0.938150–1.723666 m/s and a different trajectory
digest. The original issue's byte-identical native/typical behavior is no longer possible.

## Regression test value and base proof

The five cases in [the regression file](../tests/training/test_scenario_speed_authoring.py) are
seeded or geometry-only and exercise the actual scenario loader, parsed maps and, for speeds,
the live simulator. No production test seam is added.

| Test | Bug / credible regression caught | Why previous coverage misses it | Deterministic real path |
| --- | --- | --- | --- |
| Speed settings (3 cases) | Dropped tier, explicit fields, seed or tier precedence; removal of post-assignment normalization | Speed-tier tests cover mappings, not scenario-to-live-cap wiring | Fixed dev seed; actual loader and simulator; independently computed normal draws |
| Bottleneck | Lost route density or paths through blocks/wall faces | Loader override tests cover supplied fields, not authored geometry | Parsed successor map and body-clearance path intersections |
| Platform | Unreachable pause waypoint inside stairs | Trajectory override tests resolve waypoints without checking stair geometry | Parsed map, pause rule and complete trajectory clearance |

Fail-on-base uses the exact loader source from base `66df3de19` and selects the released scenario
matrix as pre-fix authoring input. All five cases fail: the three speed cases raise
`simulation_config contains unknown keys`, bottleneck density is 0 rather than 0.08, and the platform
path intersects an obstacle. Replacing only base density with 0.08 still fails the marker clearance
assertion. The fixed cases pass. The existing
`test_unknown_simulation_key_fails_at_real_loader` covers strict rejection and passes unchanged.
