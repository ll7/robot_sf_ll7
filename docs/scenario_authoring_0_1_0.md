# Scenario authoring successor for 0.1.0

[Back to Scenario Zoo](scenario_zoo/index.md)

Use [the 0.1.0 authoring matrix](../configs/scenarios/classic_interactions_francis2023_authoring_0_1_0_v1.yaml)
for the corrected double bottleneck and station platform. It inherits the 48 scenarios from the
0.0.8 matrix and overrides only those two scenarios. It is a new authoring input, not a published
benchmark release. This PR preserves the existing YAML, SVG maps, seed sets and manifests relative to fresh main. The freeze comparison and its pre-existing main differences are listed below.

The double bottleneck uses the high route density, 0.08 pedestrians per square metre of spawnable
sidewalk area. Its eight markers use opposing obstacle-clear lanes through both openings instead of
wall-face starts and straight paths through blocks. The station retains p1's trajectory and pause;
p3 pauses at (60, 5.5), outside the stair block, instead of the unreachable (60, 8).

## Speed settings

`simulation_config` accepts `ped_speed_tier`, `desired_speed_mean`, `desired_speed_std`, and
`desired_speed_seed`. Tier normalization and parameter derivation use the existing simulation
settings validator after all scenario fields are applied. Explicit mean/std take precedence over
the tier. Numeric values must be finite and non-negative; the seed must be a non-negative integer.
Omitting the fields preserves native speeds. Standard deviation without a mean or tier is rejected rather than accepting an
unapplied speed distribution setting.

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

The six cases in [the regression file](../tests/training/test_scenario_speed_authoring.py) are
seeded or geometry-only and exercise the actual scenario loader, parsed maps and, for speeds,
the live simulator. No production test seam is added.

| New test case | Defect caught | Fails on base? | Tests behaviour? | Cheapest meaningful check? |
| --- | --- | --- | --- | --- |
| Typical + speed seed 1002 | Tier normalization, derived distribution and dropped RNG seed | Yes: unknown speed fields | Yes: actual caps equal independent seeded draws | One seeded reset; no episode rollout |
| Typical + zero std | Explicit spread must override the tier spread | Yes: unknown speed fields | Yes: all live caps are exactly 1.3 | One reset isolates spread precedence |
| Explicit mean/std/seed | Explicit speed fields must reach the simulator without a tier | Yes: unknown speed fields | Yes: all live caps are exactly 1.1 | One reset isolates direct-field wiring |
| Typical + explicit mean/std | Explicit values must take precedence over tier defaults | Yes: unknown speed fields | Yes: all live caps are exactly 1.1 | One reset isolates mean precedence |
| Bottleneck routes/markers | Missing high density and paths intersecting body-buffered obstacles | Yes: density; density-only repair still fails marker clearance | Yes: real loaded population setting and full marker path geometry | Map geometry avoids a stochastic rollout |
| Platform pause path | Pause waypoint and connecting path inside the stair block | Yes: buffered obstacle intersection | Yes: real loaded pause and trajectory clearance | Map geometry avoids a stochastic rollout |

Existing tier tests cover mappings rather than loader-to-live-cap wiring. Existing override tests resolve supplied fields and waypoints without checking these authored paths against obstacles. The 21-row simulator diagnostic complements the geometry checks with actual pedestrian progress; it is not duplicated in every regression test.

Fail-on-base uses a detached checkout of fresh main `53f8f2666e98d00b70a8ef5376790be474a743aa` and selects the released scenario
matrix as pre-fix authoring input. The only test-file substitution is the matrix path, because the successor does not exist on base. The base source is imported from that checkout, not from the fixed environment's editable package. All six cases fail: the four speed cases raise
`simulation_config contains unknown keys`, bottleneck density is 0 rather than 0.08, and the platform
path intersects an obstacle. Replacing only base density with 0.08 still fails the marker clearance
assertion. The fixed cases pass. The existing
`test_unknown_simulation_key_fails_at_real_loader` covers strict rejection and passes unchanged.

## Released input byte proof

The [per-file byte inventory](context/evidence/scenario_authoring_0_1_0_bytes.csv) compares every one of the 1,369 files under `configs/` and `maps/`, plus packaged SVG/JSON map data under `robot_sf/maps/` and `fast-pysf/maps/` in freeze commit `66f402ba176b13e45210d0da0b2cf20fcdc0cc02` with fresh main and this PR. SHA-256 is computed from `git show <revision>:<path>` bytes, with exact byte equality checked separately. No frozen path is missing. All 1,369 head files are byte-identical to main; 1,364 are also byte-identical to the freeze. This deliberately over-inclusive inventory covers every released scenario/config/map, including include ancestors, registries, planner configs and seed sets.

Five upstream differences already exist on main, and this PR preserves them:

- `configs/adversarial/issue_8891_temporal_robustness_packet.yaml`: a working-tree diagnostic provenance pin.
- `configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml` and `benchmark_data_release_s30_h600_zenodo_metadata.template.json`: the approved future metadata successor and its metadata hash.
- `configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml` and `three_width_doorway_release_0_0_8_v1_zenodo_metadata.template.json`: the corresponding future doorway metadata successor and hash.

The metadata changes came from the separate release-description follow-up already merged on main. The scenario matrices, SVGs, scientific campaign configs, seed sets and frozen planner configs are equal to the freeze. An absolute statement that *every* current config/map byte equals the freeze would be false; preservation by this PR is the proven claim. The release branch and freeze commit are read-only comparators.

Reproduce the inventory from the repository root (standard library only):

```python
import hashlib
import subprocess
freeze = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
def blob(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"])
paths = subprocess.check_output([
    "git", "ls-tree", "-r", "--name-only", freeze, "--", "configs", "maps", "robot_sf/maps", "fast-pysf/maps"
], text=True).splitlines()
# Packaged map directories also contain Python tools and Markdown, not map data.
paths = [p for p in paths if p.startswith(("configs/", "maps/"))
         or p.endswith((".svg", ".json"))]
for path in paths:
    frozen, base, head = (blob(rev, path) for rev in (freeze, "53f8f2666", "HEAD"))
    assert base == head, path
    print(path, hashlib.sha256(frozen).hexdigest(),
          hashlib.sha256(head).hexdigest(), frozen == head)
```

## Per-scenario effective speed requests

No checked-in YAML under `configs/scenarios/` requests `ped_speed_tier`, including the resolved 48-row successor. To make the before/after check non-vacuous, every row below explicitly requests `typical` on dev seed 1001. Fresh main rejects all 48 requests before constructing a simulator. The before native column is a separate no-tier control, not the effective speed of the refused request. All after distributions resolve to mean 1.3 and std 0.2 m/s; the table measures live caps at reset, not observed trajectory velocity. The bottleneck population changes with the authored density.

| Scenario | Before native mean m/s | Before typical | After typical count | After live cap mean [min, max] m/s |
| --- | ---: | --- | ---: | --- |
| classic_bottleneck_low | no actors | rejected | 0 | no actors [no actors, no actors] |
| classic_bottleneck_medium | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| classic_bottleneck_high | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_realworld_double_bottleneck_high | 0.910000 | rejected | 31 | 1.324000 [0.938150, 1.723666] |
| classic_station_platform_medium | 0.650000 | rejected | 27 | 1.337676 [0.995988, 1.723666] |
| classic_cross_trap_low | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_cross_trap_medium | 0.650000 | rejected | 8 | 1.260927 [0.995988, 1.541863] |
| classic_cross_trap_high | 0.650000 | rejected | 12 | 1.316614 [0.995988, 1.723666] |
| classic_doorway_low | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_doorway_medium | 0.650000 | rejected | 6 | 1.267435 [0.995988, 1.541863] |
| classic_doorway_high | 0.650000 | rejected | 9 | 1.258330 [0.995988, 1.541863] |
| classic_group_crossing_low | 0.650000 | rejected | 2 | 1.379748 [1.273032, 1.486464] |
| classic_group_crossing_medium | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_group_crossing_high | 0.650000 | rejected | 4 | 1.345451 [1.080446, 1.541863] |
| classic_head_on_corridor_low | 0.650000 | rejected | 2 | 1.379748 [1.273032, 1.486464] |
| classic_head_on_corridor_medium | 0.650000 | rejected | 4 | 1.345451 [1.080446, 1.541863] |
| classic_merging_low | 0.650000 | rejected | 4 | 1.345451 [1.080446, 1.541863] |
| classic_merging_medium | 0.650000 | rejected | 9 | 1.258330 [0.995988, 1.541863] |
| classic_overtaking_low | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_overtaking_medium | 0.650000 | rejected | 6 | 1.267435 [0.995988, 1.541863] |
| classic_t_intersection_low | 0.650000 | rejected | 2 | 1.379748 [1.273032, 1.486464] |
| classic_t_intersection_medium | 0.650000 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| classic_urban_crossing_medium | 0.650000 | rejected | 5 | 1.275558 [0.995988, 1.541863] |
| francis2023_frontal_approach | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_pedestrian_obstruction | 0.390000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_pedestrian_overtaking | 1.040000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_robot_overtaking | 0.390000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_down_path | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_intersection_no_gesture | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_blind_corner | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_narrow_hallway | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_narrow_doorway | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_entering_room | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_exiting_room | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_entering_elevator | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_exiting_elevator | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_intersection_wait | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_intersection_proceed | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_following_human | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_leading_human | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_accompanying_peer | 0.650000 | rejected | 1 | 1.486464 [1.486464, 1.486464] |
| francis2023_join_group | 0.216667 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| francis2023_leave_group | 0.216667 | rejected | 3 | 1.279981 [1.080446, 1.486464] |
| francis2023_crowd_navigation | 0.650000 | rejected | 9 | 1.258330 [0.995988, 1.541863] |
| francis2023_parallel_traffic | 0.650000 | rejected | 8 | 1.260927 [0.995988, 1.541863] |
| francis2023_perpendicular_traffic | 0.650000 | rejected | 7 | 1.274588 [0.995988, 1.541863] |
| francis2023_circular_crossing | 0.650000 | rejected | 6 | 1.267435 [0.995988, 1.541863] |
| francis2023_robot_crowding | 0.650000 | rejected | 24 | 1.336439 [0.995988, 1.723666] |


The [speed receipt](context/evidence/scenario_authoring_0_1_0_speeds.json) records the requested and effective values and a live-cap digest for each scenario. No tier is added to released or successor YAML. The no-actors scenario has no live speed caps even when the distribution setting is accepted.

```bash
# In the PR checkout:
uv run python scripts/validation/probe_scenario_authoring.py \
  --mode speed-caps --output speed-caps-head.json
# In a detached fresh-main checkout, with the same environment and this probe script:
PYTHONPATH="$PWD" <environment-python> <probe-script> \
  --mode speed-caps \
  --matrix configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml \
  --output speed-caps-base.json
```

## Behaviour gate and adversarial review

The [paired receipt](context/evidence/scenario_authoring_0_1_0_empty_world.json) covers 1,344 actor-free cells on base and the same 1,344 on head: 14 arms × 48 scenarios × dev seeds 1001 and 1002. Every robot trajectory digest, outcome, step count and declared guard fallback count is identical. Each cohort has 1,131 successes, 140 collisions and 73 timeouts. All 213 non-successes are individually classified as pre-existing actor-free collision or timeout with the paired base evidence; matching base is not a causal explanation or a claim that these planners solve every scenario.

The successor's two corrected scenarios contribute another 56 cells (14 arms × 2 scenarios × 2 seeds): 45 successes, 2 collisions and 9 timeouts. All outcomes and trajectories equal the corresponding released-input actor-free controls, and every non-success is listed. All three execution receipts are complete with no missing, duplicate, unavailable, failed or incomplete-trace slots. Pedestrian arrays are empty at reset and throughout every step trace.

No unexpected checkpoint or predictor substitution is observed. The guarded PPO arm does use its declared safety-controller fallback: 13,267 `fallback_safe` decisions in each full cohort, with identical per-cell counts on base and head. These are inventoried as part of that composite arm's normal behaviour, not described as absent or promoted as independent PPO evidence. This preservation gate passes with zero changed cells; it does not certify goal completion, ranking, safety or release eligibility.

| Arm | Cells | Success | Collision | Timeout |
| --- | ---: | ---: | ---: | ---: |
| goal | 96 | 63 | 25 | 8 |
| guarded_ppo | 96 | 87 | 0 | 9 |
| hybrid_rule_v4_fast_progress_static_escape | 96 | 87 | 0 | 9 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 96 | 84 | 0 | 12 |
| orca | 96 | 94 | 0 | 2 |
| ppo | 96 | 62 | 34 | 0 |
| prediction_planner | 96 | 79 | 17 | 0 |
| predictive_mppi | 96 | 92 | 0 | 4 |
| risk_dwa | 96 | 92 | 0 | 4 |
| sacadrl | 96 | 29 | 64 | 3 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 96 | 91 | 0 | 5 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 96 | 91 | 0 | 5 |
| social_force | 96 | 86 | 0 | 10 |
| socnav_sampling | 96 | 94 | 0 | 2 |

Reproduce the full actor-free sweep from an immutable checkout of the named source, separately for base and head:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PWD" <environment-python> scripts/validation/run_empty_world_sweep.py \
  --head-sha HEAD --suite main --seeds 1001 1002 --workers 4 \
  --output-dir <diagnostic-output>
```

For the successor diagnostic, copy the main campaign template to an external temporary file, change only `scenario_matrix` to the successor, bind `SUITES["main"]` to that temporary template and filter `--scenarios classic_realworld_double_bottleneck_high classic_station_platform_medium`. The runner still removes every actor and uses its ordinary campaign path. No released template is modified. Base used four workers and head eight; both produced identical paired robot trajectories. Raw local episode-file hashes and source-tree bindings are retained in the compact receipt; raw traces remain in the local diagnostic archive.

The adversarial review found and corrected evidence gaps: the original broad freeze claim ignored five pre-existing main differences; the first inventory omitted packaged maps; the speed comparison needed the actual base loader and a clear distinction between rejected requests and native controls; the probe needed executing-revision and source-file hashes. A release-checklist edit changed a pinned assurance example, so both files were restored exactly to main and retirement guidance stays here. No golden assurance digest was weakened. The bottleneck test was renamed to describe its actual checks (route density and marker clearance), avoiding a claim that it independently audits route geometry.

Two early sweep attempts observed moving Git revisions while documentation commits were being made. Their mixed-commit receipts were rejected and excluded, not repaired or resealed. Accepted head and successor sweeps use one immutable detached checkout; the receipt verifies that its runtime/config/map/dependency trees equal the final PR trees. Initial numerical-library thread oversubscription was corrected by setting OMP, MKL and OpenBLAS thread counts to one before repeating execution.
