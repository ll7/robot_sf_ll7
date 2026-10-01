<!-- AI-GENERATED NEEDS-REVIEW -->
# Pedestrian obstacle-force development fit for 0.0.9

**Rejected diagnostic prototype; adoption blocked by footprint penetration and narrow-door flow.** `calibrated_v2` is an opt-in candidate,
not an accepted calibration or release setting. The missing selector remains `legacy_v1` and
preserves the released defaults and configuration digest. No held-out seeds were stepped.

## Released law

`fast-pysf/pysocialforce/forces.py` selects a closest wall point and defines
`q = max(d - b, 1e-5)`, where `d` is centre-to-wall distance and
`b = threshold + agent_radius * sigma`. The nominal potential is inverse-square,
`U = factor/(2 q²)`, but the legacy gradient divides the displacement by **q**, not d.
Consequently the implemented force away from an endpoint/straight wall is
`F = factor * d / q⁴ * n`. It is not the true gradient of that nominal potential.
There is no activation cutoff; far-field magnitude falls approximately as `d⁻³`.
The released parameters are factor 10, sigma 0, threshold -0.57 m, force radius 0.35 m.
A negative offset both weakens contact response and retains a long tail.

For two jamb tips at `(8, ±0.6)`, upstream centreline distance s gives opposing x acceleration
`20 s / (sqrt(s² + 0.6²) + 0.57)⁴`. At s≈1.51 m this is 1.30 m/s², exactly the
released rest drive `0.65/0.5`. Social acceleration is zero for a lone walker. Wider and
finite-thickness benchmark walls add several segments, explaining geometry-dependent
stand-offs of roughly 1.5–2.3 m; summing the actual segments is necessary.

A fresh like-for-like legacy comparator confirms the geometry dependence. Thin walls stall all
10 seeds at 1.2 m (1.511 m upstream) and 2.0 m (0.982 m upstream), while the three wider doors
already pass. With 0.1 m polygon walls, **0/50** pass and stand-offs are 2.086–2.898 m across
1.2–3.6 m. The candidate passes **100/100** across both thicknesses. These fixtures differ from
the source report's geometry; their distances should not be presented as a replay of that report.
The source-bound rows are preserved in `legacy_comparator.json`.

The production selection is applied in `_build_pysf_simulation` in
`robot_sf/sim/simulator.py`, before pedestrian spawning/force construction. It affects the
pedestrian substrate only. Planner wall laws and pedestrian preferred speeds stay as configured.
The prototype profile retains the legacy force law and selects factor **0.003**,
threshold **+0.375 m**, sigma **0**. The positive offset restores strong contact repulsion
while the much smaller factor reduces long-range braking. This profile is compatible only
with `legacy_shifted_gradient_v1`; contradictory law/profile selection fails closed.

## Literature and target boundaries

- [Helbing & Molnár, 1995, equation 13](https://arxiv.org/html/cond-mat/9805244):
  exponential wall potential `U0 exp(-d/R)`, U0=10 m²/s², R=0.2 m; relaxation time 0.5 s.
  Their model demonstrates lanes and alternating door passage. This motivates separating
  contact strength from short interaction range rather than multiplying the entire legacy tail.
- [Helbing, Farkas & Vicsek, 2000](https://arxiv.org/html/cond-mat/0009448):
  wall term `A exp((r-d)/B)n` plus normal body-contact and tangential friction terms;
  A=2000 N, B=0.08 m, mass 80 kg (A/m=25 m/s²). Those contact terms matter: the paper's
  constants cannot simply be copied into this substrate's inverse-power acceleration factor.
- [Moussaïd, Helbing & Theraulaz, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3084058/):
  free-path direction choice and time-to-collision speed choice, with physical wall forces on
  contact. Their individual-trajectory comparison uses 1.3 m/s and tau=0.5 s and their model
  generates bidirectional lane organization. A free path through a doorway should not create a
  distant equilibrium; this is a model-design target, not an empirical lone-walker clearance fit.
- [Seyfried et al., 2009, Table 2](https://arxiv.org/html/physics/0702004):
  normal walking through 0.8–1.2 m bottlenecks, N=20/40/60, measured specific flows 1.61–2.31
  pedestrians/(m·s). For N=60: 1.61 at 0.8 m, 1.90 at 1.0 m, 1.97 at 1.2 m. Flow grows
  approximately linearly with width. The paper warns that finite-N transients and initial density
  influence the absolute flow; these laboratory measurements are not universal design capacities.
- [Zhang & Seyfried, 2014](https://arxiv.org/abs/1508.06789): bottleneck shape and narrowing
  length affect flow. A thin-jamb check alone does not establish finite-wall passage.
- [FHWA, 2006, object clearances](https://www.fhwa.dot.gov/publications/research/safety/pedbike/05085/chapt9.cfm):
  desired wall/fence clearance 0.5 m and obstacle allowance 0.3–0.5 m. The task's **0.2–0.5 m
  body-surface clearance** is an engineering screening band, not a universal measured distribution.
  The real benchmark collision footprint is **0.40 m**, distinct from the **0.35 m force radius**.
  Wall penetration must be judged against 0.40 m.

The 0.8 m lone-passage target is an inference from observed normal flow through real 0.8 m doors.
The 1.9/(m·s) target is compared at 1.3 m/s, never imposed on the released 0.65 m/s speed.
The archived 60-agent queue used 0.35 m force-model discs on a jittered grid;
those starts overlap under the actual 0.40 m collision footprint. The checked-in driver now
uses 0.85 m horizontal spacing and nine rows, avoiding that initial overlap. The archived flow
loss is a provisional diagnostic, **not a valid accepted physical calibration**. The queue it does not exactly
match the experiment's 3.3/m² initial density or body geometry. This is a plausibility screen,
not full empirical validation of pedestrian dynamics.

## Grid, fitting and reproducibility

Only seeds 1001–1010 were used for calibration. The sequential grids were:

1. factor `[0.03,0.1,0.3,1,3,10]` × offset `[-0.57,-0.3,-0.1,0,0.1]` (30 pairs).
2. factor `[0.001,0.003,0.01,0.03,0.1]` × offset `[0.2,0.3,0.35]` (15 pairs).
3. factor `[0.0001,0.0003,0.001,0.003,0.006]` × offset `[0.375,0.4,0.425,0.45]` (20 pairs).

The first two phases screened thin walls. The third screened every width at both zero and
0.1 m thickness after a production regression exposed a thick-wall force barrier. Every phase
screened corridor clearance and used a direct wall-approach control at both walking speeds.
Crowd-flow survivors were checked at widths 0.8, 1.0, 1.2, 2.0 and 3.6 m, both speeds,
60 pedestrians and 120 s. Counts are unique first passages; flow uses N/(last-first time),
matching the finite-N convention used in the cited table, not repeated crossings or full-horizon
occupancy averages. No wall collision clipping, force disabling or social-force tuning was used.

The initial grids used the historical default pair kernel. A matched refit of all 13 direct-wall
survivors uses the matrix's explicit `wrapped_v2` pair kernel, keeping all other parameters and
seeds fixed. Eight preserve centre clearance >=0.35 m in all matched flow runs.
That initial safety screen used the 0.35 m force radius, an incorrect proxy for the benchmark
footprint. Among candidates passing that **superseded** screen, `(0.003,0.375)` minimized
specific-flow MSE against the N=60 targets at 0.8/1.0/1.2 m (matched-kernel MSE=1.031667; initial historical-kernel MSE=1.011542). That loss is large:
**every safe third-phase candidate had zero crossings at 0.8 m.** Choosing the least-bad candidate
for a paired preview does not pass the calibration gate. The final footprint audit resolves all
48 scenarios at `ped_radius=0.40`: **none of the 13 matched survivors is footprint-safe**. The
preview candidate reaches centre distance 0.3593 m, or **0.0407 m of physical wall penetration**.
No accepted parameter choice exists. At 0.8 m a 0.40 m disc has zero geometric clearance; body
geometry and integration must be reconciled before treating normal-flow targets as a fit objective.

The 0.1/-0.57 style scale-only candidates weaken contact along with the tail. The first survivor
0.1/+0.1 penetrated walls under crowd forces, reaching centre distance 0.2464 m. Another thin-wall
survivor, 0.01/+0.35, produced 1.824 m/s² peak braking in the real finite-wall regression and was
rejected. The final production regression is force-only, so it can run in fast CI without episodes.
On the base it fails with **9.170178 >= 1.30**; the opt-in peak is **0.781480 m/s²**; legacy hash and released-parameter regressions pass.

The canonical emergent builders on the base commit still contain the #10057 wall-order defect.
The calibration harness explicitly converts xyxy geometric walls to the xxyy substrate input
and uses the canonical builder and lane-segregation metric. It does not silently assume #10057
was merged. The raw archive contains the exact producer scripts for all three grid phases;
their hashes match the original manifests. The readable checked-in driver adds the thick-wall
screen to all phases, binds safety to the real 0.40 m footprint, starts corridor walkers at
0.25 m body clearance, and corrects initial crowd spacing for the 0.40 m footprint. Rerunning it applies the corrected acceptance condition. Archived producer
scripts preserve the original 0.35 m interpretation; subtract 0.05 m from their body-clearance
fields to obtain the benchmark footprint clearance. The preserved raw trajectories/distances
remain usable, but their provisional safety passes are superseded by `footprint_audit.json`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python \
  scripts/validation/calibrate_obstacle_force_10061.py --stage screen --workers 1 \
  --pairs configs/benchmarks/obstacle_force_calibration_0_0_9/thick_grid.json --out /tmp/wallcal-screen
```

## Acceptance

| Gate | Candidate result | Decision |
|---|---|---|
| (a) lone passage, 1.2/2.0/2.2/2.8/3.6 m | 100/100 passages, thin + 0.1 m walls, 10 seeds; 10.9–11.5 s to x=door+1 | Pass for listed widths |
| (b) corridor clearance / penetration | actual 0.40 m footprint: steady corridor clearance 0.3181–0.3236 m; crowd minimum centre 0.3593 m, penetration 0.0407 m | **Fail** |
| (c) bottleneck flow | normal-speed specific flow: 0.000/1.312/1.573/1.670/1.543 at widths 0.8/1.0/1.2/2.0/3.6 m | **Fail** at narrow widths |
| (d) canonical lane formation | matched kernel: released-speed mean 0.199, 5/10 weak-or-clear, 1/10 clear; typical-speed mean 0.178, 6/10 weak-or-clear, 0/10 clear | Partial; no robust-lane claim |
| (e) reset overlap | 240/240 paired resets identical; zero overlaps under either profile; zero new overlaps | Pass |

For widths 1.2–3.6 m, normal-speed flow has linear-fit R²=0.9950 and slope 1.508/(m·s).
That restricted fit does not excuse zero 0.8 m flow. Released-speed specific flows are
0.000/0.550/0.633/0.651/0.601; all 60 walkers cross 1.2–3.6 m openings. The 1.0 m released-speed
queue averages 58.8 crossings by 120 s. Samples are development diagnostics, with no held-out,
release, paper or dissertation admission.

## Benchmark preview and validation

The production driver runs goal, ORCA and social_force × the unchanged 48-scenario 0.0.8 matrix
× seeds 1001–1005 × both profiles (1,440 episodes), at the authored 0.0.8 horizons and 0.1 s dt.
The copied horizon schedule is a new preview input; no frozen or historical artifacts were edited.
This is a preview on the recorded 0.0.9 implementation source, not a new 0.0.8 release campaign.
All rows must record the actual force factor/offset; fallback or degraded rows fail the driver.
The reset audit compares each profile's complete initial pedestrian state and robot poses.

The completed preview contains exactly **1,440 episodes**, with no fallback/degraded rows.
Every cell records the actual force parameters; every JSONL digest was verified before aggregation.

| Arm | Legacy success | Candidate success | Change | Legacy / candidate collision |
|---|---:|---:|---:|---:|
| goal | 94/240 (39.17%) | 101/240 (42.08%) | +2.92 points | 57.92% / 55.83% |
| ORCA | 206/240 (85.83%) | 198/240 (82.50%) | -3.33 points | 11.25% / 13.75% |
| social_force | 167/240 (69.58%) | 170/240 (70.83%) | +1.25 points | 0% / 0% |

These headline changes conceal large scenario regressions. `classic_doorway_high` ORCA success
falls 80% to 0%; `classic_doorway_medium` social_force falls 100% to 0%. In contrast,
`francis2023_narrow_hallway` improves from 0% to 80% for goal, 80% to 100% for ORCA,
and 0% to 100% for social_force. `francis2023_narrow_doorway` remains 0% for all three arms.
Five seeds per cell are a diagnostic sample, not a statistical release-admission result.
All 144 scenario/arm comparisons are in `preview_summary.json`; exact rows are in
`preview_raw.tar.gz`, along with the paired reset records and legacy lane comparator.

With the matched pair kernel, released-speed legacy lane index mean is 0.138 (5/10 weak-or-clear,
0/10 clear), versus candidate 0.199 (5/10 weak-or-clear, 1/10 clear). At typical speed, legacy
mean is 0.167 (5/10 weak-or-clear, 0/10 clear), versus candidate 0.178 (6/10 weak-or-clear,
0/10 clear). This preserves a weak lane signal; neither fixture establishes robust lane emergence.
Historical-kernel lane/crowd runs remain in the original raw archive and are not substituted for
the matched acceptance results. `matched_kernel_fit.json` and `matched_kernel_raw.tar.gz` carry
the final fit ranking, exact producer wrapper/helper bytes and complete matched runs.

An additional 25-pair diagnostic exponential family uses `A exp((r-d)/B)n`, amplitude
`[10,25,50,100,200]` m/s² and range `[0.01,0.02,0.04,0.08,0.16]` m, the same seeds and
0.1 s integration, including lone 0.8/1.0 m doors. No pair passed the combined screen.
Short ranges permitted narrow passage but the direct-wall control penetrated the 0.35 m radius
(e.g. A=10, B=0.01: minimum centre distance 0.296 m). Safer broader ranges blocked narrow
passage (e.g. A=50, B=0.04: minimum distance 0.372 m, 120/140 passages). These are bounded
negative results, not proof that every exponential/contact formulation is infeasible.
The exact diagnostic implementation and raw runs are in `exponential_raw.tar.gz`.

The evidence-registry checker initially identified 288 cell hashes without adjacent locations.
This was remediated by explicit archive-member URIs and locations in the cell manifest, binding
each digest to the tracked raw archive; the producer verifies the individual member bytes.
No new finding is grandfathered by the refreshed baseline.

Validation receipts and every full-suite failure disposition are in `validation_summary.json`
and `validation_raw.tar.gz`. The initial full lane is not represented as a clean pass.
Default/adoption remains legacy; the release must not adopt this candidate while (b) and (c) fail.
The next scientific step is a separately fitted short-range/contact formulation rather than a
scale-only change, with the same seed and wall-penetration gates retained.

## Validation disposition

The full default lane completed: 13,862 passed, 94 failed, 10 existing skips and 7 existing
xfails. Eleven failures are conservative seed-guard refusals before any step. The other 83
reproduce on base 93ba0d75f in the original cluster setup. Full history and batch command paths
resolve 79; three output-retirement/cleanup cases fail on the shared compute filesystem on
both heads and pass on this host. One mocked diagnostic-release test activates the inherited
SLURM private-source gate on the cluster and passes naturally outside SLURM on this host.
The gate was not bypassed. All 83 non-guard cases therefore have a passing focused control
in the appropriate environment; this does not rewrite the initial full lane as green.

The corrected direct-importer lane passes 1,421 tests across 92 files (4 existing xfails).
Final force/config/profile tests pass 11 tests in 2.36 s without environment steps. The force
barrier regression is real-byte production construction, and the static driver coupling checks
the matrix's wrapped pair kernel and actual footprint radius. No new skip, xfail or timeout
was introduced. The initial collection excluded 31 files; runtime refusals led to 40 conservative
exclusions for future reruns. No `--ignore` was used and no held-out seed was stepped.

Changed-line coverage meets the 80% gate: profile 96.6%, settings 90.0%, simulator 83.3%; loader
has no executable changed lines. Routing, Ruff, documentation, registry and PR-contract checks
are separate from scientific acceptance and domain approval. Slurm accounting confirms a maximum
of 16 allocated CPUs across this lane; every submission used the pending-queue/idle-capacity
admission script. All new trajectories and episodes ran on Slurm; local checks were static,
force-only, or metadata/render contract tests. Installed fast-pysf force/config bytes match the
checkout on both hosts. Preserve both owned worktrees and external logs; versioned archives
provide durable raw custody.
