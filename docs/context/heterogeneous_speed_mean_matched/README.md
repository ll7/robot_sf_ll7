# Mean-matched heterogeneous pedestrian speeds (development-seed diagnostic)

**Diagnostic only: development seeds, new current-version execution on the 0.0.8 release
source (`373dbfde`), not release evidence.** This packet admits nothing into the thesis.
The recorded classifier labels all 1,600 main episodes (4 planners x 4 cells x 2 conditions x
50 seeds) native; its limits and counts for every configuration are described below. Every
recorded realized-input check passes.

Question: do the robot outcomes depend on every pedestrian sharing one preferred speed?
The **uniform** condition puts all 12 pedestrians at 0.65 m/s. The **mixed** condition uses
3 x 0.325, 6 x 0.65 and 3 x 0.975 m/s. The arithmetic mean gives each of the 12
pedestrians equal weight and targets 0.65 m/s in both conditions. The checked reset fields
and numerical tolerances are listed below. The release's speed-coupled reset rule moves
some start positions (see Limits).

## Result under the predeclared rule

Outcome: **B, selective deterioration (ORCA)**, with qualifier **C, slow-pedestrian delay**.
The sensitivity run that excludes relocation-affected pairs reaches the same verdict.

For each planner and cell, the point effect averages mixed-minus-uniform episode outcomes
over the 50 paired seeds. The pooled value gives each of the four cell means equal weight;
it does not weight cells by episode duration. Uniform and mixed levels are calculated the
same way within each condition. Intervals are percentile 95 % from 10,000 seed-block
bootstrap draws: a draw resamples the 50 seeds while retaining their planner, cell, and
condition observations. For the relocation sensitivity, excluded pairs are removed first;
the retained pairs are equally weighted within a cell, then available cell means are
equally weighted. Time-to-goal uses only pairs successful in both conditions.

| Planner | Success uniform -> mixed, pp | Effect [95 % CI] | Collision uniform -> mixed, pp | Effect [95 % CI] |
| --- | --- | ---: | --- | ---: |
| hybrid v4 continuous | 96.0 -> 96.5 | +0.5 [-2.5, +3.5] | 0.0 -> 0.0 | 0.0 [0.0, 0.0] |
| ORCA | 82.5 -> 70.5 | **-12.0 [-19.0, -5.0]** | 17.0 -> 28.5 | **+11.5 [+4.5, +18.5]** |
| social force | 37.5 -> 31.0 | -6.5 [-14.0, +1.5] | 0.5 -> 1.0 | +0.5 [-1.0, +2.0] |
| release PPO | 34.0 -> 33.0 | -1.0 [-7.0, +5.0] | 65.0 -> 66.5 | +1.5 [-4.5, +7.5] |

- ORCA's extra collisions are pedestrian contacts: +12.0 pp [+5.5, +19.0]. Its largest losses
  are in group crossing (success -20 pp [-38, -2]) and head-on corridor
  (-16 pp [-32, -2]). The hybrid-ORCA success gap widens from 13.5 to 26.0 pp
  (interaction +12.5 [+4.5, +20.5]).
- Social force's pooled effect is small, but its cell effects point in opposite directions:
  crossing +12 pp [0, +26], group crossing -26 pp [-42, -10], overtaking -14 pp [-26, -4].
  These are timeout shifts, not collisions.
- PPO's total collision effect is +1.5 pp [-4.5, +7.5], while contact types shift: pedestrian contacts
  +7.5 pp [+1.0, +14.5], obstacle contacts -6.0 pp [-12.5, +0.5].
- The success order is hybrid > ORCA > {social force, PPO} in both conditions. Social force
  and PPO swap places, but their uniform difference (+3.5 pp [-2.5, +9.5]) was never
  resolved, so this is not an order change.
- Pedestrian lost-time fraction rises under mixing for every planner (+0.04 to +0.06 overall).
  In each episode, lost time is summed over steps, divided by episode duration, then
  averaged equally across all 12 pedestrians or the specified speed stratum. The resulting
  episode values are averaged over retained paired seeds within each cell, then the four
  cell means receive equal weight. The near-robot component includes only steps with
  centre distance less than 3 m, but still divides by full episode duration. For the slow
  stratum it rises for hybrid
  (+0.041 [+0.023, +0.058]), ORCA (+0.018 [+0.008, +0.029]) and social force
  (+0.021 [+0.008, +0.035]), but not for PPO (+0.001 [-0.003, +0.005]).

All per-cell effects and endpoints are in [`evidence/primary/paired_effects.csv`](evidence/primary/paired_effects.csv).
Planner-pair differences are in [`planner_pair_interactions.csv`](evidence/primary/planner_pair_interactions.csv)
and the decision in [`summary.json`](evidence/primary/summary.json). The same outputs with the
relocation-affected pairs removed are in [`evidence/sensitivity_excl_relocation/`](evidence/sensitivity_excl_relocation/).

## Limits

- **Speed-coupled reset relocation.** At reset, the release moves pedestrians away from the
  robot start by a buffer that grows with the population's maximum preferred speed. Route
  respawns use the same exclusion. Mixed therefore keeps pedestrians 0.325 m further away.
  This changes start positions in 26 of 50 crossing-cell seeds and 1 of 50 overtaking seeds
  (the same seeds for every planner; 108 of 800 pairs). It is
  part of the treatment as the release implements it. Excluding these pairs retains the same outcome-rule verdict (ORCA success -13.7
  [-20.9, -6.8], collision +13.2 [+6.2, +20.4]).
- **No pedestrian-only control.** Much of the pedestrian delay increase occurs far from the
  robot, which points to pedestrian-pedestrian interaction (fast walkers behind slow ones).
  The near-robot share is a proxy, not a robot-attributable delay. It also mixes exposure
  time with slowdown.
- **Strata are not validated human categories.** The design does not identify which
  stratum drives the ORCA effect and does not address AMMV interaction calibration.
- **Scope.** Four authored cells and 50 shared development seeds; intervals condition on this
  roster. The mixed condition also scales initial walking velocity by the same factor.

## Realized-input verification

[`evidence/pairing_checks.json`](evidence/pairing_checks.json) records the checks below,
taken from per-episode reset receipts and per-step pedestrian state:

- 12 pedestrians in every episode.
- Sorted preferred speeds match the targets with `numpy.allclose` (absolute tolerance
  1e-9 and default relative tolerance 1e-5): mixed 3/6/3, uniform 12 x 0.65. The
  equal-pedestrian mean differs from 0.65 by at most 1e-9.
- `desired_speed_mean` is None and no explicit desired speeds are set.
- Within each paired planner/cell/seed, `groups`, `behaviors`, `agent_radius`, and `n_peds`
  match exactly between conditions. The numeric `goal`, `tau`, `robot_pose`, and `robot_goal`
  fields match under `numpy.allclose` (absolute tolerance 1e-6, default relative tolerance
  1e-5). `start_pos` differs only in rows marked relocated, and unrelocated `start_vel`
  matches the uniform velocity times the mixed speed factor under the same tolerance.
- Per-step preferred speeds do not change after reset beyond 1e-6, excluding start-delay
  zeros.
- For each cell, condition and seed, the checked cross-planner fields are `start_pos`,
  `goal`, `max_speeds`, `robot_pose`, and `robot_goal`; each matches the ORCA reference
  under the same 1e-6 absolute and 1e-5 relative tolerances. Other reset fields are not
  covered by this cross-planner assertion.

The scenario loader rejects an explicit `desired_speed_mean` as an unknown key
(`qualification/override_probe.txt`). In code, setting one would resample every speed and
erase the mixture. The pilot (seeds 1001-1010) is in [`qualification/`](qualification/) and is
never pooled.

## Execution-mode accounting

[`evidence/mode_counts.csv`](evidence/mode_counts.csv) gives native, fallback and degraded
counts for all 32 planner/cell/condition configurations: each has 50 rows classified native.
The classifier checks the recorded checkpoint, planner, effective-view, pedestrian-model,
spawn and failure indicators; only rows classified native enter effects. All 1,600 records
contain effective-view degradation, pedestrian-model status and spawn-invalid indicators;
the 400 hybrid records also contain a planner fallback counter. ORCA, social force and PPO
records do not contain that counter, which the classifier treats as zero, so the absence of
a planner fallback cannot be independently established from that counter for those planners.
All 400 PPO records have checkpoint provenance recording a successful load without checkpoint
fallback. Checkpoint provenance is absent for the other, non-checkpoint planners. Thus
"native" above means classified native under these recorded fields and defaults, not proof
that every possible fallback mechanism was instrumented.

## Raw archive custody

- W&B artifact: `wandb://ll7/robot_sf/xhet-hetspeed-raw-20261028:v0`.
- W&B artifact digest: `cacc7e8dc9d2c4200f5ab71d51a344f9`.
- Host copy: lab host archive.
- Readback: the artifact was `COMMITTED`; all 4,124 paths, sizes, and file MD5 digests
  matched the host archive. The artifact digest matched the local and remote manifests.

## Acquisition provenance

The runtime base `373dbfde4f39667cf9e8732dabe7df5118bdeab1` and the acquisition and
preregistration commit `94081a52ef8a7fddd9cf163a75a0d73a5bb7cf65` are preserved on the
remote original branch. All 1,600 extracted episode rows record the latter Git hash. That
commit differs from the runtime base only inside this packet; no runtime patch was needed.
The completed evidence commit is `95d39f91b6afa36f4d3deff801e5047d210689b6`. These are
recorded in [`provenance/environment.json`](provenance/environment.json).

## Reproduction

Analysis only (seconds; needs NumPy and PyYAML; output must go outside the packet):

```bash
P=docs/context/heterogeneous_speed_mean_matched
gunzip -c $P/evidence/episodes.csv.gz > /tmp/xhet_episodes.csv
python $P/analyze.py analyze --episodes /tmp/xhet_episodes.csv --contract $P/contract.yaml --out /tmp/xhet_primary
cmp /tmp/xhet_primary/summary.json $P/evidence/primary/summary.json
(cd $P && sha256sum -c SHA256SUMS)
```

Full campaign rerun (about 1 h on 8 cores; the script starts 20 processes):
`bash docs/context/heterogeneous_speed_mean_matched/rerun_campaign.sh`. It runs at acquisition commit
`94081a52` (whose runtime is the unchanged `373dbfde` base) with this packet, `uv sync --frozen --extra benchmark --extra training`
(uv.lock sha256 `def82098...d023`, Python 3.13.17) and the PPO checkpoint `764a7d88...24bc`.

### Reconstruction comparison contract

For a given development seed and the same frozen inputs, compare the full set of 32
planner/cell/condition combinations. Each combination must occur once in both runs. In
the extracted episode CSV, every field must match exactly except `git_hash` and
`wall_time_s`: this includes the configuration and seed key, native classification and
reason, command adapter, status and termination, step and contact outcomes, config hash,
checkpoint load, all recorded robot and pedestrian metrics, observer integrity and
relocation flag. Compare every field of each reset receipt exactly, including the realized
positions, velocities, goals, speeds, groups, behaviors, robot pose/goal, relocation and
captured-step count. The comparator rejects missing or duplicate combinations and columns.
These are exact serialized/numeric comparisons; a failed comparison is not silently rounded.

The recording checkout commit can differ when the same runtime and preregistered packet
are reconstructed elsewhere. Paths, timestamps, process IDs, elapsed/wall-clock time,
archive metadata and compressed-container bytes can also differ; these are not evidence
of a changed trajectory and are not compared. Environment differences still need to be
reported. This comparison does not rerun or validate the entire 50-seed campaign.

To repeat the single-seed comparison, check out acquisition commit `94081a52`, install
the pinned release environment and checkpoint above, run `driver.py --planner <planner>
--seeds 1121 --out <runs>/<planner>/sample` for all four planners, then use that commit's
`analyze.py extract --runs <runs> --out <extracted>`. From the packet checkout run:

```bash
python docs/context/heterogeneous_speed_mean_matched/provenance/compare_reconstruction.py \
  --archive ~/archive/experiments/xhet/main --rerun <runs> \
  --episodes docs/context/heterogeneous_speed_mean_matched/evidence/episodes.csv.gz \
  --rerun-episodes <extracted>/episodes.csv --seed 1121 --out <comparison.json>
```

The committed analysis CSVs use LF line endings; their archived counterparts use CRLF.
After normalizing only line endings the historical values match, and the current analyzer
reproduces the committed CSVs and JSON summaries byte-for-byte.

## Files

- `contract.yaml`: preregistered execution contract and outcome rule. It was committed in
  `94081a52` before any main-seed episode ran.
- `driver.py`, `run_chunk.sh`, `rerun_campaign.sh`: acquisition through the normal campaign
  runner `run_map_batch` with the frozen release `algo_config_path`, plus a read-only observer.
- `analyze.py`: extraction, checks, paired effects, decision.
- `evidence/episodes.csv.gz`: one row per episode. Columns include mode, typed contacts,
  outcomes, pedestrian delay and the relocation flag. Trajectories are not stored here.
- `provenance/`: input digests, environment, and sha256 manifests of the raw archive. The raw
  archive holds full episode JSONL, reset receipts, and per-step pedestrian
  position/velocity/goal/preferred speed with robot position and command.
