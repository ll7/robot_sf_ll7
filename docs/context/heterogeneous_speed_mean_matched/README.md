# Mean-matched heterogeneous pedestrian speeds (development-seed diagnostic)

**Diagnostic only: development seeds, new current-version execution on the 0.0.8 release
source (`373dbfde`), not release evidence.** This packet admits nothing into the thesis.
Of the 1,600 main episodes (4 planners x 4 cells x 2 conditions x 50 seeds), all 1,600 are
native: no fallback and no degraded rows. Every realized-input check passes.

Question: do the robot outcomes depend on every pedestrian sharing one preferred speed?
The **uniform** condition puts all 12 pedestrians at 0.65 m/s. The **mixed** condition uses
3 x 0.325, 6 x 0.65 and 3 x 0.975 m/s, so the realized mean is exactly 0.65 m/s. Identities,
goals, groups, radii (0.35 m), robot start and robot goal are identical between conditions.
Start positions are also identical, except where the release's speed-coupled reset rule moved
a pedestrian (see Limits).

## Result under the predeclared rule

Outcome: **B, selective deterioration (ORCA)**, with qualifier **C, slow-pedestrian delay**.
The sensitivity run that excludes relocation-affected pairs reaches the same verdict.

Mixed minus uniform, equal-weight mean over the four cells. Intervals are percentile 95 %
from 10,000 seed-block bootstrap draws over the 50 seeds. Each cell-level rate has
n = 50 pairs:

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
- PPO's total collisions are unchanged, but their type shifts: pedestrian contacts
  +7.5 pp [+1.0, +14.5], obstacle contacts -6.0 pp [-12.5, +0.5].
- The success order is hybrid > ORCA > {social force, PPO} in both conditions. Social force
  and PPO swap places, but their uniform difference (+3.5 pp [-2.5, +9.5]) was never
  resolved, so this is not an order change.
- Pedestrian lost-time fraction rises under mixing for every planner (+0.04 to +0.06 overall).
  The near-robot share (within 3 m) for the slow stratum rises for hybrid
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
  part of the treatment as the release implements it. Excluding these pairs leaves every
  conclusion unchanged (ORCA success -13.7 [-20.9, -6.8], collision +13.2 [+6.2, +20.4]).
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
- Sorted preferred speeds equal the target arrays exactly: mixed 3/6/3, uniform 12 x 0.65.
- `desired_speed_mean` is None and no explicit desired speeds are set.
- Goals, tau, groups, behaviors, radius, robot pose and robot goal are equal between conditions.
- Start positions are equal except in relocated rows.
- Unmoved initial velocities equal the uniform velocity times the speed factor.
- Preferred speeds never change during an episode.
- The same reset is identical across all four planners.

The scenario loader rejects an explicit `desired_speed_mean` as an unknown key
(`qualification/override_probe.txt`). In code, setting one would resample every speed and
erase the mixture. The pilot (seeds 1001-1010) is in [`qualification/`](qualification/) and is
never pooled.

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
`bash docs/context/heterogeneous_speed_mean_matched/rerun_campaign.sh`. It runs on commit
`373dbfde` with this packet, `uv sync --frozen --extra benchmark --extra training`
(uv.lock sha256 `def82098...d023`, Python 3.13.17) and the PPO checkpoint `764a7d88...24bc`.

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
