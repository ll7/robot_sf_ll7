# Matched pedestrian-perception degradation (diagnostic)

**Development-seed diagnostic, new current-version execution.** This is not release evidence and does not admit any claim. Four planner configurations were run on six release cells and 50 development seeds (1121–1130 and 1161–1200). Each of four perception conditions changes one effect at a time, giving 4,800 main episodes. All main episodes ran at execution commit `4e16d46908789c37651a4b851a23f10bf1540e7b`: release `373dbfde4f39667cf9e8732dabe7df5118bdeab1`, plus the guard-escape ablation switch, plus the perception hook. The design, outcomes, analysis and decision rule are frozen in [`contract.yaml`](contract.yaml), committed before any main episode ran.

## What was degraded

One perceived pedestrian snapshot per step (`robot_sf/sensor/pedestrian_perception.py`) feeds every pedestrian channel a planner can read:
- the SocNav pedestrian rows (positions, ego-frame velocities, count);
- the occupancy-grid pedestrian layer read by the PPO policy.

Observation shapes are unchanged, and metrics use simulator truth. The conditions are:
- **Oracle:** control; the same code path, and a test proves it is bitwise identical to the release observation.
- **Noise:** 0.05 m Gaussian position noise per axis, indexed by episode, step and pedestrian slot.
- **Delay:** two-step (0.2 s) delay of positions and velocities.
- **Occlusion:** static-wall occlusion (line of sight blocked by an obstacle interior; 360° FOV, unlimited range).

These are declared synthetic stressors, not a sensor model. Static map knowledge and robot state remain exact.

`tests/sensor/test_pedestrian_perception.py` fails when the occupancy grid is rasterized from simulator truth (the release behaviour) and passes with the snapshot.

## Validity (all 4,800 rows; see [`evidence/validity.json`](evidence/validity.json))

- **Execution:** ORCA and both hybrids ran in `adapter` mode and PPO in `native` mode (every row has a `mode` field). There are 0 fallback or degraded markers, 0 non-ok statuses and 0 checkpoint-load failures. PPO checkpoint `ppo_release_robot_b1002_last_20261001` has sha256 `764a7d88…24bc`.
- **Coverage:** both channels consumed the snapshot on every step of every episode.
- **Noise and delay:** every one of the 2,400 episodes was perturbed. Mean 2-D error was 0.063 m for noise and 0.18–0.20 m for delay.
- **Occlusion:** active in four cells (40–83 % of steps). It never occurs in `classic_head_on_corridor_medium` or `classic_bottleneck_high`, which have no wall between robot and pedestrians. In those two cells, as predeclared, the occlusion rows are a negative control: all 400 seed × planner pairs reproduce the oracle outcome exactly.
- **Distance of hidden pedestrians:** in the pilot, occluded pedestrians were never closer than 5.9 m to the robot, and in the blind corner never closer than 14.5 m ([`evidence/pilot/occluded_pedestrian_distances.json`](evidence/pilot/occluded_pedestrian_distances.json)). In these authored cells, wall occlusion therefore hides only distant pedestrians.

## Results (equal-cell pooled, paired by seed and cell; seed-block bootstrap 95 % intervals)

Oracle success: ORCA 93.0 % [90.0, 95.7]; hybrid v4 guard on 98.3 % [97.0, 99.7]; guard off 56.0 % [52.7, 59.7]; PPO 55.3 % [51.7, 59.0].

Condition effect on success, in percentage points (occlusion pooled over its four active cells):

| Planner | Noise 0.05 m | Delay 0.2 s | Static occlusion |
| --- | ---: | ---: | ---: |
| ORCA | +2.7 [−0.3, +6.0] | +5.0 [+2.0, +8.0] | −1.5 [−3.5, 0.0] |
| Hybrid v4, guard on | −0.7 [−2.7, +1.0] | −0.3 [−2.0, +1.3] | 0.0 (all 200 trajectories identical) |
| Hybrid v4, guard off | −1.3 [−4.7, +2.0] | −1.3 [−4.3, +1.7] | 0.0 (all 200 trajectories identical) |
| PPO | −1.3 [−3.7, +1.0] | −4.3 [−7.0, −1.7] | −28.0 [−32.0, −23.5] |

- **Guard on, collisions:** 0 % in every condition. Its timeout effect is at most +0.7 pp (interval within ±3 pp).
- **Guard benefit:** guard on minus guard off is +42.3 pp success under oracle. Its change under degradation (`guard_dependence`) is +0.7 [−3.0, +4.3] for noise, +1.0 [−2.3, +4.3] for delay and 0 for occlusion.
- **PPO under occlusion:** the drop is one cell. Blind-corner success fell from 50/50 to 0/50, every episode ending in a wall collision, after the robot lost sight of a pedestrian 14.5 m or more away. The other cells change by −8 to +2 pp. This is the learned policy failing when no pedestrian is visible, not a response to a nearby hidden pedestrian.
- **ORCA vs hybrid under delay:** ORCA's oracle success deficit to hybrid guard on (−5.3 pp [−8.3, −2.3]) closes to 0.0 [−2.3, +2.3] under delay, because ORCA's pedestrian collisions fall by 4.7 pp.

Per-cell values for every planner, condition and endpoint are in [`evidence/rates.csv`](evidence/rates.csv) and [`evidence/paired_effects.csv`](evidence/paired_effects.csv). The predeclared rule evaluation is in [`evidence/summary.json`](evidence/summary.json). Intervals are pointwise, not simultaneous. They condition on these six authored cells and these stressor magnitudes.

## Reproduce

Analysis only (no simulation; NumPy and PyYAML), from the repository root:

```bash
uv run python docs/context/perception_degradation/analyze.py \
  --episodes docs/context/perception_degradation/evidence/episodes.jsonl.gz --output /tmp/xperc-recomputed
cmp /tmp/xperc-recomputed/effects.csv docs/context/perception_degradation/evidence/paired_effects.csv
sha256sum -c docs/context/perception_degradation/SHA256SUMS
```

Full acquisition, at the execution commit, with `uv sync --frozen --extra training` (Python 3.13.14, `uv.lock` sha256 `def82098…d023`) and the PPO release asset placed under `output/model_cache/` (see `model/registry.yaml`):

```bash
docs/context/perception_degradation/run_campaign.sh main <fresh-output-root>
uv run python docs/context/perception_degradation/project.py --raw-root <fresh-output-root> --output <dir>/episodes.jsonl.gz
```

Configs are generated by [`make_configs.py`](make_configs.py) into [`inputs/`](inputs/). Pilot and qualification evidence (seeds 1001–1010, never pooled) is in [`evidence/pilot/`](evidence/pilot/). It includes true-vs-perceived traces and [`qualify_pilot.py`](qualify_pilot.py) output with 0 trace violations in 231,917 steps.

## Custody

[`evidence/episodes.jsonl.gz`](evidence/episodes.jsonl.gz) holds one row per episode: identity, mode, fallback fields, perception accounting, endpoints, and the source path, line and SHA-256 of the exact producer line. [`evidence/episodes_sources.json`](evidence/episodes_sources.json) binds the 16 producer files by digest. The 4.2 GB raw bundle (step traces, commands, typed contacts and resolved configs) is held outside Git in a private archive with a sha256 manifest. Its W&B upload ([`wandb_upload.py`](wandb_upload.py)) is pending because no W&B credentials were available to this lane.

This packet changes no frozen artifact, release, claim or dissertation text.
