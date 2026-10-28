# Matched pedestrian-perception degradation (diagnostic)

**Development-seed diagnostic, new current-version execution.** This is not release evidence and does not admit a thesis claim. Four planner configurations were run on six release cells and 50 development seeds (1121–1130 and 1161–1200). Each of four perception conditions changes one effect at a time, giving 4,800 main episodes. All main episodes record execution commit `4e16d46908789c37651a4b851a23f10bf1540e7b`: release `373dbfde4f39667cf9e8732dabe7df5118bdeab1`, plus the guard-escape ablation switch and perception hook. The design, outcomes, analysis and decision rule are in the unchanged [`contract.yaml`](contract.yaml). Its commit timestamp predates the main environment record and first raw episode; an independently observed pre-episode push has not been established, as detailed in [provenance](provenance/README.md). Compared with the release, tracked configs and maps are byte-identical; five runtime files differ for the diagnostic changes. The original branch retains the execution commit and the reconstruction patch is in [`provenance/`](provenance/).

## What was degraded

One perceived pedestrian snapshot per step (`robot_sf/sensor/pedestrian_perception.py`) feeds every pedestrian channel a planner can read:
- the SocNav pedestrian rows (positions, ego-frame velocities, count);
- the occupancy-grid pedestrian layer read by the PPO policy.

Observation shapes are unchanged, and metrics use simulator truth. The conditions are:
- **Oracle:** control; the same code path. A test at the execution commit checks that every observation key and its array values match the unperturbed environment exactly for its tested reset and steps.
- **Noise:** 0.05 m Gaussian position noise per axis, indexed by episode, step and pedestrian slot.
- **Delay:** two-step (0.2 s) delay of positions and velocities.
- **Occlusion:** static-wall occlusion (line of sight blocked by an obstacle interior; 360° FOV, unlimited range).

These are declared synthetic stressors, not a sensor model. Static map knowledge and robot state remain exact.

`tests/sensor/test_pedestrian_perception.py` fails in the tested case when the occupancy grid is rasterized from simulator truth (the release behaviour) and passes with the snapshot; this does not prove every possible observation path.

## Validity (all 4,800 rows; see [`evidence/validity.json`](evidence/validity.json))

- **Execution:** ORCA and both hybrids report `adapter` mode, and PPO reports `native` mode. All 16 planner-condition groups have zero recorded fallback/degraded markers and non-ok statuses; effective-view degraded is false in all 4,800 rows. The 2,400 hybrid rows record fallback count zero; the 2,400 ORCA and PPO rows omit that counter, which is not a recorded zero and cannot prove zero fallback. All 1,200 PPO rows report the checkpoint loaded and checkpoint fallback false; none contains an episode-level checkpoint hash. The condition-level archived preflights record the computed hash of the local release checkpoint `ppo_release_robot_b1002_last_20261001` as `764a7d88…24bc`. The pedestrian model reports `native` in every row.
- **Coverage:** every row reports both channels consumed the snapshot on every recorded step.
- **Noise and delay:** every one of the 2,400 episodes was perturbed. Within each planner-condition group, the reported 2-D position error is the unweighted mean of 300 per-episode means (each episode first averages over its perceived pedestrian rows and steps); cells and seeds therefore have equal weight. The four planner means round to 0.063 m for noise and span 0.18–0.20 m for delay.
- **Occlusion:** exercised in aggregate in four cells (40–83 % of steps, pooling snapshots across the four planners and 50 seeds within each cell). Nevertheless, two guard-off episodes in `classic_cross_trap_medium` and one PPO episode in `classic_overtaking_medium` had pedestrians but never differed from truth, with no occlusion step. Literal O5 therefore marks both active cell/conditions as intervention failures, not robustness evidence. In `classic_head_on_corridor_medium` and `classic_bottleneck_high`, the declared wall-free negative controls, all 400 treatment rows with pedestrians never differ from truth and no step is occluded: they also fail literal O5, as expected for these controls. The 400 paired seed × planner rows in those controls match on success, any collision, pedestrian collision, wall collision and timeout indicators.
- **Distance of hidden pedestrians:** in the pilot, occluded pedestrians were never closer than 5.9 m to the robot, and in the blind corner never closer than 14.5 m ([`evidence/pilot/occluded_pedestrian_distances.json`](evidence/pilot/occluded_pedestrian_distances.json)). These distances describe the pilot observations, not a minimum verified for every main episode.

## Results (equal-cell and equal-seed pooled; seed-block bootstrap 95 % intervals)

Each rate first averages the 50 episode indicators within a cell, then averages the six cell rates equally. Condition effects subtract the paired oracle rate. Occlusion effects use the same calculation over the four active cells, with each cell weighted equally. Those four-cell occlusion results include the two O5-failed cell/conditions above and are descriptive, not evidence of robustness. The bootstrap resamples the 50 complete seed blocks, keeping cells, planners and conditions paired.

Six-cell oracle success: ORCA 93.0 % [90.0, 95.7]; hybrid v4 guard on 98.3 % [97.0, 99.7]; guard off 56.0 % [52.7, 59.7]; PPO 55.3 % [51.7, 59.0].

Condition effect on success, in percentage points (occlusion pooled over its four active cells):

| Planner | Noise 0.05 m | Delay 0.2 s | Static occlusion |
| --- | ---: | ---: | ---: |
| ORCA | +2.7 [−0.3, +6.0] | +5.0 [+2.0, +8.0] | −1.5 [−3.5, 0.0] |
| Hybrid v4, guard on | −0.7 [−2.7, +1.0] | −0.3 [−2.0, +1.3] | 0.0 (success matches in all 200 pairs) |
| Hybrid v4, guard off | −1.3 [−4.7, +2.0] | −1.3 [−4.3, +1.7] | 0.0 (success matches in all 200 pairs) |
| PPO | −1.3 [−3.7, +1.0] | −4.3 [−7.0, −1.7] | −28.0 [−32.0, −23.5] |

- **Guard on, collisions:** 0 of 1,200 observed episodes across the four conditions collided; this is not a guarantee of zero risk. Its timeout effect is at most +0.7 pp (interval within ±3 pp).
- **Guard benefit:** guard on minus guard off is +42.3 pp success under the six-cell oracle. Its six-cell change under degradation (`guard_dependence`) is +0.7 [−3.0, +4.3] for noise and +1.0 [−2.3, +4.3] for delay. In the four active occlusion cells, the matched oracle and occlusion benefits are both +57.5 pp [53.0, 62.0], giving a change of 0; this four-cell result includes O5-failed cell/conditions.
- **PPO under occlusion:** blind-corner success fell from 50/50 to 0/50, with a wall collision in every occlusion episode. The other active cells change by −8 to +2 pp. The pilot observed hidden pedestrians at least 14.5 m away in the blind corner; those pilot distances do not establish the distance for every main episode or isolate a causal mechanism for the policy failure.
- **ORCA vs hybrid under delay:** ORCA's oracle success deficit to hybrid guard on (−5.3 pp [−8.3, −2.3]) closes to 0.0 [−2.3, +2.3] under delay, because ORCA's pedestrian collisions fall by 4.7 pp.

Per-cell values for every planner, condition and endpoint are in [`evidence/rates.csv`](evidence/rates.csv) and [`evidence/paired_effects.csv`](evidence/paired_effects.csv). The predeclared rule evaluation is in [`evidence/summary.json`](evidence/summary.json). Intervals are pointwise, not simultaneous. They condition on these six authored cells and these stressor magnitudes.

The inclusive 5 pp O1 threshold classifies ORCA's delay effects on success (+5.0 pp [2.0, 8.0]) and collision (−5.0 pp [−8.0, −2.0]) as relevant and narrow, as well as the two descriptive PPO occlusion effects. The earlier analysis omitted the ORCA classifications because its floating-point estimate fell just below the threshold. The timeout endpoint now follows the contract's `max_steps` termination definition; all 24 recorded timeouts agree with the earlier event flag, so no observed timeout classification changes.

For O2, the general six-cell pooling rule gives guard-off minus PPO success of +0.7 pp [−4.0, +5.3] under oracle and +19.3 pp [+14.3, +24.3] under occlusion; their matched change is +18.7 pp [+15.7, +21.3]. The oracle interval includes zero, so this pair does not trigger O2 in that population. **Secondary matched four-active-cell analysis:** the same contrast is −18.0 pp [−24.5, −11.5] under oracle and +10.0 pp [+3.0, +17.0] under occlusion; the matched change is +28.0 pp [+23.5, +32.0]. Success and collision both reverse under the four-cell O2 calculation. This secondary result includes the two active O5-failed cell/conditions and is not robustness evidence.

## Known limitations of the preregistration

The contract is retained unchanged. Its general pooling clause calls for six cells, while its occlusion-specific clause calls for four active cells. The primary O2 evaluation applies the six-cell clause consistently to both oracle and occlusion; the four-cell matched analysis above is a labelled secondary reading. Prior code combined the six-cell oracle difference with the four-cell occlusion difference, which describes neither population's matched O2 comparison.

The contract also declares two wall-free occlusion negative controls but O5 applies its no-difference and no-occlusion failure rules to every treatment cell. Literal O5 therefore labels those intended controls as failures; they are identified as controls rather than silently exempted. An **alternative, post-hoc cell-exercise-only reading** observes that all four active cells had some occluded steps in aggregate. This weaker reading ignores the preregistered per-episode condition and does not erase the two active-cell failures.

Finally, O4's written same-sign criterion can also include improvements, although its name and stated consequence describe deterioration. The primary output applies the criterion as written. A **secondary directional reading** requires decreased success or increased collision; both readings are false for all current condition/endpoints. Numerical comparisons allow only a small floating-point tolerance at the declared inclusive thresholds.

## Thesis wording (conditional on separate admission)

The available records can support a qualified description of these development-seed diagnostics, their specified synthetic perturbations and authored cells: for example, the sampled guard-on arm had no observed collisions, and four-cell PPO success under occlusion changed by −28.0 pp [−32.0, −23.5]. The latter includes O5-failed cell/conditions and must be presented as descriptive. The four-cell ranking reversal is a secondary matched-subset analysis with the same limitation. Every planner-condition group has no recorded fallback/degraded marker, but the ORCA and PPO rows have no fallback counter; the records do not establish zero fallback for those planners. The pre-episode push remains unverified, even though the recorded commit timestamp predates the first episode.

Do not claim general safety, invariant rankings or guard benefits, exact-information independence, an identified PPO failure mechanism, a main-run minimum hidden-pedestrian distance, or sensor realism and generalization beyond these cells. The pilot distances do not establish a main-run minimum; pointwise intervals and zero observed collisions do not establish general safety. The design does not separate stressor interactions, other magnitudes or real sensor-error structure. No thesis text or claim is admitted by this packet.

## Reproduce

Analysis only (no simulation; NumPy and PyYAML), from the repository root:

```bash
uv run python docs/context/perception_degradation/analyze.py \
  --episodes docs/context/perception_degradation/evidence/episodes.jsonl.gz --output /tmp/xperc-recomputed
cmp /tmp/xperc-recomputed/effects.csv docs/context/perception_degradation/evidence/paired_effects.csv
cmp /tmp/xperc-recomputed/rates.csv docs/context/perception_degradation/evidence/rates.csv
cmp /tmp/xperc-recomputed/validity.json docs/context/perception_degradation/evidence/validity.json
cmp /tmp/xperc-recomputed/summary.json docs/context/perception_degradation/evidence/summary.json
uv run pytest docs/context/perception_degradation/test_analyze.py -q
(cd docs/context/perception_degradation && sha256sum -c SHA256SUMS)
```

Full acquisition, at the execution commit, with `uv sync --frozen --extra training` (Python 3.13.14, `uv.lock` sha256 `def82098…d023`) and the PPO release asset placed under `output/model_cache/` (see `model/registry.yaml`):

```bash
docs/context/perception_degradation/run_campaign.sh main <fresh-output-root>
uv run python docs/context/perception_degradation/project.py --raw-root <fresh-output-root> --output <dir>/episodes.jsonl.gz
```

Configs are generated by [`make_configs.py`](make_configs.py) into [`inputs/`](inputs/). Pilot and qualification evidence (seeds 1001–1010, never pooled) is in [`evidence/pilot/`](evidence/pilot/). It includes true-vs-perceived traces and [`qualify_pilot.py`](qualify_pilot.py) output with 0 trace violations in 231,917 steps.

## Reconstruction comparison contract

For one selected main seed, reproduce all six cells, four planners and four conditions at the execution commit. The expected block has 96 distinct `(seed, cell, planner, condition)` keys. The compact projection must have the same schema and exactly match every projected field except `git_hash`, `source_path`, `source_line` and `source_line_sha256`. This includes execution mode, status, fallback and checkpoint fields, guard setting, budget, steps, termination, outcome, metrics and the complete perception summary. Commits are checked against independently declared archived and replay commit IDs; for this verification both must be the execution commit. A later reconstruction may have a different commit only if its runtime patch and environment are independently verified. Source paths and line numbers change with output location and subset selection; source-line digests may change because the raw records contain timestamps, wall time and paths. These exceptions do not claim raw-byte identity.

At the execution commit, after syncing the frozen training environment and installing the verified checkpoint, the following creates a one-seed subset while retaining all other main config values:

```bash
mkdir -p /tmp/xperc-one/configs
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
out = Path('/tmp/xperc-one/configs')
for condition in ('oracle', 'noise', 'delay', 'occlusion'):
    config = yaml.safe_load(Path(f'docs/context/perception_degradation/inputs/campaign_main_{condition}.yaml').read_text())
    config['seed_policy']['seeds'] = [1121]
    (out / f'{condition}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
for condition in oracle noise delay occlusion; do
  .venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
    --config "/tmp/xperc-one/configs/$condition.yaml" --output-root /tmp/xperc-one/raw \
    --label "$condition" --log-level WARNING || exit "$?"
done
.venv/bin/python docs/context/perception_degradation/project.py \
  --raw-root /tmp/xperc-one/raw --output /tmp/xperc-one/episodes.jsonl.gz
```

From this packet checkout, compare the projection with the complete archived block:

```bash
python docs/context/perception_degradation/compare_reconstruction.py \
  --archived docs/context/perception_degradation/evidence/episodes.jsonl.gz \
  --replay /tmp/xperc-one/episodes.jsonl.gz --seed 1121 \
  --archived-commit 4e16d46908789c37651a4b851a23f10bf1540e7b \
  --replay-commit 4e16d46908789c37651a4b851a23f10bf1540e7b \
  --report /tmp/xperc-one/comparison.json
```

The comparison fails on missing, duplicate or unexpected keys, different schemas, wrong declared commits or any different required field. A rerun compares only these projected fields; it does not independently verify every raw trace field. The [recorded replay receipt](provenance/replay-1121.json) shows 96/96 normalized rows matched and 0/96 complete rows matched. Both sets of 96 recorded commits matched the execution commit; source paths matched 0/96, source line numbers 16/96, and source-line digests 0/96. The [replay projection](provenance/replay-1121-episodes.jsonl.gz) retains those custody fields. The comparator was also checked against an altered-mode negative control.

## Custody

[`evidence/episodes.jsonl.gz`](evidence/episodes.jsonl.gz) holds one row per episode: identity, mode, fallback fields, perception accounting, endpoints, and the source path, line and SHA-256 of the exact producer line. [`evidence/episodes_sources.json`](evidence/episodes_sources.json) binds the 16 producer files by digest. The 4.41 GB raw main bundle (step traces, commands, typed contacts and resolved configs) is held outside Git with a SHA-256 manifest in the lab host archive and in `wandb://ll7/robot_sf/campaign-xperc_perception_degradation_main_4e16d4690:v0` (artifact digest `dfa056b7ee2e55aff08120bcfb4eb89f`). Readback reported `COMMITTED` and matched all 228 remote file paths, sizes and MD5 digests to the local files; this checks remote metadata, not a full download. See [`provenance/preservation.json`](provenance/preservation.json). The pilot and smoke raw stages have separately verified manifests in the lab host archive; the pilot qualification projections are tracked here and smoke is not pooled into the main evidence.

This packet changes no frozen artifact, release or dissertation text.
