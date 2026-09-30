# Issue #10007: prediction planner defects, lane FXP

Base: PR #9926 branch `codex/issue-9750-physical-radius-fix-20260929`,
`b3fa204dc4eda5004d67e4fcf518f3370fcd0966`. The hunt report was treated as leads;
A1–A4 were first confirmed at `d41cceb7f` and reconfirmed at this latest base after
#9926 advanced; its relevant planner, collector and release-config bytes were unchanged.
No retraining,
held-out evaluation, frozen YAML changes, 0.0.2/0.0.7 artifact changes, release,
or merge was performed. This is implementation and scorer diagnostic evidence,
not a planner ranking, navigation-success improvement, or release admission.

## Evidence method

The [serial probe](evidence/issue_10007_fxp/probe.py) uses the released v2_full and
v1 checkpoints with verified registry SHA-256 digests, no predictor fallback,
and actual environment observations from `classic_doorway_low` and
`classic_bottleneck_medium`, seeds **1001, 1002, 1003 only**. It advances the
collector's deterministic goal policy for at most 160 steps per episode,
stopping on environment termination: 157 doorway and 269 bottleneck observations,
426 total per revision. The corrected scorer replays the base capture, preserving
every observation array
with its original dtype/shape. Per-observation SHA-256 digests are identical,
alongside scenario/seed/tick, heading, count, and wall-distance equality. At most two serial probe
processes ran concurrently; each owned one environment.

For A2, it compares costs on the actual map grid against the same grid with only
obstacle occupancy removed and the combined channel rebuilt from pedestrian cells.
A controlled wall-facing comparison changes observation yaw to point at the
nearest occupied cell, keeping the original grid-pose metadata and actual sampled
robot position; it scores a legal 1.6 m/s straight command, masking pedestrians
from both comparisons. This is a diagnostic counterfactual, not an executed
navigation rollout. Unmodified candidate scoring is also retained in the raw
captures. Most ordinary candidates did not reach wall cells within 0.8 s; four
ordinary doorway sequence scores gained 0.03125. Do not infer fewer collisions.

The [summary](evidence/issue_10007_fxp/summary.json) preserves aggregates and
concrete counterexamples. The [manifest](evidence/issue_10007_fxp/manifest.json)
pins source files, probe, checkpoint, and raw-capture digests. The gzip JSON [observation snapshot](evidence/issue_10007_fxp/observations.jsonl.gz),
[base failures](evidence/issue_10007_fxp/regression_base.txt),
[validation log](evidence/issue_10007_fxp/validation.txt), summary, and probe are
committed. Complete scorer JSON and additional raw logs remain in the lane's
external evidence directory. Existing model assets remain portable through the registry's
`artifact/models-2026-05-registry-v1` release. No new model was produced.

## A1 — confirmed collector defect; historical model exposure needs retraining

Root cause: `robot_sf/sensor/socnav_observation.py:920` rotates world pedestrian
velocities by minus robot heading. Both collectors stored those observation
velocities in a field misleadingly called `ped_velocities_world` and rotated them
again. Serving consumes the observation velocities directly. For a north-facing
robot and an eastward 1 m/s pedestrian, the correct feature is `[0,-1]`; the old
collector emitted approximately `[-1,0]`.

Fix: rename the field to `ped_velocities_ego`, remove the redundant rotation
helper, and copy the observation velocity once in both producers:
`scripts/training/collect_predictive_planner_data.py:241` and
`scripts/training/collect_predictive_hardcase_data.py:239`. Position transforms
remain separate. Existing model bytes and registry entries are unchanged.

| Dev map | Mean collector/serving velocity error, before | After |
| --- | ---: | ---: |
| Bottleneck medium | 0.728898 m/s | 0.000000 m/s |
| Doorway low | 1.196829 m/s | 0.000000 m/s |

Both collectors have the same errors and both reach zero. These are feature
errors, not model ADE/FDE improvements. Regression:
`tests/planner/test_prediction_fxp_regressions.py:45`, both collectors, nested and
flat observations. Base failure:
`ACTUAL: array([-1.000000e+00, 4.371139e-08]); DESIRED: array([0., -1.])`.
All four cases pass after the fix.

Historical exposure: the registry's v2_full source commit
`cef93136b92ddca9b0c4436bc44049412461a2fd` has the second rotation at collector
lines 179–182. The W&B-retained training summary names the same commit and model.
The exact dataset has no available collector hash here, so the provenance does
not independently prove every historical row was collected with that revision.
It strongly supports the reported training/serving mismatch, not a quantitative
claim about how much it harmed navigation or inflated validation metrics.

The provided `A_probe_velocity_skew.py` was run on both checksum-verified assets.
For v2_full, 0.8 s displacements were `[-0.369,-0.452]`, `[1.043,-0.070]`,
`[0.283,0.509]`, `[0.965,-0.293]` for velocities `(-1.2,0)`, `(1.2,0)`, `(0,1.2)`,
`(0,-1.2)`; constant velocity would give `[-0.96,0]`, `[0.96,0]`, `[0,0.96]`,
`[0,-0.96]`. The stationary probe gave `[0.050,0.004]`. Direction-dependent
responses support investigation but do not uniquely diagnose training corruption:
a learned model need not equal constant velocity. Collector fixes do not change
these already-trained checkpoint outputs.

## A2 — confirmed dropped wall term

Root cause: `_path_penalty` returns preferred combined/obstacle occupancy followed
by pedestrian occupancy. Both prediction score paths unpacked `_, occ_penalty`,
so wall-only bytes had no cost, contrary to the tutorial's static-obstacle intent.

Fix: the 0.0.8 configs select `predictive_occupancy_version: combined_v2`.
Action and sequence scoring use `obstacle_penalty + 0.5 * ped_penalty`, matching
the MPPI occupancy convention, at `robot_sf/planner/socnav_prediction.py:1284`
and `:1421`. The first term prefers combined occupancy, so this deliberately
retains the existing MPPI convention's additional pedestrian contribution.
Historical `pedestrians_v1` remains the default, preserving old config identity.

| Wall-facing dev comparison | Before, positive-cost observations | After | Maximum added cost |
| --- | ---: | ---: | ---: |
| Bottleneck medium, each score path | 0/269 | 19/269 | 0.09375 |
| Doorway low, each score path | 0/157 | 7/157 | 0.09375 |

Example: bottleneck seed 1001, tick 70, nearest occupied-cell center distance
1.303841 m: both scores gain 0.03125 after the correction, versus zero before.
Doorway seed 1001, tick 55, distance 1.272792 m: the same 0 → 0.03125 change.
These costs restore wall sensitivity; the soft term alone is no safety guarantee.

Regression: `tests/planner/test_prediction_fxp_regressions.py:63`, both real
scoring methods, obstacle/combined grid arrays occupied and pedestrian channel
empty. Base failure: `Obtained: 0.0; Expected: 0.25 ± 2.5e-07`. Both pass after.

## A3 — confirmed silent horizon cap

Root cause: MPPI first capped the request at the forecast length, then inflated
shorter requests to the anchor's effective horizon. The release declared 24 steps
at 0.1 s, but the v1 checkpoint emits eight steps. Both downloaded model payloads
independently declare `horizon_steps: 8`.

Fix: `robot_sf/planner/predictive_mppi.py:232` rejects requests outside
`1..future.shape[1]` with a descriptive `ValueError`, before optimization. Supported
requests retain their requested length. Both the config dataclass and root builder
now default to eight. The MPPI 0.0.8 release declares **8 × 0.1 s = 0.8 s**;
both predictive release configs disable ineffective horizon boosting. Historical
configs with unsupported requests now fail explicitly; their files are untouched.

At all 426 dev observations, the base accepted a 24-step request and returned
8; fixed code rejects 24 with `horizon_steps=24 ... supported horizon of 8 steps`.
A supported 4-step request returned 8 before and 4 after; the corrected release's
8-step request returns 8. No future was extrapolated.

Regressions: `tests/planner/test_prediction_fxp_regressions.py:111` and `:120`.
Base failure lines: `Failed: DID NOT RAISE ValueError` and `assert 8 == 4`.
Both pass after. Two existing MPPI fixtures with four forecast steps now explicitly
request four steps, preserving their cache and route-target checks.

Options for a longer horizon:

- Recommended now: use the honest 0.8 s model-supported horizon.
- Train a predictor with 24 native outputs at 0.1 s, after collecting and validating
  identity-consistent, correctly framed 24-step targets. It changes the artifact,
  training objective, and runtime cost; it needs separate evidence before promotion.
- An autoregressive continuation would need a documented state/velocity/frame update,
  training or calibration on repeated predictions, and long-horizon error validation.
  No such implementation or evidence was found here. Holding the last pedestrian
  point or appending constant-velocity guesses is not validated model support and
  is not implemented by this correction.

## A4 — confirmed heading saturation

Root cause: each configured heading delta was divided by a single 0.1 s timestep,
then clipped to ±1 rad/s. Seven base and nine near-field heading requests therefore
became only `{-1,0,+1}`.

Fix: `predictive_heading_lattice_version: horizon_scaled_v2` interprets deltas as
horizon yaw changes. `robot_sf/planner/socnav_prediction.py:920` uses the effective
horizon duration and, when outer headings exceed reachable yaw, scales the entire
set proportionally to the angular limit. No individual headings collapse to the
same saturated rate. Historical `per_step_v1` remains the default. MPPI's release
anchor explicitly lists the same decimal near-field angles as its base lattice,
avoiding nearly identical pi/rounded-angle duplicates.

At all dev observations the old lattice had three distinct rates. Both corrected
release planners have seven in ordinary states and eleven in near-field states
(the union of seven base and nine augmented options). Ordinary rates include
`±0.32724875`, `±0.65449875`, `±0.9817475` and zero. Near-field outer rates reach
±1 while inner rates remain distinct. These are distinct configured commands,
not proof of distinct first-tick motion under the drive's acceleration limits.

Regression: `tests/planner/test_prediction_fxp_regressions.py:95`, ordinary and
near-field states for both production release YAMLs/builders. Base failure:
`assert 3 == 7` and `assert 3 == 11` (the historical MPPI near-field union also
contains pi/decimal near-duplicates). All four cases pass after.

## A5 — index-matching code confirmed; v1 dataset attribution remains conditional

Current registry bytes at `model/registry.yaml:605` connect v1 to
[`ll7/robot_sf/geedo1po`](https://wandb.ai/ll7/robot_sf/runs/geedo1po), dataset
`predictive_rollouts_mixed_v1.npz`, and commit
`dfc4aea84e25cc83f9888c620286457bab3e1596`. The hunt's use of that registry commit
as a collector revision is **refuted**: the collector does not exist at that commit.

Live W&B readback shows `geedo1po` is `predictive_proxy_selected_v1_file_upload`,
created `2026-02-20T08:15:44Z`, with an **empty training config**. Its retained
metadata points to `a5c9ac0bd5ccb0dcfd48674f367d176f843affdd` and program `<stdin>`.
At that actual upload commit, the collector's lines 138–148 still write future
rows by list index; the observation producer sorts pedestrians by distance.
The nearest-match repair `97405065b` happened at `2026-02-20T09:19:09Z`, after
this upload. The release checkpoint payload names the mixed-v1 dataset and selected
epoch 8; recorded validation ADE/FDE are 3.357923/0.621323 m.

Thus the repository collector available before the upload used index matching,
and index corruption is a strong, temporally consistent explanation. The upload
run has no collection config, collector digest, dataset manifest, or original
training code snapshot proving which producer made the exact uploaded NPZ.
Report the checkpoint exposure as **plausible, not conclusively confirmed**;
`geedo1po` cannot settle it as a training run. Do not repair or retrain v1 here.

The v1 velocity-response probe yielded displacements `[-1.395,-0.356]`,
`[0.037,-0.044]`, `[-0.038,0.125]`, `[-0.106,-0.227]` for the same four velocity
inputs, stationary `[0.010,0.032]`. These weak/directional responses and validation
errors do not independently prove identity-switch contamination. Reopen attribution
if the original mixed-v1 dataset manifest, collector source, or training run is recovered.

## Retraining plan for the v2_full successor — prepared, not executed

The reproducible source recipe is
`configs/training/predictive/predictive_br07_all_maps_randomized_full.yaml` plus
source commit `cef93136b92ddca9b0c4436bc44049412461a2fd`.
[`ll7/robot_sf/u40parjb`](https://wandb.ai/ll7/robot_sf/runs/u40parjb) is a portable
**file-upload run**, not the original trainer run. Its config points to the
`predictive_br07_all_maps_randomized_full_20260305T123116Z` pipeline, and its
retained `training_summary.json` supplies the following observed recipe:

- 12,163 mixed samples, 24 max agents, legacy four features, eight 0.1 s targets;
  active-agent/target ratios 0.239809/0.238049.
- Hidden dimension 192, three message-passing steps, distance temperature 2.0;
  220 epochs, batch 128, learning rate 0.0002, weight decay 0.00001, seed 42.
- Retained weights selected epoch 201 by `val_loss_fallback`; ADE/FDE
  0.065778/0.121756 m on the old collected features. The summary's overall
  selection setting is `proxy`; these describe selection configuration and the
  actual saved fallback differently and must both be recorded.
- The source profile collects eight seeds per scenario, at most 240 base frames,
  and seven hardcase pairs at most 260 frames; hardcase repeat three, shuffle 42,
  validation fraction 0.2. The current scenario loader resolves 23 scenarios.
  Exact historic collection counts/seed choices require the original dataset manifest.

Use the corrected collector at the eventual merged FXP commit to **recollect both
base and hardcase components**. Do not mix fresh correct velocity features with
old double-rotated NPZ rows. A successor model ID, e.g.
`predictive_proxy_selected_v2_full_velocity_v3`, must leave existing registry IDs
and published assets intact.

A concrete dev-only collection packet should copy the full profile to a new versioned
config and set an explicit base seed manifest: all 23 scenarios on seeds 1001–1008
(184 episodes), with four original hardcase scenario families represented by seven
pairs using seeds 1009–1015. Proxy and final development evaluation must use explicit
seeds 1016–1030 and at most two simulation workers, with no inherited manifest seeds.
The original profile's seed schedule is not an admissible execution command for this
lane. Keep dt 0.1, max agents 24, horizon 8, max speed 1.2, repeat three, and the observed
training hyperparameters. Confirm the actual seed manifests/config before submission.

Data-volume estimate: 184 × 240 + 7 × 260 = **45,980 collected frames maximum**.
At most 42,688 base windows plus 1,764 hardcase windows, repeated three, gives
**47,980 mixed windows maximum**; real early termination lowers these. Start with
at least the observed 12,163-sample coverage, and retain per-scenario/frame/target
counts rather than forcing a nominal number by duplicating more rows. Float32
state/target/mask/target-mask payload is about **32.3 MiB at 12,163 samples** or
**127.4 MiB at 47,980**, before compression; reserve 5 GiB for datasets, manifests,
logs, checkpoints, and proxy traces. Hold out complete collection episodes for
validation in a successor recipe, using separate collection manifests or retained
episode IDs; the current trainer's random window split does not provide this by
itself. Overlapping windows from one episode otherwise make that split optimistic. This is a separately declared
recipe difference, not bitwise reproduction of the old model.

Resource/time plan, explicitly **reservation estimates rather than measured training
times**: collection on one CPU Slurm task, 8 CPUs, 32 GiB RAM, 4 h walltime, with
one collector process (a second independent partition is permitted). Training on
one Slurm task, one A30 or L40S GPU, 8 CPUs, 32 GiB RAM, 12 h walltime, one trainer,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `num_workers=0`, and at most one proxy
simulation process. Keep GPU device and all hyperparameters in the saved resolved
config. The pipeline does not expose a `--device` CLI override; the trainer selects
the available device, so verify CUDA in its actual startup metadata before counting
GPU execution. No scheduler job was submitted here.

Budget collection **1–4 h**, GPU training/proxies **4–12 h**, dev validation **1–4 h**
(total **6–20 h**) until a short authorized pilot measures frames/s, epoch time,
and proxy duration. No runtime is retained in the historical summary, so a tighter
estimate would be invented. At 12,163 samples with an 80/20 split and batch 128,
there are about 77 training updates/epoch, 16,940 over 220 epochs; recompute from the
actual successor split and measured pilot. Reserve checkpoint intervals and a resumable
job rather than raising timeouts to conceal failures. Current Slurm access/capacity
must be verified at execution time; access may end on 2026-09-30.

Persist the resolved config, source commit, collector and dataset SHA-256, seeds,
feature-frame marker `ego / observation-only rotation`, split/identity-match policy,
package/device metadata, training summary, and checkpoint in a new W&B **training**
run under `ll7/robot_sf`. Preserve its run ID and immutable dataset/checkpoint artifacts
in a new registry entry. No known original training W&B run was recovered; `u40parjb`
reproduces the retained provenance/recipe through its summary, not training by itself.
Validate a known heading and the A1 serving-feature equality before long training,
then compare held-out dev ADE/FDE and planner diagnostics without fallback. Model
promotion and release evidence need a separate authorized task.

## Test-value gate and validation

| Test group | Defect and credible regression | Existing coverage gap | Real bytes / independent oracle / seam |
| --- | --- | --- | --- |
| A1, four cases | Accidental second rotation of already-ego velocities | Existing collector fixture headings are zero, so both implementations agree | Actual nested/flat observation extraction and sample arrays; hand-derived north/east `[0,-1]`; no production seam |
| A2, two cases | Discarding the first occupancy result in either score path | Existing prediction caching/progress tests do not compare wall-only grids | Real grid arrays, production release YAML/builder and both scorers; wall-only mean occupancy is 1, added cost is configured 0.25; no monkeypatch/seam |
| A3, rejection | Silent truncation of configured horizon | Existing MPPI determinism/conflict tests tolerated default 12→8 capping | Real runtime forecast length and production MPPI method; explicit 24 request must raise; hashed model payload/dev probes independently prove eight outputs; no seam |
| A3, shorter request | Inflating a supported horizon through the anchor | Existing cache/target tests silently accepted short stub horizons | Request four versus returned count four; independent integer oracle, no seam; two existing four-step fixtures now declare four explicitly |
| A4, four cases | Dividing horizon heading by one tick and clipping all inner deltas | Existing candidate tests count risk-distance calls and determinism, not distinct configured rates | Both real release YAMLs/builders, actual candidate arrays, independent 7/11 counts and `0.523599/0.8` angular-rate arithmetic; no seam |

The two existing collector fixture edits only rename the frame field; they retain
all previous feature/schema assertions. The two MPPI fixture edits make their existing
cache/target behavior tests use a supported four-step forecast contract. They are not
new bug-proof tests; the new regressions and dev probes supply the red/green evidence.

The final new regression file has **12 failures on the base**, with the precise
lines shown above, and **12 passes after correction**. The broader focused lane has
**142 passes, no skips or xfails** across prediction contracts, MPPI, both collectors,
predictive model, pipeline, mixed datasets, and probabilistic prediction interface.
Ruff check/format and `git diff --check` pass. No timeout, skip, or xfail was added.
The full repository readiness pipeline was not run because its automatic broad
simulation/test selection is outside this lane's seed and serial-test constraints.
Hosted CI and domain review remain separate; this delivery is a draft, not merge-ready.

Reproduce with the project environment, from the relevant checkout root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/fast-pysf"
python -m pytest -n0 tests/planner/test_prediction_fxp_regressions.py -q
python -m pytest -n0 tests/planner/test_prediction_fxp_regressions.py \
  tests/planner/test_socnav_prediction_module.py \
  tests/planner/test_predictive_mppi_planner.py \
  tests/training/test_collect_predictive_planner_data.py \
  tests/training/test_collect_predictive_hardcase_data.py \
  tests/test_predictive_model.py tests/training/test_run_predictive_training_pipeline.py \
  tests/training/test_build_predictive_mixed_dataset.py \
  tests/planner/test_probabilistic_prediction_interface.py -q
# Download only the existing public model assets into a flat cache directory.
gh release download artifact/models-2026-05-registry-v1 --repo ll7/robot_sf_ll7 \
  --pattern 'predictive_proxy_selected_v1-predictive_model.pt' \
  --pattern 'predictive_proxy_selected_v2_full-predictive_model.pt' --dir /tmp/fxp-models
# Capture base observations in the base checkout, then replay them after fixing.
python docs/context/evidence/issue_10007_fxp/probe.py /tmp/fxp-before.json /tmp/fxp-models
# Run this command from the fixed checkout, with the same probe source.
python docs/context/evidence/issue_10007_fxp/probe.py /tmp/fxp-after.json /tmp/fxp-models \
  docs/context/evidence/issue_10007_fxp/observations.jsonl.gz
```

For the base comparator, create a detached worktree at the pinned base, copy only
this probe and the new regression file there, activate the same environment with
`PYTHONPATH` pointing to that base checkout and its `fast-pysf`, and run from its
root. Do not run base tests from the fixed checkout: Python's current-directory
imports can otherwise select corrected modules. The original import mismatch in
the shared environment was repaired by putting the checkout's `fast-pysf` ahead of
the installed dependency; that import error was not counted as a regression failure.
