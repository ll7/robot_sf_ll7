# Issue 627 SoNIC Wrapper Follow-up

Date: 2026-04-15
Related issues:
* `robot_sf_ll7#627` Prototype fail-fast Robot SF wrapper for CrowdNav/SoNIC family
* `robot_sf_ll7#626` SoNIC source-harness reproduction spike
* `robot_sf_ll7#601` CrowdNav family feasibility note

## Goal

Prototype a fail-fast Robot SF wrapper for the upstream SoNIC/CrowdNav family that preserves the
source observation/action boundary as explicitly as possible, without silently falling back to a
local heuristic planner.

## What Was Implemented

* `robot_sf/planner/sonic_crowdnav.py`
  + fail-fast source asset validation
  + SoNIC model-only import shims
  + Robot SF observation translation into the upstream dict contract
  + upstream `ActionXY` to Robot SF `unicycle_vw` projection
* `robot_sf/benchmark/algorithm_metadata.py`
  + `sonic_crowdnav` provenance and kinematics metadata
* `robot_sf/benchmark/algorithm_readiness.py`
  + readiness classification and aliases for `sonic_crowdnav` / `sonic_gst`
* `robot_sf/benchmark/map_runner.py`
  + benchmark runner wiring for the new wrapper entrypoint

## Validation

Canonical commands:

```bash
uv run pytest tests/benchmark/test_algorithm_metadata_contract.py -k 'sonic_crowdnav'
uv run pytest tests/benchmark/test_map_runner_utils.py -k 'sonic_crowdnav'
uv run pytest tests/planner/test_sonic_crowdnav.py
```

Observed result:

* metadata and map-runner wiring pass
* wrapper translation tests pass against a fake upstream repo
* fail-fast guards pass for missing assets and unsupported upstream kinematics
* the checked-in SoNIC checkout also passes a real wrapper smoke test in Robot SF

## Current Boundary

The source-harness probe in `docs/context/issue_626_sonic_source_harness_probe.md` is still the
blocking evidence surface. The upstream SoNIC source harness is still not reproducible in the
current environment, even after adding `gym` to a side environment; the next blocker is now
`matplotlib` . That means this wrapper is still a model-only / prototype-level adapter rather than a
benchmark-ready family integration, even though the Robot SF smoke path itself works.

## Verdict

 `wrapper not yet viable for benchmark spike`

Reason:

* the wrapper exists and is fail-fast, but the upstream source harness is still blocked
* the current proof only covers model-only translation and integration wiring
* benchmark promotion would require a source-faithful repro path first

## Follow-Up

If the upstream source environment becomes reproducible, re-run the source-harness probe and then
re-evaluate whether the wrapper can be promoted from prototype to benchmark-spike status.

## GenSafeNav Follow-up

Date: 2026-04-15

`GenSafeNav` is close enough to the SoNIC runtime/model stack that the existing model-only adapter
can be reused for the learned checkpoints without another wrapper class.

Observed result:

* `trained_models/Ours_GST/checkpoints/05207.pt` runs through the adapter
* `trained_models/GST_predictor_rand/checkpoints/05207.pt` also runs through the adapter
* both expose the same upstream `selfAttn_merge_srnn` policy family and holonomic `ActionXY`
  contract

Assumptions recorded:

* `Ours_GST` is still a model-only wrapper path, not a source-harness parity result
* `GST_predictor_rand` is a learned CrowdNav++-style comparison checkpoint and is wrapper-friendly
* `ORCA` and `SF` in `GenSafeNav` should not be exposed as checkpoint wrappers because their saved
  configs use source-side classical policies rather than learned checkpoint inference

Implementation boundary:

* safe to add benchmark aliases for `gensafenav_ours_gst` / `ours_gst`
* safe to add benchmark aliases for `gensafenav_gst_predictor_rand` / `gst_predictor_rand`
* not safe to add `GenSafeNav ORCA` or `GenSafeNav SF` as model-only checkpoint wrappers

## Performance Status

Current evidence now includes both smoke validation and a small benchmark probe.

* `Ours_GST` and `GST_predictor_rand` both step through the Robot SF wrapper and produce finite
  commands.
* There is now episode-level evidence on `verified_simple_subset_v1`, recorded below.
* That evidence is still too weak to support a strong performance claim; the wrappers are runnable, 
  but their benchmark quality remains limited here.

## Verified-Simple Benchmark Probe

Date: 2026-04-15

Matrix:

* `configs/scenarios/sets/verified_simple_subset_v1.yaml`
* 30 episodes total (10 scenarios x 3 seeds)

Canonical commands used:

```bash
LOGURU_LEVEL=INFO uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --out output/ai/autoresearch/gensafenav_tuning/ours_gst_baseline.episodes.jsonl \
  --algo ours_gst \
  --benchmark-profile experimental \
  --workers 1 \
  --structured-output json \
  --no-video \
  --no-resume \
  --external-log-noise suppress

LOGURU_LEVEL=INFO uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --out output/ai/autoresearch/gensafenav_tuning/gst_predictor_rand_baseline.episodes.jsonl \
  --algo gst_predictor_rand \
  --benchmark-profile experimental \
  --workers 1 \
  --structured-output json \
  --no-video \
  --no-resume \
  --external-log-noise suppress
```

Observed baseline result:

* `ours_gst`: success `0.3333`, timeout `0.4667`, mean runtime `1.2092s`
* `gst_predictor_rand`: success `0.3000`, timeout `0.1333`, mean runtime `1.0506s`

Observed failure pattern:

* both checkpoints solve the east/north/west empty-map cases
* both are weak on static-obstacle and social-interaction slices
* `ours_gst` is more timeout-dominated
* `gst_predictor_rand` is more collision-dominated

Control comparison:

* native `goal` on the same matrix also lands at success `0.3000`
* interpretation: the verified-simple slice is not a trivial free win, but the GenSafeNav wrappers
  still do not show evidence of strong benchmark behavior here

Autoresearch experiments attempted:

* `goal.next` route-waypoint substitution into the upstream goal slot
  + result: discarded
  + reason: regressed `ours_gst` to success `0.2000` and broke the empty-map east case
* increased `max_angular_speed` from `1.0` to `2.0`
  + config: `configs/algos/gensafenav_turnfast_probe.yaml`
  + result: discarded
  + reason: no aggregate success gain for either checkpoint

Current interpretation:

* the remaining weakness does not look like a simple cap-tuning problem
* the wrappers appear structurally limited on static-obstacle and social-interaction cases under the
  current model-only adapter contract
* any stronger improvement likely requires a more explicit hybrid/guarded planner path or a source-
  faithful reproduction that preserves more of the upstream runtime semantics

## Assessment: Dynamic-Only Bias vs Static Obstacles

Date: 2026-04-15

Question reviewed:

* are the current failures expected if the imported policies are primarily dynamic-obstacle behavior
  models?

Assessment:

* yes, this is a plausible and likely-major contributor to the observed behavior
* the current adapter translation in `robot_sf/planner/sonic_crowdnav.py` is pedestrian-centric:
  robot state + predicted pedestrian-relative trajectories are provided, while static-map occupancy is
  not explicitly exposed in the translated model input contract
* this aligns with the failure pattern already measured on `verified_simple_subset_v1`: weak static-
  obstacle and social-interaction performance despite acceptable empty-map behavior

Conservative caveat:

* this does not prove static-obstacle blindness is the only root cause
* projection effects (holonomic `ActionXY` to unicycle `v,w`) and source-runtime mismatch can still
  contribute materially to collision and timeout modes

## Transferability Check: issue-791 Reward Curriculum Branch

Date: 2026-04-15

Reference branch reviewed:

* `origin/codex/791-reward-curriculum-gate`

What likely transfers:

* experiment discipline: gate-style progression, explicit keep/discard criteria, and config-first
  reproducibility
* observation-contract rigor: avoid implicit claims of parity when the observation/action boundary
  differs

What does not directly transfer:

* the issue-791 changes are primarily PPO training-side reward curriculum and feature-extractor work
  inside Robot SF policy training
* they do not provide a drop-in fix for frozen external SoNIC checkpoints running through a model-
  only adapter

Working conclusion:

* treat issue-791 as process inspiration, not as a direct algorithmic fix path for issue-627 wrappers

## Next Autoresearch Contract (Recommended)

Date: 2026-04-15

Goal:

* determine whether minimal static-awareness guardrails can reduce wall-collision failures without
  over-claiming source parity

Metric command scope:

* matrix: `configs/scenarios/sets/verified_simple_subset_v1.yaml`
* planners: `ours_gst`,  `gst_predictor_rand`
* report: success rate, collision rate, timeout rate, and per-scenario deltas

Hypothesis order (one at a time):

1. guarded hybrid selector (new explicit algorithm alias)
   - when static-obstacle risk signal is high, hand off to a native safe baseline (for example

`orca` or `goal` ) under an explicit, documented guard

   - keep only if static-obstacle collision rate drops while aggregate success is non-regressive
2. projection-safety clamp (wrapper-local)
   - adjust `ActionXY -> v,w` projection with stricter turn-while-forward suppression near large

     heading error

   - keep only if timeout inflation stays bounded and collision rate decreases
3. source-faithful repro retry
   - continue upstream harness unblock path; if restored, compare model-only vs source-faithful

     behavior before further adapter complexity

Fail-closed/claim discipline:

* if hybrid guardrails are introduced, expose them as separate benchmark algorithm keys and metadata
  rather than relabeling baseline wrapper performance
* keep provenance wording explicit: model-only adapter vs source-faithful runtime

Stop condition:

* stop after two non-improving hypotheses in sequence, or after one clear keep candidate reaches
  measurable static-obstacle benefit with acceptable aggregate trade-off

## Guarded Hybrid Prototype (Implemented)

Date: 2026-04-15

Implemented scope:

* Added explicit guarded benchmark aliases as separate algorithms (no relabeling of base wrappers):
  + `ours_gst_guarded` (canonical `gensafenav_ours_gst_guarded`)
  + `gst_predictor_rand_guarded` (canonical `gensafenav_gst_predictor_rand_guarded`)
* Added mixed-mode metadata/readiness contracts for the guarded aliases.
* Implemented map-runner wiring that:
  + runs the same model-only `SonicCrowdNavAdapter` checkpoint path, 
  + applies a short-horizon guard (`GuardedPPOAdapter`) to the proposed command, 
  + uses explicit goal-policy fallback when guard vetoes the base command, 
  + records guard decision counts (`guard_stats`) in episode metadata.

Validation commands:

```bash
uv run pytest tests/benchmark/test_algorithm_metadata_contract.py -k 'guarded or gensafenav or sonic'
uv run pytest tests/benchmark/test_map_runner_utils.py -k 'guarded or gensafenav or sonic_crowdnav'
```

Observed result:

* targeted tests passed (metadata + map-runner wiring)

## Guarded Probe Results (verified_simple_subset_v1)

Date: 2026-04-15

Command pattern used:

```bash
LOGURU_LEVEL=INFO uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/verified_simple_subset_v1.yaml \
  --out output/ai/autoresearch/gensafenav_tuning/<guarded_out>.episodes.jsonl \
  --algo <guarded_algo> \
  --benchmark-profile experimental \
  --workers 1 \
  --structured-output json \
  --no-video \
  --no-resume \
  --external-log-noise suppress
```

`ours_gst_guarded` vs baseline `ours_gst` :

* baseline: success `0.3333`, collision `0.2000`, max_steps `0.4667`, mean runtime `1.2092s`
* guarded: success `0.3333`, collision `0.1000`, max_steps `0.5667`, mean runtime `1.2240s`
* interpretation: collision burden decreased, but max-steps burden increased; aggregate success was
  unchanged.

`gst_predictor_rand_guarded` vs baseline `gst_predictor_rand` :

* baseline: success `0.3000`, collision `0.5667`, max_steps `0.1333`, mean runtime `1.0506s`
* guarded: success `0.3000`, collision `0.2000`, max_steps `0.5000`, mean runtime `1.1797s`
* interpretation: strong collision reduction with unchanged success, traded for larger max-steps
  (conservative behavior / stall tendency).

Guard activity snapshot (aggregate step decisions):

* `ours_gst_guarded`: `ppo_clear=2275`,  `fallback_safe=183`,  `stop_safe=126`
* `gst_predictor_rand_guarded`: `ppo_clear=1815`,  `ppo_safe=29`,  `fallback_safe=261`, 
 `stop_safe=369`

Decision status:

* keep guarded aliases as explicit experimental alternatives (not replacements)
* do not promote as clear benchmark-quality wins yet due to max-steps inflation
* next bounded iteration should target reducing conservative stalling without regressing collision
  gains
