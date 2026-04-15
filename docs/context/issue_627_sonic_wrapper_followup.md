# Issue 627 SoNIC Wrapper Follow-up

Date: 2026-04-15
Related issues:
- `robot_sf_ll7#627` Prototype fail-fast Robot SF wrapper for CrowdNav/SoNIC family
- `robot_sf_ll7#626` SoNIC source-harness reproduction spike
- `robot_sf_ll7#601` CrowdNav family feasibility note

## Goal

Prototype a fail-fast Robot SF wrapper for the upstream SoNIC/CrowdNav family that preserves the
source observation/action boundary as explicitly as possible, without silently falling back to a
local heuristic planner.

## What Was Implemented

- `robot_sf/planner/sonic_crowdnav.py`
  - fail-fast source asset validation
  - SoNIC model-only import shims
  - Robot SF observation translation into the upstream dict contract
  - upstream `ActionXY` to Robot SF `unicycle_vw` projection
- `robot_sf/benchmark/algorithm_metadata.py`
  - `sonic_crowdnav` provenance and kinematics metadata
- `robot_sf/benchmark/algorithm_readiness.py`
  - readiness classification and aliases for `sonic_crowdnav` / `sonic_gst`
- `robot_sf/benchmark/map_runner.py`
  - benchmark runner wiring for the new wrapper entrypoint

## Validation

Canonical commands:

```bash
uv run pytest tests/benchmark/test_algorithm_metadata_contract.py -k 'sonic_crowdnav'
uv run pytest tests/benchmark/test_map_runner_utils.py -k 'sonic_crowdnav'
uv run pytest tests/planner/test_sonic_crowdnav.py
```

Observed result:

- metadata and map-runner wiring pass
- wrapper translation tests pass against a fake upstream repo
- fail-fast guards pass for missing assets and unsupported upstream kinematics
- the checked-in SoNIC checkout also passes a real wrapper smoke test in Robot SF

## Current Boundary

The source-harness probe in `docs/context/issue_626_sonic_source_harness_probe.md` is still the
blocking evidence surface. The upstream SoNIC source harness is still not reproducible in the
current environment, even after adding `gym` to a side environment; the next blocker is now
`matplotlib`. That means this wrapper is still a model-only / prototype-level adapter rather than a
benchmark-ready family integration, even though the Robot SF smoke path itself works.

## Verdict

`wrapper not yet viable for benchmark spike`

Reason:

- the wrapper exists and is fail-fast, but the upstream source harness is still blocked
- the current proof only covers model-only translation and integration wiring
- benchmark promotion would require a source-faithful repro path first

## Follow-Up

If the upstream source environment becomes reproducible, re-run the source-harness probe and then
re-evaluate whether the wrapper can be promoted from prototype to benchmark-spike status.

## GenSafeNav Follow-up

Date: 2026-04-15

`GenSafeNav` is close enough to the SoNIC runtime/model stack that the existing model-only adapter
can be reused for the learned checkpoints without another wrapper class.

Observed result:

- `trained_models/Ours_GST/checkpoints/05207.pt` runs through the adapter
- `trained_models/GST_predictor_rand/checkpoints/05207.pt` also runs through the adapter
- both expose the same upstream `selfAttn_merge_srnn` policy family and holonomic `ActionXY`
  contract

Assumptions recorded:

- `Ours_GST` is still a model-only wrapper path, not a source-harness parity result
- `GST_predictor_rand` is a learned CrowdNav++-style comparison checkpoint and is wrapper-friendly
- `ORCA` and `SF` in `GenSafeNav` should not be exposed as checkpoint wrappers because their saved
  configs use source-side classical policies rather than learned checkpoint inference

Implementation boundary:

- safe to add benchmark aliases for `gensafenav_ours_gst` / `ours_gst`
- safe to add benchmark aliases for `gensafenav_gst_predictor_rand` / `gst_predictor_rand`
- not safe to add `GenSafeNav ORCA` or `GenSafeNav SF` as model-only checkpoint wrappers

## Performance Status

Current evidence is limited to integration and smoke validation.

- `Ours_GST` and `GST_predictor_rand` both step through the Robot SF wrapper and produce finite
  commands.
- There is no episode-level success/collision/runtime report yet for these GenSafeNav aliases in the
  Robot SF benchmark stack.
- Do not infer benchmark-quality performance from the current proof set; the wrappers are runnable,
  but their comparative quality is still unmeasured here.
