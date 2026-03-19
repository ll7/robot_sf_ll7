# Issue 626 SoNIC Source Harness Probe

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#626` SoNIC source-harness reproduction spike
- `robot_sf_ll7#601` CrowdNav family feasibility note
- `robot_sf_ll7#627` Robot SF wrapper follow-up

## Goal

Establish whether the upstream `SoNIC-Social-Nav` source harness can be reproduced in the current
environment without heuristic fallback, before any Robot SF wrapper work begins.

## Canonical probe command

```bash
uv run python scripts/tools/probe_sonic_source_harness.py \
  --repo-root output/repos/SoNIC-Social-Nav \
  --model-name SoNIC_GST \
  --checkpoint 05207.pt \
  --output-json output/benchmarks/external/sonic_source_harness_probe/report.json \
  --output-md output/benchmarks/external/sonic_source_harness_probe/report.md
```

## Current result

Verdict: `source harness blocked`

Observed failure:

- The upstream entrypoint `output/repos/SoNIC-Social-Nav/test.py` fails immediately in the current
  repository Python environment with:
  - `ModuleNotFoundError: No module named 'gym'`

Why this matters:

- The issue is no longer blocked by ambiguity.
- It is blocked by a concrete source-environment dependency mismatch.
- That is exactly the kind of fail-fast evidence we needed before attempting any Robot SF adapter.

## Source contract captured from the upstream model config

For `trained_models/SoNIC_GST`:

- `robot_policy`: `selfAttn_merge_srnn`
- `human_policy`: `orca`
- `robot_sensor`: `coordinates`
- `predict_method`: `inferred`
- `action_kinematics`: `holonomic`
- `env_use_wrapper`: `True`
- `env_name` default: `CrowdSimPredRealGST-v0`

Interpretation:

- The source stack expects a wrapped predictor-aware crowd simulator and a holonomic source action
  contract.
- This remains materially different from the Robot SF benchmark contract, which is why source-side
  reproduction has to come first.

## Runtime assumptions recorded from the upstream source

- Docker base image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`
- Additional source packaging:
  - `output/repos/SoNIC-Social-Nav/gst_updated/requirements.txt`
  - `output/repos/SoNIC-Social-Nav/Python-RVO2/requirements.txt`
- The source release is Docker/NVIDIA-oriented and expects packages that are not present in the
  current repository environment.

## Conclusion

Status: `source harness blocked`

Recommended next move inside this issue:

1. keep the fail-fast probe script as the canonical reproduction entrypoint
2. use it to validate a dedicated SoNIC source environment
3. do not start `#627` until the probe returns `source harness reproducible`

## Model-only reuse follow-up

The narrower model-only reuse question is materially better than the full source-harness result, but
still not plug-and-play.

Canonical command:

```bash
uv run python scripts/tools/probe_sonic_model_inference.py \
  --repo-root output/repos/SoNIC-Social-Nav \
  --model-name SoNIC_GST \
  --checkpoint 05207.pt \
  --output-json output/benchmarks/external/sonic_model_inference_probe/report.json \
  --output-md output/benchmarks/external/sonic_model_inference_probe/report.md
```

Observed result:

- direct model import verdict: `direct model import blocked`
- direct failure: `AssertionError: Torch not compiled with CUDA enabled`
- shimmed model-only verdict: `model-only inference reproducible with shims`
- shims required:
  - `gymnasium as gym module alias`
  - stub `rl.networks.envs.VecNormalize`
  - `config.policy.constant_std = false`
- checkpoint load result:
  - missing state keys: `['dist.logstd._bias']`
  - unexpected state keys: `[]`
- synthetic forward pass result:
  - action shape: `[1, 2]`
  - value shape: `[1, 1]`

Interpretation:

- The bundled SoNIC checkpoint is reusable at inference time only with narrow compatibility shims.
- This is enough to justify future model-only adapter work.
- It is not enough to claim source-harness parity or benchmark-readiness.
