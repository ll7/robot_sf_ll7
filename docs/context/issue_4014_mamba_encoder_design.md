# Issue #4014 Mamba Encoder Primitive

This note records the first implementation slice for the Mamba state-space model policy encoder
lane: a CPU-safe feature extractor primitive that can be tested before training campaigns or
graphics processing unit (GPU) dependencies are introduced.

## Claim Boundary

- Evidence tier: implementation primitive only.
- The `torch_ssm_lite` backend is a small PyTorch state-space-model-inspired sequence encoder for
  continuous integration (CI) and smoke tests, not an exact Mamba implementation.
- The optional `mamba_ssm` backend is recorded with `backend_exact=true` only when the external
  package is installed and selected.
- Under standard Proximal Policy Optimization (PPO), both the existing long short-term memory
  (LSTM) extractor and this Mamba extractor encode a sequence within one observation. They do not
  carry hidden state across environment steps.
- A fair PPO versus PPO-LSTM versus PPO-Mamba comparison still needs later slices for training
  registration, true recurrent PPO (`RecurrentPPO`), bounded temporal-history observations, and
  matched diagnostic reporting.

## Contract Added

- `robot_sf.feature_extractors.mamba_extractor.MambaFeatureExtractor`
- `MambaFeatureExtractorConfig` with conservative defaults:
  - `backend="auto"`
  - `sequence_source="rays"`
  - `fail_if_exact_backend_missing=false`
- Runtime metadata:
  - `backend_name`: `mamba_ssm` or `torch_ssm_lite`
  - `backend_exact`: `true` only for the exact optional backend
- Sequence inputs:
  - `rays`: current CPU-safe default, matching the LSTM extractor's within-observation ray sequence
  - `temporal_history`: fail-closed planned input key for the later bounded-history wrapper

## Explicit Non-Claims

This slice does not register `feature_extractor: mamba` in PPO training, does not run a benchmark
campaign, does not submit Slurm Workload Manager or GPU jobs, and does not make paper- or
dissertation-facing performance claims.
