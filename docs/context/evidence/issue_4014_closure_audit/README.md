# Issue #4014 Closure Audit

This audit checks whether the Mamba state-space encoder and recurrent proximal policy optimization (PPO) comparison issue can close from merged pull request evidence alone.

Claim boundary: closure audit only; no benchmark, paper, or dissertation claim is promoted.

## Decision

Close #4014 when this PR merges. The merged pull requests established the primitives and registration surfaces, and this PR adds the matched configs, runs all three smoke rows, records parameter and throughput metadata, and generates the diagnostic comparison artifact.

## Criterion Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Mamba feature extractor with CPU-safe fallback | Met | PR #4162 added `MambaFeatureExtractor`, exports, tests, and design note. |
| True RecurrentPPO long short-term memory (LSTM) lane | Met | PR #4234 added the dry-run lane. This PR adds matched smoke config and non-dry-run performance-summary emission. |
| PPO-Mamba smoke lane registration | Met | PR #4475 added the first config and dry-run tests. This PR adds the executable matched `default_gym` smoke config. |
| Parameter-count metadata | Met | PR #4581 added `parameter_summary` to PPO summaries. This PR extends RecurrentPPO after training. |
| Three matched smoke runs | Met | Local validation produced real PPO, RecurrentPPO-LSTM, and PPO-Mamba summaries with `parameter_summary.available=true`, throughput, and wall-clock metadata. |
| Diagnostic comparison artifact | Met | This PR generated `docs/context/evidence/issue_4585_matched_ppo_sequence_smoke_comparison/`. |

## Local Audit Findings

- `gh issue view 4014 --comments` failed because the GitHub CLI requested the deprecated classic Projects field; the audit used `gh api` for the issue body, all comments, and timeline events.
- `sb3-contrib` was initially absent; installing the existing `recurrent` extra made RecurrentPPO importable.
- The existing `socnav_struct` config fails before learning because Stable Baselines3 rejects nested observation spaces. The matched configs use `default_gym`, which provides the non-nested `drive_state` and `rays` dictionary needed by `MultiInputLstmPolicy` and `MambaFeatureExtractor`.
- A PPO-Mamba dry run writes `parameter_summary.available=false`, so dry-run output cannot satisfy the acceptance criteria.
- The executable smoke rows are diagnostic-only evidence and do not promote benchmark, paper, or dissertation claims.

## Remaining Slice

No remaining slice is required for #4014 diagnostic smoke acceptance. The comparison artifact was generated with:

```bash
uv run python scripts/training/compare_issue_4014_ppo_sequence_models.py \
  --perf-summary ppo=<ppo-perf.json> \
  --perf-summary recurrent_ppo_lstm=<lstm-perf.json> \
  --perf-summary ppo_mamba=<mamba-perf.json> \
  --config ppo=configs/training/ppo/issue_4014_ppo_smoke_matched.yaml \
  --config recurrent_ppo_lstm=configs/training/ppo/issue_4014_recurrent_ppo_lstm_smoke_matched.yaml \
  --config ppo_mamba=configs/training/ppo/issue_4014_ppo_mamba_smoke_matched.yaml \
  --output-dir docs/context/evidence/issue_4585_matched_ppo_sequence_smoke_comparison
```
