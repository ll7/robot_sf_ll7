# Issue #835 Lightweight CNN Divergence Triage

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/835>

## Goal

Issue #193 reported a catastrophic `lightweight_cnn` final-eval drop at 32 K PPO steps
(`-64.09` final reward after a `-4.20` best checkpoint). Issue #835 is the bounded follow-up:
reproduce that curve shape, capture gradient and feature-scale diagnostics during the suspected
20-32 K window, and decide whether the extractor needs an immediate stabilization change or should
remain experimental.

## Implementation Surface

- `configs/scenarios/lightweight_cnn_issue_835_repro.yaml` isolates the `lightweight_cnn` preset
  with seed `42`, 32 K PPO steps, 4 K eval cadence, RTX-class CUDA execution, and diagnostics from
  20 K onward.
- `robot_sf/training/ppo_diagnostics.py` adds an opt-in `DiagnosticPPO` wrapper that records
  gradient norms at the PPO clipping point and appends per-update JSONL summaries.
- `robot_sf/feature_extractors/lightweight_cnn_extractor.py` can opt into forward-pass feature
  statistics with `record_feature_stats: true`.
- `scripts/multi_extractor_training.py` keeps normal PPO as the default and switches to
  `DiagnosticPPO` only when `run.training_diagnostics: true`.

## Validation Run

Command:

```bash
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/lightweight_cnn_issue_835_repro.yaml \
  --output-root tmp/issue_835_repro \
  --verbose
```

Environment:

- Date: 2026-04-29
- GPU: NVIDIA GeForce RTX 3070
- CUDA: 12.8
- Python: 3.12.8
- Worker mode: `single-thread`
- Duration: 401.1 seconds

Eval curve:

| Step | Mean reward | Std reward |
| ---: | ----------: | ---------: |
| 4,000 | -53.57 | 54.06 |
| 8,000 | -33.00 | 48.85 |
| 12,000 | 3.38 | 11.08 |
| 16,000 | -11.01 | 12.32 |
| 20,000 | 3.79 | 9.89 |
| 24,000 | -4.45 | 13.66 |
| 28,000 | -4.40 | 7.74 |
| 32,000 | -8.37 | 6.10 |

Diagnostic window summary:

| Timestep | Total grad mean | Total grad max | Extractor grad mean | Combined feature mean abs | Combined feature std | Combined feature max abs |
| -------: | --------------: | -------------: | ------------------: | ------------------------: | -------------------: | -----------------------: |
| 20,480 | 19.402 | 43.956 | 10.057 | 0.479 | 0.767 | 14.293 |
| 22,528 | 22.867 | 56.800 | 11.975 | 0.453 | 0.749 | 12.976 |
| 30,720 | 16.859 | 42.388 | 9.004 | 0.450 | 0.751 | 13.148 |
| 32,768 | 13.714 | 38.585 | 7.259 | 0.450 | 0.747 | 13.504 |

## Conclusion

The issue-193 catastrophic final drop did **not** reproduce on the same seed and 32 K budget in
this rerun. The extractor remained noisy and still ended below its best checkpoint, but the final
reward was `-8.37`, not a collapse near `-64`. Feature magnitudes stayed stable through the
diagnostic window, and gradient norms did not show a late-run spike that explains the original
cliff.

Current verdict: keep `lightweight_cnn` experimental and do not promote it, but treat the issue-193
cliff as a high-variance early-training outcome rather than a deterministic architecture failure.
The bounded triage does not justify an architecture change or learning-rate mitigation yet.

## Follow-Up Boundary

The next useful evidence would be a multi-seed comparison at a longer budget. That belongs in the
broader feature-extractor campaign, not this issue. This issue should close with diagnostics and a
documented non-reproduction verdict.
