<!-- AI-GENERATED (robot_sf#7254, 2026-08-17) - NEEDS-REVIEW -->

# Issue #7254 matched training-optimization smoke

This directory publishes the corrected compact manifest for the bounded predictive-planner
FP32/loader/AMP smoke. It supersedes the original PR #7263 handoff for provenance purposes.

## Claim boundary

- Evidence tier: `diagnostic-only`.
- Scope: one deterministic fixture, one NVIDIA GeForce RTX 3080, two repeats per arm.
- The result supports only bounded implementation-level training-signal equivalence and a
  diagnostic throughput observation.
- It does not establish policy equivalence, benchmark improvement, general GPU speedup, or a
  paper-facing claim.
- Both optimized arms were slower than the control in this run; no default-performance
  recommendation is supported.

## Frozen contract and identity

- Runtime commit: `75c8c51b9d9bbd544dd49763662c80360a2788a2`.
- Config: `configs/training/predictive/predictive_optimization_smoke_issue_7254.yaml`.
- Config SHA-256: `c9d124c935cbc8f0e444ec1e45ffbd6724d787a84883dad641716194064a1942`.
- Dataset SHA-256: `8b8f30ca85a493805277c2b12ae8390dedf60fc9671e8427974e7e2d7bdcd434`.
- Split SHA-256: `a037d8e164ea8275c7b3b52b6f5dcca0a2a19fe1e36ee798b85528158e2a9019`.
- Four identical epoch-order digests, update counts, and environment fingerprints are recorded
  in [`summary.json`](summary.json).
- The tolerance contract was copied before arm execution and used unchanged: one warm-up epoch,
  three measured epochs, two control repeats, multiplier `3.0`, and absolute floor `0.001`.

## Result

| Arm | Repeats | Examples/s after warm-up | Peak allocated bytes | Curve result |
| --- | ---: | ---: | ---: | --- |
| `fp32_control` | 2 | 30,707.54 | 17,415,168 | control |
| `fp32_loader` | 2 | 24,685.93 | 17,415,168 | equivalent |
| `amp_loader` | 2 | 22,304.83 | 17,344,000 | equivalent |

Terminal classification: `equivalent_smoke`. All six checkpoints passed strict model loading.
The complete compact machine-readable handoff is [`summary.json`](summary.json); the raw logs,
checkpoints, and fixture remain ignored worktree-local output as documented in
[`artifact_provenance.json`](artifact_provenance.json).

## Reproduction

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/training/run_predictive_optimization_smoke.py \
  --config configs/training/predictive/predictive_optimization_smoke_issue_7254.yaml \
  --output-root output/review_issue7254_corrected_final
```
