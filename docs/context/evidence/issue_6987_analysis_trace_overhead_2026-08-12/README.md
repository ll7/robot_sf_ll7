# Issue #6987 analysis-trace measurement-invariance receipt

This packet records a bounded local diagnostic for the opt-in
`analysis_trace: all` path in `robot_sf.benchmark.runner.run_episode`. It is
not benchmark, paper-facing, real-world safety, release, or campaign evidence.
No raw traces are included.

## Result

Both receipts use the same measurement commit (recorded in each receipt),
Linux 6.17.0-35-generic x86_64, and CPython 3.13.13. Each uses two paired batches,
six measured samples per arm per batch, one warmup per arm per batch, and alternating
off/on then on/off arm order.

- The first receipt had batch overheads of 15.9285% and 23.3892%, a 7.4606 percentage-point spread,
  and an aggregate 18.4691% overhead. Its integrity and repeated-batch stability checks passed,
  but one batch exceeded the 10% target, so its decision was `not_met`.
- The independent rerun had batch overheads of 3.6358% and 9.5253%, a 5.8895 percentage-point
  spread, and an aggregate 8.3275% overhead. Its integrity and repeated-batch stability checks
  passed, so its local decision was `met`.

The two sequential same-commit runs disagree on the 10% decision despite each being internally
stable. That prevents a general or campaign-facing conclusion that the target is reproducibly met
on this host. The current status is measurement variance bounded but unresolved; the analysis-trace
profile remains opt-in, and no trace-builder optimization is justified by these receipts alone.

The tracked [reconciliation packet](analysis_trace_overhead_reconciliation.v1.json) keeps both
decisions separate and classifies the comparison as `unavailable`: these legacy receipts do not
record cache state or numerical-thread settings. The reconciliation CLI therefore refuses to
average them or promote a host/order explanation. Future measurements record that context in the
receipt itself.

The earlier optimized comparison receipt in
`issue_6972_analysis_trace_overhead_2026-08-12/` is historical diagnostic evidence. Its receipt
records Linux 6.8 / CPython 3.13.14 at optimized commit `2fe5b888...`, not macOS and not the
measurement commit here, so it is not a like-for-like replacement for these repeated measurements.

## Contract

The receipts identify Issue #6987 as the measurement follow-up and Issue #6972 as the source fixture.
The harness preserves paired outcomes and metrics, requested/applied control-sequence digests,
trace Git commit hashes, trace artifact/provenance digest matches, deterministic compressed-size
summaries, and compact raw timing samples. A target decision is admissible only when the integrity
checks pass, repeated same-commit batch medians stay within the declared 25 percentage-point
absolute overhead-fraction spread tolerance, and every repeated batch is within the 10% target.
A stable batch above 10%
therefore yields `target_met: false`; otherwise the receipt reports an inconclusive decision.

## Reproduction

From this worktree, run:

```bash
LOGURU_LEVEL=ERROR scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/benchmark/measure_analysis_trace_overhead_issue_6972.py \
  --samples 6 --warmups 1 --batches 2 \
  --output /tmp/issue-6987-analysis-trace-overhead.json
```

The command records the exact repository commit, environment, fixture, arm order, per-batch
timings, compressed sizes, and integrity checks. The durable receipts are
[the first run](analysis_trace_overhead_receipt.v2.stable-run.json) and
[the independent rerun](analysis_trace_overhead_receipt.v2.rerun.json).

To reconcile multiple receipts without averaging incompatible timing contexts, run:

```bash
uv run python scripts/analysis/reconcile_analysis_trace_overhead_issue_6987.py \
  docs/context/evidence/issue_6987_analysis_trace_overhead_2026-08-12/analysis_trace_overhead_receipt.v2.stable-run.json \
  docs/context/evidence/issue_6987_analysis_trace_overhead_2026-08-12/analysis_trace_overhead_receipt.v2.rerun.json \
  --output /tmp/issue-6987-reconciliation.json
```

Exit status 2 means the comparison is unavailable or context-incomplete; it is not a failed
benchmark. The packet remains diagnostic-only and never authorizes a campaign or optimization.
