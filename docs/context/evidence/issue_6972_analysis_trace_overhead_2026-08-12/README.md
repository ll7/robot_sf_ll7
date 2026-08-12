# Issue #6972 analysis-trace overhead receipt

This packet records a bounded local performance diagnostic for the opt-in
analysis trace in `robot_sf.benchmark.runner.run_episode`. The fixture is a
short synthetic non-map episode; the result is not benchmark, paper-facing,
real-world, or safety evidence.

## Result

The median trace-on overhead fell from 88.4145% at baseline commit
`36cc9fca4c8f0b2325feffafc529c76b91c3e978` to 5.8905% at optimized commit
`2fe5b888b90f1ff16030c6d7860d6175c7ac0bbd`, below the 10% local target. The
trace-on median wall time fell from 23.864 ms to 13.596 ms. The trace-only
compressed payload was 1,070 bytes at baseline and 1,068 bytes after the
optimization; full episode-record gzip sizes are reported separately in the
receipt. Paired outcomes and metrics remained equal, and the control-sequence,
commit-provenance, and artifact-digest checks passed in both arms.

These values are local wall-clock diagnostics on one machine and one fixture,
not a claim about all scenarios or hardware.

## Reproduction

Run the same six-sample measurement at any candidate commit:

```bash
LOGURU_LEVEL=ERROR uv run python scripts/benchmark/measure_analysis_trace_overhead_issue_6972.py \
  --samples 6 --warmups 1 --batches 2 --output /tmp/issue-6972-receipt.json
```

The current script records repeated paired batches and returns
`target_met: null` when same-commit batch medians are unstable. The historical
comparison receipt below predates that repeated-batch contract and remains a
diagnostic v1 receipt:
[analysis_trace_overhead_receipt.json](analysis_trace_overhead_receipt.json).
