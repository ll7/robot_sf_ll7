# Issue #6972 analysis-trace overhead receipt

This packet records a bounded local performance diagnostic for the opt-in
analysis trace in `robot_sf.benchmark.runner.run_episode`. The fixture is a
short synthetic non-map episode; the result is not benchmark, paper-facing,
real-world, or safety evidence.

## Result

The median trace-on overhead fell from 92.2350% at baseline commit
`36cc9fca4c8f0b2325feffafc529c76b91c3e978` to 6.8860% at optimized commit
`b2cfce837e2362497d9798a65ff64439aefbfefe`, below the 10% local target. The
trace-on median wall time fell from 24.859 ms to 13.911 ms. Paired outcomes and
metrics remained equal, and the control-sequence, commit-provenance, and
artifact-digest checks passed in both arms.

These values are local wall-clock diagnostics on one machine and one fixture,
not a claim about all scenarios or hardware.

## Reproduction

Run the same six-sample measurement at any candidate commit:

```bash
LOGURU_LEVEL=ERROR uv run python scripts/benchmark/measure_analysis_trace_overhead_issue_6972.py \
  --samples 6 --warmups 1 --output /tmp/issue-6972-receipt.json
```

The script records exact samples, deterministic compressed sizes, paired
outcome/metric digests, control-sequence digests, and provenance checks. The
comparison receipt is [analysis_trace_overhead_receipt.json](analysis_trace_overhead_receipt.json).
