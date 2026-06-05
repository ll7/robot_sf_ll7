# Issue #2309 AMV Trace Export Probe

This compact evidence directory records why the #2309 AMV-specific trace export failed closed.

- `summary.json` is the durable summary.
- Raw rerun JSONL is disposable local probe output and is not committed.
- No `simulation_trace_export.v1` artifact was promoted because the regenerated benchmark JSONL had
  no step frames for conversion.
