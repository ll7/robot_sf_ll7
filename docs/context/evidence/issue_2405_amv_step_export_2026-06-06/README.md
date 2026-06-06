# Issue #2405 AMV Step-Export Evidence

This bundle preserves compact proof that the Issue #2168 AMMV/default Social Force path can now
regenerate selected aggregate rows with step frames and convert one selected row per side into
loader-valid `simulation_trace_export.v1`.

- `summary.json`: machine-readable proof summary.
- `trace_export_rows.csv`: compact row table for the default and AMMV selected exports.
- Claim boundary: diagnostic trace-export proof only; not benchmark-strength evidence and not a
  raw trace publication.
- Raw JSONL and trace JSON outputs were generated under an ignored worktree-local artifact
  directory and are not tracked.
