# Issue #1904 Scenario Perturbation Criticality Pilot Evidence

This bundle contains compact diagnostic evidence for issue #1904. It preserves the paired
no-op-versus-route-offset summary from a local pilot over the #1858 manifest after #1903
materialization support landed.

- `summary.json`: compact reviewable summary with materialized variant IDs, pair rows, pair status
  counts, and mean deltas.

Raw episode JSONL, generated scenario matrices, route override files, coverage output, and local
runner summaries remain under ignored `output/` paths and are not mirrored here.

Claim boundary: diagnostic local pilot only; not benchmark-strength or paper-facing evidence.
