# Issue #1239 Human-Model Transfer Evidence

Issue: [#1239](https://github.com/ll7/robot_sf_ll7/issues/1239)
Context note: [docs/context/issue_1239_human_model_transfer.md](../../issue_1239_human_model_transfer.md)

This bundle preserves compact evidence from the first human-model transfer smoke slice on
May 18, 2026.

## Files

* `preflight_matrix_summary.json` - Four planner rows from
  `configs/benchmarks/human_model_transfer_smoke_v1.yaml`, including explicit
  `human_model_variant` and `human_model_source` fields.
* `smoke_planner_rows.json` - Compact smoke-run row summary. The native Social Force row produced
  one episode; the three Social-Navigation-PyEnvs proxy rows failed closed because the upstream
  checkout was not present locally.

## Checksums

```text
9cb0c50815ae36525806030cf1547315ca7979d626439008cc96d858bfb2f246  preflight_matrix_summary.json
d330e2227e0c8cc6cb97770973ebfa4819fdb716526470637997d2c693fdce17  smoke_planner_rows.json
```
