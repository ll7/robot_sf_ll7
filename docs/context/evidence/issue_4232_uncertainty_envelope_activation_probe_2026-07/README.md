# Issue #4232 Uncertainty-Envelope Activation Probe

This compact evidence bundle records a CPU-local diagnostic slice for issue #4232. It demonstrates
that the `prediction_mpc` uncertainty-envelope telemetry can observe actual nonzero-alpha mechanism
activation on a named predeclared scenario row while keeping every row excluded from
benchmark-strength evidence.

Command:

```bash
uv run python scripts/benchmark/run_issue_4232_uncertainty_envelope_alpha_sweep.py \
  --packet configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml \
  --output-dir output/issue_4232_activation_francis_crowd \
  --scenario-id francis2023_crowd_navigation \
  --max-seeds 1 \
  --json
```

Acceptance evidence:

| Criterion | Evidence | Boundary |
| --- | --- | --- |
| Define benchmark or campaign scope before making uncertainty-envelope performance or safety claims. | `pre_registration_packet.json` preserves the #4232 packet used by the diagnostic run. | Diagnostic-only activation probe, not a campaign. |
| Record seeds, scenario set, planner settings, runtime constraints, and artifact provenance. | `metadata.json`, `alpha_arm_metric_table.csv`, `runtime_cost_report.csv`, and `SHA256SUMS` record seed `111`, scenario `francis2023_crowd_navigation`, planner `prediction_mpc`, alpha arms, and checksums. | Raw episode JSONL remains under ignored `output/` and is not copied here. |
| Update claim map, benchmark report, or paper-facing text only if evidence supports the boundary. | `claim_readiness.md` keeps the result diagnostic-only and not ready for benchmark-strength, conformal, real-world safety, deployment, paper, or dissertation claims. | No claim map, benchmark report, paper, or dissertation text is changed. |
| Keep fallback/degraded execution excluded from benchmark-strength evidence. | `row_status_audit.csv` shows all rows are `diagnostic_only`, with zero benchmark-strength rows. | Fallback/degraded/not-available/failed/blocked remain ineligible by packet policy. |
| Resolve the latest mechanism-activation blocker after PR #4576. | `envelope_activation_diagnostics.json` records the nonzero `envelope_on_alpha_0p10` row as `activated`, with `effective_radius_used_by_planner=true` and `envelope_activation_count=1254285`. | This proves mechanism activation exists on a suitable diagnostic row; it does not prove a safety or performance improvement. |

The probe does not run a full benchmark campaign, does not submit Slurm/GPU work, and does not
support paper, dissertation, conformal-calibration, deployment, real-world safety, or generalized
planner-superiority claims.
