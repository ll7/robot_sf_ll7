# Issue #2272 ORCA-Residual Launch Packet Status

Issue: [#2272](https://github.com/ll7/robot_sf_ll7/issues/2272)
Parent issue: [#1475](https://github.com/ll7/robot_sf_ll7/issues/1475)
Date: 2026-06-05
Status: launch-packet status correction; no SLURM submission from this issue.

Compact smoke evidence:
[evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json](evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json)

Source checksum manifest:
[evidence/issue_1475_orca_residual_bc_smoke_12749.SHA256SUMS](evidence/issue_1475_orca_residual_bc_smoke_12749.SHA256SUMS)

Smoke report:
[policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md](policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md)

## Result

Issue #2272 was extracted as a pre-submission launch-packet child, but the owning Issue #1475
smoke rerun was already submitted and finalized before this child was opened. This note therefore
closes the stale pre-submission gap by promoting the compact status artifacts and updating the
SLURM ledger, rather than submitting or rerunning anything.

Decision: `smoke_failed_closed_revise`.

The smoke packet is launchable and has now been exercised, but it must not escalate to
`nominal_sanity`. The ORCA-residual BC candidate produced one smoke row with `success_rate=0.0`,
`collision_rate=0.0`, and `timeout_low_progress`; the wrapper gate correctly failed closed because
success is required before nominal escalation.

## Launch Packet Fields

| Field | Value |
| --- | --- |
| Launch command | `scripts/dev/sbatch_orca_residual_bc_issue1475.sh --episodes 3 --seeds "111:112:113" --no-status --sbatch-arg "--job-name=gse-1475-smoke"` |
| Lineage config | `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` |
| BC config | `configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml` |
| SLURM script | `SLURM/Auxme/issue_1475_orca_residual_bc.sl` |
| Commit under smoke | `5faaa318d609f87730757d7fbda65b799178b5c5` |
| Job id | `12749` |
| Output root | `output/slurm/issue1475-orca-residual-bc-job-12749` |
| Durable status | Compact summary/report are tracked; raw output paths remain local and non-durable. |
| Nominal escalation | Blocked. |

## Diagnostics Schema

The tracked summary records the required smoke-gate fields for this status step:

- job id, branch, commit, command, configs, partition, QoS, exit state, and elapsed time;
- lineage packet and dry-run preflight status;
- `success_rate`, `collision_rate`, `near_miss_rate`, `termination_reason_counts`, and
  `failure_mode_counts`;
- shield decision count, shield override rate, hard-constraint violation rate, and
  `nominal_escalation_allowed`;
- local output paths plus checksums for the summary JSON, smoke JSONL, BC policy zip, and dataset
  NPZ; the standalone checksum manifest also records the stdout/stderr log hashes.

## Recommendation

Do not rerun the same smoke packet or submit `nominal_sanity` unchanged. The next useful action is
to revise the ORCA-residual BC candidate or the smoke target for low-progress timeout behavior, then
rerun the same bounded smoke gate only after that revision.

## Claim Boundary

This is launch-packet and diagnostic smoke status only. It is not learned-residual success
evidence, benchmark-strength evidence, or paper-facing hybrid-learning evidence. Local output files
remain non-durable unless promoted separately.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
scripts/dev/sbatch_orca_residual_bc_issue1475.sh --dry-run --no-status
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
