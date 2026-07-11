# Salvaged h600 harvest registration

Evidence status: `diagnostic-only`; registration is
`blocked_campaign_registration` because the declared compact campaign summary and episode-row
inputs were unavailable in this execution environment.

This packet records the fail-closed receipt for the salvaged trace-capable h600 rerun associated
with job 13334. The checker expected 6,480 episodes and a completed campaign execution status, but
it could not read either `reports/campaign_summary.json` or `reports/seed_episode_rows.csv` from the
declared harvest root. It therefore did not inspect trace-label coverage or authorize the issue
#4206 mechanism cross-cut.

The job was salvaged after the exit-code conflation fixed by #5240. That provenance does not
override the missing-input blockers and does not promote this packet to benchmark, planner,
paper, or dissertation evidence.

## Reproduction

```bash
uv run python scripts/validation/check_issue_5248_salvaged_trace_rerun.py \
  --campaign-root output/issue4206-13334-harvest \
  --job-id 13334 \
  --expected-total-episodes 6480 \
  --preregistration-config \
    configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml \
  --output-dir docs/context/evidence/issue_5248_salvaged_h600_registration_2026-07 \
  --generated-at 2026-07-11T05:16:28Z
```

Expected result for this packet: exit code `2` and status
`blocked_campaign_registration`.

## Next action

Run the same checker where the verified job-13334 harvest is available. Replace this blocked
receipt only after the checker observes the campaign summary, all 6,480 episode rows, completed
execution status, and the preregistered trace-label coverage. Do not copy raw campaign outputs into
Git.
