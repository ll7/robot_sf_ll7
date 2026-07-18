<!-- AI-GENERATED (robot_sf#5248, 2026-07-16) - NEEDS-REVIEW -->
# Salvaged h600 harvest registration

Evidence status: `diagnostic-only`; registration is `blocked_campaign_registration`.

This packet supersedes the 2026-07-14 receipt from #5654. That receipt concluded the campaign
"never captured trace-verified failure-mechanism labels" because every `seed_episode_rows.csv` row
carries `mechanism_label == "unknown"`. Re-inspection of the job-13334 episode JSONL on the
issue-pinned host (`imech156-u`) corrects that conclusion: the campaign **is** trace-capable, and
trace-verified labels **have** been derived — the receipt now consumes that derivation.

## What the checker actually observed

The campaign source artifacts are present and structurally valid:

- `reports/campaign_summary.json` read; `campaign.total_episodes == 6480` and
  `campaign.campaign_execution_status == "completed"`.
- `reports/seed_episode_rows.csv` read; exactly 6,480 data rows.
- Both source files are recorded with SHA-256 sums in `registration.json`.

The job was salvaged after the exit-code conflation fixed by #5240; the completed 6,480-episode
result confirms the FAILED mislabel was spurious. That provenance does **not** promote this packet
to benchmark, planner, paper, or dissertation evidence.

The host-pinned revalidation on `imech156-u` on 2026-07-16 reported:

```text
total_episodes=6480, status_completed
status: blocked_campaign_registration
```

The first line verifies that the harvest itself is complete. The second is the separate,
fail-closed trace-label registration decision described below; it does not invalidate the harvest.

## Why the campaign IS trace-capable (correction to #5654)

Independent re-inspection of all 11 arms' `episodes.jsonl` shows trace capture was functioning:

- Every one of the 6,480 episode records carries a non-empty
  `algorithm_metadata.simulation_step_trace` (`simulation-step-trace.v1`; 1–600 steps each).
- 2,375 of 6,480 records additionally carry a non-empty `planner_decision_trace`.

The `mechanism_label == "unknown"` on the raw `seed_episode_rows.csv` is therefore **not** a
trace-capture failure: mechanism labels are a *post-hoc derivation* over the trace surfaces (issue
#4831), not a write-time field. Issue #4831's builder
(`scripts/analysis/derive_issue_4206_trace_verified_failure_mechanisms.py`) has already run against
this harvest and derived trace-verified labels for all 2,612 failure episodes (100% failure
coverage; `paired_trace` evidence mode). That derivation lives in the declared sidecar
`docs/context/evidence/issue_4831_trace_verified_failure_mechanisms/mechanism_labels.csv`, which
this receipt now consumes via the checker's `--mechanism-sidecar` overlay.

## Why registration still stays blocked

Registration remains `blocked_campaign_registration`, but for the **accurate** reason rather than
the "trace capture failed" reason recorded in #5654:

- With the #4831 sidecar overlaid, trace-verified labeled rows rise from `0.000` (raw rows) to
  `2,415` at accepted confidence (`observed_mechanism` / `supported_hypothesis`) — a fraction of
  `0.373` over all 6,480 rows.
- The remaining `197` derived labels are `weak_hypothesis` (`time_budget_artifact`); they count as
  derived but are below the accepted-confidence bar and do not count toward the floor.
- The preregistered minimum (`min_trace_verified_labeled_fraction: 0.5`, measured over all rows) is
  not met. The denominator includes the 3,868 success episodes, which structurally carry no
  failure-mechanism label, so coverage cannot reach 50% of all rows for this campaign. Whether that
  denominator should be restricted to failure episodes is a metric-semantics question explicitly out
  of scope for this registration (#5248 scopes out metric semantics); the floor is honored as
  ratified in #4350.

This is not a "re-run the campaign to capture traces" blocker — the traces and the derived labels
already exist. The actionable gap is the preregistered floor/denominator, not missing capture.

## Reproduction

```bash
uv run python scripts/validation/check_issue_5248_salvaged_trace_rerun.py \
  --campaign-root output/issue4206-13334-harvest/issue4206_trace_capable_h600_rerun_20260704 \
  --job-id 13334 \
  --expected-total-episodes 6480 \
  --preregistration-config \
    configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml \
  --mechanism-sidecar \
    docs/context/evidence/issue_4831_trace_verified_failure_mechanisms/mechanism_labels.csv \
  --output-dir docs/context/evidence/issue_5248_salvaged_h600_registration_2026-07 \
  --generated-at 2026-07-16T125218Z
```

Expected result: exit code `2` and status `blocked_campaign_registration`, with
`trace_labeled_fraction == 0.373` (raised from `0.000` without the sidecar). Requires the verified
job-13334 harvest and the #4831 sidecar to be present on the issue-pinned host.

## Next action

The conditional #4206 mechanism cross-cut still cannot run in a ready state, because the
preregistered trace-label floor (over all rows) is not met. The correct next step is a
maintainer decision on the floor denominator (failure-episode vs all-episode) — a metric-semantics
change — not a new campaign run. Resolving that would let the #4831-derived labels register this
harvest and unblock the #4206 F-C4(ii) cross-cut against the genuinely trace-capable job 13334.
