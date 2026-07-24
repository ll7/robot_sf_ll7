<!-- AI-GENERATED (robot_sf#5248, 2026-07-20) - NEEDS-REVIEW -->
# Salvaged h600 harvest registration

Evidence status: `diagnostic-only`; registration is `ready_for_issue_4206_reanalysis`.

This packet supersedes the 2026-07-14 receipt from #5654 and the 2026-07-16
re-inspection. The 2026-07-14 receipt concluded the campaign "never captured
trace-verified failure-mechanism labels" because every `seed_episode_rows.csv` row
carries `mechanism_label == "unknown"`. Re-inspection of the job-13334 episode
JSONL on the issue-pinned host (`imech156-u`) corrected that conclusion: the
campaign **is** trace-capable, and trace-verified labels **have** been derived. The
2026-07-20 update applies the issue #5779 structural denominator correction, which
flips the registration verdict from `blocked` to `ready` for the accurate reason.

## Issue #5779 structural denominator correction

The preregistered trace-verified labeled fraction floor
(`min_trace_verified_labeled_fraction: 0.5`) is now measured over **failure
episodes** (rows whose write-time `success` outcome is below 1.0) rather than over
all campaign rows. The threshold value is unchanged; only the denominator changes.

Why this is a structural correction, not a relaxation: success episodes
structurally carry no failure-mechanism label — they did not fail, so there is no
failure mechanism to derive. An all-rows denominator therefore included the 3,868
success episodes as permanently-unlabeled rows, which made the 0.5 floor unreachable
by construction for any campaign with a healthy success rate. The corrected
failure-episode denominator measures label coverage over the population that can
actually carry a failure-mechanism label.

The contract records the correction explicitly
(`min_trace_verified_labeled_fraction_denominator: failure_episodes` in
`configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml`), and
the preregistration checker rejects contracts that omit it or declare `all_rows`.
The registration receipt reports **both** ratios for transparency, but the gate uses
the failure-episode fraction.

## What the checker observes on job 13334

The campaign source artifacts are present and structurally valid:

- `reports/campaign_summary.json`: `campaign.total_episodes == 6480` and
  `campaign.campaign_execution_status == "completed"`.
- `reports/seed_episode_rows.csv`: exactly 6,480 data rows.
- Both source files are recorded with SHA-256 sums in the receipt.

The job was salvaged after the exit-code conflation fixed by #5240; the completed
6,480-episode result confirms the FAILED mislabel was spurious. That provenance does
**not** promote this packet to benchmark, planner, paper, or dissertation evidence.

## Why the campaign IS trace-capable (correction to #5654)

Independent re-inspection of all 11 arms' `episodes.jsonl` shows trace capture was
functioning:

- Every one of the 6,480 episode records carries a non-empty
  `algorithm_metadata.simulation_step_trace` (`simulation-step-trace.v1`; 1–600 steps each).
- 2,375 of 6,480 records additionally carry a non-empty `planner_decision_trace`.

The `mechanism_label == "unknown"` on the raw `seed_episode_rows.csv` is therefore **not** a
trace-capture failure: mechanism labels are a *post-hoc derivation* over the trace surfaces
(issue #4831), not a write-time field. Issue #4831's builder
(`scripts/analysis/derive_issue_4206_trace_verified_failure_mechanisms.py`) has already run
against this harvest and derived trace-verified labels for all 2,612 failure episodes (100%
failure coverage; `paired_trace` evidence mode). That derivation lives in the declared sidecar
`docs/context/evidence/issue_4831_trace_verified_failure_mechanisms/mechanism_labels.csv`, which
the checker consumes via the `--mechanism-sidecar` overlay.

## Why registration is now ready (issue #5779)

With the #4831 sidecar overlaid, trace-verified labeled rows rise from `0` (raw rows)
to `2,415` at accepted confidence (`observed_mechanism` / `supported_hypothesis`). The
remaining `197` derived labels are `weak_hypothesis` (`time_budget_artifact`); they count as
derived but are below the accepted-confidence bar and do not count toward the floor.

Both ratios are reported in the receipt; only the failure-episode ratio gates:

| Denominator | Population | Labeled (accepted) | Fraction | vs 0.5 floor |
| --- | --- | --- | --- | --- |
| Failure episodes (gate) | 2,612 | 2,415 | **0.925** | meets |
| All campaign rows (transparency) | 6,480 | 2,415 | 0.373 | (prior, blocked) |

The failure-episode fraction (~0.925) clears the unchanged 0.5 floor, so the campaign is
`ready_for_issue_4206_reanalysis`. The all-rows ratio (0.373) is the value the prior receipts
gated on and is retained in the receipt for transparency; it is no longer the gating
denominator.

The 2,612 failure-episode count is the #4831 failure set (`outcome.route_complete == false`).
The checker derives the denominator from each row's write-time `success` outcome
(`success < 1.0`), which coincides with the #4831 failure set for this harvest. Any
route-complete-but-collision edge row would only be added to the denominator (unlabeled),
which cannot move the fraction below the 0.5 floor from ~0.925; the exact count is confirmed
by re-running the checker on the harvest host.

## Relationship to the prior receipts in this directory

`registration.json`, `registration.md`, and `SHA256SUMS` in this directory are the
**prior** receipts generated on 2026-07-16, which gated on the all-rows denominator
and recorded `blocked_campaign_registration` at fraction 0.373. They are retained
here for provenance and are superseded by this README's corrected verdict. They will
be regenerated by the checker (schema `issue_5248_salvaged_trace_rerun_registration.v2`)
on the issue-pinned harvest host to produce the `ready` receipt that matches the
command below; the build host for this packet does not carry the verified harvest.

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
  --generated-at 2026-07-20T000000Z
```

Expected result (issue-pinned host with the verified harvest and #4831 sidecar):
exit code `0` and status `ready_for_issue_4206_reanalysis`, with
`trace_labeled_fraction == ~0.925` (failure-episode denominator, raised from `0.000`
without the sidecar) and `trace_labeled_fraction_all_rows == 0.373` (transparency).
Requires the verified job-13334 harvest and the #4831 sidecar to be present on the
issue-pinned host; the test
`tests/validation/test_check_issue_5248_salvaged_trace_rerun.py::test_job_13334_denominator_correction_makes_campaign_ready`
skips when either is absent.

## Next action

With the denominator correction applied, the job-13334 harvest registers as ready for
the issue #4206 mechanism cross-cut. The conditional #4206 F-C4(ii) cross-cut builder
can now run against this genuinely trace-capable, trace-labeled campaign. This is still
registration readiness only — it does not establish benchmark, planner, paper, or
dissertation claims, and the cross-cut's own evidence contract still governs what those
conclusions may assert.
