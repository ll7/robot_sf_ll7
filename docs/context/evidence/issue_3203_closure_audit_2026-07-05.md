# Issue #3203 Closure Audit

Issue: [#3203](https://github.com/ll7/robot_sf_ll7/issues/3203)

This audit maps the open acceptance criteria for the Issue #3203 scenario-horizon table
re-export to merged pull requests and committed evidence as of July 5, 2026. It is a
consolidation slice only: no new benchmark campaign, Slurm job, graphics processing unit (GPU) job,
paper claim, dissertation claim, or metric-definition change was run or made.

## Current Decision

Do not close Issue #3203 yet. The payload/provenance table-readiness slice is complete, but the
broader promotion criteria remain blocked by the Social Navigation Quality Index (SNQI)
rank-alignment contract failure in the July 1 bounded rerun.

The next empirical action is one of:

- repair the SNQI rank-alignment failure and run a fresh bounded campaign, or
- predeclare a narrower claim boundary that excludes SNQI validity, then run a fresh bounded
  campaign under that boundary before any benchmark Results or dissertation wording promotion.

## Acceptance Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Re-export both scenario-horizon table payload files; stale-artifact detector no longer flags missing payloads. | Satisfied for payload/provenance readiness only. | PR [#3263](https://github.com/ll7/robot_sf_ll7/pull/3263) restored the dissertation export payload files, manifest, and checksums. PR [#3733](https://github.com/ll7/robot_sf_ll7/pull/3733) added the re-export readiness preflight. PR [#4093](https://github.com/ll7/robot_sf_ll7/pull/4093) preserved the fresh July 1 bounded rerun packet. PR [#4383](https://github.com/ll7/robot_sf_ll7/pull/4383) added the explicit narrow readiness packet and opt-in checker mode that classifies this source as `table_reexport_ready`. |
| Regenerate manifest and checksums; update ledger, gap report, catalog, and documented command. | Satisfied for diagnostic/provenance use. | PR [#3263](https://github.com/ll7/robot_sf_ll7/pull/3263) regenerated the export bundle manifest and checksums. PR [#4093](https://github.com/ll7/robot_sf_ll7/pull/4093) updated the issue evidence packet, ledger, gap-report, catalog, and reproduce command for the July 1 rerun. PR [#4479](https://github.com/ll7/robot_sf_ll7/pull/4479) corrected stale June 20 PPO-failure wording so the durable docs point at the current July 1 SNQI blocker. |
| Reclassify the dissertation ledger row away from non-claimable. | Satisfied, but only to diagnostic/provenance status. | `docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json` now marks `exported_tables` as `current` and `diagnostic`, with allowed wording limited to Discussion/Limitations and payload provenance. |
| Promote the re-export to benchmark Results, planner ranking, SNQI validity, paper claims, or dissertation claims. | Not satisfied. | `docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01/reports/campaign_summary.json` records row-complete execution (`benchmark_success`, `evidence_status=valid`, 9 rows, 1296 episodes, PPO native contract pass), but the readiness gate remains `diagnostic_only` because `snqi_contract_status=fail` and `snqi_contract_rank_alignment_spearman=-0.19999999999999998`, below the `0.3` pass threshold. |

## Linked Pull Requests Read

- [#3263](https://github.com/ll7/robot_sf_ll7/pull/3263): re-exported scenario-horizon
  dissertation table payloads, manifest, and checksums, but preserved the invalid June 20 caveat.
- [#3442](https://github.com/ll7/robot_sf_ll7/pull/3442): aligned PPO SocNav dictionary
  observations.
- [#3443](https://github.com/ll7/robot_sf_ll7/pull/3443): recorded PPO/SNQI smoke evidence for
  the blocker tracked by Issue #3266.
- [#3713](https://github.com/ll7/robot_sf_ll7/pull/3713): added fail-closed scenario-horizon
  Results readiness checking.
- [#3720](https://github.com/ll7/robot_sf_ll7/pull/3720): backfilled PPO dictionary observation
  keys instead of crashing on missing keys.
- [#3733](https://github.com/ll7/robot_sf_ll7/pull/3733): added stale dissertation table bundle
  re-export readiness preflight.
- [#3824](https://github.com/ll7/robot_sf_ll7/pull/3824): completed the PPO/SNQI smoke-tier
  blocker path referenced by the issue thread.
- [#4093](https://github.com/ll7/robot_sf_ll7/pull/4093): recorded the July 1 fresh bounded rerun
  as diagnostic/provenance evidence.
- [#4383](https://github.com/ll7/robot_sf_ll7/pull/4383): predeclared the narrow table re-export
  claim boundary and added the opt-in `table_reexport_ready` checker path.
- [#4479](https://github.com/ll7/robot_sf_ll7/pull/4479): updated the durable diagnostic boundary
  from the stale June 20 PPO partial failure to the current July 1 SNQI failure.

## Residual Blocker

The remaining blocker is empirical, not a new CPU-only guardrail/checker gap. The July 1 campaign
already repaired PPO row execution and produced complete table payload provenance, but it did not
pass the SNQI ranking contract. A closing PR for the broader issue still needs fresh bounded
campaign evidence that either passes the SNQI contract or follows a predeclared narrower boundary
with a fresh rerun before any Results or dissertation promotion.
