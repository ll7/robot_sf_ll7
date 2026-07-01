# Issue #1554 Job 13198 Constraints-First Analysis

This bundle records what can be concluded from the retained job 13198 packet context without rerunning compute or promoting any paper-facing claim.

- evidence status: `diagnostic-only evidence-boundary artifact`
- accepted packet: `issue1554-job13198-constraints-artifact-20260701`
- issue: `ll7/robot_sf_ll7#1554`
- adjacent dependency: `ll7/robot_sf_ll7#3216`
- job id: `13198`
- campaign: `2026-06-issue1554-s20-h500-split-mem180-run`
- config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`
- public evidence commit: `12a188de7246aad3b9088ea76e6a25a20029f976`
- Slurm result: `COMPLETED` / `0:0`
- new Slurm submission: `none`
- paper or dissertation claim edits: `none`

## Claim Boundary

This is not a paper-grade, dissertation-facing, or benchmark-strength ranking claim. It is a compact artifact bundle that separates completed-job constraints metadata, blocked constraints-first metric extraction, adjacent-rank classifications, diagnostic-only Social Navigation Quality Index (SNQI) observations, and follow-up blockers.

The public packet says job 13198 completed in `01:42:35` and produced nine `ok` planner rows with 960 episodes per planner. The same packet records a soft SNQI contract warning: `SNQI contract status=fail with snqi_contract.enforcement=warn; campaign marked soft contract warning.` That warning blocks SNQI-backed rank interpretation.

## Inputs Checked

- present: `docs/context/evidence/issue_1554_slurm_evidence_2026-06-30/README.md`
- present: `docs/context/evidence/issue_1554_slurm_evidence_2026-06-30/packet.json`
- present: `tests/benchmark/fixtures/issue1554_slurm_evidence/jobs_13192_13198_13203.json`
- present: `/home/luttkule/git/robot_sf_ll7-private-ops/ops/jobs/campaigns/2026-06-issue1554-s20-h500.yaml`
- missing on this worker: `/home/luttkule/git/robot_sf_ll7-private-ops/ops/jobs/metrics/13198.json`
- missing on this worker: `/home/luttkule/git/robot_sf_ll7-private-ops/ops/jobs/decision_packets/2026-06-30_slurm_ready_scout_issue1554-job13198-constraints-first-analysis-20260630.md`

Because the packet-named private metrics and decision packet were not available on this remote PC, this bundle does not reconstruct per-scenario success, collision, near-miss, or adjacent-rank values. Those rows are recorded as blocked in the CSV files instead of being inferred.

## Constraints-First Metrics

`constraints_first_metrics.csv` preserves the completed-job constraints metadata available from the public evidence packet:

- nine matrix/planner rows were reported `ok`;
- 960 episodes per planner were reported;
- no compute submission, artifact deletion, or claim edit was performed in this packet;
- success, collision, and near-miss aggregate values are marked `blocked_missing_private_retained_metrics` because the retained job metrics JSON was absent from this worker.

## Adjacent Rank Classifications

`adjacent_rank_claims.csv` classifies adjacent-rank statements without promoting any rank:

- SNQI adjacent-rank statements fail closed because the job 13198 packet carries the soft SNQI contract warning;
- constraints-first adjacent-rank statements remain blocked until the retained per-scenario constraints metrics and rank table are available;
- no manuscript table, dissertation table, or planner-family ordering claim is promoted.

## Diagnostic-Only SNQI Observation

The only SNQI observation preserved here is diagnostic: job 13198 completed but the SNQI contract warning prevents using SNQI as a rank metric for promoted claims. This supports analysis-before-rerun routing; it does not establish a rank.

## Seed-Budget Compute Decision

More seed-budget compute is not justified from this packet alone. The immediate next dependency is recovery or regeneration of the missing retained metrics/decision-packet evidence for job 13198, then a targeted #3216 constraints-first closeout only if that recovered analysis exposes a specific gap. A duplicate S20/H500 run should not be scheduled from this artifact alone.

## Files

- `packet.json`: machine-readable provenance, input availability, claim boundary, and follow-up decision.
- `constraints_first_metrics.csv`: completed-job constraints metadata plus blocked constraints metric rows.
- `adjacent_rank_claims.csv`: fail-closed or blocked adjacent-rank classifications.
- `artifact_inventory.json`: checksums and file inventory for this compact bundle.
