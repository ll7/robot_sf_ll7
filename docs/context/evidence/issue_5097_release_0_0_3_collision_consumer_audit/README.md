<!-- AI-GENERATED (robot_sf#5097, 2026-07-14) - NEEDS-REVIEW -->

# Issue #5097 — Release 0.0.3 Collision-Consumer Audit

## Plain-language summary

The public release 0.0.3 bundle satisfies the narrow collision-data criteria in
[Issue #5097](https://github.com/ll7/robot_sf_ll7/issues/5097). Its archive and all 87 signed
payload files verify, all 20,160 episode rows carry the typed event-ledger contract, and all 9,022
exact-collision rows have a positive collision count, an unsuccessful outcome, and an active
Social Navigation Quality Index (SNQI) collision penalty. The canonical SNQI implementation
reproduces every stored episode score exactly.

This is a successor slice for #5097; the broader successor-release recomputation and release
re-base remain tracked by [Issue #4364](https://github.com/ll7/robot_sf_ll7/issues/4364). It does
not certify the release as a whole: the bundle contains the release-wide inconsistencies listed
below, and its own SNQI contract status is `fail`.

## Classification and claim boundary

- `schema_version`: `release_collision_consumer_reconciliation.v1`
- `status`: `pass`
- `result_classification`: `collision_consumer_reconciled`
- `evidence_grade`: `diagnostic-only`
- `diagnostic_only`: `true`
- `benchmark_promotion`: `false`
- `paper_facing`: `false`
- `provenance`: `seeds`: S30 release seed budget (30 seeds per arm; 1,440 rows per arm); `config`:
  the signed asset's embedded `payload/release/release_manifest.resolved.json` and named
  `paper_experiment_matrix_v2_h600_s30_extended` release matrix; `hash`: archive SHA-256 and
  pinned SNQI weights/baseline SHA-256 are recorded in [`summary.json`](summary.json).
- `claim_boundary`: `Diagnostic-only reconciliation of exact per-episode collision outcomes with collision counts, the binary-success collision gate, and the canonical SNQI collision term. A pass is not release-wide benchmark success, planner-ranking evidence, SNQI contract validity, or a paper claim.`

## Acceptance criterion to evidence

| Issue #5097 acceptance criterion | Evidence | Result |
| --- | --- | --- |
| Extend the release 0.0.2 withdrawal boundary and fail-closed checker to SNQI and success rate. | Merged PR [#5133](https://github.com/ll7/robot_sf_ll7/pull/5133) added both derived consumers to `docs/context/evidence/issue_3482_release_0_0_2_collision_count_boundary/manifest.json`, validates them with `scripts/validation/check_release_0_0_2_collision_count_boundary.py`, and covers malformed/premature-promotion cases. | Met. |
| In the successor release, recompute SNQI and success from exact collision outcomes. | Release [0.0.3](https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.3) plus [`summary.json`](summary.json): 20,160/20,160 SNQI values reproduce under the pinned v3 weights/baseline; 9,022 exact-collision rows equal positive total collision counts, contain typed events, set `metrics.success=false`, and activate a positive SNQI collision penalty. | Met at the collision-consumer boundary; broader release re-base/recomputation remains with [#4364](https://github.com/ll7/robot_sf_ll7/issues/4364). |
| Align the derived collision aggregate with the runtime collision envelope, or explicitly distinguish the measures. | PR #5133 changed the executable regression to use the runtime radius-sum contact envelope and retained explicit pedestrian/obstacle/agent collision components. The release audit requires those components to sum to `total_collision_count`, its `collisions` alias, and the typed exact event. | Met. |
| Add a regression proving runtime collision termination yields `collisions >= 1` and `success = 0`. | PR #5133 added `tests/benchmark/test_collision_runtime_termination_metrics.py`, including a real runtime step at radius-sum contact and direct collision/SNQI/success assertions. The successor-release audit independently checks the same implication over every public exact-collision row. | Met. |

## Reproduction

Download the named asset from release 0.0.3 and verify its SHA-256 digest is
`3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7`, then run:

```bash
uv run python scripts/validation/check_release_collision_consumers.py \
  --bundle <downloaded-release-asset.tar.gz> \
  --expected-bundle-sha256 3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7 \
  --expected-release-tag 0.0.3 \
  --expected-rows 20160 \
  --expected-arms 14 \
  --expected-rows-per-arm 1440 \
  --source-url https://github.com/ll7/robot_sf_ll7/releases/download/0.0.3/paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final_publication_bundle.tar.gz \
  --output docs/context/evidence/issue_5097_release_0_0_3_collision_consumer_audit/summary.json
```

The checker reads the archive without extracting it and fails closed on archive/hash drift,
unsigned or missing payloads, cardinality drift, duplicate episode identities, event-ledger
inconsistency, collision-count disagreement, collision-success disagreement, stored SNQI drift, or
an inactive collision penalty.

## Release-wide warnings outside #5097

- `release/release_result.json` is stale relative to the rebuilt campaign summary (18,720 versus
  20,160 episodes; 13 versus 14 successful arms; invalid/failure versus valid/success status).
- All episode ledgers cite commit `a307ef276d701f8d14dead1aa0513f44ee97c0b0`, while the publication
  manifest/tag cites `e2ac534c9d6bb750346b1e0724638c91306e410a`.
- The campaign reports `snqi_contract_status=fail`; this audit proves only that the collision term
  consumes the exact collision outcome, not that SNQI is a valid headline scalar or ranking.
- One non-collision row records both `goal_reached=true` and `timeout=true`. The bundle omits the
  reached-goal step needed to re-evaluate that separate timing boundary; it does not affect the
  collision-implies-unsuccessful criterion.
- Publication URLs/DOI metadata still contain placeholders.

No benchmark campaign was rerun, no Slurm/GPU job was submitted, and no paper/dissertation claim
was edited or promoted for this audit.
