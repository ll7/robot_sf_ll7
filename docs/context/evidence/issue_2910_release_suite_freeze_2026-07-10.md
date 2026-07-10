# Issue #2910 Benchmark v0.1 Four-Suite Freeze (2026-07-10)

This note records the first concrete four-suite roster for benchmark v0.1 and keeps publication
blocked until the release evidence is regenerated in the maintainer-approved freeze window.

Evidence status: **configuration contract only; not benchmark-run or publication evidence**.

## Frozen Definitions

| Required suite | Frozen config | Membership | Metadata references |
| --- | --- | --- | --- |
| Nominal | `configs/benchmarks/paper_experiment_matrix_v1.yaml` | 37 `scenario_cert.v1` eligible scenarios | 6/6 resolve |
| Stress | `configs/benchmarks/issue_1344_paired_stress_primary.yaml` | 9 policy-routed stress-only scenarios | 6/6 resolve |
| Adversarial | `configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml` | `crossing_ttc`, `classic_head_on_corridor` | 6/6 resolve |
| Autonomous mobile vehicle (AMV) specific | `configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml` | 4 nominal AMV calibration scenarios | 6/6 resolve |

The canonical machine-readable surface is
`configs/benchmarks/releases/benchmark_v0_1_release_suites.yaml`. It explicitly excludes the two
geometrically infeasible scenarios from nominal and stress membership and keeps the accepted
stress-only routing separate from nominal evidence.

The reversible selection assumption is recorded in the manifest: nominal and stress membership
comes directly from the accepted PR #4671 policy; the adversarial suite reuses the frozen #1500
two-family definition and #1502 evidence; the AMV-specific suite reuses the executed #1344 paired
nominal configuration instead of the proposal-only #2456 suite. The AMV choice remains simulation
evidence and makes no hardware-calibration claim.

## Acceptance Audit

| Issue #2910 criterion | Evidence | Status |
| --- | --- | --- |
| Freeze nominal, stress, adversarial, and AMV-specific suites. | Committed four-suite manifest plus `test_committed_release_suite_manifest_freezes_expected_roster`. | Met for configuration membership. |
| Included scenarios have contracts and certification status. | Each suite has non-empty, durable `scenario_contract` and `scenario_certification` references; all 24 required references dereference. | Partial: dereference proof does not interpret owner schemas or prove per-scenario semantic coverage. |
| Planner rows fail closed on fallback, degraded, and unavailable states. | PRs #4536 and #4606; suite manifest references the tracked release/adversarial/AMV row-status surfaces. | Met for existing gate behavior; fresh release rows remain pending. |
| Claims link to durable artifacts. | Every suite declares a tracked artifact manifest and the dereference report resolves all four. | Met for pointers; fresh #4364 artifacts do not exist yet. |
| Release summary states what is and is not supported. | PR #3296 claim matrix and PR #4648 integration report. | Partial: publication badge is intentionally blocked pending #4364. |
| Unavailable intended evidence stays fail-closed. | Manifest sets `publication_status: blocked_pending_release_rebase`; PR #4671 preserves `policy_accepted_blocked_pending_rebase`. | Met. |

## Validation

```bash
uv run pytest tests/benchmark/test_release_suite_freeze_manifest.py \
  tests/benchmark/test_release_suite_contract.py \
  tests/benchmark/test_release_suite_reference_validation.py -q

uv run python scripts/validation/check_release_suite_contract.py \
  --manifest configs/benchmarks/releases/benchmark_v0_1_release_suites.yaml \
  --base-dir . --json
```

Expected integration result: four structurally complete suites, 24 resolved metadata references,
zero dereference blockers. This proves configuration and path integrity only.

## Remaining Blockers And Next Empirical Action

- Semantic validation must confirm each referenced owner covers the selected suite membership; a
  parseable file is not proof of schema or evidence completeness.
- Issue #4364 explicitly defers release regeneration and publication-badge work until the freeze
  window. The regenerated matrix must replace the current 0.0.2-derived evidence before the
  publication gate may pass.
- Next empirical action: after the #4364 timing gate opens, regenerate the release evidence from
  the reviewed release candidate, validate this frozen roster against the fresh rows, and rerun the
  existing publication gate. No campaign or release action was performed for this note.

Fallback, degraded, adapter, unavailable, and diagnostic rows remain caveats or exclusions; this
freeze does not promote them to benchmark success or support deployment-safety claims.
