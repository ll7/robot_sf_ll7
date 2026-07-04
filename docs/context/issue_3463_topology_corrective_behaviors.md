# Issue #3463 Topology Corrective Behaviors Integration Report

Issue: [#3463](https://github.com/ll7/robot_sf_ll7/issues/3463)

Status: current integration report, diagnostic-only evidence.

## Plain Summary

Issue #3463 converted the topology-guided local-planner lane from a revise-only
diagnostic into guarded, testable implementation surfaces, but it does not promote
the lane as benchmark or paper evidence.

## Claim Boundary

This report consolidates the merged implementation slices for Issue #3463 and maps them
to the issue acceptance criteria. The strongest supported claim is diagnostic-only
implementation readiness: topology candidate availability, command arbitration,
route-progress accounting, near-parity thresholds, and a bounded sensitivity packet
now have explicit code, config, or test surfaces. Broad benchmark improvement,
planner promotion, and paper-facing safety or efficiency claims remain out of
scope. The benchmark-facing enabled-versus-disabled promotion gate remains
[#3465](https://github.com/ll7/robot_sf_ll7/issues/3465).

## Delivered Evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| Topology hypothesis availability is observable and classified. | [PR #4388](https://github.com/ll7/robot_sf_ll7/pull/4388), merge `85cc54750f660b92ecb82b4b04a2e0ec26ff0a05`, added episode-level `topology_guided_episode` diagnostics in `robot_sf/benchmark/map_runner_episode.py` with tests in `tests/benchmark/test_map_runner_utils.py`. | Covered as diagnostic metadata. |
| Topology candidate influence is measurable when eligible. | [PR #4426](https://github.com/ll7/robot_sf_ll7/pull/4426), merge `3483ba35ff2a37abfb87e422f5695e001a58d2b0`, made `arbitration_weight` explicit, added `blend_topology_command`, and records `topology_command_influence`; [PR #4176](https://github.com/ll7/robot_sf_ll7/pull/4176), merge `7291576e8ee321651dd75c38d6ca80ff94eb544c`, exposed topology corrective controls. | Covered by code and focused tests. |
| Fallback-only operation stays diagnostic and is not counted as improvement. | `topology_guided_episode` records fallback status from PR #4388; diagnostic manifests and candidate registry rows keep `claim_boundary`/`claim_eligibility` diagnostic-only; [PR #4444](https://github.com/ll7/robot_sf_ll7/pull/4444), merge `14147ee8b7b66b295e1167111e7ecc5acbd25ec3`, registers the Issue #3463 sensitivity packet as `diagnostic_only_not_benchmark_or_paper_evidence`. | Covered as a claim-boundary guard; not benchmark evidence. |
| Route-progress accounting distinguishes real stall from near-parity selection churn. | [PR #3622](https://github.com/ll7/robot_sf_ll7/pull/3622), merge `91a34df2c2763d2b31ca9b854950a7584e22b825`, hardened near-parity progress accounting; [PR #4411](https://github.com/ll7/robot_sf_ll7/pull/4411), merge `31dd50467ee9a7361cdc236bb4682358d713afde`, added `topology_route_progress_state.v1` and route-progress/churn metadata with tests in `tests/planner/test_topology_guided_local_policy.py` and `tests/validation/test_run_topology_hypothesis_diagnostics.py`. | Covered by diagnostic metadata and focused tests. |
| Near-parity gate thresholds are explicit and reproducible. | PR #4411 added `topology_near_parity_thresholds.v1`; PR #4426 added the nested `topology_guided_hybrid_rule.v1` config block with fail-closed validation for finite non-negative thresholds and bounded arbitration weight; config lives in `configs/policy_search/candidates/topology_guided_hybrid_rule_v0.yaml`. | Covered by code, config, and tests. |
| Horizon scenario-slice sensitivity has a bounded CPU-only path. | PR #4444 registered `configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml`, candidate `topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone`, registry entries, and validator tests. | Covered as a launch packet only; runtime execution remains the next empirical action. |
| Issue #3465 remains the benchmark-facing promotion gate. | The issue thread records Issue #3465 as the successor gate; this report preserves that boundary and does not close or supersede Issue #3465. | Covered. |

## Consolidated Contract

The current #3463 contract is:

- topology-guided corrective behavior remains diagnostic-only until #3465 or a
  successor explicitly runs a paired enabled-versus-disabled gate;
- fallback, degraded, missing-candidate, or fallback-only rows must not count as
  topology-lane success;
- topology-guided config fields must stay finite and bounded by their documented
  validation rules;
- the issue #3463 sensitivity packet is a reproducible local run plan, not
  completed benchmark evidence.

## Remaining Blockers

| Blocker | Why it remains |
| --- | --- |
| No full benchmark promotion claim. | Issue #3463 intentionally stops at guarded diagnostic implementation; Issue #3465 owns benchmark-facing enabled-versus-disabled comparison. |
| No completed Issue #3463 cross-slice runtime result in this report. | PR #4444 registers the bounded packet and tests the validator, but does not run the CPU scenario-slice diagnostics. |
| Two schema nitpicks from PR #4426 are separate follow-up work. | The issue thread spun off global-versus-per-step command limits and uniform topology-guided config schema handling to Issue #4430. |

## Next Empirical Action

Run the registered CPU-only diagnostic packet, then record compact evidence under
`docs/context/evidence/` if it produces a useful result:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_reselection_cross_slice.py \
  --manifest configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml \
  --output-dir output/diagnostics/issue_3463_topology_reselection_cross_slice
```

Expected interpretation remains diagnostic-only unless a separate benchmark gate
updates the claim boundary.
