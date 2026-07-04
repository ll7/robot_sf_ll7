# Issue #3207 Integration Status

Plain-language summary: Issue #3207 is ready for an authorized empirical fixed-scope
simulation-fidelity campaign, but it is not ready for a simulator-dependence, sim-to-real, paper,
or dissertation claim. The tracked code and evidence now cover launch packet, preflight,
fixed-scope plan materialization, per-cell execution wiring, planner prerequisite checks, and the
post-run rank-identifiability contract. The remaining work is to run the fixed-scope campaign and
classify the measured rank-identifiability result.

This note is a consolidation report for
[issue #3207](https://github.com/ll7/robot_sf_ll7/issues/3207). It adds no benchmark result and
does not change runtime behavior.

## Delivered Contract

| Contract surface | Durable evidence | Status |
| --- | --- | --- |
| Study contract and launch packet | [`issue_3207_fidelity_sensitivity_launch_packet_2026-06-20`](issue_3207_fidelity_sensitivity_launch_packet_2026-06-20/README.md) and [`configs/research/fidelity_sensitivity_v1.yaml`](../../../configs/research/fidelity_sensitivity_v1.yaml) define the internal fidelity axes and no-claim boundary. | Delivered. |
| Bounded diagnostic actual slices | [`issue_3207_fidelity_sensitivity_actual_slice_2026-06-20`](issue_3207_fidelity_sensitivity_actual_slice_2026-06-20/README.md) and [`issue_3207_fidelity_sensitivity_actual_slice_2026-06-23`](issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/README.md) record compact local campaign slices only. | Delivered as diagnostic-only evidence, not full #3207 acceptance evidence. |
| No-claim validity-boundary packet | [`issue_3207_simulator_dependence_validity_boundary_packet_2026-06-29`](issue_3207_simulator_dependence_validity_boundary_packet_2026-06-29/README.md) classifies the existing bounded slice as `no_claim` because the scope is not full fixed-scope evidence and rank evidence is non-identifiable. | Delivered; claim remains blocked. |
| Full fixed-scope preflight | [`issue_3207_fidelity_fixed_scope_preflight_2026-07-04`](issue_3207_fidelity_fixed_scope_preflight_2026-07-04/README.md) records the 108-cell-per-scenario plan and preflight readiness boundary. | Delivered as launch/readiness evidence only. |
| Planner prerequisite and post-run contract closure | [`issue_4401_closure_audit_2026-07-04.md`](issue_4401_closure_audit_2026-07-04.md) summarizes merged prerequisite and rank-identifiability checker slices. | Delivered; no campaign output included. |

## Current Boundary

The current repository state supports only this claim:

> The full fixed-scope #3207 fidelity-sensitivity campaign has a tracked config, launch packet,
> preflight, fixed-scope execution wiring, planner prerequisite checks, and post-run
> rank-identifiability contract.

It does not support any claim that planner rankings are stable across fidelity perturbations. It
also does not support simulator-realism, simulator-dependence, sim-to-real, paper-facing, or
dissertation wording. Those claims require measured full fixed-scope campaign output and a passing
post-run rank-identifiability report.

## Remaining Work

| Remaining item | Why it remains |
| --- | --- |
| Run the authorized full fixed-scope campaign. | The latest issue guidance names empirical fixed-scope campaign execution as the remaining unmet input; this PR did not run benchmarks, Slurm jobs, GPU jobs, or campaign execution. |
| Produce `fidelity_rank_stability_report.json`. | The post-run checker needs measured rows to verify the configured `runtime_rank_identifiability_recheck` contract. |
| Re-run the simulator-dependence validity-boundary checker over the full-scope summary. | Existing packet is intentionally `no_claim`; it was generated over a bounded diagnostic slice, not the full fixed-scope campaign. |
| Promote or block claim language based on measured result. | Stable identifiable ranks can support a bounded validity-boundary statement; rank flips or non-identifiable ranks require caveat or calibration follow-up instead. |

## Validation Boundary

Recommended cheap validation for this consolidation note:

```bash
test -f docs/context/evidence/issue_3207_integration_status_2026-07-05.md
git diff --check
```

No full benchmark campaign was run, no Simple Linux Utility for Resource Management (SLURM) or GPU
submission was made, and no paper or dissertation claim text changed.
