# Issue #4401 Closure Audit

Plain-language summary: issue #4401 asked for the fixed-scope fidelity-sensitivity launch plan to stop failing on three prerequisite gaps before any campaign run. Merged PRs #4420 and #4446 now cover the CPU-only launch and post-run contract pieces. The only remaining action is empirical: a later authorized campaign must produce and pass the rank-identifiability report before any ranking claim is promoted.

This audit is an integration report for [issue #4401](https://github.com/ll7/robot_sf_ll7/issues/4401). It is not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, and not a paper-facing claim.

## Acceptance Mapping

| Acceptance criterion from #4401 | Delivered evidence | Status |
| --- | --- | --- |
| Hybrid fixed-scope opt-in is explicit and tested while the fail-closed default remains unchanged. | PR #4420 added `fixed_scope.planner_opt_ins.hybrid_rule_v0_minimal` in `configs/research/fidelity_sensitivity_v1.yaml`, records `explicit_opt_in_satisfied: true` in `robot_sf/benchmark/fidelity_fixed_scope_preflight.py`, and tests the shipped plan in `tests/benchmark/test_fidelity_fixed_scope_run_plan.py`. | Delivered by merge commit `dd2910a31103050c205e939656c31558a4aadb93`. |
| ORCA `rvo2` dependency failure names the exact remedy instead of a bare import failure. | PR #4420 added the actionable `planner_requires_rvo2:orca` remediation text in `robot_sf/benchmark/fidelity_fixed_scope_preflight.py`, including `uv sync --all-extras` and stale `third_party/python-rvo2/build` cleanup guidance. | Delivered by merge commit `dd2910a31103050c205e939656c31558a4aadb93`. |
| Post-run rank-identifiability recheck is a concrete machine-readable contract. | PR #4420 added backward-compatible `post_run_contract_specs` with `runtime_rank_identifiability_recheck`, report path, builder, metric, threshold, output path, and `blocks_claims_when_failed: true`. | Delivered by merge commit `dd2910a31103050c205e939656c31558a4aadb93`. |
| Plan mode no longer blocks on unresolved hybrid opt-in or undefined rank-identifiability contract. | PR #4420 tests the shipped plan as `preflight_ready`, `executable`, with empty `gate_reasons`, explicit hybrid opt-in, and the rank-identifiability contract in the run plan. | Delivered by merge commit `dd2910a31103050c205e939656c31558a4aadb93`. |
| The post-run contract has a concrete checker and report artifact path, not just a plan declaration. | PR #4446 added `write_rank_identifiability_report()` and `check_rank_identifiability_contract()` in `robot_sf/benchmark/fidelity_rank_stability.py`, wires the campaign runner to emit `fidelity_rank_stability_report.json`, and tests pass, fail-closed, unsupported-threshold, serialization, and runner wiring behavior. | Delivered by merge commit `92733e5d50a6ffca55dc52047b5fc60f86dc5d88`. |
| No campaign execution, no new planner arm, no threshold change, and no claim update included. | PR #4420 was preflight/plan wiring only. PR #4446 was deterministic report/checker wiring only. This audit adds no runtime behavior and no campaign output. | Satisfied for the merged implementation slices and this audit. |

## Residual Work

The remaining item named in the issue thread is not a missing CPU-only implementation criterion. A future authorized full fixed-scope campaign must write `fidelity_rank_stability_report.json`, run `check_rank_identifiability_contract()` against the `runtime_rank_identifiability_recheck` spec, and keep any planner-ranking claim blocked unless that report satisfies `non_zero_variance_and_rank_identifiable`.

That campaign action is intentionally out of scope here: no full benchmark campaign was run, no Slurm or GPU job was submitted, and no paper or dissertation claim text was changed.

## Local Verification

Audit-time verification should rerun the focused tests and plan-only command:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_fidelity_fixed_scope_run_plan.py \
  tests/benchmark/test_fidelity_rank_stability.py -q

scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/benchmark/run_fidelity_sensitivity_campaign.py \
  --fixed-scope-plan-only \
  --plan-out output/issue_4401_closure_audit_plan \
  --require-launchable
```

The generated `output/issue_4401_closure_audit_plan/` directory is a disposable local validation artifact and is not durable evidence.
