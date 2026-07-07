# Issue #3285 Closure Audit

Date: 2026-07-07
Issue: <https://github.com/ll7/robot_sf_ll7/issues/3285>

## Claim Boundary

This is a closure audit for the Robot SF to SocNavBench/HuNavSim scenario converter issue. It maps the
live issue acceptance criteria and merged pull requests to evidence, then records the current closure
decision.

Conclusion: **keep #3285 open**. The local, asset-free converter contract is covered: intermediate
representation, schema validation, unsupported-field reporting, target compatibility reports, target
export manifests, target previews, and schema validation for target handoff artifacts all exist on
`origin/main`. The runnable SocNavBench/HuNavSim export criterion is not met because it still depends
on staged external assets/adapters.

This report is evidence-generation only. It is not a full benchmark campaign, not a SocNavBench or
HuNavSim run, not a simulator-equivalence claim, and not paper or dissertation evidence.

## Live Audit Inputs

Live issue thread read on 2026-07-07. The latest owner comments record:

- PR #4562 merged on 2026-07-05: target compatibility reporting exists, but every target remains
  `ready=false` with named blockers and no external artifacts emitted.
- PR #4711 merged on 2026-07-07: asset-free target export previews exist, but payloads stay
  fail-closed/`blocked`.
- PR #4718 merged on 2026-07-07: target handoff schemas and validation helpers exist.
- Remaining closure in the live thread: real SocNavBench/HuNavSim artifact export remains blocked on
  staged assets/adapters tracked by #1456, #1498, #2414, and #1134.

Open-PR dedupe on 2026-07-07 found no open PR directly covering #3285. Fragmentation guard applies:
#4562, #4711, and #4718 all merged recently. This report is therefore a consolidation/integration
artifact rather than another checker or packet refresh.

## Acceptance Criteria To Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Converter output is schema-validated for deterministic fixture scenarios. | **Met** | PR #3735 added `robot_sf/benchmark/scenario_interop.py`, `scenario_interop_ir.v1.json`, deterministic serialization tests, and `validate_interop_ir()`. The current context note documents the dry-run command and validation path in `docs/context/issue_3285_scenario_interop_converter.md`. |
| Unsupported fields are reported explicitly rather than silently dropped. | **Met** | PR #3735 added `unsupported_fields` to the intermediate representation. The context note states every top-level source key is classified exactly once, with recognized simulator-specific keys and unknown keys recorded with reasons. |
| Smoke dry-run command is documented. | **Met** | PR #3735 documented `scripts/tools/convert_scenario_interop.py --matrix configs/baselines/example_matrix.yaml`. PR #4673 extended the documented smoke path with `--target-out-dir`; PR #4711 extended it with preview output. |
| Target compatibility is visible before runnable export. | **Met** | PR #4562 added `build_target_compatibility_report()` and `convert_scenario_interop.py --target socnavbench|hunavsim`. Reports fail closed with `ready=false` and named blockers. |
| Target export handoff artifacts are concrete and fail closed when runnable prerequisites are absent. | **Met** | PR #4673 added target export manifests; PR #4711 added target-shaped export previews for SocNavBench and HuNavSim. Both remain blocked rather than pretending to be runnable external assets. |
| Target handoff artifacts are machine validated. | **Met** | PR #4718 added JSON Schema files and validation helpers for target compatibility reports, export manifests, and export previews, with tamper tests. |
| Issue links the external prerequisites required for a full runnable export. | **Met, but still blocking closure** | The issue thread and `docs/context/issue_3285_scenario_interop_converter.md` link #1456, #1498, #2414, and #1134. As of this audit, #1498 and #2414 are closed, but #1456 remains `state:blocked-external-input` and #1134 remains open pending real SocNavBench ETH source assets and generated map validation. |
| Real SocNavBench/HuNavSim artifact export is runnable. | **Not met** | The latest #3285 owner comment after PR #4718 says runnable export remains blocked on staged external assets/adapters. #1456 still reports missing licensed SocNavBench control-pipeline data. #1134 remains open after its own closure audit because the official ETH `data.pkl`, generated `socnavbench_eth.svg`, checksum/provenance, and parser/smoke validation are still absent on the execution host. |

## Remaining Work

- Stage or hydrate the licensed SocNavBench control-pipeline assets required by #1456.
- Complete #1134 with the official ETH source assets: generate the real Robot SF SVG map, record
  checksum/provenance, and pass parser/smoke validation.
- Re-run the #3285 target export path after those inputs exist, and only then decide whether the
  runnable SocNavBench/HuNavSim export criterion is met.

## Closure Decision

Use `Refs #3285`, not `Closes #3285`, for this audit. All agent-executable local converter contract
criteria are covered by merged PRs, but the issue should remain open while the runnable external
export criterion is blocked on external input.
