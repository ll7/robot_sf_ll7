# Issue #3285 Dry-Run Scenario Interop Converter (2026-06-27)

Status: implementation of the local, asset-free slice (`smoke evidence`). Not benchmark evidence and
not a cross-benchmark validity claim.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/3285
- Conceptual mapping seed: [`issue_2928_socnavbench_hunavsim_metric_correspondence.md`](issue_2928_socnavbench_hunavsim_metric_correspondence.md)
- Source-side contract: `robot_sf/benchmark/scenario_contract.py`, `robot_sf/benchmark/schema/scenarios.schema.json`
- External-asset prerequisites (block the full external run): #1456, #1498, #2414, #1134

## What this slice delivers

A deterministic, schema-validated **intermediate representation (IR)** for Robot SF scenario-matrix
entries, plus an explicit unsupported-field report. This is the buildable-local part of #3285; it
does **not** require external assets and does **not** emit any SocNavBench or HuNavSim file.

- Module: `robot_sf/benchmark/scenario_interop.py` (`convert_scenario_to_ir`, `dump_ir`,
  `validate_interop_ir`).
- IR schema: `robot_sf/benchmark/schemas/scenario_interop_ir.v1.json`
  (`robot_sf.scenario_interop_ir.v1`).
- Dry-run CLI: `scripts/tools/convert_scenario_interop.py`.
- Tests: `tests/benchmark/test_scenario_interop.py`,
  `tests/benchmark/test_scenario_interop_target_compatibility.py`.

## Target Compatibility Report

The converter also emits an asset-free target compatibility report
(`robot_sf.scenario_interop_target_compatibility.v1`) for `socnavbench` and `hunavsim`.
This is a fail-closed readiness projection, not a target artifact exporter.

- `ready`: false until target assets/adapters are staged and required IR fields are present.
- `blockers`: explicit missing asset, adapter, map, agent, flow, or unsupported-field blockers.
- `warnings`: non-blocking provenance or target-semantics gaps, such as missing seeds or HuNavSim
  ROS/Gazebo launch semantics outside the target-neutral IR.

Use `--target socnavbench` or `--target hunavsim` on the dry-run CLI to include a selected target
report in the stderr summary. Omitting `--target` reports both targets.

## IR contract

The IR captures the scenario fields that are common across social-navigation testbeds:

| IR section | Source fields | Notes |
|---|---|---|
| `provenance` | `id`/`scenario_id`/`name`, `metadata`, all top-level keys, source file | Records source id (fixed precedence), `source_kind` (`axis`/`explicit_map`), full `source_fields`, and `source_metadata`. |
| `geometry` | `obstacle`, `map_file` | `obstacle` is also mapped to a coarse `environment_type` (`open` → `open_space`, `bottleneck` → `constrained_passage`, `maze` → `cluttered`). |
| `environment` | `density`, `flow`, `groups`, `speed_var`, `goal_topology`, `robot_context` | Target-neutral environment semantics. |
| `agents` | `single_pedestrians[]` | start/goal POIs, preferred speed, role, role target, wait points; specs without an `id` get a deterministic positional id `agent_<index>`. |
| `timing` | `repeats`, `seeds` | |
| `unsupported_fields` | everything else | See below. |

### Unsupported-field reporting (explicit, never silent)

Every top-level source key is classified exactly once. Recognized simulator-specific keys
(`simulation_config`, `robot_config`, `route_overrides_file`, `amv`, `multi_amv`, `supported`) and
any unrecognized key are recorded in `unsupported_fields` with a reason. Nothing is silently dropped.

### Determinism

For a given scenario dict the IR is a pure function of the input: section/key order is fixed by
construction, `provenance.source_fields` is sorted, and `unsupported_fields` is sorted by field name.
`dump_ir` therefore yields byte-identical JSON across runs.

## Dry-run command

```bash
# Print IR + per-scenario validity summary
uv run python scripts/tools/convert_scenario_interop.py --matrix configs/baselines/example_matrix.yaml

# Write one <scenario_id>.ir.json per scenario
uv run python scripts/tools/convert_scenario_interop.py \
    --matrix configs/baselines/example_matrix.yaml --out-dir output/interop_ir
```

The CLI prints the IR to stdout and a `{"dry_run_summary": [...]}` block (scenario id, IR validity,
unsupported-field count) to stderr. Exit code is non-zero if any scenario IR fails schema validation.

## Validation

```bash
uv run python -m pytest tests/benchmark/test_scenario_interop.py -q          # 10 passed
uv run python -m pytest tests/benchmark -k "scenario or convert or socnav" -q # 239 passed (issue cmd)
```

## Claim boundary / blocked-until

- This is a **dry-run + IR validation** slice only. It makes no claim of simulator equivalence,
  planner transferability, or benchmark-score parity (see the boundary in
  [`issue_2928_...`](issue_2928_socnavbench_hunavsim_metric_correspondence.md)).
- Emitting real SocNavBench/HuNavSim assets and running them is **blocked** on staged external
  assets (#1456 / #1498 / #2414 / #1134) and is intentionally out of scope here.
