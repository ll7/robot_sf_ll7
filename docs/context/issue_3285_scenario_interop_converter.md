# Issue #3285 Dry-Run Scenario Interop Converter

Status: local, asset-free implementation slice (`smoke evidence`). This is not benchmark
evidence and does not claim cross-benchmark validity.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/3285
- Conceptual mapping seed: [`issue_2928_socnavbench_hunavsim_metric_correspondence.md`](issue_2928_socnavbench_hunavsim_metric_correspondence.md)
- Source-side contract: `robot_sf/benchmark/scenario_contract.py`,
  `robot_sf/benchmark/schema/scenarios.schema.json`
- External-asset prerequisites that block an external run: #1456, #1498, #2414, #1134

## Slice Delivers

The local converter emits deterministic, schema-validated intermediate representation (IR)
for Robot SF scenario-matrix entries, plus explicit unsupported-field reports. It does not
require external assets and does not emit runnable SocNavBench or HuNavSim files.

- Module: `robot_sf/benchmark/scenario_interop.py` (`convert_scenario_to_ir`, `dump_ir`,
  `validate_interop_ir`, `build_target_export_manifest`, `build_target_export_preview`).
- IR schema: `robot_sf/benchmark/schemas/scenario_interop_ir.v1.json`
  (`robot_sf.scenario_interop_ir.v1`).
- Dry-run CLI: `scripts/tools/convert_scenario_interop.py`.
- Tests: `tests/benchmark/test_scenario_interop.py`,
  `tests/benchmark/test_scenario_interop_target_compatibility.py`.

## Target Compatibility

The converter emits asset-free target compatibility reports
(`robot_sf.scenario_interop_target_compatibility.v1`) for `socnavbench` and `hunavsim`.
These are fail-closed readiness projections, not target artifact exports.

- `ready`: false until target assets/adapters are staged and required IR fields are present.
- `blockers`: explicit missing asset, adapter, map, agent, flow, and unsupported-field blockers.
- `warnings`: non-blocking provenance or target-semantics gaps, including missing seeds and
  HuNavSim ROS/Gazebo launch semantics outside the target-neutral IR.

## Target Export Manifest And Preview

`--target-out-dir` writes deterministic, asset-free target export manifests
(`robot_sf.scenario_interop_target_export_manifest.v1`). A manifest is a JSON handoff
contract for downstream adapters; it is not a runnable SocNavBench or HuNavSim scenario file.
When prerequisites are missing, it remains `status: blocked`, `ready: false`, with named
blockers, warnings, source scenario, and source IR schema.

`--target-preview-out-dir` writes deterministic, asset-free target-shaped preview payloads
(`robot_sf.scenario_interop_target_export_preview.v1`). Preview payloads preserve the
target-specific sections that a real writer must resolve: `scenario`/`pedestrians` for
SocNavBench and `world`/`agents` for HuNavSim. They carry the same fail-closed blockers as
the manifest and are adapter-development artifacts, not runnable external benchmark inputs.

## IR Contract

| IR section | Source fields | Notes |
|---|---|---|
| `provenance` | `id`/`scenario_id`/`name`, `metadata`, all top-level keys, source file | Records source id, `source_kind` (`axis`/`explicit_map`), full `source_fields`, and `source_metadata`. |
| `geometry` | `obstacle`, `map_file` | `obstacle` maps to coarse `environment_type` (`open` -> `open_space`, `bottleneck` -> `constrained_passage`, `maze` -> `cluttered`). |
| `environment` | `density`, `flow`, `groups`, `speed_var`, `goal_topology`, `robot_context` | Target-neutral environment semantics. |
| `agents` | `single_pedestrians[]` | Start/goal points of interest, preferred speed, role, role target, wait points; specs without an `id` get deterministic id `agent_<index>`. |
| `timing` | `repeats`, `seeds` | Preserves repeat count and explicit seed set when present. |
| `unsupported_fields` | everything else | Every unmapped top-level source key is reported with a reason; nothing is silently dropped. |

## Dry-Run Commands

```bash
# Print IR plus per-scenario validity summary.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml

# Write one <scenario_id>.ir.json per scenario.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --out-dir output/interop_ir

# Write fail-closed target export manifests.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --target socnavbench \
  --target-out-dir output/issue_3285_target_export_manifest_smoke

# Write fail-closed target export previews.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --target socnavbench \
  --target-preview-out-dir output/issue_3285_target_export_preview_smoke
```

The CLI prints IR to stdout only when no output directory is requested. It always writes a
`{"dry_run_summary": [...]}` block to stderr and exits non-zero when a scenario IR fails schema
validation.

## Claim Boundary

- This dry-run, IR validation, target export manifest, and target export preview slice makes no
  claim about simulator equivalence, planner transferability, or benchmark-score parity.
- Emitting real runnable SocNavBench/HuNavSim scenario assets and running them remains blocked on
  staged external assets/adapters (#1456, #1498, #2414, #1134).
