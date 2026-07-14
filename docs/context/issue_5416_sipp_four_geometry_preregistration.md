# Issue #5416: four-geometry SIPP evidence packet

This packet freezes the four scenario rows, five-planner comparator roster, paired seeds, and
fail-closed evidence rules before any campaign run. It is a preregistration contract; it does not
make a planner-quality, safety, liveness, benchmark, or paper-facing claim.

## Claim boundary and status

- Evidence status: `diagnostic-only` / launch-packet contract; no benchmark campaign ran in this PR.
- Claim boundary: future rows are `exploratory_synthetic_benchmark_only` until native execution,
  geometry certification, complete provenance, and paired-row checks pass.
- Current CPU geometry preflight: corridor, platform, and merging are `hard_but_solvable` and
  eligible; doorway is `knife_edge` and `stress_only`. The stress-only caveat remains visible and
  is not headline evidence.
- Confidence in the packet contract: 99%; confidence in any future planner comparison: not
  estimable until the paired five-seed matrix exists.

The machine-readable source of truth is
[`configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml`](../../configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml).
The executable gate is
[`scripts/validation/check_issue_5416_sipp_four_geometry_packet.py`](../../scripts/validation/check_issue_5416_sipp_four_geometry_packet.py).

## Frozen comparison

The exact scenario rows are `classic_head_on_corridor_low`, `classic_doorway_low`,
`classic_station_platform_medium`, and `classic_merging_low`. The exact planner keys are:

| Role | Planner key | Execution/config boundary |
| --- | --- | --- |
| Candidate | `sipp_lattice` | Experimental explicit opt-in, `configs/algos/sipp_lattice_slice2_smoke.yaml` |
| Comparator | `hybrid_rule_v0_minimal` | `hybrid_rule_local_planner`, explicit opt-in |
| Comparator | `teb` | Explicit opt-in, native TEB-style adapter |
| Comparator | `nmpc_social` | Explicit opt-in, native NMPC-style adapter |
| Comparator | `dwa` | Explicit opt-in, classical Dynamic Window Approach adapter |

Each planner uses the same observation/action/kinematics contract. Construction failures, missing
rows, fallback/degraded execution, schema mismatches, and incomplete pairing remain visible and
cannot count as successful evidence.

## Seeds, outcomes, and decision rule

The smoke seed is `111`; the result-producing paired seeds are `111, 112, 113, 114, 115`, with a
fixed horizon of 500 steps and `dt=0.1`. Primary outcomes are collision-free completion, collision
counts/rates, deadlock/timeout/max-step rate, paired progress/time-to-goal conditional on
collision-free completion, and median/p95 planner-step runtime. SIPP diagnostics retain expansion
limit hits, runtime-bound exits, fallback count, and commitment invalidations.

The candidate advances only if the paired slice has no collision regression and either improves
collision-free completion or materially reduces deadlock. Runtime improvement alone does not
qualify. Five seeds are a diagnostic slice, not a record claim.

## Geometry and artifact gate

Before execution, run the packet checker. It invokes the repository's `scenario_cert.v1` API on
each selected source manifest. An excluded or unresolved row remains visible with its reason and
blocks interpretation; platform/merging failures must not be attributed to a planner. A reviewed
replacement may be used only if it already exists; this PR does not repair scenario geometry.

Future result-producing work must retain the frozen configs, public SHA and dirty-state proof,
exact command/environment/resource provenance, per-episode JSONL, aggregate summary, planner
diagnostics, denominator/exclusion table, paired comparison, and the explicit claim-boundary note.
Raw JSONL, videos, checkpoints, and large logs stay out of Git; durable derived evidence belongs
under `docs/context/evidence/issue_5416_sipp_four_geometry/` before any interpretation.

## Paired-analysis handoff

[`scripts/analysis/analyze_issue_5416_sipp_four_geometry.py`](../../scripts/analysis/analyze_issue_5416_sipp_four_geometry.py)
consumes native JSONL plus per-planner provenance and writes summary, exclusion, and paired-comparison
artifacts; missing/fallback rows, collision components, diagnostics, or provenance fail closed.

## Validation and exclusions

```bash
uv run python scripts/validation/check_issue_5416_sipp_four_geometry_packet.py --json
uv run pytest -q tests/validation/test_check_issue_5416_sipp_four_geometry_packet.py
uv run pytest -q tests/planner/test_sipp_lattice.py tests/benchmark/test_sipp_lattice_policy_builder.py
```

The completed local seed-111 smoke used the real map-runner path and wrote one ignored row:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios, select_scenario

scenario_path = Path("configs/scenarios/archetypes/classic_head_on_corridor.yaml")
scenario = dict(select_scenario(load_scenarios(scenario_path), "classic_head_on_corridor_low"))
scenario["seeds"] = [111]
run_map_batch(
    [scenario],
    Path("output/benchmarks/issue_5416_smoke/seed111_sipp.jsonl"),
    schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
    scenario_path=scenario_path,
    algo="sipp_lattice",
    algo_config_path="configs/algos/sipp_lattice_slice2_smoke.yaml",
    horizon=500,
    dt=0.1,
    workers=1,
    resume=False,
    benchmark_profile="experimental",
)
print(json.dumps({"status": "smoke_complete", "seed": 111}))
PY
```

Observed smoke result: one row, `algorithm_metadata.algorithm=sipp_lattice`,
`algorithm_metadata.status=ok`, and `termination_reason=terminated`. This is execution/contract
proof only, not a planner outcome or benchmark success claim.

No full benchmark campaign run, Slurm/GPU submission, target-host routing, transient packet
lineage, metric-definition change, planner-logic change, or paper/dissertation claim edit is part
of this preregistration slice.
