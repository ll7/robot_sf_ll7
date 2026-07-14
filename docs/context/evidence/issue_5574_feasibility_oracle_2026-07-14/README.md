<!-- AI-GENERATED (robot_sf#5574, 2026-07-14) - NEEDS-REVIEW -->

# Issue #5574 Feasibility Oracle Evidence

This packet uses the planner-free oracle to decide whether the two named Francis 2023 zero-success candidate cells are infeasible under the nominal collision envelope or merely unresolved. It is diagnostic-only evidence, not a benchmark result.

## Command

```bash
uv run python scripts/tools/run_feasibility_oracle_issue_5574.py \
  configs/scenarios/francis2023.yaml \
  --output docs/context/evidence/issue_5574_feasibility_oracle_2026-07-14/verdicts.json
```

The default candidate set is `francis2023_narrow_doorway` and `francis2023_blind_corner`. The report runs the same scripted `goal` traversal and horizon at the nominal `1.0 m` radius and reduced `0.5 m` radius, using the first manifest seed for each cell.

## Provenance and claim boundary

- Report schema: `issue_5574_feasibility_oracle_report.v1`.
- Per-cell schema: `envelope_sensitivity_axis.v1`; per-radius schema: `scenario_feasibility_oracle.v1`.
- Source manifest: `configs/scenarios/francis2023.yaml`.
- Seeds: `225` for the narrow doorway and `219` for the blind corner.
- Run mode: local central processing unit (CPU) diagnostic; no learned planner or benchmark campaign.
- Claim boundary: `diagnostic_only_not_benchmark_evidence`.

## Observed verdicts

| Cell | Nominal 1.0 m | Reduced 0.5 m | Interpretation |
| --- | --- | --- | --- |
| `francis2023_narrow_doorway` | `infeasible_by_construction`; `geometrically_infeasible`, corridor/envelope margin `0.0 m` | Geometry becomes `hard_but_solvable` with margin `1.0 m`, but the scripted traversal collides before the 400-step horizon | Exclude the cell from nominal-envelope planner attribution; the reduced probe shows envelope-sensitive geometry but does not establish a completed traversal. |
| `francis2023_blind_corner` | `time_truncated`; geometry is `hard_but_solvable` with margin `2.0 m`, scripted traversal collides | `time_truncated`; geometry remains `hard_but_solvable` with margin `3.0 m`, scripted traversal terminates without completion | Do not label this cell geometrically infeasible; retain it as unresolved/scripted-unsolved diagnostic evidence. |

The nominal narrow-doorway verdict is enough to declare that `1.0 m` configuration geometrically excludes the cell. The reduced-radius run is sensitivity evidence only: its positive static margin does not override the failed scripted traversal. The blind-corner runs provide no geometric infeasibility proof, so a zero planner-success rate remains a planner/interaction question rather than an automatic map exclusion.

## Boundaries

This packet does not alter benchmark denominators, campaign metadata, planner rankings, or paper-facing claims. A downstream campaign may pass its zero-success per-cell rates to `annotate_zero_completion_cells(...)`; missing or blocked oracle output remains fail-closed rather than being treated as planner-limited success evidence.
