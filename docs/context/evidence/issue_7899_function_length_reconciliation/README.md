# Function-Length Reconciliation — issue #7899

**Status:** audit + reconciliation delivered; no production change.
**Issue:** [#7899](https://github.com/ll7/robot_sf_ll7/issues/7899) (parent #6456, programme #4770).
**Scanner:** `scripts/dev/audit_function_lengths.py` (AST-based; `function_length_audit.v1`).

## Audit

- Root: `robot_sf/` (933 files, 932 scanned; 1 explicit exclusion with rationale).
- Threshold: 200 inclusive source lines (def line through end of body; decorators excluded).
- Findings: **32 functions** over 200 lines on current `main` (top: 1349-line
  `issue_5303_search_promotion_preregistration_v2.preflight_issue_5303_powered_contract`).
- Full report (SHA-256 `81da165d…`): `output/function_length_audit.json` (ignored, worktree-local).
- Byte-stable for an unchanged tree; fixture tests pin inclusive-count semantics.

## Reconciliation vs closed #6456 children

- None of the 32 closed-child symbols (e.g. `_run_campaign_orchestrator`, `run_map_batch`,
  `write_report_artifacts`) remain over 200 lines on current `main` — all resolved below
  threshold or renamed/moved.
- The 32 current findings are **not** covered by any existing open issue → `remaining_without_child`.
- **Parent disposition: `residual_children_required`** — the 25 original child targets are
  resolved, but 32 current functions exceed the threshold and need new non-overlapping child
  plans (one module/owner per PR, Domain-Aware Approval required for benchmark/evidence owners).

## Boundary

No production Python under `robot_sf/`, metric, planner, scenario, benchmark output, schema,
evidence status, or public API changed. No benchmark executed.
