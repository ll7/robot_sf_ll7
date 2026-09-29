<!-- AI-GENERATED (robot_sf#9735) - NEEDS-REVIEW -->
# VV-5 ten-fault diagnostic matrix (#9735)

**Scope:** Ten controlled synthetic fault classes were replayed on source commit
`03f96dbf4907622ea0ec7f865db4adf1630ede7a` (current `origin/main`). The applicable checks
detected **8/10**. This is fixture-level sensitivity across different checks,
not a pooled statistical estimate, nominal campaign evidence, or 0.0.8
release admission. The machine-readable controls, mutants, checker outputs, and
boundaries are in [the compact matrix](issue_9735_vv5_full_matrix_2026-09-28.json).

| Injected fault | Check exercised | Result | Boundary |
| --- | --- | --- | --- |
| Reset pedestrian overlap | Spawn validity | Detected | Synthetic reset checker input; `route_complete=False` |
| Respawn onto robot | Spawn validity | Detected | Event plus collision injection; no simulator placement |
| Robot start inside wall radius | Spawn validity | Detected | Synthetic reset checker input |
| Per-cell wall-force sum | Obstacle-force grid-resolution invariance | Detected | One component fixture; clean relative change 0, mutant 0.2784 against 0.25 limit |
| Centre-distance threshold and wrong braking | Hybrid clearance/speed-cap invariant | Detected | Component speeds only; no hybrid episode |
| Infeasible gap | Route-clearance gate and spawn preflight | Detected | 3.0 m clean / 1.8 m mutant, seed 111; full feasibility oracle inconclusive |
| Planner-only radius halved | Prediction planner physical-unit audit | Detected | Config-level audit; release-row anomaly checker misses it |
| Planner goal direction flipped | Paired pedestrian-free regression check | Detected | One short kinematic trace at seed identity 111; diagnostic `min_paired_cells=1` |
| Pedestrian–robot force disabled | Force reduction telemetry | **Missed** | Factory wiring and one force evaluation; no contact trial |
| Collision total doubled | Release-row anomaly checker | **Missed** | Synthetic row; no release aggregation |

The infeasible-gap full feasibility oracle did **not** supply a clean-pass /
mutant-reject pair: its clean full verdict was blocked by unrelated ancillary
`metrics.distributional_disruption.missing_data.slow_speed_tier.status=unavailable`.
Only the geometric subcheck and two explicit route gates count toward the
detected result. The planner-radius audit is a test-owned unit rule against the
production planner config, not a complete production release gate. The goal-flip
trace is a one-seed diagnostic, not evidence that the nominal matrix detector
would catch a single changed planner.

## Misses and next checks

The disabled pedestrian–robot-force case reports zero force as valid because
the metric has no independently approved expectation that this component is
active. Bind the approved scenario-level PRF state and expected component roster
to the resolved runtime config, verify component capture before integration
across exposed steps, and then confirm with a stopped-robot multi-seed trace.
Legitimate zero magnitude must remain possible. This is a proposed release
gate, not a result obtained here.

For the doubled collision total, recompute `total_collision_count` and
`collisions` from typed components and the event ledger, and reject a mismatch
before aggregation. Issue #9855 tracks that release blocker. The earlier
[four-fault packet](issue_9735_vv5_fault_injection.md) also identifies
outcome-dependent reset-overlap and unavailable-clearance gaps owned by #9861;
those are distinct from this ten-class sensitivity count.

## Reproduction and custody

The four source scripts and the compactor are tracked under
`scripts/validation/`. Use the shared-venv wrapper from the repository root
and a disposable artifact root outside worktree `output/`:

```sh
vv5_artifact_root=/tmp/issue-9735-vv5-replay
mkdir -p "$vv5_artifact_root"/first
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_issue_9735_fault_injection.py --out-json "$vv5_artifact_root"/first/report.json --out-md "$vv5_artifact_root"/first/report.md
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_issue_9735_geometry_radius_diagnostic.py --root "$PWD" --out "$vv5_artifact_root"/geometry --fault infeasible_gap --fault planner_only_radius_halved
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_issue_9735_force_diagnostic.py --output-dir "$vv5_artifact_root"/force
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_issue_9735_braking_goal_diagnostic.py --output-dir "$vv5_artifact_root"/remaining
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/build_issue_9735_full_matrix.py --first "$vv5_artifact_root"/first/report.json --geometry-radius "$vv5_artifact_root"/geometry/report.json --force "$vv5_artifact_root"/force/report.json --remaining "$vv5_artifact_root"/remaining/remaining_report.json --source-commit 03f96dbf4907622ea0ec7f865db4adf1630ede7a --output "$vv5_artifact_root"/full_matrix.json
```

The compactor verifies all ten unique classes, exact 8/10 disposition, and
source/config and replay-script file hashes against the pinned source commit
or tracked script bytes. It records packet execution HEADs as provenance;
these may differ from the source commit when the evidence-only PR advances.
The current packets were rerun at execution head
`052b03e395d63dc377923feb588a9b85f1e94e24`, a local append-only merge of
the PR head into current `origin/main`; their source/config bytes are bound
to `03f96dbf4907622ea0ec7f865db4adf1630ede7a`. Those file hashes are retained
in the compact JSON so packet identity can be rechecked without local raw logs.
Current-source packet SHA-256 values, in command order:
`cefb32e94e5b3bf01f1c3fbf001b06cb5607fcf90c441272930fd6ac6c39d8f6`,
`61f8daa4f4a401ce682c3e83c3f89007bb40cb289b21e0e0f7ed76d261ccde43`,
`1269776a2a86e6a54b30961a725eab49cd210261c78ad35b6e8431856b0cc21b`,
`eed50a085bf460e91dbdeb9145fa4eda73206d3c8819fac5b6277c8681345b9a`.
The tracked compact JSON SHA-256 is
`1b9b111a187e523ef98527a28713c11c5dd992b56f1a996ee945832119f7bffb`.
These packet hashes identify local replay inputs; the raw logs and generated
fixtures are ignored caches and are **not** independently durable. The tracked
scripts and compact matrix are the reviewable, reproducible evidence. No
historical map, release bundle, or preregistered manifest was edited.

Artifact disposition (`artifact_provenance_summary.v1`): compact JSON and this
report are `tracked-compact-evidence`; generated fixtures and packet logs are
`ignored-cache`; nominal release rows and full-oracle success evidence were not
produced. Keep #9735 open until independent review and the missed gates receive
their own evidence.
