# Issue #1569 AMV Actuation Smoke Evidence

This bundle preserves a **small tracked summary** of the local 2026-05-27 review run for the
synthetic AMV actuation-envelope stress slice from `issue_1556_amv_actuation_stress_slice_v0`.

## Scope

- Verdict: `compact smoke run`
- Claim boundary: local, non-paper-facing, synthetic diagnostic only
- Claim-map effect: unchanged
- Source config: `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml`
- Source campaign id:
  `issue_1556_amv_actuation_stress_slice_v0_issue1569-smoke_20260527_092250`

## Provenance

- Smoke command:
  `source .venv/bin/activate && uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml --label issue1569-smoke --output-root <local camera-ready output root> --log-level WARNING`
- Analyzer command:
  `source .venv/bin/activate && uv run python scripts/tools/analyze_camera_ready_campaign.py --campaign-root <local issue1569 smoke campaign root>`
- Commit: `51eb075d0bd7208efa8eede9fcbcd9a9aa908b68`

## Artifact classification

- local camera-ready campaign tree for the source campaign id: `non-evidence-local-only`
- `docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json`:
  `tracked-compact-evidence`
- `docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/checksums.txt`:
  `tracked-compact-evidence`

## Why This Bundle Is Durable Enough

The local smoke produced a valid camera-ready campaign plus analyzer output, but the raw campaign
tree remains under `output/`. This bundle keeps only the compact review summary and checksums needed
for issue-facing handoff without mirroring the generated campaign artifacts into git.

## Key Takeaways

1. All three planner rows were executable benchmark-success rows by contract (`successful_evidence`)
   with no fallback, degraded, unavailable, or failed rows.
2. Automated analysis reported no internal consistency findings.
3. Episode-level `success_mean` remained `0.0` for `goal`, `orca`, and `social_force`, so the smoke
   does not strengthen any AMV performance claim.
4. `amv_coverage_status` stayed `warn` because the selected scenario rows still carry empty `amv`
   metadata blocks in the scenario matrix.
5. Adapter diagnostics remain caveated: ORCA required command projection for most steps in this
   smoke, and the `social_force` campaign row still reports unknown command-space/projection-policy
   metadata in the underlying campaign output. Follow-up issue #1572 tracks that metadata cleanup.

See `summary.json` for the compact planner-level snapshot promoted from the local campaign.
