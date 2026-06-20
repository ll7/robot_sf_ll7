# Issue #3181 AMV Feasibility Ranking Evidence

This directory contains compact tracked evidence for the local #3181 diagnostic slice. Raw
campaign output remains disposable under `output/`.

- Config: `configs/benchmarks/issue_3181_amv_feasibility_ranking_slice_v0.yaml`
- Campaign id: `issue_3181_amv_feasibility_ranking_slice_v0_local_20260620_191559`
- Scope: 2 scenarios (`classic_bottleneck_high`, `classic_cross_trap_high`) x 2 seeds (`101`,
  `102`) x 2 variants.
- Durable summary: `summary.json` is the exact output of
  `scripts/validation/run_multi_amv_smoke.py --actuation-ranking-episodes ... --actuation-ranking-campaign-summary ...`.
  The campaign summary is required because raw episode rows do not carry planner-row
  `readiness_status` or `availability_status`; without explicit campaign status enrichment, the
  ranking helper fails closed.
- Result: bounded diagnostic feasibility direction only. `actuation_aware_hybrid_rule_v0` reduced
  or tied command clipping in every paired row, but success stayed zero, one pair collided in both
  variants, and one cross-trap row traded clipping improvement against worse final progress.
- Parent validation included `tests/benchmark/test_multi_amv.py -q`,
  `tests/benchmark/ -k "amv or actuation" -q`, `ruff check`, `ruff format --check`, JSON parsing,
  and camera-ready preflight for the config.

Do not use this artifact as benchmark-strength, paper-facing, planner-improvement, or
hardware-calibrated AMV evidence.
