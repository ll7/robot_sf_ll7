# Issue #5308 doorway QD campaign — capability archive fixture

This directory holds the archived artifacts from a bounded (<=4h CPU) doorway-family
MAP-Elites / CMA-ME quality-diversity campaign for [issue #5308](https://github.com/ll7/robot_sf_ll7/issues/5308).

It is a **capability artifact, not benchmark evidence** (claim boundary
`capability_not_evidence`). The artifacts demonstrate that the existing capability
(`run_map_elites` + `production_qd_evaluator` + `CMaMeEmitter` + knife-edge warm
start) populates the `(distance_to_human_min, time_to_collision_min)` behavior grid
with distinct certified failure mechanisms when wired to the real production
adversarial pipeline. They are not camera-ready findings, a planner-performance
claim, or a validated QD-superiority result without a held-out comparator.

## Contents

| File | Schema | Purpose |
| --- | --- | --- |
| `archive.json` | `adversarial_qd_archive.v1` | Populated MAP-Elites grid: filled cells, coverage, QD score, distinct certified failure modes. |
| `comparison.json` | `adversarial_qd_archive.v1` (comparison) | Equal-budget MAP-Elites vs single-objective diversity comparison. |
| `campaign_summary.json` | `adversarial_qd_archive.v1` | Run summary provenance block. |
| `run_manifest.json` | `adversarial-qd-campaign-run.v1` | Budget, wall-clock, git head, claim boundary. |

## How it was produced

```bash
uv run python scripts/adversarial/run_qd_campaign_issue_5308.py \
    --config configs/adversarial/issue_5308_qd_doorway.yaml \
    --output-dir <tmp> --budget 120
```

then the four JSON artifacts were copied here as a tracked capability fixture. The
per-candidate bundles (`qd_candidates/candidate_*`) are worktree-local and are not
tracked; `archive.json` carries the replayable candidate geometry for every elite.

## Stop-condition summary (issue #5308 contract)

- `filled_cell_count > 0`: met (8 of 25 cells).
- `>= 2 distinct certified failure modes`: met (`collision`, `timeout`).
- `archive.json schema == adversarial_qd_archive.v1`: met.
- Wall-clock CPU: ~63 s for 120 candidates, far inside the 4h budget.
