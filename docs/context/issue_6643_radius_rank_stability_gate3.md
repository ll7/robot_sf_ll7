# Issue #6643 — Gate 3 radius rank-stability analysis and durable evidence bundle (#6600)

## Plain-language summary

Issue #6643 is the **Gate 3 (analysis and durable evidence)** child of the
maintainer-approved radius-sensitivity campaign #6600. Once the Gate 2 production
sweep (#6642) lands, Gate 3 must turn the immutable sweep into:

- planner-ranking tables for success, typed collisions, and SNQI;
- Kendall rank correlation and rank-flip counts versus the 1.0 m baseline;
- per-planner paired changes with uncertainty;
- scenario-family and feasibility transitions, including the narrow-doorway family;
- a fail-closed missingness/degradation ledger;
- a durable evidence bundle with immutable config, command, commit, seed roster,
  artifact checksums, and reproduction instructions.

The final verdict must be exactly one of `stable_within_tested_radii`,
`radius_dependent`, `non_identifiable`, or `invalid_missing_or_inconsistent_evidence`,
posted on #6600 and propagated to #3207. A ranking flip is a valid boundary result,
not a failed experiment.

## Current state (2026-08-03)

- **Tooling: delivered and merged** via PR #6664
  (`robot_sf/benchmark/radius_rank_stability.py`,
  `scripts/benchmark/analyze_radius_rank_stability_issue_6643.py`,
  `tests/benchmark/test_radius_rank_stability.py`). The tool fails closed: without a
  Gate 2 sweep summary it returns `blocked_pending_gate2` (exit 2) and promotes no
  ranking interpretation.
- **Analysis: blocked pending Gate 2.** The Gate 2 production sweep (#6642) has not
  run: Gate 1 (#6641) has not reported a passing binding-canary verdict, and no sweep
  summary or fail-closed missingness ledger exists anywhere (checked `output/`,
  `experiments/`, and the artifact store). Per the #6643 stop rule, no scientific
  verdict may be produced before complete row identities or a fail-closed missingness
  ledger exists.
- **Registered diagnostic bundle:** this worktree ran the tool without a sweep
  summary, which registered a durable `blocked_pending_gate2` evidence bundle
  (`diagnostic-only` evidence tier, `interpretation_promoted: false`). The bundle is
  recorded under the run artifact directory used by this lease; see the reproduction
  command below to regenerate or to rerun once Gate 2 lands.

## Reproduction

Blocked-mode run (registers the diagnostic bundle; exit 2 is the expected
fail-closed status, not a scientific verdict):

```bash
uv run python scripts/benchmark/analyze_radius_rank_stability_issue_6643.py \
  --output-dir <bundle-dir> \
  --config configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml \
  --print-comments
```

Scientific run (only after Gate 2 lands, at the immutable campaign commit):

```bash
uv run python scripts/benchmark/analyze_radius_rank_stability_issue_6643.py \
  --output-dir <bundle-dir> \
  --sweep-summary <gate2-sweep-summary.json> \
  --gate1-canary-receipt <gate1-canary-receipt.json> \
  --config configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml \
  --campaign-commit <immutable-gate2-commit> \
  --print-comments
```

Promotion requirements enforced by `write_evidence_bundle`: checksum-covered config,
sweep summary, and passing Gate 1 canary receipt whose digests match the sweep's own
declared campaign provenance; otherwise the bundle stays `diagnostic-only`.

## Unblock condition

1. Gate 1 (#6641) reports a passing radius-binding-canary verdict on all five binding
   surfaces.
2. Gate 2 (#6642) produces the 3-radius x 14-planner x 48-cell x seeds 111-140 sweep at
   one immutable campaign commit with complete row identities or a fail-closed
   missingness ledger, excluding fallback/degraded/failed/missing/duplicate/
   provenance-invalid rows.
3. Re-run the scientific command above on that immutable summary, then post exactly
   one verdict on #6600 and propagate to #3207.

## Evidence tier

`diagnostic-only` today (blocked pending Gate 2). A complete, identifiable verdict on
valid native rows may become `nominal_benchmark_radius_sensitivity` for radius
sensitivity only. Manuscript admission remains a separate author step via the diss#535
watcher; issue or PR closure alone does not promote a dissertation claim.
