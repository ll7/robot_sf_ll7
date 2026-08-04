# Issue #6643 — Gate 3 radius rank-stability analysis and durable evidence bundle (#6600)

## Plain-language summary

Issue #6643 is the Gate 3 analysis child of the approved collision-envelope
radius-sensitivity campaign #6600. The merged analyzer will turn an admitted Gate 2
production sweep into planner-ranking tables for success, typed collisions, and the Social
Navigation Quality Index (SNQI); Kendall rank correlation and rank-flip counts against the
1.0 m baseline; paired per-planner changes with uncertainty; scenario-family and feasibility
transitions, including the narrow-doorway family; and a checksum-bound evidence bundle.

The final scientific verdict vocabulary is exactly `stable_within_tested_radii`,
`radius_dependent`, `non_identifiable`, or `invalid_missing_or_inconsistent_evidence`.
A ranking flip is a valid boundary result. The verdict must be posted once on #6600 and
propagated to #3207 only after the Gate 2 input gate is satisfied.

## Current state (2026-08-04)

- **Analysis tooling: delivered and merged.** The current `origin/main` contains
  `robot_sf/benchmark/radius_rank_stability.py`,
  `scripts/benchmark/analyze_radius_rank_stability_issue_6643.py`, and focused tests.
  The analyzer requires the exact three-arm scope, fixed 48-cell matrix, 14-planner roster,
  seeds 111–140, row accounting, paired observations, family feasibility, and matched
  campaign provenance before it can promote a result.
- **Gate 1: admitted as a runtime-binding prerequisite, not as campaign evidence.** The
  passing receipt proves the declared radius reaches the required simulator and output
  surfaces; it does not establish rank stability, radius dependence, safety, physical
  footprint, simulator realism, sim-to-real validity, or dissertation evidence.
- **Gate 2: not produced.** The repository and the lease’s shared factory artifact store
  contain preparation/admission manifests and preflight material, but no production sweep
  summary with complete row identities and no fail-closed missingness ledger. This lease has
  no compute-submit authorization, so no production campaign was run here.
- **Scientific verdict: not emitted.** The analyzer’s `blocked_pending_gate2` result is a
  pre-analysis gate status, not one of the four scientific verdicts. No ranking result is
  promoted, and no verdict is posted to #6600 or propagated to #3207 while the required Gate 2
  input is absent.

## Diagnostic bundle and evidence boundary

The blocked-mode analyzer invocation registers a diagnostic-only bundle outside the product
worktree under the lease’s external artifact directory. It records the exact analysis commit,
command, config path when supplied, and the blocked status. It is control-plane handoff evidence,
not benchmark evidence and not a substitute for Gate 2 rows.

Because no campaign has run, the blocked bundle leaves `campaign_commit` unavailable instead of
substituting the analysis commit. A promoted bundle must carry a real campaign commit that matches
all three Gate 2 arms and the checksum-covered provenance.

```bash
uv run python scripts/benchmark/analyze_radius_rank_stability_issue_6643.py \
  --output-dir /home/luttkule/.local/state/ll7-factory/runs/ll7-lease-6643-6dfddeb98161/blocked-evidence-bundle \
  --config configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml \
  --print-comments
# exit 2 is expected: blocked_pending_gate2
```

The bundle must remain `diagnostic-only` with `interpretation_promoted: false`. Its claim
boundary is within-simulator radius sensitivity only—not physical-footprint validation,
simulator-realism evidence, sim-to-real evidence, or a safety guarantee. Manuscript admission
remains a separate author step via the diss#535 watcher.

## Scientific reproduction after the unblock

Run this only after Gate 2 provides complete native row identities or an explicit fail-closed
missingness ledger, with all arms at one immutable campaign commit:

```bash
uv run python scripts/benchmark/analyze_radius_rank_stability_issue_6643.py \
  --output-dir <bundle-dir> \
  --sweep-summary <gate2-sweep-summary.json> \
  --gate1-canary-receipt <gate1-canary-receipt.json> \
  --config configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml \
  --campaign-commit <immutable-gate2-commit> \
  --print-comments
```

Before posting anything, verify the bundle’s config, command, campaign commit, analysis commit,
seed roster, input/output SHA-256 checksums, and reproduction instructions. Exclude fallback,
degraded, failed, missing, duplicate, and provenance-invalid rows. Then post exactly one valid
scientific verdict on #6600 and propagate the same decision to #3207; do not infer or promote a
dissertation claim from issue closure.

## Unblock condition

1. Gate 2 (#6642) produces the 0.5/0.8/1.0 m × 14-planner × 48-cell × seeds 111–140
   production result at one immutable campaign commit.
2. Every declared row is present and valid, or the summary carries a complete fail-closed
   missingness/degradation ledger that the analyzer can classify as invalid evidence.
3. The Gate 1 receipt, config checksum, campaign commit, and summary provenance match.
4. Rerun the scientific command, review the durable bundle, and post the one verdict plus
   #3207 propagation comment.
