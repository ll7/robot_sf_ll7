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

## Versioned report contract

The current producer emits `radius_rank_stability.v2`. The v2 envelope is required because the
report now carries the required `paired_inference_contract` block and per-contrast `support`
diagnostics. The historical `radius_rank_stability.v1` shape remains a legacy identifier; it is
not upgraded implicitly and is rejected by the v2 writer-side validator when its new blocks are
missing.

The durable bundle provenance emits
`issue_6643_radius_rank_stability_bundle.v2` and pins `report_schema_version` to
`radius_rank_stability.v2`. This is a compatibility contract only: no radius campaign, numeric
result, evidence admission, ranking, safety conclusion, or paper-facing claim is created by the
version change.

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
   the Issue #3207 propagation comment.

## Gate 2 summary composer contract (2026-09-22)

The deterministic composer reads exactly three complete camera-ready campaign roots and writes the
`issue_6642_radius_sweep_summary.v1` input expected by Gate 3:

```bash
uv run python scripts/benchmark/compose_radius_sweep_summary_issue_6643.py \
  --campaign-root <complete-0.5m-campaign-root> \
  --campaign-root <complete-0.8m-campaign-root> \
  --campaign-root <complete-1.0m-campaign-root> \
  --gate1-canary-receipt <exact-original-gate1-receipt.json> \
  --output <gate2-sweep-summary.json>
```

Each root must contain its canonical `campaign_manifest.json`,
`preflight/validate_config.json`, `reports/campaign_summary.json`, and one complete
`runs/<planner>__differential_drive/episodes.jsonl` for every frozen planner. The composer verifies
the exact 3-radius × 14-planner × 48-scenario × 30-seed identities, rejects duplicate rows, and
derives success, typed-collision, and SNQI aggregates and seed-keyed pairs from the episode records.
Each episode must carry at least one planner identity in `algo`,
`algorithm_metadata.algorithm` / `canonical_algorithm`, or `result_provenance.planner_key`; every
present identity must canonicalize to its run-directory planner, and missing or conflicting
identities fail closed.
It also reconciles each planner row's serialized success, pedestrian-collision,
obstacle-collision, total-collision, and SNQI means against those same records at the camera-ready
four-decimal precision; status/count metadata alone is insufficient.
Typed collisions mean the recorded `ped_collision_count + obstacle_collision_count`; the composer
requires that sum to equal `total_collision_count` on every episode rather than reinterpreting the
untyped CSV collision column.

The three arm configs are intentionally different tracked treatment configs, so their SHA-256
digests must remain recorded per arm. Gate 3 requires the exact frozen #6642 campaign commit
(`aabad2e2a82cd8dcca93cc78a01493ec6ead5212`) and one shared Gate 1 receipt digest; it does not
require the three arm config digests to be equal. The bundle's `--config` path remains the 1.0 m
baseline config. For each arm, the composer reads the config blob from the frozen commit, verifies
its bytes against the pinned digest, and then compares the preflight's digest with those bytes. A
missing Git object or mismatch fails closed. The receipt must also match the exact frozen digest in
the tracked Gate 2 manifest/config contract; coordinated edits to gate and arm metadata cannot admit
a different passing receipt. Composed summaries record stable campaign IDs and repository-relative
config paths, not host-specific campaign roots.

Family feasibility is not inferred from success rates, route-clearance warnings, or the separate
#6644/#6645 narrow-doorway diagnostics. Every campaign root must instead carry a checksum-bound
`reports/radius_family_feasibility.json` receipt with the same explicitly pinned rule identity and
family roster across arms:

```json
{
  "schema_version": "issue_6642_family_feasibility.v1",
  "radius_m": 0.5,
  "source_campaign_id": "<exact campaign id>",
  "source_campaign_commit": "<exact 40-character campaign commit>",
  "source_config_sha256": "<exact radius-arm config SHA-256>",
  "approved_rule": {
    "definition_id": "<reviewed owner-approved rule id>",
    "authority_sha256": "<digest of the durable approved rule artifact>"
  },
  "definition": "<approved preregistered aggregation rule>",
  "families": {
    "narrow_doorway": "feasible"
  }
}
```

Statuses are limited to `feasible` and `infeasible`, and `narrow_doorway` is mandatory. The expected
rule ID and authority digest are currently unset in source because the live #6600/#6642 contracts do
not define or pin an approved family-level aggregation. The composer also has no in-tree evaluator
that recomputes family results from the exact admitted episode rows, so production family-receipt
acceptance is deliberately impossible in this revision. A receipt or checksum alone binds bytes; it
does not establish that the family labels follow an approved rule. A separate reviewed change must
supply both the durable owner-approved rule artifact and an evaluator that independently recomputes
the family results from the exact source rows. Pinning identity strings alone cannot enable the
path. Synthetic evaluator stubs in focused unit tests exercise summary mechanics only and are not
campaign or benchmark evidence. The current preserved job 15504 (complete 0.5/0.8 m arms) and
recovery job 15516 (complete 1.0 m arm) also do not contain an approved family-feasibility block.
The exact original Gate 1 receipt bytes whose declared SHA-256 is
`88ab630a555ce4a0a6e0b273e6808bc56bffbfa16c57ac3b579c97eb179d9922` are also absent from the
preserved campaign trees. A source replay can test the canary behavior, but a byte-different replay
cannot replace that receipt for promoted provenance.

Therefore the remaining unblock inputs are: (1) an owner-approved, preregistered family-feasibility
aggregation and checksum-bound per-arm mappings produced from the preserved rows, and (2) recovery
of the exact original passing Gate 1 receipt bytes. Until both exist, no Gate 3 scientific verdict
or downstream propagation is valid.
