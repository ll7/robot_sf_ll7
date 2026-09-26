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
`runs/<planner>__differential_drive/episodes.jsonl` plus the adjacent
`episodes.jsonl.provenance.json` for every frozen planner. Each sidecar must validate as
`benchmark_result_provenance.v1`, name the frozen source commit and
`map_runner.run_map_batch`, be complete, and contain exactly one available `episodes_jsonl`
artifact whose path identifies that planner run. Its `campaign_identity.algorithm` must match that
planner's commit-pinned launch algorithm (which may differ from scenario-resolved algorithms). The
composer hashes the exact episode bytes that it parses and requires that digest to equal the runner
receipt's artifact digest; receipt row links and counts must also match. Episode and receipt files
cannot be reached through symlinked `runs` or planner-run directories. A missing, malformed,
wrong-algorithm, wrong-runner, wrong-commit, wrong-path, incomplete, or stale receipt fails closed.
Both the episode digest and exact sidecar-byte digest are carried per planner into
`campaign_provenance` and Gate 3's evidence provenance.

The composer also binds scenario-matrix provenance to the frozen source input: the manifest and
campaign summary matrix hashes must agree, the receipt's matrix path and digest must identify the
matrix blob at the pinned source commit, and its campaign identity must repeat that matrix hash and
the expected `classic_interactions` suite key. Every receipt row must bind to the corresponding
episode's seed and declare the frozen simulator settings (`horizon: 600`, `dt: 0.1`, and
`record_forces: true`); the episode itself must carry the matching horizon, run horizon, timestep,
and force-recording setting. These checks prevent a self-consistent but unrelated receipt, or
receipt settings that diverge from the frozen campaign inputs, from admitting rows.

This is a runner-produced digest receipt under the repository's current trust model, not a
signature or durable-storage custody attestation. The summary therefore records
`source_integrity_status: runner_receipt_matched`, `artifact_custody_status: unattested`, and
`promotion_allowed: false`. Gate 3 may not convert that state into promoted evidence: the evidence
builder preserves both digests and the durable bundle writer refuses promotion unless a separately
trusted custody attestation is added by a reviewed contract change. A digest or an attestation flag
copied only into `campaign_summary.json` or the composed sweep summary is not independent custody
proof. A raw analyzer `VerdictDecision` is computational output only and must not be propagated
or published without the final evidence object's `promotion_allowed` guard. Synthetic receipts
used by focused unit tests validate plumbing only and are never benchmark evidence.

The composer verifies
the exact 3-radius × 14-planner × 48-scenario × 30-seed identities, rejects duplicate rows, and
derives success, typed-collision, and SNQI aggregates and seed-keyed pairs from the episode records.
It also classifies each episode's runtime status, planner-runtime and foresight fallback markers,
and explicit evidence-eligibility flags; a clean campaign-summary fallback count alone cannot admit
a row.
Each episode's algorithm fields (`algo`, `scenario_params.algo`, and any
`algorithm_metadata.algorithm` / `canonical_algorithm`) must canonicalize to the algorithm resolved
for that planner key and scenario from the campaign config at the pinned source commit.
`scenario_params.algo_config_hash` must match the effective per-scenario runtime config hash from
that same commit: candidate base config plus `params`, then family/scenario overrides, or the
scenario-level algorithm override where present. The resolver is shared with map-runner so raw
candidate-manifest hashes are not mistaken for the config identities recorded in episode rows.
Any explicit planner-key carrier (`planner_key`, `scenario_params.planner_key`, or
`result_provenance.planner_key`) must exactly match the run-directory planner key. If two roster
keys resolve to the same algorithm and config for a scenario, a planner-key carrier is mandatory;
missing, conflicting, or mismatched identity inputs fail closed.

The frozen #6642 hybrid manifests contain real effective-identity collisions. In particular, the
`scenario_adaptive_hybrid_orca_v1` and `scenario_adaptive_hybrid_orca_v2_collision_guard` rows both
resolve to the same ORCA algorithm/config for `francis2023_leave_group`. They share the same
effective identity on 47 of the 48 frozen scenarios; `classic_merging_low` is the sole scenario
where the v2 guard override distinguishes them. The current map-runner episode JSONL producer does
not serialize a planner-key carrier (the camera-ready campaign adds it only to in-memory
annotations), so those rows cannot be disambiguated from episode bytes and are rejected by the
composer. This is an identity-provenance gap, not a claim that their measured outcomes are invalid.
Recovering authoritative row-level keys or producing new episodes with a serialized key is required
before those ambiguous rows can enter a Gate 3 summary.
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

Gate 3 independently validates that each radius has the expected family-provenance schema, exact
radius, non-empty rule definition and identity, unique receipt digest, status-map digest, and
source campaign ID, commit, and config digest matching that radius's campaign provenance. All
arms must use one rule identity, and production rule identity pins must be present. Because those
pins and the evaluator remain unavailable, valid-looking or self-declared family statuses still
produce a blocking `family_feasibility_rule_identity_unpinned` reason; they cannot set
`interpretation_promoted: true` or authorize a Gate 3 verdict. This provenance gate does not admit
family feasibility or promote Gate 3.

The exact original Gate 1 receipt bytes whose declared SHA-256 is
`88ab630a555ce4a0a6e0b273e6808bc56bffbfa16c57ac3b579c97eb179d9922` are also absent from the
preserved campaign trees. A source replay can test the canary behavior, but a byte-different replay
cannot replace that receipt for promoted provenance.

Therefore the remaining unblock inputs are: (1) an owner-approved, preregistered family-feasibility
aggregation and checksum-bound per-arm mappings produced from the preserved rows, and (2) recovery
of the exact original passing Gate 1 receipt bytes. Until both exist, no Gate 3 scientific verdict
or downstream propagation is valid.
