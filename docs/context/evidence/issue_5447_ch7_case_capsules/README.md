<!-- AI-GENERATED (robot_sf#5447 cheap-lane worker) - NEEDS-REVIEW -->
# Issue #5447 — Chapter 7 causal-trajectory case capsules

**Status:** data dependency resolved; capsule artifact set materialized and
pinned. Evidence grade: `descriptive-only` (no validated causal/risk reports
yet). Author-pending narrative/figure fields remain; an independent pinned-SHA
visual/evidence review is still required before dissertation integration.

## What this slice delivers

A **reproducible, pinned case-capsule artifact set** built from the validated
#5446 candidate manifest (which now exists at a frozen SHA under
`docs/context/evidence/issue_5446_release_0_0_3_candidates/`):

- `scripts/analysis/materialize_issue_5447_capsules.py` — reproducible driver
  that reads the pinned candidate manifest and writes the frozen artifact set.
- `docs/context/evidence/issue_5447_ch7_case_capsules/ch7_case_capsule_manifest.v1.json`
  — the schema-versioned capsule manifest.
- `docs/context/evidence/issue_5447_ch7_case_capsules/pre_selection_ledger.v1.json`
  — full candidate-pool → chosen/unavailable audit (rejects cannot be hidden).
- `docs/context/evidence/issue_5447_ch7_case_capsules/build_command.v1.txt`
  — exact build + downstream figure/export commands.
- `docs/context/evidence/issue_5447_ch7_case_capsules/SHA256SUMS` — per-directory
  `sha256sum -c` checksum sidecar for the emitted files (verify from this dir:
  `sha256sum -c SHA256SUMS`). The pinned #5446 candidate-manifest hash is recorded
  inside the manifest `inputs.candidate_manifest_sha256` and the ledger
  `candidate_manifest` block, not as a cross-directory SHA256SUMS line.

The builder/validator itself (`robot_sf/benchmark/case_capsules.py`) and the
one-off CLI (`scripts/analysis/build_ch7_case_capsules_issue_5447.py`) were
delivered earlier (PR #5478). This slice runs them on real data and freezes the
output.

## Honest result (run against the pinned #5446 manifest)

- **Status:** `insufficient_evidence` (3 admitted of 4 minimum).
- **Admitted (descriptive-only):** `hard_vs_easy_seed` (seed_flip /
  classic_doorway_medium / ppo), `strong_fail_weak_success` (planner_upset /
  classic_realworld_double_bottleneck_high / goal>ppo), `unexpected_recovery`
  (seed_flip / classic_cross_trap_high / ppo).
- **Unavailable (never substituted, labelled with a concrete reason):** `paired_first_unsafe_action`
  and `ambiguous_abstention` → `causal_report_unavailable`; `near_miss_online_risk`
  → `risk_report_unavailable`. These require validated reports from #5441–#5445,
  which are still open.
- Per the issue #5447 stop rule, the set is reported honest at 3 rather than
  broadened to a count it cannot defend.

## Pinned inputs (reproducibility)

- Candidate manifest: `docs/context/evidence/issue_5446_release_0_0_3_candidates/seed_flip_inversion_candidates.v1.json.gz`
  (gz SHA256 `1021e2fa…83dd64`; canonical dict SHA256
  `cf9caa96…5984f45`, recorded in the emitted manifest's `inputs`).
- Validated causal / online-risk reports: **unavailable** (issues #5441–#5445
  still open). Capsules are therefore graded `descriptive-only`, not `causal`.

## Reproduce

```bash
uv run python scripts/analysis/materialize_issue_5447_capsules.py
```

## Remaining work (downstream, not blocked on data)

1. Complete author-pending narrative/figure fields (competing explanation, what
   failed, generalisation limits, marked times, "why this time matters") and the
   pair shared-axis specs — per-capsule `AUTHOR_REQUIRED` sentinels.
2. Vector-figure rendering from pinned episode trajectories (reuses
   `robot_sf.benchmark.figures`; requires the #5615/#5446 trace resolution).
3. One independent pinned-SHA visual/evidence review before dissertation
   integration.
4. If #5441–#5445 deliver validated causal/risk reports, first extend the
   materialize driver with explicit, validated report-input paths, then regenerate
   the set; this slice intentionally exposes no causal/risk CLI options.

<!-- /AI-GENERATED -->
