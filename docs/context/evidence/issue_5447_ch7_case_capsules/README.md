<!-- AI-GENERATED (robot_sf#5478, 2026-07-13) - NEEDS-REVIEW -->
# Issue #5447 — Chapter 7 causal-trajectory case capsules (builder scaffold)

**Status:** builder/schema/validator delivered; **capsule set is blocked on data**
(the validated #5446 candidate manifest does not exist yet). Evidence grade:
`diagnostic-only` tooling — no benchmark or paper-facing claim is made here.

## What this slice delivers

A reproducible, fail-closed **case-capsule manifest builder** for the Chapter 7
worked examples:

- `robot_sf/benchmark/case_capsules.py` — builder + structural validator.
  Emits a `ch7_case_capsule_manifest.v1` from a *validated*
  `seed_flip_inversion_candidates.v1` candidate manifest (issue #5446) plus
  optional causal / online-risk reports (issues #5441–#5445).
- `scripts/analysis/build_ch7_case_capsules_issue_5447.py` — thin CLI.
- `configs/analysis/issue_5447_ch7_case_capsules.yaml` — frozen selection
  contract (archetype targets, honest floor/ceiling, build command).
- `tests/benchmark/test_case_capsules_issue_5447.py` — contract tests.

## Honesty contract (issue #5447)

- **Fail closed.** An empty / wrong-schema / candidate-free manifest raises
  `CaseCapsuleError`. Fewer than `min_capsules` admissible archetypes →
  `status: insufficient_evidence` (stop at a smaller honest set; do not broaden).
- **Never fabricate.** An archetype whose source candidate or required
  causal/risk report is missing is emitted `status: unavailable` with a concrete
  reason — never replaced by an attractive unvalidated row.
- **Descriptive-only unless validated causal report.** Causal labels come only
  from a supplied validated causal report; otherwise the capsule is graded
  `descriptive-only` and its caption must say so. Risk archetypes require a
  validated online-risk report or they are `unavailable`.
- **Author-pending, not fabricated, narrative.** Subjective fields (competing
  explanation, what failed, generalisation limits, marked times, "why this time
  matters") are set to the `AUTHOR_REQUIRED` sentinel and reported by the
  validator as `author_pending` — structurally valid, honestly incomplete.
- **Input-hash provenance.** The emitted manifest records the canonical SHA-256
  of the candidate manifest so the capsule set is pinned to its inputs.

## Why the capsule set itself is not built here

Per the issue-#5447 dependency (`Depends on: validated candidate manifest from
#5446; causal/risk reports from #5441–#5445 where available`) and the maintainer
triage comment (2026-07-13), the frozen `seed_flip_inversion_candidates.v1`
candidate manifest **does not yet exist**: the #5446 slice (PR #5466) delivered
only the miner *tooling*; running it on a real eligible native-execution campaign
table is the remaining blocker there. Building capsules from fabricated or
degraded rows is forbidden. This slice therefore delivers the reproducible
builder/validator so the capsule set drops out deterministically the moment the
pinned candidate manifest exists.

## Reproducible build command (once inputs exist)

```bash
uv run python scripts/analysis/build_ch7_case_capsules_issue_5447.py \
    --candidates ${CANDIDATE_MANIFEST} \
    --causal ${CAUSAL_REPORTS} \
    --risk ${RISK_REPORTS} \
    --json output/issue_5447/ch7_capsules.json \
    --validate
```

## Validation performed

```bash
uv run pytest -q tests -k 'case_capsule or trajectory_visualization'
git diff --check
```

## Remaining work (blocked / downstream)

1. **Blocked on #5446:** a pinned `seed_flip_inversion_candidates.v1` manifest
   from a real eligible native-execution campaign table.
2. **Blocked on #5441–#5445:** validated causal / online-risk reports (otherwise
   causal/risk archetypes stay `unavailable`; non-causal capsules are
   `descriptive-only`).
3. **Downstream:** vector-figure rendering from pinned episode trajectories
   (reuse `robot_sf/benchmark/figures`), author-field completion, and the
   independent frozen-SHA visual/evidence review before dissertation integration.

<!-- /AI-GENERATED -->
