# Issue 3190 Closure Audit

This audit maps the acceptance criteria for issue
[#3190](https://github.com/ll7/robot_sf_ll7/issues/3190) to merged pull requests and current
local validation. Issue #3190 covers context-note freshness checking and a review-gated archival
sweep. This note does not add a new checker, move additional notes, or change benchmark claims.

## Acceptance Evidence

| Criterion | Evidence |
| --- | --- |
| `check_context_note_freshness.py` implements Rules A/B/C with `--max-age-days`, `--strict`, and `--json-output`. | PR [#3208](https://github.com/ll7/robot_sf_ll7/pull/3208) added the checker and focused tests. PR [#3436](https://github.com/ll7/robot_sf_ll7/pull/3436) added overlapping checker/test coverage. Local validation: `tests/tools/test_check_context_note_freshness.py` passed on 2026-07-04. |
| Rule A exits non-zero; Rules B/C remain warnings unless `--strict`. | PR [#3208](https://github.com/ll7/robot_sf_ll7/pull/3208) covered these exit semantics with tests. A non-strict repository run on 2026-07-04 exited 0 while reporting warning-only orphan findings. |
| Staleness uses real Git last-touch dates, not file modification time. | PR [#3208](https://github.com/ll7/robot_sf_ll7/pull/3208) implemented Git last-touch date handling and covered it in `tests/tools/test_check_context_note_freshness.py`. |
| Orphan detection cross-references both `docs/context/INDEX.md` and `docs/context/catalog.yaml`. | PR [#3208](https://github.com/ll7/robot_sf_ll7/pull/3208) implemented this rule. PR [#4442](https://github.com/ll7/robot_sf_ll7/pull/4442) added an additional freshness-decay checker with the same Rule A/B/C scope and dry-run archive proposal checks. |
| `tests/tools/test_check_context_note_freshness.py` covers each rule with fixtures. | The test file exists on `origin/main`; the focused local test run passed on 2026-07-04. |
| Archival sweep preserves notes by moving, not deleting; catalog, index, and links are updated; docs proof passes. | PR [#4385](https://github.com/ll7/robot_sf_ll7/pull/4385) archived five catalog-approved superseded notes into `docs/context/archive/`, updated `docs/context/catalog.yaml`, `docs/context/INDEX.md`, `docs/context/README.md`, and inbound links. Local archival planner validation on 2026-07-04 reported `OK no archival candidates found`. |
| Future review-gated sweeps have a fail-closed approval path. | PR [#3694](https://github.com/ll7/robot_sf_ll7/pull/3694) added the archival planner. PR [#4152](https://github.com/ll7/robot_sf_ll7/pull/4152) added approval-manifest validation. Local planner tests passed on 2026-07-04. |
| Freshness proof is available through the docs-proof workflow without making the repository-wide warning backlog a per-PR blocker. | PR [#4241](https://github.com/ll7/robot_sf_ll7/pull/4241) added docs-proof freshness integration. PR [#4330](https://github.com/ll7/robot_sf_ll7/pull/4330) added diff-scoped gating. Local diff-scoped validation passed on 2026-07-04. |

## Current Validation

Commands run from a clean worktree based on `origin/main`:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync pytest \
  tests/tools/test_check_context_note_freshness.py \
  tests/tools/test_plan_context_note_archival.py \
  tests/tools/test_check_context_note_freshness_decay.py \
  tests/validation/test_check_docs_proof_consistency.py -q

scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync python \
  scripts/tools/check_context_note_freshness.py --max-age-days 180 --max-human-findings 20

scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync python \
  scripts/tools/plan_context_note_archival.py --max-age-days 180 --max-human-moves 20

scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync python \
  scripts/tools/check_context_note_freshness_decay.py --max-age-days 180

scripts/dev/run_worktree_shared_venv.sh -- uv run --no-sync python \
  scripts/validation/check_docs_proof_consistency.py \
  --check-context-note-freshness --freshness-scope diff --base origin/main
```

Results:

- Focused tests passed: 88 tests.
- `check_context_note_freshness.py` exited 0 in non-strict mode and reported 465 warning-only
  orphan findings.
- `plan_context_note_archival.py` exited 0 with `OK no archival candidates found`.
- `check_context_note_freshness_decay.py` exited 0 and reported no new archival candidates.
- Diff-scoped docs-proof freshness check exited 0.

## Closure Boundary

This audit supports closing issue #3190 as criteria-complete. It does not claim that the
repository-wide orphan context-note backlog is resolved. Those findings remain warning-only unless
`--strict` is requested, and they are not maintainer-approved archival moves.

No full benchmark campaign, Slurm or GPU submission, paper claim, dissertation claim, release,
merge, or deletion was performed for this audit.
