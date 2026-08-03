# Resume state: issue #6706 tie-aware ranking

Task ID: `RSF-FB-2026-08-01`<br>
Issue: https://github.com/ll7/robot_sf_ll7/issues/6706<br>
Branch: `codex/issue-6706-tie-aware-ranking`<br>
Worktree: `/tmp/robot_sf_fb6706`

## Safe resume point

The branch was created from RobotSF `origin/main` at
`4d91ace3b1f5c24eb4a8da19b01eb22ab75b0b27`. The main RobotSF checkout at
`/Users/lennart/git/robot_sf_ll7` was not edited. The dissertation checkout was
not edited in this pause operation.

Current implementation is local and not yet pushed or opened as a PR. The
focused tests pass: `8 passed`.

## Files changed on this branch

- `robot_sf/benchmark/tie_aware_ranking.py`
- `robot_sf/benchmark/schemas/tie_aware_ranking.v1.json`
- `scripts/analysis/export_tie_aware_ranking.py`
- `tests/benchmark/test_tie_aware_ranking.py`
- `tests/analysis/test_export_tie_aware_ranking.py`

The legacy `robot_sf/benchmark/ranking.py` was not changed.

## Implemented behavior

- Exact canonical score equality produces `exact_tie` and tie groups.
- Strict order is emitted only from an approved paired comparison.
- Disjoint marginal intervals without that approval remain `non_identifiable`.
- Interval overlap or contact produces `non_identifiable`.
- Incomplete uncertainty produces `non_identifiable`.
- Missing or invalid support, excluded rows, and invalid evidence produce
  `incomparable` and do not enter rank ranges.
- Approved paired comparisons can provide a directional override.
- Rank ranges are derived from the strict relation graph; display order is
  serialized separately and is never a scientific rank.
- JSON output is deterministic and a Markdown summary is available.

## Resume commands

```bash
cd /tmp/robot_sf_fb6706
git status --short --branch
uv run ruff check robot_sf/benchmark/tie_aware_ranking.py scripts/analysis/export_tie_aware_ranking.py tests/benchmark/test_tie_aware_ranking.py tests/analysis/test_export_tie_aware_ranking.py
uv run ruff format --check robot_sf/benchmark/tie_aware_ranking.py scripts/analysis/export_tie_aware_ranking.py tests/benchmark/test_tie_aware_ranking.py tests/analysis/test_export_tie_aware_ranking.py
uv run pytest tests/benchmark/test_tie_aware_ranking.py tests/analysis/test_export_tie_aware_ranking.py tests/test_ranking.py tests/test_cli_ranking.py -q
python scripts/validation/check_broad_exceptions.py
git diff --check
```

Then review the public contract and schema, commit, fetch, and push:

```bash
git diff --stat
git diff -- robot_sf/benchmark/tie_aware_ranking.py scripts/analysis/export_tie_aware_ranking.py robot_sf/benchmark/schemas/tie_aware_ranking.v1.json
git add robot_sf/benchmark/tie_aware_ranking.py robot_sf/benchmark/schemas/tie_aware_ranking.v1.json scripts/analysis/export_tie_aware_ranking.py tests/benchmark/test_tie_aware_ranking.py tests/analysis/test_export_tie_aware_ranking.py docs/context/issue_6706_tie_aware_ranking_resume.md
git commit -m "feat(benchmark): add tie-aware ranking export"
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git push -u origin codex/issue-6706-tie-aware-ranking
```

Open a PR for issue #6706 only after the checks pass. Do not merge until the
PR has been rebased or otherwise admitted at an exact green head under the
current repository privacy/merge gates.

## Review and stop conditions

- Confirm the output field names and support requirement against the issue
  contract before filing the PR.
- Do not add a tolerance, rounded-display tie, practical-equivalence margin,
  or statistical-equivalence rule.
- Do not modify metric semantics, fairness gates, campaign outputs, release
  data, dissertation wording, or rendering frameworks.
- Stop for maintainer review if the representation requires a new statistical
  equivalence rule or changes the ranking-claim policy.
