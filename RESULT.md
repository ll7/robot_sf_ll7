# PR #6059 review-blocker remediation

## Fixed comments

- Fixed the provenance-semantics blocker across all 18 Issue #4848 and #4891 exemplar bundles.
  `metadata.json` and embedded `trace_series.json` now preserve
  `git_commit` as `missing:original-generation-commit-not-recoverable`; the reachable
  `0b0214ced856eac77fa9a4c15b02921eabab1661` is recorded only as the distinct
  `published_integration_commit` pointer. README files now label generation provenance as
  unavailable and identify the integration pointer as not the generation commit.
- Refreshed all 18 per-directory `SHA256SUMS` files.
- Updated the PR description to match the remediation: `git_commit` remains unavailable,
  while `published_integration_commit` identifies the reachable byte-identical integration
  source; stale repointing language and file-count text were removed.
- Corrected the durable baseline-review addendum in
  `scripts/validation/evidence_registry_baseline_review.yaml`: the orphaned generation commit
  remains unavailable, and `0b0214ced856eac77fa9a4c15b02921eabab1661` is described only as the
  separate published integration pointer, not as the generation commit.
- No unresolved inline review threads were present before the fix; no thread was resolved.
- The PR head was verified as `cb537c6c661a1347baa4248870b38eb3810f2712` before this report refresh.

## Validation

- `git fetch origin main` and `git merge origin/main` — passed; base was current.
- Provenance consistency probe over 18 bundles — passed.
- `uv run python scripts/tools/lint_evidence_registry.py --strict --strict-exclusion-policy docs/context/evidence/evidence_registry_strict_ci_policy.yaml` — passed; active findings 0.
- `uv run python scripts/dev/evidence_registry_ratchet.py --check` — passed; findings 414, baseline 414.
- `uv run pytest tests/dev/test_evidence_registry_ratchet.py tests/test_export_issue_4848.py tests/tools/test_audit_exemplar_bundles.py -q` — passed; 78 tests.
- `uv run pytest tests/dev/test_evidence_registry_ratchet.py -q` — passed; 24 tests after the
  baseline-review wording fix.
- `uv run python scripts/dev/check_docs_evidence_integrity.py --files <72 changed evidence files>` — passed.
- `git diff --check` — passed.
- PR body verification — passed; the stale `Repointed the producer pointer` claim is absent,
  and the description names `missing:original-generation-commit-not-recoverable` and
  `published_integration_commit`.
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` — blocked before validation by the five
  lease-injected untracked task files (`.ll7_task_*`); changed-file proof cannot be established
  while they remain present. They were preserved because this lease does not authorize deletion.
  The focused evidence gates above passed.

## Commit and push

- Fix commit SHA: `301f8ae2905d46c7c9530651295d789ee2d5de40`.
- Reporting commit SHA: `79b1021188e85263a1cad953fb2a9b7d8d1992b9`.
- Push target: existing PR branch `cheap/cheap-issue-5986-e4ebaff0d9c5`.
- No new PR, merge, worktree deletion, or Slurm submission was performed.

## Post-push review threads

- Pre-push unresolved inline review threads: none.
- Resolved thread IDs: none.
- Post-push unresolved threads: re-queried at head `cb537c6c661a1347baa4248870b38eb3810f2712`; none were present.

## Live PR state after push

- PR head at the final pre-refresh verification: `cb537c6c661a1347baa4248870b38eb3810f2712` on
  `cheap/cheap-issue-5986-e4ebaff0d9c5`.
- Hosted `reproducibility-check` is `SKIPPED` because the workflow is intentionally
  `workflow_dispatch`-only; this lease cannot submit compute or dispatch it.
- GitHub `reviewDecision` is empty/null; no approving human review is recorded.
- Other hosted checks were newly queued or in progress at capture time.

## Blockers

- Hosted `reproducibility-check` remains `SKIPPED`; this lease forbids compute submission, so it cannot be replaced here.
- GitHub `reviewDecision` remains null/empty: the only submitted review is Gemini Code Assist `COMMENTED`; an authorized human approval is still required.
- The five lease task files remain untracked and intentionally untouched; full local changed-file
  proof therefore remains unavailable in this worktree.
