---
name: gh-pr-opener
description: "Open a conservative Robot SF PR with scope verification, freshness checks, and artifact discipline."
---

# GH PR Opener

Use this when a branch is ready for PR handoff and must follow Repository-grade evidence rules.

## Key Guardrail

Fail-closed policy: do not open a PR until scope is implemented, proof is fresh, and artifacts are
classified.

## Preconditions

- Branch corresponds to a single clear issue scope.
- Issue contract and PR diff match (or deferred work is captured by follow-up issues).
- Current branch head differs from stale readiness stamps (freshness required).

Freshness check:
- `uv run python scripts/dev/pr_ready_freshness.py status --base-ref origin/main`
- If stale/failing, rerun `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` and write freshness.

## Workflow

1. Confirm branch/issue alignment.
2. Verify scope completion and linked issue status.
3. Sync latest `origin/main`, then rebase/merge according to repo policy.
4. Recheck readiness freshness post-sync.
5. Classify generated artifacts from `output/` (discard/ignored/cache/durable evidence).
6. Run the review audit checklist for changed workflow/skill area.
7. Build PR body from `.github/PULL_REQUEST_TEMPLATE/pr_default.md`.
8. Open draft PR:
   `gh pr create --draft --base main --head <branch> --title "<type>: <summary> (#<n>)" --body-file <prepared_body.md>`.
9. Keep parent issue open unless repository policy indicates closure wording in PR description.

## Proof and Artifact Rules

- PR body must state:
  - implementation summary,
  - validation evidence,
  - artifact classification and provenance decision,
  - follow-up issues, if any.
- Do not commit large temporary artifacts from `output/`; use manifests or external artifact pointers.

## Anti-Loop / Race Rules

- Never rely on stale validation after branch/head changes.
- If issue/branch linkage changes mid-flow, stop and recompute handoff state.
- Avoid force-push during PR open flow.

## Required Output

- PR opened URL,
- branch SHA at open time,
- freshness evidence source,
- artifact decision,
- follow-up issues created.
