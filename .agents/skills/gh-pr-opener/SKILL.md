---
name: gh-pr-opener
description: Open a conservative Robot SF PR with scope verification, freshness checks, and artifact discipline.
category: github-pr
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
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
- `uv run python scripts/dev/pr_ready_freshness.py status --base-ref origin/main --require-clean-tree`
- If stale/failing, prepare the PR body first, then rerun:

  ```bash
  pr_body_file=/absolute/path/to/prepared-pr-body.md
  PR_READY_MODE=final BASE_REF=origin/main \
    PR_READY_PR_BODY_FILE="$pr_body_file" scripts/dev/pr_ready_check.sh
  ```

  The wrapper records clean committed-HEAD freshness after all gates pass.
- Use plain `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` only for interim dirty-tree
  feedback before PR handoff.

Remote-state check (issue #6916):
- Capture a baseline before the expensive readiness run with:

  ```bash
  uv run python scripts/dev/check_prepublication_state.py capture --repo <owner/repo> \
    --issue <number> --branch <head-branch> \
    --snapshot-path output/validation/prepublication/<head>.json
  ```
- Immediately before opening or updating the PR, run:

  ```bash
  uv run python scripts/dev/check_prepublication_state.py check \
    --snapshot-path output/validation/prepublication/<head>.json
  ```

  Exit 0 means `ready`; exit 2 means `refresh-required`; exit 3 means `superseded`; exit 4 means
  the state could not be established. A superseded or blocked result must stop publication. Only
  exit 2 authorizes the synchronization path below; do not use it to bypass blocked undeclared or
  mismatched ancestry.
- When `check` returns exit 2 for a moved remote base or branch tip, run the following from a clean
  worktree:

  ```bash
  uv run python scripts/dev/check_prepublication_state.py sync \
    --snapshot-path output/validation/prepublication/<head>.json --integrate
  ```

  It fetches the current refs and uses ordinary Git merge operations without resetting or deleting
  local state. Resolve any conflict, rerun readiness for the integrated head, push that exact head,
  capture a new baseline, and run the final check again. Do not treat sync's self-comparison receipt
  as publication proof. If `check` instead reports blocked ancestry, preserve that signal and
  reconstruct only the intended commits on current `origin/main`; merging main can hide inherited
  parent commits and is not valid remediation.

## Workflow

1. Confirm branch/issue alignment.
2. Verify scope completion and linked issue status.
3. Sync latest `origin/main`, then rebase/merge according to repo policy.
4. Build PR body from `.github/PULL_REQUEST_TEMPLATE/pr_default.md` before final readiness so the
   PR contract gate can read it.
   - For evidence-producing PRs, fill `Downstream Propagation` instead of leaving it implicit.
     Check the parent issue, claim map or benchmark report, leaderboard or artifact catalog,
     registry or config index, context index or memory note, and follow-up issue rows.
   - For low-risk or non-evidence PRs, write a short `Not applicable because:` rationale so the
     omission is intentional and reviewable.
   - Recent example: PR #2044 promoted compact trace-viewer screenshot evidence and updated the
     context index/catalog so the visual proof survived worktree cleanup.
5. Classify generated artifacts from `output/` (discard/ignored/cache/durable evidence).
6. Run the review audit checklist for the changed workflow/skill area.
7. Push the clean committed head that will become the PR head.
8. Capture the remote-state baseline from that synchronized local/remote head.
9. Run final readiness with `PR_READY_PR_BODY_FILE="$pr_body_file"`, then run the remote-state
   check immediately before publication. If sync integrates drift, rerun readiness, push the
   integrated head, recapture the baseline, and check again.
10. Open a ready PR by default using
   `gh pr create --base main --head <branch> --title "<type>: <summary> (#<n>)" --body-file <prepared_body.md>`.
   Use `--draft` only when the user explicitly requests draft status or when the branch is an
   intentional handoff with incomplete validation, unresolved scope, or another clearly documented
   reason that should block review.
    For an existing PR, reconcile its final title and body with
    `uv run python scripts/dev/gh_pr_body_rest.py <pr-number> --reconcile --title "<final title>" --repo ll7/robot_sf_ll7 --body-file <prepared_body.md>`;
    this is an explicit no-op when both fields already match. Keep the existing
    body-only mode for compatibility, but use reconciliation for any final-state
    handoff. Do not use `gh pr edit --body-file` while it queries retired Projects
    Classic fields.

    For label operations on issues or PRs, use
    `uv run python scripts/dev/gh_pr_label_rest.py add <number> --label <name> --repo ll7/robot_sf_ll7`
    or
    `uv run python scripts/dev/gh_pr_label_rest.py remove <number> --label <name> --repo ll7/robot_sf_ll7`
    instead of `gh pr edit --add-label` / `gh issue edit --label` which route through the same
    deprecated Projects Classic GraphQL path.
11. Keep parent issue open unless repository policy indicates closure wording in PR description.

## Proof and Artifact Rules

- PR body must state:
  - implementation summary,
  - validation evidence,
  - artifact classification and provenance decision,
  - downstream propagation decisions for evidence-producing changes,
  - follow-up issues, if any.
- Do not commit large temporary artifacts from `output/`; use manifests or external artifact pointers.

## Final-State Metadata

After any revision or fix push, rebuild the PR title/body from the final diff, validation, claims,
and follow-ups before handoff. Always reconcile the body; change the title only when the final scope,
intent, type, or issue linkage changed. The REST helper is idempotent and patches title and body
together when either differs. Review evidence must then carry the exact `pr-metadata: reconciled @
<digest>` trailer. A stale or missing digest is a handoff blocker, not a reason to invent a summary.

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
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.


## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.


## Output

Return the schema named by the `output_schema` frontmatter field, or a compact equivalent when the caller does not require YAML.
