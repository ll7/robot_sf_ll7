---
name: gh-pr-comment-fixer
description: "Fix GitHub PR review comments using the gh CLI: fetch review threads, implement requested changes, run the full test suite, commit and push fixes, then resolve the review threads. Use when asked to address PR comments or review feedback on the current branch."
---

# Gh Pr Comment Fixer

## Overview

Use the gh CLI to retrieve PR review comments for the current branch, apply fixes, verify with
the full test suite, and resolve the review threads after pushing.

## Workflow

1. Identify the PR for the current branch.
   - Prefer `gh pr view --json number,headRefName` and confirm it matches the branch.

2. Retrieve review threads and comments.
   - Use GraphQL for review threads:
     `gh api graphql -F owner=<owner> -F repo=<repo> -F number=<pr_number> -f query='query($owner:String!,$repo:String!,$number:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$number){reviewThreads(first:100){nodes{id isResolved comments(first:100){nodes{id body path line url}}}}}}}'`
   - Keep the review thread `id` with each requested change. GitHub resolves review threads by
     thread id, not by the inline comment id or a reply id.
   - Use `gh pr view --json comments` for top-level discussion comments.
   - Summarize the requested changes and note any questions or ambiguities.

3. Evaluate each comment.
   - Decide if it is reasonable, needs clarification, or should be declined.
   - If clarification is needed, ask before changing code.

4. Implement fixes.
   - Make minimal, targeted edits.
   - Add or update tests if the change warrants it.
   - Keep commit scope aligned with a single comment or logical group.

5. Run the full test suite.
   - Use the repo standard test command unless instructed otherwise.
   - Report failures and decide whether to continue or request input.

6. Check local artifact persistence before committing or pushing.
   - Inspect ignored/generated outputs:
     - `git status --ignored --short -uall output`
   - For likely durable artifacts, inspect size, timestamps, and hashes with `find output ...`,
     `du -sh`, and `sha256sum`.
   - Decide whether each relevant artifact is disposable, an ignored cache, represented by a
     tracked manifest/registry entry, or must be uploaded to durable storage.
   - Treat benchmark bundles, model checkpoints, W&B run outputs, policy-analysis reports, and
     config dependencies under `output/model_cache` as durable-candidate artifacts.
   - If the review fix introduces or discovers a dependency on an ignored local artifact, make it
     persistent before pushing, preferably via `model/registry.yaml` with W&B artifact metadata or
     another explicit durable reference.
   - Mention the artifact persistence decision in the PR reply or follow-up comment.

7. Commit and push fixes.
   - Use a conventional commit message.
   - Mention which comments were addressed.

8. Resolve review threads.
   - Replying to a review thread is not the same as resolving it. After the fix is committed,
     pushed, and validated, resolve each addressed review thread with GraphQL:

     ```bash
     gh api graphql \
       -F thread_id=<review_thread_id> \
       -f query='mutation($thread_id:ID!){resolveReviewThread(input:{threadId:$thread_id}){thread{id isResolved}}}'
     ```

   - Resolve only after the fix is pushed. Do not resolve threads that were merely acknowledged,
     still need reviewer input, failed validation, or were converted into a follow-up issue without
     satisfying the current PR.
   - Re-query review threads after resolution and verify that each addressed thread reports
     `isResolved: true` before saying the review comments are resolved:

     ```bash
     gh api graphql \
       -F owner=<owner> \
       -F repo=<repo> \
       -F number=<pr_number> \
       -f query='query($owner:String!,$repo:String!,$number:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$number){reviewThreads(first:100){nodes{id isResolved comments(first:100){nodes{id body path line url}}}}}}}'
     ```

   - If GraphQL auth, rate limits, or permissions block resolution, leave the PR comment or final
     handoff explicit: fixes were pushed, but the named review thread ids remain unresolved.

## Notes

- Prefer `gh pr view` and `gh pr diff` over manual web browsing.
- If no PR exists for the current branch, ask the user to open one or provide a PR number.
- Post multiline comments using a body file, not escaped `\n`:
  - Preferred helper: `scripts/dev/gh_comment.sh pr --current <<'EOF' ... EOF`
  - Alternative: `gh pr comment <num> --body-file <file>`
