# Robot SF Skills

This directory contains repo-local skills for Codex-style agents. Use this file as a quick index;
read the specific `SKILL.md` before applying a skill.

## Selection Guide

- Not sure which skill applies: `skill-picker`
- Implement an issue end-to-end: `gh-issue-autopilot`
- Address PR review comments: `gh-pr-comment-fixer`
- Open a PR from a ready branch: `gh-pr-opener`
- Verify a branch against `origin/main`: `implementation-verification`
- Run only the standard gate: `pr-ready-check`
- Review benchmark-sensitive changes: `review-benchmark-change`
- Update docs after code changes: `update-docs-on-code-change`
- Create or refresh linked context notes: `context-note-maintainer`
- Gather repository context before editing: `context-map`, `benchmark-overview`, or
  `experiment-context`
- Submit Auxme issue-791 jobs reliably: `auxme-slurm-reliable-submit`

## Skill Routing

- `quality-playbook` plans proof for non-trivial work; `implementation-verification` audits whether
  branch claims are actually proven against `origin/main`.
- `clean-up` is a branch-tidying workflow; `pr-ready-check` is the narrower readiness gate.
- `agentic-eval` evaluates skills/prompts/instructions; `auto-improvement` iterates on a small
  prompt, docs, or code improvement target.
- `review-benchmark-change` reviews benchmark-sensitive semantics; `implementation-verification`
  can include benchmark checks but starts from branch claims and evidence.
- `context-map` discovers the relevant surface before work; `skill-picker` chooses which skill to
  use when routing is unclear.

## GitHub Skill Policy

- Prefer GitHub MCP / GitHub app tools for interactive reads and comments when available.
- Use `gh` for deterministic batch/project operations, auth debugging, and GraphQL review-thread
  state that MCP does not expose cleanly.
- Batch issue cleanup separately from Project #5 metadata writes; follow
  `docs/context/issue_713_batch_first_issue_workflow.md`.
- Use Project #5 `Priority Score` as the issue-ordering source; use `gh-issue-priority-assessor`
  when the score inputs need review.
- Use `scripts/dev/gh_comment.sh` for multiline PR/issue comments.

## Maintenance

- Run `uv run python scripts/dev/check_skills.py` after adding, renaming, or removing skills.
- Keep skill directory names aligned with `name:` frontmatter.
- Keep this README in sync with every repo-local `SKILL.md`.

## Available Skills

| Skill | Use When |
| --- | --- |
| `agentic-eval` | Evaluating or improving skills, prompts, instructions, or other AI workflow artifacts. |
| `analyze-camera-ready-benchmark` | Auditing a camera-ready benchmark campaign for consistency, runtime, fallback, and reproducibility signals. |
| `analyze-latest-policy-sweep` | Comparing recent policy-analysis sweep outputs and optional video/frame artifacts. |
| `auto-improvement` | Running a short measurement-aware refinement loop for prompts, docs, or small code changes. |
| `auxme-slurm-reliable-submit` | Submitting Auxme issue-791 jobs with explicit config, live partition checks, and max-time-safe wrapper routing. |
| `autoresearch` | Running an autonomous experiment loop with baseline, variants, and keep/discard decisions. |
| `benchmark-overview` | Getting fast benchmark-faithful orientation for scenario splits, baselines, metrics, and artifacts. |
| `clean-up` | Tidying the current branch with Ruff, tests, changed-file gates, and docstring TODO checks. |
| `context-map` | Building a focused map of relevant files, docs, commands, and risks before multi-file work. |
| `context-note-maintainer` | Creating or refreshing linked `docs/context/` notes so reusable agent knowledge stays discoverable and current. |
| `experiment-context` | Finding canonical config-first training/evaluation paths, artifact lineage, and validation gates. |
| `gh-issue-autopilot` | Selecting or executing an issue through implementation, validation, push, and draft PR. |
| `gh-issue-clarifier` | Tightening ambiguous GitHub issues and marking decision-required items when needed. |
| `gh-issue-creator` | Creating structured GitHub issues from rough prompts with repo template conventions. |
| `gh-issue-priority-assessor` | Reviewing Project #5 priority inputs and proposing field values with uncertainty. |
| `gh-issue-template-auditor` | Auditing existing issues against the issue-template contract and repairing clear gaps. |
| `gh-pr-comment-fixer` | Fetching PR review comments, implementing fixes, validating, pushing, and resolving threads. |
| `gh-pr-opener` | Opening Robot SF PRs with issue-scope verification and PR-readiness freshness checks. |
| `implementation-verification` | Comparing the current branch to `origin/main` and proving each claimed feature works as designed. |
| `paper-facing-docs` | Drafting or reviewing benchmark/manuscript-support docs with conservative provenance handling. |
| `planner-integration` | Assessing planner-family adapter burden, provenance safety, and benchmark readiness. |
| `pr-ready-check` | Running the standard PR readiness pipeline through shared `scripts/dev` entry points. |
| `quality-playbook` | Planning and validating non-trivial changes with proof-first risk classification. |
| `review-and-refactor` | Inspecting a small code/docs surface and making a narrow justified improvement. |
| `review-benchmark-change` | Reviewing benchmark-sensitive patches for semantic, normalization, reproducibility, or provenance regressions. |
| `skill-picker` | Choosing the most appropriate repo-local skill when the task is ambiguous or the user asks for skill routing. |
| `svg-inspection` | Debugging SVG map parser issues such as route-only mode, zone mismatches, and obstacle-crossing routes. |
| `update-docs-on-code-change` | Keeping docs aligned with code changes that alter workflows, commands, contracts, or user behavior. |
| `what-context-needed` | Asking for the minimum missing context when a task cannot be answered safely. |

## Notes

- Prefer the most specific skill that matches the task.
- Combine skills only when they cover different phases, such as `context-map` before
  `implementation-verification`, then `pr-ready-check` before PR handoff.
- This README is an index, not a replacement for each skill's `SKILL.md`.
