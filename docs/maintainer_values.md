# Maintainer Values And Hard Contracts

[Back to Documentation Index](./README.md)

This is the compact source of truth for current maintainer preferences. Other agent docs should
link here instead of repeating the full value hierarchy.

## Hard Rule

Be honest, transparent, and reproducible.

Do not present a result as established when it only worked once, cannot be reproduced, or cannot be
implemented in the repository. Benchmark, metric, schema, artifact-provenance, and paper-facing
claims require enough evidence for another contributor to understand what ran, what did not run,
and what remains uncertain.

Fallback or degraded benchmark execution is never success evidence. It may be useful diagnostic
information only when labeled that way.

When the intended proof fails or cannot be gathered, close the work as `blocked`, `diagnostic`, or
`not benchmark evidence` as appropriate, and record the next smallest proof step. Do not relabel an
unproven result as complete just to preserve momentum.

## Exploration

Exploration is encouraged, including new planner families, research directions, and workflow ideas.
Agents may open exploratory branches, issues, or PRs with incomplete proof when the status is clear.

Use explicit status language:

- `exploratory`: promising direction, not yet validated as benchmark evidence.
- `diagnostic`: useful for debugging or understanding behavior, not a claim.
- `candidate`: plausible next implementation or experiment target.
- `blocked`: needs an artifact, dependency, environment, or maintainer decision.
- `not benchmark evidence`: ran outside the benchmark contract or under fallback/degraded mode.
- `paper-grade`: fully reproducible and suitable for paper-facing claims.

For substantive claims, recommendations, benchmark conclusions, and prioritization judgments below
roughly 95 percent confidence, include a short numeric uncertainty estimate, caveat, or condition
that would change the conclusion. Do not spend reasoning effort quantifying ordinary status updates
or low-impact implementation narration.

## Validation

Use validation proportional to risk.

- Docs-only and instruction-only changes use the cheap validation path by default: inspect the diff
  and verify changed links or referenced paths where practical.
- Runtime, benchmark, metric, schema, model-provenance, and paper-facing changes need executable
  proof appropriate to the claim.
- Low-value tests may be removed without maintainer approval when the reason is unambiguous and
  documented, provided repository coverage expectations still hold or the remaining gap is tracked.
  Do not assume flaky tests are common; classify each failure before broad policy changes.
- Paper-grade or benchmark-strength claims should name the exact claim, command/config/seed path,
  artifact provenance, metric/schema mode, sample size or statistics, fallback/degraded exclusions,
  limitations, and reproduction path before they are treated as established.
- When validation cannot prove the intended claim, report the observed evidence and the failed or
  missing proof separately, then hand off the next concrete proof step.

## Work Collection

GitHub issues are the central collection system for deferred work. Prefer better filtering,
priority discussion, and issue splitting over creating separate backlog stores.

Autonomous issue-to-PR loops may pick work themselves. When the ranking depends on a real tradeoff,
use a priority-discussion workflow to ask one focused question, then record the answer in the issue
or Project metadata. Priority discussion follow-up is tracked in issue #1729.

Project #5 scoring remains useful as an advisory, quota-aware prioritization tool. Treat precise
scores as sortable approximations from coarse inputs, not as hard authority.

## Optional Tooling

Spec-kit is optional. Use it for large, multi-contract designs where the overhead buys clarity; do
not make it the default governance layer for ordinary work.

`.agents/` is the canonical source for agent workflow content. Remove stale compatibility surfaces
when they no longer provide value; Claude cleanup is tracked in issue #1728.

The `docs/context/` tree should be optimized for fast, token-efficient agent retrieval. The next
architecture should favor Markdown notes with machine-readable metadata, a compact generated
catalog, and a CLI/query surface before considering heavier retrieval infrastructure. Context
architecture follow-up is tracked in issue #1714.
