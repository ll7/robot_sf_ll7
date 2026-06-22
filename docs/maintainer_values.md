# Maintainer Values And Hard Contracts

[Back to Documentation Index](./README.md)

This is the compact source of truth for current maintainer preferences. Other agent docs should
link here instead of repeating the full value hierarchy.

## Hard Rule

Be honest, transparent, reproducible, and understandable.

Do not present a result as established when it only worked once, cannot be reproduced, or cannot be
implemented in the repository. Benchmark, metric, schema, artifact-provenance, and paper-facing
claims require enough evidence for another contributor to understand what ran, what did not run,
and what remains uncertain.

Changes to benchmark scenarios, metric semantics, model profiles, or release-bound evidence must
follow the standing [benchmark scenario and model governance](./benchmark_governance.md) contract so
versioning, comparability, deprecation, and reproduction requirements are reviewable before claims
are interpreted.

When evidence is mixed, limited, or partly degraded, every report must open with:
1. claim boundary,
2. evidence status (`diagnostic-only`, `smoke evidence`, `nominal benchmark evidence`,
   `paper-grade`),
3. major caveats (including fallback/degraded rows),
4. uncertainty or confidence below ~95%.
Only after this ordering can result interpretation and recommendation appear.

Fallback or degraded benchmark execution is never success evidence. It may be useful diagnostic
information only when labeled that way.

## Clarity

Understandability is a first-class value, not a nice-to-have. If a newcomer cannot tell what a
feature does or why it matters, the work is effectively unusable no matter how correct it is.

- **Clarity wins on human-facing surfaces.** Token-efficiency (see [`AGENTS.md`](../AGENTS.md)) governs
  agent-internal prompts, scratch, and handoff notes. It does **not** override clarity on the
  README, `docs/`, feature names, `CHANGELOG.md`, public docstrings, and PR/issue titles. When the
  two conflict on a human-facing surface, prefer the version a newcomer understands.
- **Define jargon on first use.** Every acronym and project-specific term (for example VRU, AMV,
  AMMV, SNQI, occluder, "claim boundary") must be expanded on first use in a document or linked to
  [`docs/glossary.md`](./glossary.md), which is the canonical source for term definitions.
- **Lead with a plain-language summary.** Each user-facing feature, doc section, and changelog
  entry should open with one plain sentence — what it does and why a reader should care — before any
  precise-but-dense terminology. Keep the exact technical term; add a short gloss alongside it.
  Example: "Occluder timing perturbations: randomly vary *when* obstacles block the robot's view,
  so planners are tested against realistic sensing gaps."
- **Prefer one canonical name.** When the repository expands the same acronym several ways, fix the
  canonical expansion in the glossary and converge on it instead of leaving readers to guess.

This is a standing rule plus a glossary, not a one-time rewrite: re-apply it whenever a human-facing
surface is touched so dense prose does not silently re-accumulate.

Reports with mixed or limited benchmark evidence should put the claim boundary first: evidence
status, major caveats, fallback/degraded exclusions, and uncertainty belong before result
interpretation, rankings, or success language. Use the lightweight ladder consistently:
`diagnostic-only` for debugging or contract probes, `smoke evidence` for narrow execution proof,
`nominal benchmark evidence` for predeclared benchmark-matrix results, and `paper-grade` only for
fully reproducible claims suitable for manuscript-facing use.

When the intended proof fails or cannot be gathered, close the work as `blocked`, `diagnostic`, or
`not benchmark evidence` as appropriate, and record the next smallest proof step. Do not relabel an
unproven result as complete just to preserve momentum.

## Exploration

Exploration is encouraged, including new planner families, research directions, and workflow ideas.
Agents may open exploratory branches, issues, or PRs with incomplete proof when the status is clear.

Hypotheses should be tracked close to experiments by default: configs, launch packets, issue
comments, private ops ledgers, or issue-specific context notes are acceptable while the question is
what to run next. Central hypothesis ledgers are opt-in for research families that need cross-run
belief management, such as many related runs, conflicting or repeated negative results, duplicate
variant risk, claim-boundary movement, or dissertation/paper synthesis.

Use explicit status language:

- `exploratory`: promising direction, not yet validated as benchmark evidence.
- `diagnostic`: useful for debugging or understanding behavior, not a claim.
- `candidate`: plausible next implementation or experiment target.
- `blocked`: needs an artifact, dependency, environment, or maintainer decision.
- `not benchmark evidence`: ran outside the benchmark contract or under fallback/degraded mode.
- `paper-grade`: fully reproducible and suitable for paper-facing claims.

For substantive claims, recommendations, benchmark conclusions, and prioritization judgments below
roughly 95 percent confidence, include a short numeric uncertainty estimate, caveat, or condition
that would change the conclusion near the claim boundary. Do not spend reasoning effort
quantifying ordinary status updates or low-impact implementation narration.

## Validation

Use validation proportional to risk.

- Docs-only and instruction-only changes use the cheap validation path by default: inspect the diff
  and verify changed links or referenced paths where practical.
- Runtime, benchmark, metric, schema, model-provenance, and paper-facing changes need executable
  proof appropriate to the claim.
- PRs that alter evidence classification, experimental comparison methodology, figure eligibility,
  benchmark interpretation, or paper-facing claim surfaces need explicit domain-aware approval or a
  stated blocker before they are treated as merge-ready.
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

Interesting research paths that are no longer the best next step should usually stay open at lower
priority instead of being closed as parked or superseded. When deprioritizing one, record a short
reason and a revival condition, such as the artifact, synthesis result, benchmark gap, or maintainer
priority change that would make it relevant again. Close issues when they are duplicate, invalid,
fully superseded by a merged/current issue, or no longer useful as a research option.

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
