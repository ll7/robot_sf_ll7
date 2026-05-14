# PR First-Pass Review Audit (2026-05-14)

## Scope

This audit sampled the roughly 30 most recent closed pull requests returned on
2026-05-14 by:

```bash
gh pr list --repo ll7/robot_sf_ll7 --state closed --limit 30 \
  --json number,title,createdAt,closedAt,mergedAt,author,headRefName,baseRefName,url
```

The sample covered PRs #1197 through #1131. All sampled PRs were merged. The goal was to identify
why PRs were not perfect on first opening and to turn repeated review findings into a pre-opening
review habit.

Supporting inspection used:

```bash
gh pr view <n> --repo ll7/robot_sf_ll7 \
  --json number,title,createdAt,mergedAt,reviews,comments,commits,files,additions,deletions
gh api --paginate repos/ll7/robot_sf_ll7/pulls/<n>/comments
gh api --paginate repos/ll7/robot_sf_ll7/issues/<n>/comments
```

## Quantitative Signals

- 29 of 30 PRs had more than one commit after opening.
- 26 of 30 PRs had at least one inline review comment.
- The sample contained 91 inline review comments.
- Review states were mostly `COMMENTED`, not `CHANGES_REQUESTED`, so the issue was usually
  first-pass polish, missed edge cases, or proof consistency rather than complete implementation
  failure.
- PR #1197 was the cleanest recent example: one commit, no inline comments, and only summary/status
  comments.

## Main Failure Modes

### 1. Durable docs and proof drift

Repeated examples: #1192, #1178, #1171, #1166, #1161, #1160, #1133, #1139, #1137, #1140.

Typical findings:

- PR body validation counts did not match the context note.
- New `docs/context/` pages were not discoverable from `docs/README.md` or `docs/context/README.md`.
- Context notes still said no validation had run after the PR body claimed tests or readiness checks.
- Durable evidence bundles kept local `output/` paths or absolute local paths instead of tracked,
  repository-relative pointers.
- Follow-up wording referenced stale files or omitted a previously agreed requirement.

Why this escaped first pass:

- The readiness gate checks tests and coverage, but it does not compare PR body claims against
  context-note validation sections or docs indexes.
- Agents treated docs updates as prose-only edits instead of checking them as durable navigation and
  evidence surfaces.

### 2. Fail-closed and edge-case validation gaps

Repeated examples: #1184, #1177, #1176, #1166, #1164, #1161, #1160, #1159, #1158, #1135, #1132,
#1131.

Typical findings:

- Optional hook commands masked real failures with `&& ... || true`.
- JSON/YAML/path readers accepted malformed values, `None`, directories, absolute paths, traversal,
  or unsupported types too late or too silently.
- Numeric contracts did not reject `NaN`, `inf`, negative sentinels, or empty arrays with the wrong
  shape.
- Benchmark/planner status defaults allowed accidental smoke or degraded labels instead of making
  status explicit.
- Tests covered the happy path but missed malformed payloads, empty batches, or route/geometry
  boundary relations.

Why this escaped first pass:

- Implementations were scoped to the requested feature path, while automated reviewers checked
  adversarial inputs and contract boundaries.
- The pre-PR workflow lacked an explicit "what would a reviewer break?" pass for public helpers,
  schemas, and artifact readers.

### 3. Resource and scalability hygiene

Repeated examples: #1177, #1173, #1166, #1160, #1159, #1158, #1157.

Typical findings:

- JSONL loaders used `read_text().splitlines()` where streaming would avoid memory spikes.
- Inner loops did repeated `np.linalg.norm` work instead of squared-distance calculations.
- Global `np.random.seed` could leak state across environments.
- Dynamic Gymnasium observation spaces risked wrapper/vector-env incompatibility.
- Step recordings repeated static metadata each tick.
- `np.inf` bounds were used with integer spaces.

Why this escaped first pass:

- Targeted tests proved correctness on small fixtures, but the self-review did not check the
  repository's expected artifact sizes, simulation-loop frequency, or Gymnasium compatibility
  contracts.

### 4. Scope and wording ambiguity

Repeated examples: #1183, #1182, #1178, #1176, #1170, #1137.

Typical findings:

- Agent skill instructions used broad wording such as "all open issues" without preserving the
  selected issue set.
- Review handoff language was ambiguous about whether evidence should be committed, commented, or
  added to the PR description.
- Documentation cleanup PRs missed nearby placeholders or created sentence fragments.

Why this escaped first pass:

- Review focused on satisfying the immediate issue rather than reading changed instructions as
  executable prompts for future agents.

## First-Pass Self-Review Checklist

Run this checklist before opening draft PRs, after implementation and before final readiness proof:

1. Evidence consistency
   - Compare PR body validation, context-note validation, and actual terminal output.
   - Remove stale "not run" statements once validation has run.
   - Use repository-relative paths in docs, PR text, JSON evidence, and issue comments.
   - If a new durable note is useful beyond the PR, link it from `docs/context/README.md` and
     `docs/README.md`.

2. Reviewer adversarial pass
   - For parsers, schemas, config readers, artifact readers, and public helpers, check malformed
     payloads, missing values, `None`, empty collections, wrong shapes, directories, absolute paths,
     path traversal, `NaN`, `inf`, and negative sentinels.
   - Add targeted tests for any boundary that could silently corrupt benchmark, planner, training,
     or recording semantics.
   - Do not mask real command failures when the dependency exists; optional dependencies should be
     optional only when missing.

3. Resource and runtime pass
   - Stream large JSONL or artifact files instead of reading them fully when the format is line
     oriented.
   - Avoid global random-state mutation in environment reset paths.
   - Keep Gymnasium spaces static after construction unless the PR is explicitly changing that
     contract.
   - Avoid repeated static metadata in per-step output records.

4. Agent-instruction pass
   - Treat skill and workflow docs as executable instructions.
   - Preserve user-selected issue sets, scope boundaries, and handoff-only blockers explicitly.
   - State where missing evidence belongs: committed note, PR comment, or PR body.

## Recommended Process Change

The readiness gate remains necessary, but it is not sufficient. Add a mandatory first-pass
self-review step to AI-assisted PR opening workflows. The step should be lightweight and targeted to
the diff:

- docs/context changes trigger the evidence consistency checks,
- schema/parser/path/JSON changes trigger the adversarial boundary checks,
- simulation-loop, recording, benchmark, or environment changes trigger the resource/runtime checks,
- skill or agent workflow changes trigger the executable-instruction checks.

This audit is now linked from the AI workflow and PR-opening skill so future agents can run the
checklist before opening PRs instead of waiting for Gemini or CodeRabbit to identify the same
classes of issues.
