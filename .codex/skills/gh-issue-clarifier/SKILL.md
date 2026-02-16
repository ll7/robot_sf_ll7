---
name: gh-issue-clarifier
description: "Clarify ambiguous GitHub issues by tightening scope and acceptance criteria, proposing solution options with pros/cons, and marking decision-required issues when maintainer input is needed."
---

# GH Issue Clarifier

## Overview

Use this skill to make ambiguous issues implementation-ready with minimal churn and clear
decision records.

## When To Use

- Issue description is vague, contradictory, or missing acceptance criteria.
- Multiple solution paths exist with meaningful tradeoffs.
- Scope is too broad and needs decomposition.

## Clarification Workflow

1. Gather context
   - `gh issue view <n> --comments --json title,body,labels,milestone,assignees,comments,url`
   - Review linked PRs/commits if present.
   - Inspect relevant code paths and tests before proposing changes.

2. Detect ambiguity types
   - Problem ambiguity: unclear bug or objective.
   - Scope ambiguity: too broad for one PR.
   - Solution ambiguity: multiple valid designs with different tradeoffs.
   - Validation ambiguity: missing test or acceptance criteria.

3. Produce option set (mandatory when ambiguity exists)
   - For each option include:
     - short approach name
     - `Pros`
     - `Cons`
     - risk profile (speed vs reliability)
   - Mark one `Recommended` option with concrete rationale.

4. Decide path
   - If a recommendation is clear and low risk:
     - tighten issue body directly (problem, scope, acceptance criteria, test plan).
     - keep/move project status to `Ready`.
   - If maintainer decision is required:
     - ensure label exists:
       - `gh label create "decision-required" --color B60205 --description "Needs maintainer decision" || true`
     - `gh issue edit <n> --add-label decision-required`
     - move project status to `Tracked`
     - post a structured decision comment with options and recommendation.

5. Split follow-up work when needed
   - Keep current issue narrowly implementable.
   - Create follow-up issues for deferred scope:
     - `gh issue create --title "<follow-up>" --body-file <body.md> --label "<labels>" --milestone "<milestone>"`
   - Add follow-up items to project and assign `Priority`.

6. Finalize clarified issue contract
   - Ensure issue contains:
     - crisp problem statement
     - in-scope / out-of-scope bullets
     - acceptance criteria
     - validation commands (tests/checks)
     - links to follow-ups and dependencies

## Comment Template (decision-required)

Use `scripts/dev/gh_comment.sh issue <n> <<'EOF' ... EOF`:

- `Context`
- `Options`
  - `Option A`: pros/cons
  - `Option B`: pros/cons
- `Recommendation`
- `Decision needed from maintainer`
- `Impact on timeline/risk`

## Metadata Policy

- Add or fix labels that improve routing (`bug`, `enhancement`, `validation`, `performance`, etc.).
- Align milestone to roadmap when known.
- Set project `Priority` explicitly for actionable issues.
- Remove `decision-required` once a decision is made and move status to `Ready`.

## Output Requirements

- Report:
  - what was ambiguous
  - options evaluated
  - recommended path
  - metadata changes made
  - follow-up issues created (if any)
