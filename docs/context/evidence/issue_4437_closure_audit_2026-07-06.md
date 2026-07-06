# Issue #4437 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/4437>
Audited at: `origin/main` @ `8b306cad5`
Tooling: [`scripts/dev/open_issue_closure_audit.py`](../../../scripts/dev/open_issue_closure_audit.py),
[`scripts/dev/closure_mechanics.py`](../../../scripts/dev/closure_mechanics.py)

## Purpose

Closure audit for #4437 (`hygiene: closure audit — close open issues fully covered by merged PRs,
annotate residuals`). This note maps each acceptance criterion in the maintainer's implementation
plan to merged-PR evidence and to a locally reproduced validation run, then records the closure
decision. It asserts only that the closure-audit **enablement tooling** is complete and validated; it
does not perform the audit's write pass (issue comments / closures), run a benchmark, submit
Slurm/GPU, or promote any model/benchmark/paper claim.

## Authoritative scope

The maintainer's pinned comment
([#4437 implementation plan](https://github.com/ll7/robot_sf_ll7/issues/4437#issuecomment-4882036018),
2026-07-04) is the authoritative scoping and overrides the terser issue body. It splits the work into
five phases: (1) build the candidate set (open issues with merged title-linked PRs), (2) classify
each candidate, (3) apply closure / residual / parent-ledger comment templates, (4) closure mechanics
behind human confirmation, and (5) a final summary comment on #4437.

Three merged PRs delivered the **enablement tooling** for these phases:

- **#4440** (`Refs #4437`, merged 2026-07-04) — `open_issue_closure_audit.py`: read-only candidate
  enumeration + classification (`open_issue_closure_audit.v1`). Never comments, closes, or mutates
  queue state.
- **#4503** (merged 2026-07-04) — `closure_mechanics.py`: comment templates + dry-run preview; the
  close path was intentionally inert scaffolding.
- **#4571** (merged 2026-07-05, squash `b53bd47c4`) — wires the close path behind human confirmation:
  closing requires BOTH `--close-issues <n>` (explicit numbers) AND `--apply`; default stays dry-run.

The maintainer's latest ruling
([#4437 comment](https://github.com/ll7/robot_sf_ll7/issues/4437#issuecomment-4884761136),
2026-07-05) fixes the residual boundary: *"#4437 remains open for any remaining audit-execution
passes."* The audit **execution** (running the tool over live issues, then posting comments and
closing fully-covered issues) is a GitHub write pass and has not been performed.

## Acceptance criteria → evidence

| Acceptance criterion (issue plan "Acceptance criteria") | Status | Evidence |
| --- | --- | --- |
| Tooling to enumerate open issues with ≥1 merged title-linked PR and classify each (parent/roadmap vs. closure-review) | Met | #4440 `open_issue_closure_audit.py` — read-only search (`gh search issues` / `gh search prs "<n> in:title" --merged`), title-link regex, `parent_or_roadmap` vs. `closure_review_required` classification; 12 tests in `tests/dev/test_open_issue_closure_audit.py` |
| Comment templates for closure / residual-checklist / parent-ledger, applied from the audit report | Met | #4503 `closure_mechanics.py` — `CLOSE_FULLY_COVERED_TEMPLATE`, `RESIDUAL_TEMPLATE`, `PARENT_LEDGER_TEMPLATE`; dry-run default; 12 tests in `tests/dev/test_closure_mechanics.py` |
| Fully-covered closures are gated behind explicit human confirmation (not automatic) | Met | #4571 requires BOTH `--close-issues <n>` AND `--apply`; default is dry-run; parent/roadmap candidates can never be marked `should_close` without an explicit number |
| **Audit execution pass**: run the audit over live issues, post closure/residual/parent-ledger comments, close fully-covered issues with `completed` reason | **Not met** | No evidence of an executed write pass; maintainer confirms "#4437 remains open for any remaining audit-execution passes." Requires GitHub issue comment + close authority |
| **#4437 receives a final summary comment** listing closed / residual / parent counts + issue numbers | **Not met** | Downstream of the execution pass above |
| No queue edits / new issues / benchmark execution / claim updates during the audit | Met (this lane) | This audit performs read-only GitHub queries + a docs artifact only |

## Reproduced validation (2026-07-06, `origin/main` @ `8b306cad5`)

```bash
uv run pytest tests/dev/test_closure_mechanics.py tests/dev/test_open_issue_closure_audit.py -q
# 24 passed in 0.07s
```

Live read-only audit run (demonstrates end-to-end operation + fail-closed behavior):

```bash
uv run python scripts/dev/open_issue_closure_audit.py --issue-limit 100 --pr-limit-per-issue 8
# Iterated open issues + per-issue merged-PR searches, then the GitHub *search* API
# rate-limited (HTTP 403, 30 req/min shared limit). The tool failed CLOSED with a
# schema-valid error packet, preserving: schema=open_issue_closure_audit.v1,
# read_only=true, issue_writes=false, project_writes=false, candidate_count=null.
```

The 403 is a transient GitHub search rate-limit, not a tool defect; the value here is that the
read-only tool degrades to a fail-closed error packet rather than a partial or misleading result.
Independent confirmation that #4437 is itself a valid candidate: `gh pr list --search "4437 in:title"
--state merged` returns exactly its three enablement PRs (#4440, #4503, #4571), so a completed audit
run would surface #4437 as `closure_review_required` (not `parent_or_roadmap` — its title contains no
parent marker).

## Closure decision

**Keep #4437 open — `Refs #4437`, not `Closes`.** The closure-audit *enablement tooling* (Phases 1–4
scaffolding + close-path wiring) is complete and validated, but two acceptance criteria remain open:
the audit **execution** write pass and the final #4437 **summary comment**. Both require GitHub issue
comment/close authority, which this implementation lane does not hold
(`comment_issue_or_pr: false`). This is the correct fail-closed state: tooling ready, write execution
deferred to an authorized pass — closing #4437 now would drop the still-pending execution and summary
criteria.

### Residual checklist (dispatchable — for an authorized comment/close lane)

- [ ] Run `open_issue_closure_audit.py` to produce the live candidate packet (read-only; retry if the
      GitHub search API is rate-limited).
- [ ] For each candidate, read the issue acceptance criteria + merged PR body/diff; decide
      close-fully-covered / residual / parent-ledger.
- [ ] `closure_mechanics.py --apply --close-issues <human-verified numbers>` for fully-covered issues;
      dry-run/comment-only for residual + parent-ledger candidates.
- [ ] Post the #4437 summary comment: closed / residual-annotated / parent-updated / no-action counts
      and issue numbers.

### Observed tooling note (optional consolidation follow-up — not a #4437 acceptance criterion)

`closure_mechanics.build_comment_command` / `build_close_command` (tested in
`test_closure_mechanics.py`) omit the `--repo` flag, whereas the live `execute_actions` path inlines
its own `gh` commands **with** `--repo`. The helpers are currently unused by the execution path, so
the two can drift. A future consolidation slice could route `execute_actions` through the builders
(single source of truth) and assert the `--repo` flag in the builder tests. Flagged here for
visibility; intentionally not changed in this closure-audit PR to keep scope to evidence only.
