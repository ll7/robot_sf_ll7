# Copilot Instructions

`AGENTS.md` is the canonical repository instruction source. Treat this file as the Copilot-facing
entrypoint and keep it limited to Copilot-specific pointers that are not already covered there.
For current maintainer values and hard contracts, read
[`docs/maintainer_values.md`](../docs/maintainer_values.md).

ALWAYS use the official [dev_guide](../docs/dev_guide.md) as the primary reference for development-related tasks.
It is everyones guide on how to use this repository effectively.

## Additional Instructions

- For machine-specific limits and execution routing, check local machine context files at repo
  root when present: `local.machine.md` or `local.machine.<name>.md` (template:
  `docs/templates/local.machine.example.md`). If absent, use conservative defaults.

- Use scriptable interfaces instead of cli interfaces when possible.
- Make everything honest, transparent, and reproducible.
- In benchmark work, use the canonical fail-closed fallback policy in
  `docs/context/issue_691_benchmark_fallback_policy.md`.
- Before generating PRs, apply these authoring guardrails:
  - Diagnostic producers that return floats must reject non-finite input and fail closed; use
    `robot_sf/benchmark/counterfactual_pair.py` as the local example.
  - Do not add absolute home-directory paths under `configs/**` unless the path is explicitly
    annotated as machine-local and not portable.
  - When cloning a config, update the provenance or issue comment so the copied config points at
    the correct issue, source, and intended run context.
- For GitHub issue batches and Project #5 writes, follow the batch-first workflow in
  `docs/context/issue_713_batch_first_issue_workflow.md`: clean up issues first, route project
  metadata second, run derived score sync last, use REST/local `git` for non-project state, reserve
  GraphQL for Projects v2/review threads, and cache IDs once per shell session or in local
  `.github/cache/project5.json` for long-running work.
- For measurement-driven improvement loops, use `.agents/skills/autoresearch/SKILL.md`.
  For shorter refinement passes, use `.agents/skills/auto-improvement/SKILL.md`.
  See `docs/ai/awesome_copilot_adaptation.md` for the selection guide.
- For the end-to-end AI coding workflow from prompt to PR, see `docs/ai/ai-workflow.md`.
- For the repository's cross-agent compatibility stance (retrieval → planning → execution →
  verification discipline, accepted and rejected external concepts), see
  `docs/context/issue_728_coding_agents_compatibility.md`.
- For stable cross-session repo memory, start with `memory/MEMORY.md`, then open only the relevant
  linked topic files under `memory/`. Use `docs/context/` for issue-specific execution notes rather
  than storing those narratives in `memory/`.
- For context discovery, use `.agents/skills/context-map/SKILL.md` before multi-file changes and
  `.agents/skills/what-context-needed/SKILL.md` when the task is underspecified.
- For non-trivial work, persist reusable insights, decisions, reasoning, validation notes, and
  handoff context in linked Markdown notes under `docs/context/` when that context would otherwise
  be lost. Prefer updating the canonical existing note over creating a duplicate, and use
  `.agents/skills/context-note-maintainer/SKILL.md` when creating or refreshing those notes.
- For risk-proportional validation planning, use `.agents/skills/quality-playbook/SKILL.md` for
  non-trivial changes, `.agents/skills/agentic-eval/SKILL.md` for AI-workflow artifacts, and
  `.agents/skills/review-and-refactor/SKILL.md` for narrow review-then-refactor passes.
- Treat delegated or sidecar model output as `route_evidence`: useful planning/review signal that
  still needs local validation before it supports merge, benchmark, paper-facing, or issue-close
  claims. Sparse or missing worker result/status artifacts are route failures, not negative
  findings.
- For doc synchronization, use `.agents/skills/update-docs-on-code-change/SKILL.md` when code
  changes would make docs stale.
- Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project work.
- Keep `gh` for deterministic batch automation, score sync, and auth/debugging fallback.
- Before opening a PR, follow the latest-main sync rule in
  `docs/dev_guide.md`: fetch `origin/main`, merge or rebase it into the feature branch, then run
  `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` for higher-risk branches. Docs-only and
  low-risk instruction branches may use the cheaper path from `docs/ai/ai-workflow.md` when the PR
  clearly states which checks were intentionally skipped.
- Central point to link new documentation pages is `docs/README.md`.
  - Link new documentation (sub-)pages in the appropriate section.
- For any changes that affect users, update the `CHANGELOG.md` file.
- Source the environment before using python or uv `source .venv/bin/activate`.

## Test Failure Evaluation

**When encountering test failures, always evaluate test significance before fixing:**

1. **Verify test value first**: Not all tests are equally important. Ask:

   - Does this test verify a core public contract (factories, schemas, metrics)?
   - Would this failure impact users in production?
   - Is this testing a known regression or critical edge case?
   - Is the test brittle/flaky (timing, display, environmental issues)?
   - Is the same logic already covered by other tests?

2. **Priority classification**:

   - **High** (fix immediately): Core features, user-facing behavior, schema compliance
   - **Medium** (fix within sprint): Known regressions, important edge cases
   - **Low** (consider archiving): Rare scenarios, redundant coverage, no real-world incidents
   - **Flaky** (refactor or remove): Frequently fails without indicating real bugs

3. **Actions based on priority**:

   - High priority: Fix immediately to protect critical invariants
   - Medium priority: Fix or document deferral with tracking issue
   - Low priority: Consider archiving with documented rationale
   - Flaky: Stabilize with retries/mocks if valuable, else remove

4. **Documentation requirements**:
   - Removing tests requires commit message explaining why
   - New tests need docstring: (1) what is verified, (2) why it matters
   - Deferred failures need "test-debt" label and value assessment

**Reference**: See Constitution Principle XIII and dev_guide.md Testing Strategy for complete criteria.
