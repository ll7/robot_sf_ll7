# Relocated Workflow Topic Index

This file indexes long-form workflow guidance that previously lived here. Each topic has one
canonical owner; this index adds no policy and holds no procedure. Repository-internal precedence is
owned by the `Instruction Precedence` contract in `AGENTS.md`. The pre-decomposition text remains in
Git history, and historical issue references stay discoverable through `docs/context/INDEX.md`.

| Topic | Canonical owner | Trigger |
| --- | --- | --- |
| Token-efficient active thread profile, phase audits, meta-workflow PR gate | `docs/templates/token_efficient_thread_profile.md` | long autonomous or token-saving work |
| Autonomous usage stop guard | `docs/templates/token_efficient_thread_profile.md` and the `goal-autopilot` skill | usage-bounded goal loops |
| SLURM lane priority and submission preparation | `SLURM/AGENTS.md` and the `goal-slurm-experiment` skill | cluster campaign work |
| Shared knowledge graph | `docs/ai/understand_anything.md` | architecture discovery before broad reads |
| Local machine context | `docs/templates/local.machine.example.md` | expensive or host-specific commands |
| Worktree bootstrap, branch synchronization, teardown, stash safety | `docs/dev/worktree_lifecycle.md` | linked worktree creation, sync, or cleanup |
| Knowledge capture, context notes, durable memory | `docs/context/README.md` and `memory/MEMORY.md` | reusable findings or handoffs |
| Cross-agent compatibility stance | `docs/context/issue_728_coding_agents_compatibility.md` | adapting external agent workflow sources |
| Project structure and module organization | `docs/ai/repo_overview.md` | orienting in the repository |
| Canonical owner check before new modules | `AGENTS.md` Project Structure And Ownership | adding a module, script, or config |
| Build, test, style, and config-first workflows | `docs/dev_guide.md` | contributor execution |
| Research claim validation and benchmark fallback | `docs/benchmark_governance.md` and `docs/context/issue_691_benchmark_fallback_policy.md` | benchmark or paper-facing work |
| Commit and pull request workflow | `docs/dev_guide.md` and `docs/code_review.md` | requested delivery or review |
| Communication depth | `AGENTS.md` Planning And Communication | user-facing reporting |
| Planning convention | `.agents/PLANS.md` | coordinated or evidence-critical work |
| GitHub workflow batching and Project #5 metadata | `docs/context/issue_713_batch_first_issue_workflow.md` | issue batches and metadata |
| Exact-head REST review publication | `scripts/dev/gh_pr_review_rest.py` | publishing review verdicts |
| Key Codex skills | `.agents/skills/README.md` | choosing a delivery skill |
| Shared model routing | `.agents/README.md` | delegated provider selection |
| Dependency management | `AGENTS.md` Donts | changing dependencies or the environment |
