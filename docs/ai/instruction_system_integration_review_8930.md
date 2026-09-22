# Instruction-system integration review (#8930)

Final integration review for the instruction-compaction epic. All twelve child
issues merged via the stacked chain (Issue #8957, Issue #8959, Issue #8960, Issue #8961, Issue #8964, Issue #8966,
Issue #8967, Issue #8969, Issue #8970). Verified on `origin/main` at `45ddddc88` (2026-09-14).

## Before / after

| Surface | Audit baseline (`46ac2c85`, 2026-09-10) | Current (`45ddddc88`) |
| --- | --- | --- |
| Root `AGENTS.md` non-blank lines | 228 | 106 (bound: 137, enforced by `tests/dev/test_compact_boot_router.py`) |
| Root `AGENTS.md` bytes | 20402 | 8441 |
| Required instruction references resolving | 175, 0 errors | 176 (+9 optional), 0 errors |
| Precedence owners | Competing hierarchies | Exactly one (`AGENTS.md` owns the `instruction-precedence` block; `test_repo_has_single_precedence_owner` passes) |
| Task-scope route | Missing file (`.agents/task_scope_manifest.yaml` absent) | Present, machine-readable (`version: 1`, owner `docs/ai/agent_workflow_entrypoints.md`, profiles observe/local/coordinated/...) |
| Risk-scoped execution | Universal startup ceremony | Profiles in manifest; read-only tasks require no venv, plan, or worktree |

## Conflict count

One repository-internal precedence contract is authoritative. The only
"takes precedence" language in the boot surfaces is the single
`instruction-precedence` block in `AGENTS.md`, which names the owner order and
explicitly denies competing runtime precedence to the other surfaces. No second
precedence model was found in `AGENTS.md`, `.agents/README.md`,
`.agents/PLANS.md`, or `docs/maintainer_values.md`.

## Representative-task behavior

- Read-only (`observe` profile): required context is `AGENTS.md` +
  `docs/ai/agent_workflow_entrypoints.md` only; execution plans, worktrees, and
  release gates are forbidden ceremony.
- Localized edit (`local` profile): targeted checks only, no plan or worktree.
- Multi-module change (`coordinated` profile): plan, environment, worktree, and
  PR required.
- Provider adapters are scoped to provider mechanics
  (`test_agent_entrypoints.py`, adapter-scope checks pass).

## Proof

- `uv run python scripts/dev/check_instruction_references.py` →
  `instruction references checked: 176 (+9 optional references resolved)`, no errors.
- Instruction behavior fixtures
  (`test_agent_entrypoints`, `test_compact_boot_router`,
  `test_instruction_precedence`, `test_instruction_references`,
  `test_instruction_task_fixtures`, `test_token_routing_entrypoints`,
  `test_check_agent_instructions`) → **43 passed**.
- Broader instruction batch (`-k "instruction or compact_boot or precedence or
  profile or adapter or values or delivery or friction or compendium or
  entrypoint or constitution"`) → **137 passed**.

## Residual

The clean-room agent trial from the epic success criteria is tracked as
Issue #9232. It needs an independent fresh agent session and is outside a
docs/verification closeout. It does not block epic closure because every
mechanically verifiable criterion above passes on current main.

## Safeguards preserved

No release, benchmark-evidence, research-claim, or exact-head delivery rule
was weakened by this review: this note adds no policy, changes no procedure,
and only records the before/after comparison the epic deliverable requires.
