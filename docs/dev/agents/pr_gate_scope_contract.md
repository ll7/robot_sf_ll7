# PR Gate Scope Contract (Disjoint Modulo Scopes)

_Issue #5059._ The autonomous PR-gate orchestrator can run more than one **gate** at a time — each
gate is a worker prompt that shepherds a set of pull requests. This note defines the dispatch
contract that keeps two concurrent gates from ever owning the same PR, and points at the checker that
enforces it.

## The hazard

When a gate claims PRs by an ad-hoc **range** ("gate A owns #5044/#5037", "gate B owns #5047–#5057")
the ranges can look disjoint in one snapshot yet **cross** on a later re-list pass. A gate whose
prompt tells it to "process newly appearing PRs" has, in effect, an *open-ended* range: on its final
re-list it will happily pick up a number that a newer gate already owns. Two gates shepherding the
same PR is a double-dispatch bug (duplicate reviews, racing pushes, conflicting merges).

## The contract

An **active gate owns a residue class**: it claims PR `n` iff `n % modulus == residue`.

- **Immutable.** A residue class does not depend on which PR numbers currently exist, so a gate can
  re-list as often as it likes and never reach into another gate's residue.
- **Disjoint by construction.** Two residue classes `(N_a, r_a)` and `(N_b, r_b)` share a PR iff the
  Chinese Remainder Theorem solvability condition holds: `(r_a - r_b) % gcd(N_a, N_b) == 0`. The
  simplest safe cohort is a shared modulus `N` with distinct residues (e.g. two gates split even/odd
  PRs under `modulus = 2`).
- **Second gate rule.** Before admitting a second gate, either give it a residue class disjoint from
  every active gate, **or** revoke/rewrite the old gate first. Never leave a legacy range scope
  active alongside a new gate.

## The checker

`scripts/tools/pr_gate_scopes.py` is the canonical implementation and the fail-closed regression
check:

```bash
# Manifest: a JSON list of {gate_id, scope}. Residue scope: {"modulus": N, "residue": r}.
python -m scripts.tools.pr_gate_scopes --gates gates.json
```

`validate_active_gates(gates)`:

1. rejects any non-residue (range) scope as **non-immutable** (`non_residue_scope` violation), and
2. reports any pair of live gates that could own the same PR (`overlap` violation, with the smallest
   crossing PR as a witness).

Exit codes: `0` disjoint & conforming, `1` violation found, `2` malformed manifest / IO error. Pass
`--allow-range-scopes` only to diagnose a legacy cohort mid-migration (overlap is still checked);
`--json` emits the schema-tagged report.

The module is pure arithmetic over gate descriptors — no GitHub calls, no state mutation — so it is
safe to call inline in a dispatcher or run as a pre-admission gate. Tests:
`tests/tools/test_pr_gate_scopes.py` (includes the exact issue #5059 scenario).
