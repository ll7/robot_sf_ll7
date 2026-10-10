# October 8 ruling and regression proof

The author ruled in chat to the orchestrator on 2026-10-08: new hybrid inputs
default to goal validity and its required sensor on, with static exclusion off.
The October 5 all-on decision is superseded. Released inputs keep their original
false fill-in, complete constructor dumps, mappings and frozen bytes.

Opt in through the algorithm configuration:

```yaml
physical_static_exclusion_enabled: true
```

Goal validity and its sensor remain independently configurable using
`goal_next_validity_enabled` and environment `include_goal_next_valid`.
An enabled validity scorer without `next_valid` still fails closed.

`test_010_defaults_follow_author_ruling_20261008` fails on parent
`8e6ab04a0c219af3cab085ffd813421772696989` before the implementation:

```text
assert cfg.physical_static_exclusion_enabled is False
AssertionError: assert True is False
1 failed in 7.88s
```

Command: `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest -n 2 tests/planner/test_hybrid_default_compatibility.py::test_010_defaults_follow_author_ruling_20261008 -q`.
The broader focused check passes 183 tests; receipts are in
[ruling_validation.json](ruling_validation.json).

Test value, answering both the requested and repository questions:

- Bug caught: missing fields reverting to all-on, validity losing its sensor,
  or a diagnostic roster silently dropping the all-on counterfactual.
- Credible future edit: changing a factory, constructor fill-in or arm label.
  Existing explicit-on tests do not pin the chosen current defaults.
- Fail-on-base proof: the real typed planner builder returned true on the parent;
  the assertion above failed before implementation, rather than during setup.
- Deterministic and real path: no random inputs or simulation steps; actual
  planner builder, direct dataclass constructors and executable diagnostic
  roster are checked, without replacing the production default selector.

All explicit override cases still pass in both directions. Unknown sources use
the current defaults. Registered release inputs compare every full constructor
and environment field, canonical mapping and observation contract to base.
The released audit is construction-only and does not execute release seeds.

The original raw campaign and labels are immutable. Its `current_defaults` arm
means historical all-on; the new default selects the already measured explicit
`goal_validity_with_sensor` arm. [Applicability](ruling_applicability.json)
records the changed bindings without claiming a new exact-head campaign.
