# Issue #3474 Seed-Overlap Policy For Held-Out Adversarial Evidence

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/3474>
- <https://github.com/ll7/robot_sf_ll7/issues/3275>

## Summary

For the issue #3275 adversarial proposal-vs-random gate, held-out evidence is eligible only when
the fit and evaluation archive partitions are disjoint by scenario family, scenario seed, and
archive ID. Seed overlap is therefore a disjointness failure, not merely advisory metadata.

This is a conservative methodology choice for the current gate. A future paired or dependent
inference design may intentionally reuse seeds, but it must define a separate contract, null test,
and report wording before such rows can be treated as held-out evidence.

## Implementation Note

`robot_sf/adversarial/disjoint_evaluation.py` is the canonical owner. Its overlap provenance keeps
the explicit `seed_overlap` list and count, sets `disjointness_checks_passed` to `false` when the
list is non-empty, and records `seed_overlap` in `disjointness_failure_reasons`.

No #3275 campaign was run for this policy update. The change is a gate/methodology correction with
focused unit coverage only; it makes no benchmark, paper, or planner-performance claim.
