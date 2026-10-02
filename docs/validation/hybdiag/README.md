# HYBDIAG development experiments (PR #10099)

AI-GENERATED/NEEDS-REVIEW. These paired development experiments use seeds
1001–1030 only. They are exploratory evidence, not release evaluation results.
The compact episode summaries contain outcomes and metrics, without raw traces.
Their manifests identify the measured source revision and SHA256 hashes.

Regenerate the CSV/JSON tables from the committed summaries:

```sh
python -m scripts.validation.summarize_hybdiag
python -m scripts.validation.summarize_hybdiag --check
```

CSV files begin with a normal header. Success, collisions, timeouts and freezing
have two-sided 95% Wilson intervals over episodes. Stopped time and time without a
feasible moving candidate are weighted by robot exposure. A near-miss event starts
on entry into the plant's `near_misses` predicate (surface gap below 0.5 m); consecutive
near-miss steps count as one event. Event rates use actual robot-seconds; minimum
pedestrian separation is the executed center-to-center minimum. Robot radius is
1.0 m, pedestrian radius 0.4 m. No inference of independence between repeated
near-miss events is made.

`off` retains the frozen behavior. `static` enables physical wall exclusion;
`platform` extends speed candidates while retaining predicted pedestrian exclusion;
`both` enables both. Round 2 couples the absent-successor guard to static exclusion.
Round 3 explicitly enables `goal_next_validity_enabled` and the optional sensor
field for static/both, independently of wall exclusion. Missing validity fields
retain legacy behavior. Source defaults, observation keys and spaces are unchanged.

The 1.60 m pedestrian threshold is a **candidate rejection radius** against
constant-velocity predicted positions at rollout endpoints. It is not a hard limit
on executed separation. Round 3 retains the current-position pedestrian braking
cap, normalizes speed preference by drive maximum, checks wall stopping distance
past the finite rollout, and conservatively covers turning arcs during swept checks.

The empty-world gate compares each enabled arm to paired off episodes. Acceptance
requires no new empty-world failure and collision Wilson bounds not above off
(per scenario and pooled). This does not certify identical collision risk or
improved proximity; assess near misses and exposure separately.
