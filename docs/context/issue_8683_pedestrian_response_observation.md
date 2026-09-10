# Pedestrian-Response Observation Contract (`pedestrian_response_observation`)

**Status:** diagnostic / analysis-only — a typed observation surface for structured-indoor
biased-route encounters, not benchmark evidence or a human-behavior claim.
**Review hold:** diagnostic-only and domain-review held; this note does not authorize benchmark or
paper-facing interpretation.
**Issue:** [#8683](https://github.com/ll7/robot_sf_ll7/issues/8683) (child of research
[#7883](https://github.com/ll7/robot_sf_ll7/issues/7883); builds on the canonical route observability
contract in [#7890](https://github.com/ll7/robot_sf_ll7/issues/7890) and the deterministic
biased-route fixtures in [#8033](https://github.com/ll7/robot_sf_ll7/issues/8033)).
**PR:** [#8690](https://github.com/ll7/robot_sf_ll7/pull/8690).
**Canonical contract:** [`route_choice_observability.v1`](issue_7890_route_choice_observability.md)
and [`RouteSideReport`](../../robot_sf/benchmark/route_choice_observability.py#L68).
**Owner module:** [`robot_sf/benchmark/pedestrian_response_observability.py`](../../robot_sf/benchmark/pedestrian_response_observability.py).
**Tests:** [`tests/benchmark/test_pedestrian_response_observability.py`](../../tests/benchmark/test_pedestrian_response_observability.py).

## Plain-language summary

The contract records what is already observable for one encounter: the minimum passing clearance
supplied by the caller, the side offered by the biased route, the side taken by the observed route,
and whether a response was present. It reuses the canonical [`RouteSideReport`](../../robot_sf/benchmark/route_choice_observability.py#L68)
and its side vocabulary; it does not create a second route-side schema or infer a pedestrian
preference.

## Record contract

`PedestrianResponseObservation` is immutable and serializes to
`pedestrian_response_observation.v1` with:

| Field | Meaning |
| --- | --- |
| `encounter_id` | Stable caller-supplied identity for one encounter. |
| `minimum_passing_clearance_m` | Caller-supplied observed minimum surface clearance in metres; no threshold is introduced. |
| `offered_side` | Side classified for the biased/offered route using `route_choice_observability.v1`. |
| `route_reference` | Typed coordinate-frame, endpoint, units, tolerance, neutral-band, and progress metadata copied from the route-side reports. |
| `taken_side` | Side classified for the observed taken route using the same existing contract. |
| `response_present` | `True` or `False` when presence or absence was observed; `None` is missing by default and is explicitly unavailable only when listed in `unavailable_fields`. |
| `status` | `available` only when all required fields are available; otherwise `not_available`. |
| `missing_fields` | Required fields that were not supplied (`None`), including `response_present=None` by default. |
| `unavailable_fields` | Required fields explicitly unavailable, including a caller-listed response flag or an unavailable predecessor route report. |

`False` is a valid response-presence observation and is not treated as missing. `None` is missing
unless the caller explicitly includes `response_present` in `unavailable_fields`; those states do
not overlap and no missing field is filled with a default. A side label is valid only when a usable
route reference supports the start-to-goal axis. If that reference is absent or invalid, the
builder normalizes offered and taken labels to `unavailable`, propagates the upstream
route-reference reason, and does not construct fallback-looking `RouteReference` metadata. An
upstream report with any non-`None` failure reason is also unavailable when paired with a
non-`unavailable` side; the builder normalizes that side to `unavailable` and propagates the
failure reason. An unavailable route side retains the predecessor value `unavailable` and its
reason in the record-level `unavailable_reason`. The builder fails closed when the offered and
taken reports use incompatible reference metadata.

## Validation commands

The focused contract and related regression set uses the clickable
[`scripts/dev/run_worktree_shared_venv.sh`](../../scripts/dev/run_worktree_shared_venv.sh) wrapper:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_pedestrian_response_observability.py \
  tests/benchmark/test_route_choice_observability.py \
  tests/nav/test_biased_route_generator.py \
  tests/benchmark/test_passing_clearance.py -q
```

Additional clickable validation commands are the
[`scripts/dev/check_context_notes.sh`](../../scripts/dev/check_context_notes.sh) docs/evidence gate
and [`scripts/dev/pr_ready_check.sh`](../../scripts/dev/pr_ready_check.sh) readiness gate; the
owner module also uses `python -m py_compile` and `git diff --check`.

## Deterministic proof and boundaries

The focused tests replay the existing canonical corridor and dual-doorway fixtures with identical
inputs and assert byte-stable JSON-ready records. The fixtures are structured indoor diagnostics
only. Their source limitation is that they do not establish behavior outside structured indoor
scenes; there is no autonomous mobility vehicle (AMV) evidence, human-subject validation, or basis
for human-predictability or social-compliance claims.

This contract does not change planners, metric semantics, campaigns, preregistration, social-
compliance scalars, or paper-facing claims. It is not a benchmark score, a response law, a
calibrated clearance requirement, or evidence that a pedestrian understood or complied with an
offered route. The diagnostic-only/domain-review hold remains in force.
