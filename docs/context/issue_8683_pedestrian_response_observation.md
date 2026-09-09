# Pedestrian-Response Observation Contract (`pedestrian_response_observation`)

**Status:** diagnostic / analysis-only — a typed observation surface for structured-indoor
biased-route encounters, not benchmark evidence or a human-behavior claim.
**Issue:** #8683 (child of research #7883; builds on the route observability contract in #7890
and the deterministic biased-route fixtures in #8033).
**Owner module:** `robot_sf/benchmark/pedestrian_response_observability.py`.
**Tests:** `tests/benchmark/test_pedestrian_response_observability.py`.

## Plain-language summary

The contract records what is already observable for one encounter: the minimum passing clearance
supplied by the caller, the side offered by the biased route, the side taken by the observed route,
and whether a response was present. It reuses `RouteSideReport` and its side vocabulary; it does
not create a second route-side schema or infer a pedestrian preference.

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
| `response_present` | `True` or `False` when presence or absence was observed; `None` is unavailable. |
| `status` | `available` only when all required fields are available; otherwise `not_available`. |
| `missing_fields` | Required fields that were not supplied (`None`). |
| `unavailable_fields` | Required fields explicitly unavailable, including an unavailable predecessor route report. |

`False` is a valid response-presence observation and is not treated as missing. A missing field is
never filled with a default. An unavailable route side retains the predecessor value
`unavailable` and its reason in the record-level `unavailable_reason`. The builder carries the
nested route reference and fails closed when the offered and taken reports use incompatible
reference metadata; it never treats an unreferenced side label as available evidence.

## Deterministic proof and boundaries

The focused tests replay the existing canonical corridor and dual-doorway fixtures with identical
inputs and assert byte-stable JSON-ready records. The fixtures are structured indoor diagnostics
only. Their source limitation is that they do not establish behavior outside structured indoor
scenes; there is no autonomous mobility vehicle (AMV) evidence, human-subject validation, or basis
for human-predictability or social-compliance claims.

This contract does not change planners, metric semantics, campaigns, preregistration, social-
compliance scalars, or paper-facing claims. It is not a benchmark score, a response law, a
calibrated clearance requirement, or evidence that a pedestrian understood or complied with an
offered route.
