# Deterministic Biased Route Condition Generator (`biased_route_generator`)

**Status:** diagnostic / research leaf — deterministic route condition generation and fixture evaluation;
not evidence of human preference, pedestrian intent, comfort, or social compliance.
**Issue:** [#8033](https://github.com/ll7/robot_sf_ll7/issues/8033) (parent research [#7883](https://github.com/ll7/robot_sf_ll7/issues/7883),
prerequisite contract [#7890](https://github.com/ll7/robot_sf_ll7/issues/7890)).
**Owner module:** `robot_sf/nav/biased_route_generator.py`.
**Observability contract:** `robot_sf/benchmark/route_choice_observability.py` (`route_choice_observability.v1`).
**Source reference:** Kim, S., Gu, Y., & Cheong, H. (2026). Bubble-chain biased global path planning in structured indoor environments. *Journal of Intelligent & Robotic Systems* (19 August 2026). DOI: [10.1007/s10846-026-02444-3](https://doi.org/10.1007/s10846-026-02444-3).

Plain-language summary: this module provides pure, typed, deterministic utilities to generate
route alternatives under explicit side biases (`neutral`, `left`, `right`) across canonical
multi-homotopy environments (corridor, doorway, crossing). It integrates directly with the
`route_choice_observability.v1` diagnostic contract to enable reproducible passing-side and route
predictability diagnostics without requiring heavy simulation or stochastic sampling.

## 1. Source Method and Scope Limits

The route-bias concept is adapted from bubble-chain biased global planning described by Kim, Gu,
and Cheong (*Journal of Intelligent & Robotic Systems*, 19 August 2026,
DOI: [10.1007/s10846-026-02444-3](https://doi.org/10.1007/s10846-026-02444-3)):

- **Structured-indoor environments only in the source:** The source methodology is strictly scoped
  to structured indoor layouts (such as hallways, rooms, and regular passages) where geometric bubble
  chains can constrain alternative homotopy channels. It does not establish validity for unstructured,
  open outdoor spaces, or complex multi-agent urban flows.
- **No autonomous mobile vehicle (AMV) evidence:** The source paper does not report empirical
  physical autonomous mobile vehicle (AMV) platform trials, hardware deployments, or onboard sensor
  experiments; its validation is limited to simulation/algorithmic evaluation.
- **Planner-route observability only — never human predictability:** In this repository, the
  generated conditions serve exclusively to test whether a robot navigation policy exhibits observable,
  consistent topological decisions under controlled geometric bias. Synthetic routes and simulator
  traces do not model or demonstrate general human predictability, pedestrian comfort, or social
  compliance (consistent with the boundary in [`docs/context/issue_7890_route_choice_observability.md`](issue_7890_route_choice_observability.md)).

## 2. Route Bias Modes

Route generation operates relative to a declared directed reference axis from scenario start to goal:

- `neutral`: direct centerline path along the reference axis (zero lateral offset).
- `left`: standard counter-clockwise perpendicular bias (+Y in reference frame rotated to face goal).
- `right`: standard clockwise perpendicular bias (-Y in reference frame).

The lateral displacement profile is smoothly shaped over normalized progress $t \in [0, 1]$
using configurable profile shapes (`smooth_sine`, `hann`, `cubic`, `trapezoid`), ensuring continuous
geometry where the generated path begins exactly at `start` ($t=0$) and terminates exactly at `goal` ($t=1$).

## 3. Canonical Fixture Topologies

The module provides deterministic canonical multi-homotopy environments:

- `build_corridor_fixture`: straight corridor with bounding walls and an optional central static barrier,
  yielding distinct left-pass, center, and right-pass routes.
- `build_doorway_fixture`: wall divider with symmetric left and right doorway openings.
- `build_crossing_fixture`: open crossing area with a central interaction zone requiring lateral avoidance.

Convenience generators `generate_corridor_homotopy_routes` and `generate_doorway_homotopy_routes`
produce full suites of `(neutral, left, right)` `BiasedRouteResult` instances for comparative evaluation.

## 4. Integration with Observability Contract

Each `BiasedRouteResult` encapsulates:
- the sampled 2D path coordinates;
- arc length, maximum lateral offset, and mean lateral offset;
- a `RouteSideReport` evaluated against `route_choice_observability.v1` (`classify_route_side`).

The helper `evaluate_route_observability_sequence` evaluates temporal consistency across a series
of replanned routes or candidate variations, emitting a versioned `route_choice_observability.v1`
diagnostic record.

## 5. Grid Rasterization

`rasterize_route_to_grid` converts continuous 2D trajectories into step-valid 8-connected grid cell
sequences for occupancy-grid-based planners and topological signature evaluation without non-adjacent
diagonal jumps.

## 6. Claim Boundaries and Non-Goals

- **Structured-indoor scope only**: The bubble-chain biasing method was evaluated in structured indoor
  environments; its assumptions do not transfer to unstructured, open outdoor, or mixed-traffic settings.
- **No autonomous mobile vehicle (AMV) evidence**: Neither the source method nor this synthetic generator
  provides physical AMV field trial or real-world vehicle deployment evidence.
- **No human preference or intent claims**: Generates deterministic synthetic routes for controlled
  benchmarking; does not model or assert actual pedestrian choice mechanics. Simulator traces and
  synthetic biased routes in this module evaluate planner observability only.
- **No general human predictability**: Synthetic route alternatives and simulation traces do not
  establish general human predictability, pedestrian comfort, or social acceptance (consistent with the
  "planner-route observability only — never human predictability" boundary in
  [`docs/context/issue_7890_route_choice_observability.md`](issue_7890_route_choice_observability.md)).
- **No social compliance score**: Evaluates observability and geometry only; never merges outputs
  into subjective human-likeness ratings.
- **Diagnostic only**: Designed for reproducible local testing and research validation.
