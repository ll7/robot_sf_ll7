# Issue #2550 Navground Adapter Spike Report

Issue: [#2550](https://github.com/ll7/robot_sf_ll7/issues/2550)

Date: 2026-06-11

Status: design/spike report only. No Robot SF adapter, dependency, benchmark row, or benchmark claim
was added.

## Executive Summary

Navground is a viable prototype-only external framework for social-navigation behavior reference, but
not currently worth integrating as a benchmark row. The #2442/#2514 assessment evidence proves
importability, behavior-level command emission, and MIT licensing. The redundancy surface (ORCA,
SocialForce) dominates the most accessible behaviors, and the non-redundant candidate (HL) lacks a
Robot SF adapter or benchmark runner path. Verdict: **viable prototype, not benchmark-ready now.**

## API Mapping

| Robot SF contract | Navground contract | Mapping quality |
| --- | --- | --- |
| Robot pose `(x, y, theta)` | `navground.core.Pose2([x, y], theta)` | Direct; 1:1 field mapping |
| Robot velocity `(vx, vy)` | `behavior.velocity = [vx, vy]` | Direct for holonomic; differential-drive projection needed for 2WDiff |
| Robot goal `(gx, gy)` | `navground.core.Target()` with `target.position = [gx, gy]` | Direct; 1:1 field mapping |
| Robot radius | `behavior.radius` | Direct; float meters |
| Nearby pedestrian pose/velocity | `navground.core.State` in local environment | Requires constructing Navground neighbor state list from Robot SF pedestrian set |
| Static obstacle segments | Navground obstacle state (wall/segment API) | Not proven in #2442 smoke; requires constructing Navground obstacle representation |
| Action output: `unicycle_vw` `(v, omega)` | `behavior.compute_cmd(dt, neighbors, static)` returns `Twist2` with `velocity` and `angular_speed` | Direct for 2WDiff; `Twist2.velocity` maps to linear `v`, `angular_speed` maps to `omega` |

## Action/Observation Compatibility

**Action side (proven in #2442):** Navground `compute_cmd()` returns finite `Twist2` commands for
ORCA, HRVO, HL, and SocialForce with `2WDiff` kinematics, a single `Target`, and no local
environment. The command surface is compatible with Robot SF `unicycle_vw` action space through a
direct `Twist2`-to-`unicycle_vw` projection.

**Observation side (not proven):** The #2442 smoke did not construct Navground local-environment
state from Robot SF structured state. Encoding nearby pedestrians as Navground `State` objects and
static obstacles as Navground wall/segment state remains an adapter design task, not proven evidence.
This is the primary adapter burden.

**Kinematics:** Navground exposes `2WDiff`, `2WDiffDyn`, `4WOmni`, `Ahead`, `Bicycle`, and `Omni`.
Robot SF `unicycle_vw` is closest to `2WDiff` (differential-drive). Holonomic Navground behaviors
(`Omni`) would require a `holonomic-to-unicycle` projection not currently implemented in Robot SF.

## Licensing/Runtime Constraints

| Constraint | Status |
| --- | --- |
| License | MIT, as recorded by #2442 from PyPI/repository metadata |
| Python compatibility | `>=3.10`; #2442 recorded CPython 3.10-3.14 wheels on PyPI as observed on 2026-06-07 |
| Core dependencies | `numpy>=1.21`, `PyYAML` (no heavy native deps for core) |
| Optional dependencies | `navground-learning` adds `gymnasium>=1.0.0`, `pettingzoo>=1.24.3` |
| Installation method | #2442 proved `uv pip install navground==0.7.0` in an isolated linked worktree |
| Robot SF env constraint | `uv pip install` works outside `pyproject.toml`; adding navground to pyproject.toml would add a persistent dependency for a prototype-only surface |

## Expected Baseline Role

Navground would serve as an **external framework reference anchor**, not a primary benchmark row:

- **ORCA/SocialForce**: source-reference or parity sanity check for existing Robot SF baselines;
  low standalone benchmark value due to behavioral overlap.
- **HRVO**: external parity comparator for Robot SF experimental `algo=hrvo`; useful only if local
  HRVO needs a source-backed cross-check.
- **HL (Human-like)**: the least redundant candidate; best candidate for a narrow source-harness
  prototype if a non-redundant classical behavior is needed.
- **navground-learning**: monitor/source-reference only for learned-policy intake; not relevant to
  classical benchmark rows.

## Failure Modes

1. **Neighbor encoding gap**: The primary adapter burden is constructing Navground local-environment
   state from Robot SF pedestrian observations. If this encoding is lossy or semantically different,
   behavior outputs may diverge from expected social-navigation patterns.

2. **Static obstacle handling**: Navground obstacle APIs are not proven in the #2442 smoke. If
   Navground obstacles require a different representation (e.g., continuous walls vs. segmented
   obstacles), adapter complexity increases significantly.

3. **Kinematics mismatch**: Behaviors configured as `2WDiff` may produce different heading-tracking
   dynamics than Robot SF `unicycle_vw`. The differential-drive projection from `Twist2` to
   `unicycle_vw` may need heading-velocity coupling that Navground handles internally.

4. **Action normalization divergence**: Navground `compute_cmd()` returns physical velocity commands.
   Robot SF adapters normalize actions to `[0, 1]` for some planner paths. If the Navground adapter
   bypasses normalization, behavior semantics may differ from native baselines.

5. **Dependency surface expansion**: Adding `navground` to `pyproject.toml` introduces a persistent
   dependency for prototype-only value. This violates the current conservative dependency policy.

## Claim Boundary

This report proves:

- `navground==0.7.0` is importable in the Robot SF worktree environment (from #2442 evidence).
- Navground ORCA, HRVO, HL, and SocialForce emit finite `Twist2` commands through `2WDiff`
  kinematics in a minimal source-harness smoke (from #2442 evidence).
- Navground is MIT-licensed with compatible Python version support (from #2442 evidence).
- No Robot SF adapter, benchmark row, or dependency was added (by design).

This report does not prove:

- That Navground behaviors produce semantically equivalent social-navigation behavior to existing
  Robot SF baselines in benchmark scenarios.
- That Navground local-environment encoding is compatible with Robot SF structured state.
- That Navground static-obstacle handling is compatible with Robot SF obstacle segments.
- That a Navground adapter would be benchmark-ready or benchmark-value-positive.

## Verdict

**Viable prototype, not worth integrating now.**

Navground is technically importable and behavior-compatible at the command level, but:

1. The redundancy surface (ORCA, SocialForce) dominates the most accessible behaviors.
2. The non-redundant candidate (HL) requires an unproven adapter for neighbor/obstacle encoding.
3. The dependency cost is disproportionate to the expected benchmark value.
4. No evidence exists that Navground behaviors would produce distinct benchmark outcomes.

The framework is viable as an external reference anchor or parity comparator if a future issue needs
a source-backed cross-check for a specific behavior family, but it should not become a benchmark row
or dependency until a concrete non-redundant use case is identified.

## Smoke Scenario Status

**Blocked with reason:** No smoke scenario was executed. The #2442 source-harness smoke proved
importability and single-target command emission, but a full Robot SF scenario-level smoke requires:

1. A Navground adapter that maps Robot SF structured state to Navground behavior state.
2. Navground local-environment construction from pedestrian and obstacle observations.
3. Benchmark runner integration to step Navground behaviors within the Robot SF episode loop.

These are adapter tasks, not spike-report tasks. The spike report concludes that the adapter burden
is not justified by the expected benchmark value.

## Observed Evidence vs. Inference

### Observed (from #2442/#2514)

- `navground==0.7.0` imported successfully in the #2442 linked worktree.
- `navground-learning==0.2.0` imports successfully.
- `core.Behavior.types` returns `['', 'Dummy', 'HL', 'HRVO', 'ORCA', 'PyDummy', 'SocialForce']`.
- `core.Kinematics.types` returns `['2WDiff', '2WDiffDyn', '4WOmni', 'Ahead', 'Bicycle', 'Omni']`.
- ORCA command: `[1.0, 0.0]` angular `0.0` for `2WDiff` with target at `[1, 0]`.
- HRVO command: `[1.0, 0.0]` angular `0.0` for `2WDiff` with target at `[1, 0]`.
- HL command: `[0.55067104, 0.0]` angular `0.0` for `2WDiff` with target at `[1, 0]`.
- SocialForce command: `[0.2, 0.0]` angular `0.0` for `2WDiff` with target at `[1, 0]`.
- Navground learning exports `NavgroundEnv`, `BaseEnv`, `env`, `MultiAgentNavgroundEnv`,
  `shared_parallel_env`, `parallel_env`.
- License is MIT. Python `>=3.10`.

### Inferred (not proven)

- Navground HL behavior would produce meaningfully different benchmark outcomes from existing Robot SF
  baselines.
- Navground neighbor/obstacle encoding is straightforward to implement in a Robot SF adapter.
- Navground 2WDiff kinematics are semantically equivalent to Robot SF `unicycle_vw` for
  social-navigation scenarios.
- The dependency cost is proportional to the research value of a Navground comparator.

## Follow-Up Recommendation

No immediate follow-up is recommended. If the research lane later requires a source-backed
non-redundant classical behavior comparator:

> Open a bounded source-harness issue that installs `navground==0.7.0` in an isolated optional
> environment, maps one Robot SF structured state (with pedestrians and obstacles) to Navground HL +
> 2WDiff, computes one finite `Twist2`, and records the exact adapter caveats. Stop before adding a
> planner-zoo row.

Do not add `navground` to `pyproject.toml` or the benchmark planner registry until that source
harness demonstrates non-redundant behavior and an adapter boundary worth maintaining.

## Related Surfaces

- [#2442 assessment](issue_2442_navground_assessment.md): canonical upstream probe evidence
- [Planner zoo external framework anchors](../../planner_zoo/index.md): Navground row in the external
  framework assessment table
- [Benchmark planner family coverage](../../benchmark_planner_family_coverage.md): Navground row in
  the external family anchors section
