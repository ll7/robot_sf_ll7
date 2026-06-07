# Issue #2442 Navground Planner-Zoo Assessment

Issue: [#2442](https://github.com/ll7/robot_sf_ll7/issues/2442)

Date: 2026-06-07

Status: assessment-only. This note does not add a Robot SF wrapper, benchmark row, or dependency.

## Decision

Classify Navground as `prototype only`, not `integrate next`.

The upstream package is active enough to assess seriously: `navground==0.7.0` installs from PyPI in
this worktree, exposes Python bindings for the local-navigation core, and has MIT license metadata.
Its built-in behaviors overlap heavily with existing Robot SF baselines, especially ORCA and Social
Force, so a direct benchmark-row integration would probably duplicate current coverage before it
adds new research value.

The most useful next step, if this family is reopened, is a tiny source-harness prototype for the
non-redundant behavior surface: Navground `HL` (Human-like) first, with HRVO as a possible parity
comparator. The prototype should map one Robot SF structured state into Navground behavior state,
request a `2WDiff` command, record a finite `Twist2`, and stop before adding a planner-zoo row.

## Upstream Surface

Sources checked:

- Navground docs: <https://idsia-robotics.github.io/navground/>
- PyPI `navground`: <https://pypi.org/project/navground/>
- Navground repository: <https://github.com/idsia-robotics/navground>
- Navground learning docs: <https://idsia-robotics.github.io/navground_learning/0.2/index.html>
- PyPI `navground-learning`: <https://pypi.org/project/navground-learning/>

Observed upstream facts:

- `navground==0.7.0` is the current PyPI release observed on 2026-06-07, with
  `Requires-Python: >=3.10` and CPython 3.10/3.11/3.12/3.13/3.14 wheels listed on PyPI.
- PyPI and the repository report MIT licensing for Navground.
- The docs describe Navground as a two-dimensional navigation-behavior API: behaviors consume ego
  state, local environment, and target, then output a control command.
- The documented built-in behavior set includes ORCA, HRVO, Human-like (`HL`), and Social Force.
- The documented kinematics family includes two-wheeled differential drive (`2WDiff`) in addition
  to omni, ahead, bicycle, and related variants.
- `navground-learning==0.2.0` is a separate package. It interfaces Navground with Gymnasium and
  PettingZoo for imitation-learning and reinforcement-learning workflows.
- Local package metadata showed the core `navground` package depends only on `numpy>=1.21` and
  `PyYAML` by default. `navground-learning` is the package that adds `gymnasium>=1.0.0` and
  `pettingzoo>=1.24.3`.

## Local Probe

Environment: issue #2442 linked worktree on Python 3.13, after a normal `uv sync --all-extras`.

The worktree virtualenv intentionally did not expose `pip` as a module, so the first metadata probe
failed:

```bash
uv run python -m pip index versions navground
```

Result:

```text
No module named pip
```

The install/import/source-harness probe used `uv pip` instead:

```bash
uv pip install 'navground==0.7.0'
uv pip install 'navground-learning==0.2.0'
uv run python - <<'PY'
from importlib.metadata import metadata, version

from navground import core
import navground.learning.env as learning_env
import navground.learning.parallel_env as parallel_env

print("navground", version("navground"), metadata("navground").get("Requires-Python"))
print("navground-learning", version("navground-learning"), metadata("navground-learning").get("Requires-Python"))
print("behavior_types", core.Behavior.types)
print("kinematics_types", core.Kinematics.types)
print("learning_env_exports", [name for name in ("NavgroundEnv", "BaseEnv", "env") if hasattr(learning_env, name)])
print("parallel_env_exports", [name for name in ("MultiAgentNavgroundEnv", "shared_parallel_env", "parallel_env") if hasattr(parallel_env, name)])
for behavior_type in ["ORCA", "HRVO", "HL", "SocialForce"]:
    behavior = core.Behavior.make_type(behavior_type)
    behavior.kinematics = core.Kinematics.make_type("2WDiff")
    behavior.radius = 0.3
    behavior.optimal_speed = 1.0
    behavior.max_speed = 1.0
    behavior.max_angular_speed = 1.0
    behavior.pose = core.Pose2([0.0, 0.0], 0.0)
    behavior.velocity = [0.0, 0.0]
    target = core.Target()
    target.position = [1.0, 0.0]
    behavior.target = target
    behavior.prepare()
    cmd = behavior.compute_cmd(0.1, None, True)
    print("cmd", behavior_type, list(cmd.velocity), cmd.angular_speed)
PY
```

Result:

```text
navground 0.7.0 >=3.10
navground-learning 0.2.0 >=3.10
behavior_types ['', 'Dummy', 'HL', 'HRVO', 'ORCA', 'PyDummy', 'SocialForce']
kinematics_types ['2WDiff', '2WDiffDyn', '4WOmni', 'Ahead', 'Bicycle', 'Omni']
learning_env_exports ['NavgroundEnv', 'BaseEnv', 'env']
parallel_env_exports ['MultiAgentNavgroundEnv', 'shared_parallel_env', 'parallel_env']
cmd ORCA [np.float32(1.0), np.float32(0.0)] 0.0
cmd HRVO [np.float32(1.0), np.float32(0.0)] 0.0
cmd HL [np.float32(0.55067104), np.float32(0.0)] 0.0
cmd SocialForce [np.float32(0.2), np.float32(0.0)] 0.0
```

This proves importability and a behavior-level command path for the tested release in this local
environment. It is not a Robot SF adapter proof, benchmark proof, or Python 3.11 runtime proof. The
PyPI metadata indicates Python 3.11 should be compatible, but this branch did not run the source
harness under Python 3.11.

## Candidate Value

| Navground surface | Robot SF overlap | Assessment |
| --- | --- | --- |
| `ORCA` | Strong overlap with `algo=orca` and Robot SF's reciprocal-avoidance adapter. | Useful as a source reference or parity sanity check, but low standalone benchmark value. |
| `SocialForce` | Strong overlap with `algo=social_force` and the native Social Force baseline. | Parameter-reference value only; not worth a new planner row. |
| `HRVO` | Overlaps Robot SF's experimental `algo=hrvo`, but comes from a packaged external framework. | Possible parity comparator if the local HRVO implementation needs a source-backed cross-check. |
| `HL` | No direct Robot SF baseline with the same upstream behavior surface. | Best non-redundant candidate for a tiny source-harness prototype. |
| `navground.learning` | Adjacent to learned-policy intake and Gymnasium/PettingZoo bridges. | Monitor/source-reference only until a separate issue proves artifact, observation, and adapter value. |
| Narrow-space modulation extensions | Adjacent to route-clearance and bottleneck failure work, but outside the installed base package probe. | Interesting research direction; needs separate source check before any claim. |

## Adapter Boundary

A Robot SF wrapper would need to bridge these contracts explicitly:

- Robot SF structured state: robot pose, velocity, goal, radius, nearby pedestrian pose/velocity,
  pedestrian radius, and static obstacle segments.
- Navground behavior state: `Pose2`, current velocity, `Target`, kinematics object, and a local
  environment representation such as neighbor/obstacle state.
- Navground action: `Twist2` velocity plus angular speed.
- Robot SF action: normalized or physical `unicycle_vw`, depending on the planner adapter boundary.

The local smoke showed `2WDiff` commands can be finite for ORCA, HRVO, HL, and SocialForce with a
single target and no local environment. It did not prove neighbor encoding, static-obstacle
encoding, route following, action normalization, or benchmark runner integration.

## Recommended Follow-Up

Open one bounded follow-up only if the policy-search lane needs a new classical external behavior
candidate:

> Add a Navground `HL` source-harness smoke that installs `navground==0.7.0` in an isolated
> optional environment, maps one Robot SF structured state to Navground `HL + 2WDiff`, computes one
> finite `Twist2`, records adapter caveats, and stops before adding a planner-zoo row.

Do not add `navground` to `pyproject.toml` or the benchmark planner registry until that source
harness demonstrates non-redundant behavior and an adapter boundary worth maintaining.

## Validation

Commands run during the assessment:

```bash
uv sync --all-extras
uv run python -m pip index versions navground
uv run python -m pip index versions navground-learning
uv pip install 'navground==0.7.0'
uv pip install 'navground-learning==0.2.0'
uv run python - <<'PY'
from importlib.metadata import metadata
for dist in ["navground", "navground-learning"]:
    print(dist, metadata(dist).get_all("Requires-Dist"))
PY
uv run python - <<'PY'
# import, registry, and behavior-command probe shown above
PY
```

The `pip index` commands failed because this worktree's virtualenv has no `pip` module. The
`uv pip install` and Python import/command probe succeeded.
