# Issue #1368 NeuPAN Point-Obstacle Comparator Assessment

Date: 2026-05-20

Related issue:

* <https://github.com/ll7/robot_sf_ll7/issues/1368>

## Goal

Assess whether NeuPAN is a useful point-obstacle local-planner comparator for Robot SF, or whether
its non-social point-obstacle abstraction, GPL-3.0 license, and runtime stack make it monitor-only
for AMV social-navigation benchmarking.

This is a source-side assessment only. It does not add a Robot SF adapter, candidate-registry entry,
or benchmark config.

## Primary Sources Checked

External sources:

* NeuPAN upstream repository: <https://github.com/hanruihua/NeuPAN>
* NeuPAN project page: <https://hanruihua.github.io/neupan_project/>
* NeuPAN arXiv entry: <https://arxiv.org/abs/2403.06828>

Local source clone:

* `output/repos/NeuPAN`
* Upstream revision checked: `579e7af` (`2026-02-05`, `Update README.md`)

Repository guidance:

* `docs/context/external_planner_reuse_checklist.md`
* `docs/dev/holonomic_action_contract.md`
* `docs/dev/planner_adapter_template.md`
* `docs/benchmark_planner_family_coverage.md`

## Source Contract

NeuPAN is a map-free MPC-style motion planner that maps obstacle point data to control actions. The
upstream README describes Python `>=3.10`, an editable package install, optional `ir-sim` examples,
and example runs such as:

```bash
python run_exp.py -e corridor -d diff
```

The source package exposes:

* `neupan.neupan.forward(state, points, velocities=None)`
* robot state shaped as at least `(3, 1)` with `x, y, theta`
* obstacle points shaped `(2, N)` in global coordinates
* optional obstacle point velocities shaped `(2, N)`
* example kinematics modes: `diff`, `acker`, and `omni`

For `diff`, the upstream robot config uses linear and angular speed bounds. For `acker`, the second
command is steering angle. For `omni`, the source converts an internal speed/orientation pair into
`vx, vy`.

## Source-Side Smoke

Commands run:

```bash
git clone --depth 1 https://github.com/hanruihua/NeuPAN output/repos/NeuPAN
uv venv output/neupan_venv --python 3.12
uv pip install --python output/neupan_venv/bin/python -e 'output/repos/NeuPAN[irsim]'
cd output/repos/NeuPAN/example
timeout 120s ../../../neupan_venv/bin/python run_exp.py -e corridor -d diff -m 3
```

The install completed in an ignored side environment under `output/neupan_venv`, not the Robot SF
`.venv`. That separation matters because NeuPAN brings a large solver and Torch/CUDA dependency
surface into the source environment, including `cvxpy`, `cvxpylayers`, `ecos`, `gctl==1.2`,
`ir-sim`, `scipy==1.13.0`, `torch==2.12.0`, and CUDA packages.

The 3-step source example did not reach planner initialization. It failed at import time because
`gctl` imports `tkinter` and this local Python 3.12 environment does not provide it:

```text
ModuleNotFoundError: No module named 'tkinter'
```

This is a source-environment blocker, not a Robot SF benchmark result. No runtime/control-rate
measurement was collected because the source harness stopped before the first NeuPAN step.

## Robot SF Mapping Assessment

Observation mapping:

* Static obstacles could be represented as sampled boundary points, but that would be a lossy
  point-cloud abstraction rather than Robot SF's structured map/obstacle contract.
* Pedestrians could be represented as current 2D points or sampled discs. That is deployment-
  observable if it uses current pose/radius only, but it removes group, intent, personal-space,
  social-role, and reciprocal-interaction semantics.
* Obstacle velocities are supported by the source API. For Robot SF they would have to come only
  from current/past tracked pedestrian velocities, never future trajectory labels.

Action mapping:

* The `diff` variant is the closest match to Robot SF `unicycle_vw` because it returns linear and
  angular speed.
* The `acker` variant is not directly compatible with `unicycle_vw`; steering-angle semantics would
  need an explicit bicycle-to-unicycle projection or a bicycle-robot benchmark path.
* The `omni` variant returns world-frame `vx, vy`, which can map only through the holonomic bridge
  or a lossy heading-safe projection for differential-drive profiles.

Kinematics and runtime:

* Upstream examples cover differential, Ackermann, and omnidirectional robots.
* The optimizer is CPU-oriented in the README; the source smoke installed a heavy optimization and
  Torch dependency stack before hitting the Tk blocker.
* Control-rate feasibility remains unproven here. Any future adapter issue should first create a
  reproducible source-side environment and measure latency as a function of obstacle-point count.

License:

* The upstream repository is GPL-3.0.
* Do not vendor or copy NeuPAN code into Robot SF without an explicit maintainer/legal decision.
* A future experiment, if accepted, should keep NeuPAN as an optional external source checkout or
  isolated source-harness dependency.

## Verdict

Verdict: `monitor / source-side only`.

NeuPAN is interesting as a non-social, point-obstacle reactive comparator, but it should not be
started as a Robot SF adapter now.

Reasons:

* GPL-3.0 blocks straightforward vendoring into this repository.
* Source harness reproduction is not yet complete on this machine because the upstream stack needs
  `tkinter` through `gctl`.
* Runtime/control-rate feasibility is unmeasured after the import blocker.
* The point-obstacle abstraction is not social-navigation evidence. It can test reactive geometric
  collision avoidance, but it cannot support claims about social compliance, group behavior,
  personal-space modeling, or human-intent reasoning.
* Robot SF already has non-learning local-planner comparators behind explicit experimental guards;
  NeuPAN would need a clear value proposition beyond another adapter-heavy local planner.

## Follow-Up Rules

Safe follow-up:

* Keep NeuPAN in external-planner watchlists as a point-obstacle, model-based-learning comparator.
* Revisit only if a maintainer wants a GPL-isolated source-harness experiment.
* Before any wrapper, prove the source example in a disposable side environment with `tkinter` and
  record per-step latency for representative obstacle-point counts.
* If a wrapper is later approved, start with `diff` only and fail closed on missing source assets,
  unsupported kinematics, or unavailable optional dependencies.

Do not:

* add NeuPAN to `docs/context/policy_search/candidate_registry.yaml` yet;
* cite NeuPAN as social-navigation support;
* vendor GPL-3.0 source code into Robot SF without an explicit decision;
* treat a future point-cloud wrapper smoke as benchmark readiness or social-behavior evidence.

## Validation

Commands run for this assessment:

```bash
rg -n "NeuPAN|point-obstacle|point obstacle|reactive comparator|GPL|local-planner comparator|SAGE|Tentabot|NavDP|NoMaD|policy_search" docs/context docs/benchmark_planner_family_coverage.md robot_sf tests scripts configs -g '!output/**'
git clone --depth 1 https://github.com/hanruihua/NeuPAN output/repos/NeuPAN
uv venv output/neupan_venv --python 3.12
uv pip install --python output/neupan_venv/bin/python -e 'output/repos/NeuPAN[irsim]'
cd output/repos/NeuPAN/example
timeout 120s ../../../neupan_venv/bin/python run_exp.py -e corridor -d diff -m 3
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
uv run pytest tests/benchmark/test_algorithm_readiness_contract.py tests/benchmark/test_algorithm_metadata_contract.py -q
uv run pytest tests/benchmark/test_map_runner_utils.py tests/benchmark/test_planner_inclusion.py -q
uv run pytest tests/benchmark/test_planner_command_contract.py -q
```

Result: local docs and contract surfaces were identified; upstream clone and side-environment
install succeeded; source example was blocked by missing `tkinter` before runtime measurement.
Docs proof passed for the four changed files; whitespace diff check passed; targeted Robot SF
contract tests passed with `28 passed`, `90 passed`, and `5 passed`.

No Robot SF benchmark was run because this issue is source-side assessment and no adapter exists.
