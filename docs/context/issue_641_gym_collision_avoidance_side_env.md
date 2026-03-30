# Issue 641 gym-collision-avoidance Side-Environment Reproduction

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#605` gym-collision-avoidance reference assessment
- `robot_sf_ll7#639` fail-fast main-runtime source-harness probe
- `robot_sf_ll7#641` isolated side-environment reproduction

## Goal

Re-run the upstream `gym-collision-avoidance` learned-policy source harness in an isolated legacy
runtime without reintroducing `gym` into the main Robot SF environment.

This issue exists to answer a narrower question than `#639`:

- if the main-runtime blocker was just legacy dependencies,
- does the upstream CADRL / GA3C-CADRL harness become reproducible in a side environment,
- or do deeper upstream runtime issues appear next?

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/gym_collision_avoidance_side_env_probe/report.json`
- Markdown report:
  `output/benchmarks/external/gym_collision_avoidance_side_env_probe/report.md`

Generated with:

```bash
uv run python scripts/tools/probe_gym_collision_avoidance_side_env.py \
  --repo-root output/repos/gym-collision-avoidance \
  --side-env-python output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python \
  --output-json output/benchmarks/external/gym_collision_avoidance_side_env_probe/report.json \
  --output-md output/benchmarks/external/gym_collision_avoidance_side_env_probe/report.md
```

## Side environment recipe

The isolated side environment used for the probe was created under:

- `output/benchmarks/external/gym_collision_avoidance_side_env/.venv`

Representative setup commands:

```bash
PYTHON_INTERPRETER=/path/to/python3.10

uv venv --python "${PYTHON_INTERPRETER}" \
  output/benchmarks/external/gym_collision_avoidance_side_env/.venv

uv pip install --python output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python \
  pip setuptools wheel gym tensorflow scipy imageio==2.4.1 Pillow PyOpenGL pyyaml matplotlib pytz moviepy pandas pytest

output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python -m pip install -e \
  output/repos/gym-collision-avoidance
```

This intentionally keeps the legacy `gym` runtime out of the main `robot_sf_ll7` environment.

## Commands rerun

1. side-environment version check

```bash
output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python -c "import gym, json, tensorflow as tf; print(json.dumps({'gym': gym.__version__, 'tensorflow': tf.__version__}))"
```

Observed result:
- `gym==0.26.2`
- `tensorflow==2.21.0`

2. learned-policy import path

```bash
cd output/repos/gym-collision-avoidance
../../benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python -c "from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy; policy = GA3CCADRLPolicy(); policy.initialize_network(); print('ga3c_ready')"
```

Observed result:
- succeeds
- prints `ga3c_ready`

3. upstream example entrypoint

```bash
cd output/repos/gym-collision-avoidance
../../benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python gym_collision_avoidance/experiments/src/example.py
```

Observed result:
- blocked
- first concrete blocker:
  `ImportError: Failed to import tkagg backend. You appear to be using an outdated version of uv's managed Python distribution which is not compatible with Tk.`

4. upstream example pytest path

```bash
cd output/repos/gym-collision-avoidance
../../benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python -m pytest -c /dev/null -q gym_collision_avoidance/tests/test_collision_avoidance.py -k test_example_script
```

Observed result:
- blocked on the same TkAgg backend path
- `-c /dev/null` is required so pytest does not inherit unrelated `robot_sf_ll7` coverage args

## Current result

Verdict: `source harness still blocked`

Primary blocker:
- upstream macOS visualization path forces `TkAgg` backend

What changed versus `#639`:
- the legacy runtime dependency blocker is materially reduced,
- `gym` no longer blocks import,
- TensorFlow-backed learned-policy initialization now works,
- the first blocker is now in upstream visualization/backend handling, not the CADRL-family runtime.

Why this matters:
- this is a stronger result than the main-runtime probe,
- it shows the CADRL / GA3C-CADRL policy family is not dead here,
- but it still does **not** prove full upstream source-harness parity.

## Extracted implication

Observed implication from the side-environment run:

- learned policy path: `reproducible`
- full example/test harness: `still blocked`
- blocker class: `GUI/backend rather than missing planner dependency`

Interpretation:
- a benchmark-side wrapper issue is still not justified from full source-harness parity,
- but this family now has a narrower, more concrete remaining blocker than before,
- and any follow-up should focus on headless/non-visual upstream reproduction rather than generic dependency rescue.

## Recommendation

Recommendation: `wrapper still not justified from full source-harness parity`

Reason:
- the learned policy now boots in an isolated side environment,
- but the canonical upstream example/test path still fails before a full harness run completes,
- and that failure is still upstream-harness-visible.

What is justified next if this family remains interesting:

1. a narrow non-visual/headless upstream reproduction attempt that keeps planner logic unchanged,
2. only after that, reconsider whether a wrapper/parity issue is justified.

What is not recommended yet:

- reintroducing legacy `gym` into the main repository runtime,
- benchmark-side adapter work based only on the current side-environment result,
- overclaiming CADRL-family source parity from the current partial reproduction.
