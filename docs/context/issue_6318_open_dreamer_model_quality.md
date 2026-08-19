# Issue #6318 Open Dreamer-style model-quality gate

Plain-language summary: this gate checks whether the clean-room, structured-observation dynamics
model predicts held-out Robot SF trajectories better than simple references before any imagined
replay or policy experiment is attempted.

## Scope and boundary

The gate consumes `RLTrajectoryDataset.v1` through the existing structured-observation adapter. It
fits the action-conditioned latent transition, reward head, and continuation head on one explicit
training split, then reports one-step and short multi-step prediction errors on a separate held-out
split. It keeps the observation, reward, and continuation errors separate and reports per-scenario
held-out metrics.

The references are a persistence predictor and a deterministic random-feature multilayer perceptron
(MLP). The MLP is a diagnostic comparator, not a claim about the best predictor. A fitted model
passes only when it strictly improves every required one-step and multi-step metric against every
configured reference; observation, reward, and continuation errors cannot be omitted from the gate.
Otherwise the gate stops as `failed_model_quality`.

The gate returns `blocked_insufficient_data` when the dataset does not contain the configured
minimum training and holdout episodes/transitions. It returns `blocked_contract` when the dataset,
split, action, or structured-observation contract cannot be validated. Neither blocked state is
model-quality evidence. It also blocks as insufficient data when either split lacks both continuing
and terminal transition targets; a max-horizon trace with no terminal label cannot validate the
continuation head.

This surface does not run the matched soft actor-critic (SAC) arms, inject imagined transitions,
change a released configuration, admit benchmark evidence, promote a policy, or make a paper or
dissertation claim. Those remain separately gated follow-up work in #6318.

## Entry points

- `robot_sf/research/open_dreamer_model_quality.py` — typed config, fitted model, baselines, and
  report contract;
- `scripts/validation/run_open_dreamer_model_quality.py` — config-first command;
- `scripts/validation/collect_open_dreamer_quality_fixture.py` — bounded native map-runner trace
  collector for local diagnostics;
- `configs/research/issue_6318_open_dreamer_model_quality.yaml` — example pointing at the tiny
  committed dataset preview.

Run the example from a linked worktree:

```bash
UV_NO_SYNC=1 .venv/bin/python scripts/validation/run_open_dreamer_model_quality.py \
  --config configs/research/issue_6318_open_dreamer_model_quality.yaml \
  --output-dir output/issue6318_model_quality_preview
```

The committed preview contains one two-step training episode and no held-out episode. Therefore,
the expected result is `blocked_insufficient_data`; it is not a successful model-quality run.

For a bounded local diagnostic, collect distinct train/holdout scenario families and override the
dataset path without changing the committed config:

```bash
LOGURU_LEVEL=WARNING UV_NO_SYNC=1 .venv/bin/python \
  scripts/validation/collect_open_dreamer_quality_fixture.py \
  --scenario-seed classic_doorway_medium:6318 \
  --scenario-seed classic_doorway_medium:6320 \
  --scenario-seed classic_cross_trap_low:6326 \
  --scenario-seed classic_cross_trap_low:6331 \
  --output-dir output/issue6318_quality_native_fixture \
  --dataset-id issue6318_quality_native_probe

set +e
UV_NO_SYNC=1 .venv/bin/python scripts/validation/run_open_dreamer_model_quality.py \
  --config configs/research/issue_6318_open_dreamer_model_quality.yaml \
  --dataset-path output/issue6318_quality_native_fixture/issue6318_quality_native_probe.jsonl \
  --output-dir output/issue6318_model_quality_native_probe
```

The collector is a local diagnostic path. In particular, a run that reaches
`failed_model_quality` is a negative readiness result only until its dataset, split manifest,
native execution route, and durable artifact pointer are independently reviewed.

## Data and provenance rules

- The dataset must be a validated `RLTrajectoryDataset.v1` JSONL source.
- The adapter's deterministic scenario/seed split and no-cross-scenario leakage rules are
  enforced before fitting.
- Feature normalization is learned from the training split only and recorded in the report.
- Physical recorder actions are inverted through the configured adapter bounds into the dynamics
  model's bounded `[-1, 1]` action space and the transform is recorded.
- Continuation targets use the next observed step's terminal/truncated marker. Each split must
  contain both continuing and terminal transitions before the head is judged.
- Raw datasets and generated reports under `output/` are worktree-local until separately promoted
  with a durable URI and checksum.
- A future diagnostic run must record its exact dataset digest, code commit, config, source policy,
  scenario/seed scope, native/adapter/fallback status, and artifact durability before its result
  can be independently reviewed.

## Validation

```bash
UV_NO_SYNC=1 .venv/bin/python -m pytest tests/research/test_open_dreamer_model_quality.py -q
UV_NO_SYNC=1 .venv/bin/python scripts/validation/run_open_dreamer_model_quality.py --help
UV_NO_SYNC=1 .venv/bin/python scripts/validation/collect_open_dreamer_quality_fixture.py --help
UV_NO_SYNC=1 .venv/bin/ruff check robot_sf/research/open_dreamer_model_quality.py scripts/validation/run_open_dreamer_model_quality.py scripts/validation/collect_open_dreamer_quality_fixture.py tests/research/test_open_dreamer_model_quality.py
UV_NO_SYNC=1 .venv/bin/ruff format --check robot_sf/research/open_dreamer_model_quality.py scripts/validation/run_open_dreamer_model_quality.py scripts/validation/collect_open_dreamer_quality_fixture.py tests/research/test_open_dreamer_model_quality.py
git diff --check
```
