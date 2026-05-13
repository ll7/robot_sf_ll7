# Issue 1190 DreamerV3 checkpoint import boundary probe

Related issue: [#1190](https://github.com/ll7/robot_sf_ll7/issues/1190)
Related design note: [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md)
Program boundary: [dreamerv3_program_close_out_2026_04_30.md](dreamerv3_program_close_out_2026_04_30.md)
Runbook: [docs/training/dreamerv3_rllib_drive_state_rays.md](../training/dreamerv3_rllib_drive_state_rays.md)

## Decision

Fail closed. Do not run the imported-vs-scratch BR-08 gate comparison from #1190 yet.

The current repository and Ray 2.53.0 surface do not expose a clean, repo-owned DreamerV3
world-model import boundary. The available checkpoint path is RLlib's full Algorithm/RLModule
checkpoint restore path. The Dreamer world model exists as nested PyTorch module state inside
RLlib's DreamerV3 RLModule, so selectively transferring it into a fresh BR-08 run would require
RLlib-specific checkpoint surgery rather than a stable Robot SF command/config contract.

## Canonical Path Checked

The canonical BR-08 gate command remains:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml
```

Expected run artifacts are under:

```text
output/dreamerv3/<run_id>_<timestamp>/
output/dreamerv3/<run_id>_<timestamp>/run_summary.json
output/dreamerv3/<run_id>_<timestamp>/result.jsonl
output/dreamerv3/<run_id>_<timestamp>/checkpoints/
```

This checkout does not contain a retained source DreamerV3 checkpoint under `output/dreamerv3/`.
That means there is no local source checkpoint artifact to inspect, import, or compare from.

## Boundary Evidence

Local inspection used the optional RLlib environment:

```bash
uv run --extra rllib python - <<'PY'
import inspect
import ray
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3

print(ray.__version__)
print(inspect.signature(DreamerV3.from_checkpoint))
print(inspect.getsource(DreamerV3.get_checkpointable_components))
PY
```

Observed surface:

- Ray/RLlib version: `2.53.0`.
- `DreamerV3.from_checkpoint(...)` restores an Algorithm checkpoint.
- `DreamerV3.get_checkpointable_components(...)` exposes checkpointable RLlib components such as
  `learner_group`, `env_runner`, connector components, and evaluation env runner.
- `DreamerV3.get_module(...)` returns the RLModule from the EnvRunner/RLModule stack, not a
  repo-level world-model import endpoint.
- `DreamerV3TorchRLModule` wraps `dreamer_model`, and `DreamerModel` contains
  `dreamer_model.world_model`.

The candidate tensor namespace for a manual transfer would therefore be nested under the RLModule
state, conceptually `dreamer_model.world_model.*`. That is not a stable import contract:

- key names and module construction are RLlib internals,
- the module shape depends on the Ray version, model size, action space, observation contract,
  encoder, decoder, and connector stack,
- the Robot SF launcher does not accept a `--world-model-init` or equivalent config field,
- a fresh BR-08 run would need custom load/filter/remap code before RLlib builds or trains the
  Algorithm.

That crosses the #1190 stop condition: the path requires RLlib-specific checkpoint surgery instead
of a clean boundary.

## Comparison Decision

No imported-vs-scratch gate comparison was launched.

That is intentional. The issue only allows the comparison when the checkpoint boundary is clean.
Running the gate now would either:

- compare scratch-only behavior, which does not answer #1190, or
- rely on an unversioned custom tensor transfer, which would weaken provenance and make the result
  hard to reproduce.

## Artifact Decision

No durable model artifact was produced. No `output/dreamerv3/` checkpoint from this probe is a
downstream dependency.

Future work must promote or reference any source DreamerV3 checkpoint through a durable artifact
pointer before downstream runs depend on it.

## Reopen Criteria

Reconsider this only if all of the following exist:

- a retained source DreamerV3 checkpoint with a durable pointer,
- an RLlib-supported or Robot-SF-owned world-model/RLModule initialization API,
- a config-first launcher field that declares the source checkpoint and import scope,
- a smoke test proving shape/schema mismatch failures are actionable,
- a same-seed imported-vs-scratch BR-08 gate command that records execution mode as `native` or
  `not_available`, never fallback success.

Until then, #1190 is complete as a fail-closed no-action recommendation rather than a training
campaign.

## Validation

This note is guarded by:

```bash
uv run pytest tests/docs/test_dreamerv3_checkpoint_import_boundary.py -q
```

The proof is inspection-level, not benchmark evidence. It should not be used as a DreamerV3
performance claim.
