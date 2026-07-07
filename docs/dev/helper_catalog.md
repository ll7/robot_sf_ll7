# Helper Catalog

The helper catalog is the contributor-facing index for reusable Robot SF helper
surfaces that were extracted from examples and benchmark scripts. Use these
helpers when adding demos, benchmark wrappers, or documentation automation so
new files stay as orchestration glue instead of reimplementing setup, policy
loading, recording, and output-directory behavior.

Source inventory: [`specs/140-extract-reusable-helpers/helper_inventory.yaml`](../../specs/140-extract-reusable-helpers/helper_inventory.yaml).
Contract notes: [`specs/140-extract-reusable-helpers/contracts/helper_catalog.md`](../../specs/140-extract-reusable-helpers/contracts/helper_catalog.md).

## When To Use Helpers

- Prefer `robot_sf.gym_env.environment_factory` for new environment creation.
- Prefer `robot_sf.benchmark.helper_catalog` when a benchmark or example needs a
  classic environment, a trained PPO policy, or a deterministic episode loop.
- Prefer `robot_sf.render.helper_catalog` for reusable output-directory handling
  and frame sampling utilities.
- Prefer `robot_sf.docs.helper_catalog.register_helper` when a helper-specific
  docs index entry should be inserted programmatically.
- Keep examples focused on sequencing, arguments, and user-facing narrative.

## Implemented Helpers

### `prepare_classic_env`

Module: `robot_sf.benchmark.helper_catalog`

Initializes a classic robot environment through the factory path and returns a
small deterministic seed list for example and smoke-test orchestration.

Use this when an example needs the standard classic setup without manually
constructing `RobotSimulationConfig`, calling `make_robot_env`, and choosing
default seeds itself.

### `load_trained_policy`

Module: `robot_sf.benchmark.helper_catalog`

Loads a Stable-Baselines3 PPO policy from a model path and caches successful
loads by absolute path. Missing model files raise `FileNotFoundError` with
actionable guidance instead of failing later inside a demo loop.

Use this in examples or benchmark scripts that replay bundled or externally
staged PPO checkpoints.

### `run_episodes_with_recording`

Module: `robot_sf.benchmark.helper_catalog`

Runs a deterministic list of seeded episodes with a policy and returns
structured episode summaries. When recording is enabled, it ensures the output
directory exists before episode execution.

Use this when a script would otherwise duplicate the reset, predict, step,
termination, and summary loop.

### `ensure_output_dir`

Module: `robot_sf.render.helper_catalog`

Normalizes a path through the repository artifact-path policy, creates the
directory with parent directories, and returns the resolved path.

Use this before writing videos, figures, converted recordings, benchmark
summaries, or other generated artifacts.

### `capture_frames`

Module: `robot_sf.render.helper_catalog`

Provides a shared frame-sampling entry point with stride validation and
render-method checks. The current implementation is intentionally conservative:
it validates the interface and returns an empty list until a caller-specific
render contract is supplied.

Use this as the reusable frame-capture seam for recording helpers instead of
adding ad hoc stride validation in scripts.

### `register_helper`

Module: `robot_sf.docs.helper_catalog`

Adds a deduplicated helper entry to `docs/README.md` under a `Helper Catalog`
section when helper-specific documentation should be registered from tooling.

Use this for automation-driven helper index updates. Broad narrative pages, such
as this landing page, can still be maintained manually when that keeps the docs
clearer.

## Inventory-Tracked Follow-Ups

The inventory also contains helper candidates that document repeated
orchestration patterns but are not standalone helper functions yet. Keep their
anchors here so inventory links resolve while making their state explicit.

### `prepare_pedestrian_env`

Status: inventory-tracked candidate.

The inventory records repeated pedestrian environment setup patterns involving
map conversion, pedestrian environment configuration, and robot policy
injection. Prefer the current environment factory and existing pedestrian demos
until this candidate is promoted to an implemented helper.

### `benchmark_batch_runner`

Status: inventory-tracked candidate.

The inventory records repeated batch-orchestration patterns around
`robot_sf.benchmark.runner.run_batch` and
`robot_sf.benchmark.full_classic.orchestrator.run_full_benchmark`. Prefer those
canonical runner APIs directly until this candidate is promoted to a wrapper.

## Example Author Checklist

Before adding or updating an example:

1. Check the helper list above for environment, policy, output, recording, or
   docs-index behavior.
2. Keep new example code limited to user-facing configuration and sequence.
3. Record generated outputs under `output/` through the artifact-path helpers.
4. Update `examples/examples_manifest.yaml` and regenerate
   `examples/README.md` when example visibility or metadata changes.
5. Add focused tests around any new helper behavior before relying on it from an
   example or benchmark script.
This page lists small reusable helpers that scripts and examples should call
instead of duplicating rendering, artifact, or benchmark glue.

## Render Helpers

Module: `robot_sf.render.helper_catalog`

- `ensure_output_dir(path: Path) -> Path`
  - Resolves the path through the canonical artifact-root rules, creates it,
    and returns the resolved directory.
- `capture_frames(env, stride: int = 1) -> list[np.ndarray]`
  - Samples RGB frame arrays from common render buffers such as `env.frames` or
    `env.sim_ui.frames`.
  - If no buffer is populated, it calls `env.render()` once and returns the RGB
    frame array when that render method provides one.
  - Returns an empty list when a render-capable input produces no frame data.
  - Raises `ValueError` for invalid stride values and `AttributeError` when the
    input has no render method.

## Contact Sheets

Module: `scripts.generate_video_contact_sheet`

- `generate_contact_sheet(episodes_jsonl: Path, output_path: Path, *, columns: int = 3) -> Path`
  - Reads episode JSONL rows with image frame paths in either `frame_paths` or
    `video.frame_paths`.
  - Resolves relative frame paths next to the JSONL file.
  - Writes a deterministic PNG grid to `output_path` and returns that path.
  - Raises a clear error when no frame image paths are available. Rows that only
    contain MP4 paths should be pre-extracted to frame images before calling
    this helper.
