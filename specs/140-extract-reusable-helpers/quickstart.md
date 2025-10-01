# Quickstart: Using Reusable Helpers from `robot_sf`

## Goal
Demonstrate how an example script should orchestrate reusable helpers after consolidation.

## Prerequisites
- Repository setup per `docs/dev_guide.md` (`uv sync`, activate virtualenv).
- Feature branch `140-extract-reusable-helpers` checked out.
- Inventory of helper capabilities published in `docs/README.md` (to be added during implementation).

## Steps
1. **Discover helper**
   - Browse the helper catalog in `robot_sf/benchmark/utils/helper_catalog.py` (proposed location) or its documentation entry.
   - Identify functions such as `prepare_classic_env(config_override=None)` and `run_benchmark_episode(...)`.

2. **Import helpers**
   ```python
   from robot_sf.benchmark.helper_catalog import (
       prepare_classic_env,
       load_trained_policy,
       run_episodes_with_recording,
   )
   ```

3. **Configure orchestrator**
   - Define scenario-specific overrides (e.g., `scenario_name`, `max_episodes`).
   - Use helper return values (env instance, policy, output paths) instead of reproducing setup code.

4. **Run orchestration**
   ```python
   env, seeds = prepare_classic_env(scenario_name="classic_interactions")
   policy = load_trained_policy("model/run_043.zip")
   results = run_episodes_with_recording(
       env=env,
       policy=policy,
       seeds=seeds,
       record=True,
       output_dir="tmp/vis_runs",
   )
   ```

5. **Handle results & cleanup**
   - Helpers return structured data (TypedDict) documented in `helper_catalog.py`.
   - Orchestrator prints/visualizes summaries using provided formatting helpers (e.g., `format_episode_summary_table`).
   - Ensure environment cleanup is handled by helper contexts (e.g., using context managers).

## Validation
- Run `uv run pytest tests` and `scripts/validation/test_basic_environment.sh` after integrating helpers.
- Confirm the orchestrator script now contains only  glue code and imports from `robot_sf/`.

## Next Steps
- Update the relevant example README with references to the helper catalog.
- Add new helpers to the documentation index (`docs/README.md`).
