# Root Layout Structured Migration - 2026-06-01

Status: Current

This note records the maintainer-directed migration that removes several first-level compatibility
paths after earlier inventory notes had classified them as keep/defer. The current maintainer
priority is a more structured repository root, so the migration updates the path contracts directly
instead of keeping root-level symlinks.

## Moved Paths

| Previous root path | New path | Validation focus |
| --- | --- | --- |
| `.agent/PLANS.md` | `.agents/PLANS.md` | Agent context stack and plan-writing references. |
| `contracts/` | `docs/contracts/` | Contract tests and feature-extractor documentation. |
| `model_ped/` | `model/pedestrian/` | Pedestrian PPO scripts, examples, and checkpoint lookup tests. |
| `test_pygame/` | `tests/pygame/` | Headless pygame tests, JSONL recording fixtures, IDE config, and lint config. |
| `test_scenarios/` | `tests/fixtures/scenarios/` | OSM fixture consumers and example defaults. |

No root compatibility symlinks are kept; the goal is to reduce first-level repository clutter rather
than preserve legacy aliases.

## Superseded Notes

The following notes remain useful provenance but no longer define the current path decision for the
moved roots:

- `docs/context/issue_1573_root_layout_inventory.md`
- `docs/context/issue_1583_high_risk_root_boundaries.md`
- `docs/context/issue_1598_1599_root_compatibility_decisions.md`
- `docs/context/issue_1690_root_layout_inventory.md`

## Proof Plan

The migration should be accepted only after:

- exact-path searches show the removed first-level directories no longer exist;
- active path references use the new locations;
- targeted contract, pedestrian-model, pygame, and OSM tests pass;
- docs/proof consistency checks pass for changed context notes;
- full PR readiness passes after the final branch sync.

## Validation Evidence

Interim dirty-tree validation before commit:

- `uv sync --all-extras` completed in the fresh worktree and built `robot-sf`, `pysocialforce`,
  and `pyrvo2` from this checkout.
- `git diff --check` passed after removing trailing whitespace from touched spec notes.
- Removed-root search passed: no first-level `.agent`, `contracts`, `model_ped`, `test_pygame`,
  `test_scenarios`, or `test_scenario` directory remained.
- `uv run pytest tests/contract/test_multi_extractor_summary_json.py
  tests/contract/test_multi_extractor_summary_markdown.py tests/test_training_ped_ppo.py -q`
  passed: 3 passed.
- `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame -q` passed:
  15 passed.
- `uv run pytest tests/test_osm_map_builder.py tests/test_osm_background_renderer.py
  tests/test_osm_backward_compat.py -q` passed: 31 passed.
- `uv run pytest tests/test_jsonl_recording.py tests/test_load_states_and_record_video.py
  tests/test_manual_control_pygame_runner.py -q` passed: 23 passed.
- `uv run python examples/osm_map_quickstart.py` passed and wrote ignored demo outputs under
  `output/maps/osm_demo/`.
- `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh` passed for changed
  context files.
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` passed on the dirty tree:
  4,938 passed, 10 skipped, 10 warnings. The wrapper marked the stamp as interim because the
  branch was not yet committed.

Generated outputs from the validation runs were ignored local artifacts under `output/coverage/`,
`output/maps/osm_demo/`, and `output/validation/`; they are not source evidence and should not be
committed.
