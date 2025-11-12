# Archived Examples

Purpose: document deprecated or manual examples that remain in the repository for historical reference while keeping the active catalog reproducible.

## Archival Criteria

- Requires manual interaction (e.g., pygame UI) that cannot run in CI.
- Superseded by a newer curated example covering the same workflow.
- Depends on heavyweight or external assets that are no longer maintained.

Each archived script retains its history but is excluded from automated smoke tests (`ci_enabled: false` in `examples/examples_manifest.yaml`). When functionality is needed again, prefer the replacement example or migrate the script back into an active tier with an updated docstring and manifest entry.

## Replacement Map

| Archived Path | Replacement | Reason |
| --- | --- | --- |
| `examples/_archived/classic_interactions_pygame.py` | `examples/benchmarks/demo_full_classic_benchmark.py` | Interactive pygame workflow kept only for manual debugging. |
| `examples/_archived/demo_pedestrian.py` | `examples/advanced/06_pedestrian_env_factory.py` | Legacy pedestrian env API replaced by factory-based demo. |
| `examples/_archived/interactive_playback_demo.py` | `examples/advanced/15_view_recording.py` | Manual playback UI superseded by view-recording utility. |

## Maintainer Notes

- Update this README whenever additional scripts move into `_archived/`.
- Provide a `ci_reason` in the manifest so automation reports the skip rationale.
- If no suitable replacement exists, set the replacement column to ``None`` and document the rationale in the `Reason` field.
