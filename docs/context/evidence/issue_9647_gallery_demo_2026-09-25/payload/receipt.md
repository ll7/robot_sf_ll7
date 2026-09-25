<!-- AI-GENERATED (robot_sf#9647, 2026-09-25) - NEEDS-REVIEW -->
# #9647 current-head replay gallery demo

This is a one-episode rendering smoke from the refreshed gallery code at
`7844d08e509e0362bddfb791a71d20672278b4fe`. It uses the tracked #1501 `failure_0002`
compatibility fixture, not a new #9645 discovery or a persisted search-run manifest.

## Reproduction

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/materialize_adversarial_replay_gallery.py tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json --out output/adversarial-replay-gallery/issue9647_current_smoke_20260925 --top-k 1
```

The run selected one case, used zero search candidates, replayed one episode, and held out zero
episodes. The source manifest SHA-256 is
`289bb94735ec7a1e325690e00069d761765b9f7ce0b7c06a1e6459a2e26966f9`; the source episode SHA-256
is `67bbf65dddbe440bde284564356acb4ab18e95a3869db42b5e367c151768c469`.

## Result and interpretation

The `goal` planner replay produced a collision. Source/replay identity, collision event, and the
`constraints_first_lexicographic_v1` objective matched; the objective value was `4.833333333333333`.
The current revision differs from the historical source revision `58e516aa4f69ff3098bf518199f483006589758c`,
so the classification is `outcome_reproduced_revision_changed`, not exact-source replay verification.
The source episode does not attest its map-registry digest, leaving source-input binding **unknown**.

The source's static route certificate passed as `hard_but_solvable`; dynamic task feasibility remains
**unknown**. The recorded minimum surface clearance is `-0.015686402836192603 m` at critical frame 9.
The short trace contains 10 steps, and one discontinuous actor track was split for rendering.

## Figures

The small reviewable figures are included in this bundle. The raw episode stream, case manifests,
copied inputs, and logs remain in ignored `output/`; their digests are listed in `summary.json`.

![Trajectory view](figures/trajectory.png)

![Critical frame](figures/still_9.png)

![Replay filmstrip](figures/filmstrip.png)

The map overlay is unavailable because the existing image reader cannot decode the SVG map. The
canonical runner emitted no video artifact. These are historical outcome-reproduction visuals,
not evidence that the source case is dynamically feasible, that the planner is safe, or that a new
falsification round discovered this case.
<!-- AI-GENERATED (robot_sf#9647, 2026-09-25) - NEEDS-REVIEW -->
