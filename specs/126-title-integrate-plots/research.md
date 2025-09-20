# Phase 0 Research: Plots & Videos Integration

## Decisions

### D1: Invoke visuals only after adaptive loop finalization
Rationale: Avoid redundant regeneration each iteration; reduces overhead and preserves deterministic artifact set.
Alternatives: Per-iteration refresh (higher cost, limited added value). Rejected.

### D2: Deterministic episode selection for videos
Rationale: Reproducibility (first N in chronological execution order). Matches FR-007.
Alternatives: Random sample each run (breaks reproducibility). Rejected.

### D3: Graceful degradation on missing deps
Rationale: Constitution Principle I (reproducibility) & VI (transparency). Artifacts must show reason for absence.
Alternatives: Hard fail if deps missing (blocks CI). Rejected.

### D4: No new config flags
Rationale: Reuse existing `disable_videos`, `max_videos`, `smoke` to avoid contract expansion (Principle VII - backward compatibility).
Alternatives: Add separate `generate_plots` flag (unnecessary complexity). Rejected.

### D5: Manifest JSON shape
Rationale: Keep simple arrays of objects with stable keys: `kind/path/status/note` for plots; `artifact_id/episode_id/scenario_id/path_mp4/status/note` for videos.
Alternatives: Embed inside existing summary.json (mixes concerns). Rejected.

### D6: Performance envelope
Target overhead: plots <2s, video generation default (1 video) <5s on reference CI. Aligns with FR-014 guidance.

## Resolved Unknowns
- Performance thresholds: Adopt target (plots <2s, videos <5s) as soft gates — not failing runs if exceeded but document in manifest if > threshold.

## Remaining Clarifications
- None (FR-014 now concretized; remove NEEDS CLARIFICATION marker during design update).

## Risk Mitigations
| Risk | Mitigation |
|------|------------|
| Missing ffmpeg / moviepy | Status=skipped with note; do not raise. |
| Large video max | Enforce cap via existing `max_videos`; advise doc update for higher counts. |
| Slow plotting due to matplotlib config | Use minimal placeholder plotting functions already optimized for low overhead. |

## Dependencies
- Optional: matplotlib, moviepy
- Core: existing orchestrator, plot & video modules

## Alignment with Constitution
- Reproducibility: deterministic selection & stable filenames
- Transparency: manifest JSON + skip notes
- Backward compatibility: no schema breaks, no new config fields

