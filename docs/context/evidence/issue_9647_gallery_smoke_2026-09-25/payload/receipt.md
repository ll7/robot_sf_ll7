# Issue #9647 historical gallery smoke receipt

The exact-head command exercised the gallery selector, canonical replay runner, outcome comparator, and existing renderer using the tracked #1501 `failure_0002` compatibility fixture. It selected and replayed one case with a matching outcome projection. This is a historical collision replay, not a new discovery or a falsification search.

- Code checkout: `1357ebd4084a5da077e10a5ab9ed74e5a4d9a9e4` (clean at execution).
- Source case: #1501 `failure_0002`; source episode revision `58e516aa4f69ff3098bf518199f483006589758c`.
- Replay revision: `1357ebd4084a5da077e10a5ab9ed74e5a4d9a9e4`; verification is `outcome_reproduced_revision_changed` because the replay revision differs from the source revision. `replay_match: match` means the observed identity/outcome/objective comparison matched; it does not mean exact-source replay verification.
- The gallery accepted the source static certificate as `admissible_by_source_certificate`. Dynamic task feasibility remains **unknown**.
- Renderer status: `rendered`; map context is `unavailable`; video is `unavailable` because `canonical_runner_did_not_emit_video_artifact`.
- This archived run predates map-registry input binding. Its generated case manifest says `source_input_binding: bound`, but the selected `map_id` registry bytes were not captured, so this receipt reclassifies source-input binding as **unknown**. The revision-changed replay is retained as plumbing evidence only.
- The archived `trajectory.png` joins simulator slot `simulator-slot-0` across a discontinuous occupant reuse (roughly `(36, 36)` to `(4, 3)`). Do not interpret that line as one pedestrian's path. The repaired renderer preserves actor IDs and splits discontinuous tracks; the historical figures were not regenerated. SVG map overlay was unavailable because the existing renderer image reader cannot decode the SVG.
- Rendered figures are local ignored output at `output/adversarial-replay-gallery/issue9647_historical_compat_smoke_1357ebd/cases/case_0000_7e144c8113a9/figures/`. Their sizes and SHA-256 hashes, along with other smoke output paths, are in `summary.json`. Raw logs and images are not copied into this evidence bundle.

Earlier attempts are preserved in the attempt ledger with their output-manifest hashes and dispositions, including candidate rejections, one unavailable replay, and prior revision matches. The first rejected fixture candidate carried `candidate_index` and `spawn_time_s_note` as unsupported candidate fields; neither reached replay. Only the final run at the recorded exact code head is counted as the CLI/rendering smoke result. No benchmark or real-world safety claim is supported.
