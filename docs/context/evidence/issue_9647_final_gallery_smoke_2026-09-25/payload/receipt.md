# #9647 final-head replay gallery smoke

This smoke ran the documented gallery CLI at the clean code revision recorded below. It used one tracked historical #1501 `failure_0002` compatibility fixture, not a new search result or discovery.

- Code revision: `b5e8af66c4b5b59df2ad60f8a2ae515c587bc14f`.
- Source manifest SHA-256: `289bb94735ec7a1e325690e00069d761765b9f7ce0b7c06a1e6459a2e26966f9`; recorded source revision: `58e516aa4f69ff3098bf518199f483006589758c`.
- Budget: one canonical replay episode, zero search candidates, no held-out evaluation.
- Replay result: `match`; identity, categorical outcome, and objective matched. The code revision differs from the recorded source revision, so status is `outcome_reproduced_revision_changed`.
- Source-input binding: `unknown` because the historical source episode does not attest the map-registry digest.
- Source certificate: `admissible_by_source_certificate`; dynamic task feasibility remains **unknown**.
- Rendering: trajectory, still, and filmstrip figures were produced. Map overlay is `unavailable` because the existing image reader cannot decode the source SVG. Video is `unavailable` because the canonical runner emitted no video file. The track-identity discontinuity split is recorded in the case manifest.
- The replay outcome is a historical collision; the runner reports it available/native. This is plumbing and outcome-reproduction evidence only. It is not exact-source verification, search evidence, a feasibility proof, planner-performance evidence, or a real-world safety claim.

The raw source/replay streams and images remain in ignored local output at `output/adversarial-replay-gallery/issue9647_final_smoke_b5e8af66`. The tracked bundle contains only this receipt and its compact machine-readable summary; it does not mirror the output tree. Artifact digests are listed in `summary.json`.
