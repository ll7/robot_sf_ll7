# PR #9884 corrected setup preflight pair

Diagnostic only: static setup feasibility; no planner action or outcome metric.

- Before checker: `4665cd13fc205761a4edb256a04d23d37ffbc235` (PR ancestor).
- After checker: `b626384207ca88422d3f0014d53785976d10fc0b` (PR fix commit).
- Reviewed head: `67b5a8a336c4bfc29c957c552e27d84675e72767`.
- Reproduction source commit: `03957faa332577008169d957b4fbe83ce66de0a9`.
- Committed goal-clearance diagnostic overlay SHA-256: `b578efe33e5f30f56241362afc5bb45b0aed05c398af83421ca51fbedffb04cc`.
- Manifest SHA-256: `9b4020d6fed428c8c1045a45e20e97eef1e8f51a497f5e8323ea25590dca9516`.
- Matrix SHA-256: `89e38c84af12a467599f0b76a60e5f80f7f5946b5438116378cfc4b2dee414ad`.
- Seed-set SHA-256: `3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6`.
- 79-file input closure SHA-256: `e8e55c35c186e0a514e8061f8f645bc8dfbc8c80806a1fd710cbdd23b3be6cd6`.
- Identities: 48 scenarios × 30 seeds = 1,440 paired rows.
- Before: 75 blocked; after: 0 blocked. Transitions: 75 blocked→valid, 1,365 valid→valid, zero newly blocked.
- Each recovered cell retains grid failure and has a passing continuous oracle.
- Reset clearance, respawn safety, step-1 collision diagnostic, relocation, and radius fields match for every identity.
- Separate guard run: 71/71 unsafe nominal goals and 30/30 doorway cells blocked (28 disconnected, two unsafe goals).
- Guard manifest SHA-256: `c031a3dc3933f6562a74a731eed4ede31ed860af861981c504483ceec3fc58ed`.
- Guard matrix SHA-256: `8a8aa9adb2b098f039a3e92a4849e16eb8d3665e928211f08e0c689584c9f389`.

| Report | Raw JSON SHA-256 | Timing-independent SHA-256 |
| --- | --- | --- |
| Before | `19ba97f27500fd494e17d71e691891ebbeb2c0226424dae070bb7ed232a22e16` | `cacb5e552020e96a8e4624cebc796e563a7dec1749ba89c8f48495b5cabc734f` |
| After | `94b884e7f7272a24fb923520f8c8bd57435be5970e3e9673c1e0994119193dd1` | `7bc8c72a4aec5804fd9443da466a3bbc19178ce5f703d5dcd04d535343494ac7` |
| Guard | `da7d47a7c9f34b2ed1a48398f29193dfddd10625c46357895a77810be9464884` | `9cd32a2d5f306262a4325c5a2d52c6395edac410294539e0d7a5afc51ae051df` |

Full reports, inputs, and the exact reproduction command are tracked here and in `README.md`.
This physical diagnostic input is not the final DOI-free 0.0.8 campaign candidate or release admission.
