# PR #9884 corrected setup preflight pair

Diagnostic only: static setup feasibility; no planner action or outcome metric.

- PR checker head: `b626384207ca88422d3f0014d53785976d10fc0b`
- Before local source: `2c55cd31f22132e925ce798237e988dce21e6f78` (base checker with identical corrected inputs)
- After local source: `d0f514a61ff33450621d68e42686dd28e13a86cc` (PR checker with identical corrected inputs)
- Manifest SHA-256: `9b4020d6fed428c8c1045a45e20e97eef1e8f51a497f5e8323ea25590dca9516`
- Matrix SHA-256: `89e38c84af12a467599f0b76a60e5f80f7f5946b5438116378cfc4b2dee414ad`
- Seed-set SHA-256: `3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6`
- 79-file input closure SHA-256: `e8e55c35c186e0a514e8061f8f645bc8dfbc8c80806a1fd710cbdd23b3be6cd6`
- Local source bundle SHA-256: `e0a8eaeb51d504b0c961637f4e919d1185b61dff8b4d3f0ac21e19bdc1c6f858`
- Identities: 48 scenarios x 30 seeds = 1,440 paired rows.
- Before: 75 blocked; after: 0 blocked. Exactly 75 blocked-to-valid, 1,365 valid-to-valid, zero newly blocked.
- Each recovered cell retains grid failure and has a passing continuous oracle.
- Reset clearance, respawn safety, step-1 collision diagnostic, relocation, and radius fields match for every identity.
- Historical unsafe goals and 2 m doorway probes remain blocked in focused regression tests on the legacy matrix.
- Separate manifest-bound guard run: 71/71 known unsafe nominal goals and 30/30 historical 2 m doorway cells blocked (28 disconnected, two unsafe goals).
- Guard manifest SHA-256: `c031a3dc3933f6562a74a731eed4ede31ed860af861981c504483ceec3fc58ed`
- Guard matrix SHA-256: `8a8aa9adb2b098f039a3e92a4849e16eb8d3665e928211f08e0c689584c9f389`
- Guard JSON SHA-256: `da7d47a7c9f34b2ed1a48398f29193dfddd10625c46357895a77810be9464884`
- Guard Markdown SHA-256: `67edd4c454a30815d570fc778a5e201ec0b129345c1076997a2e995ba777bb30`

- Before JSON SHA-256: `19ba97f27500fd494e17d71e691891ebbeb2c0226424dae070bb7ed232a22e16`
- After JSON SHA-256: `94b884e7f7272a24fb923520f8c8bd57435be5970e3e9673c1e0994119193dd1`
- Before Markdown SHA-256: `e3ca141f5c6fba6543ce9bb6a6409fd59137e85799725ebeba2d1016104dfe1b`
- After Markdown SHA-256: `21675a62faecfe56a3f6f9c37ca90c6d49368167ed7933755db6d7338312a198`

Full before/after reports, input copies, and the pairing script are retained in the common Git-dir `codex-agent-runs/active/pr-9884-fix-20260929/` evidence folder.
The manifest is a physical diagnostic v0.1 input assembled from open PR heads. It is not the final DOI-free 0.0.8 campaign candidate or release admission.
