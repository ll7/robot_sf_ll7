# 0.0.8 decisions

This file adds D-049 while the existing decision ledger in PR #10032 is pending.
Retain that ledger when integrating the two PRs.

## D-049 — Fresh sealed evaluation seeds

Author approved on 2026-09-30. The history check found that social force,
socnav_sampling and hybrid v4 were corrected, and their fixes checked, on
outcomes from the 0.0.7 evaluation seeds 111..140. Consequently 0.0.8 uses a
fresh sealed set. The retired band stays held out; 0.0.7 artifacts stay unchanged.

Derivation label: `robot_sf_ll7 release 0.0.8 evaluation seeds v1 (sealed 2026-09-30)`.
SHA-256: `166597da1e0e813d8a9cdc810f4b85db1407e9286c3113821423e17c50908dc0`.
Initialize `random.Random(int(hash, 16))`, sample 30 from `range(50000, 60000)`,
then sort. The explicit tuple is `EVAL_SEEDS_0_0_8` in
`robot_sf/benchmark/seed_bands.py`; its YAML transport is
`configs/benchmarks/seed_sets_0_0_8.yaml`. Development seeds remain 1001..1030.

Enforcement: `tests/benchmark/test_newseeds.py` recomputes the derivation,
resolves the real release grid (48 x 30 identities per arm), rejects the retired
candidate band, and checks fresh-seed literal-range refusals. The holdout diff
checker, #9748 validator and tuning runner protect both bands. The private mint
pins the fresh list by value and independently cross-checks the derivation hash.
No planner or environment step on either band is permitted in development,
tests, diagnostics, reviews, rehearsals or admission gates. The ONLY exception
is the sealed 0.0.8 evaluation campaign itself, which steps `EVAL_SEEDS_0_0_8`
at the freeze commit. The retired 111..140 band has no exception. The full-release
stress gate is a mechanical check before the run and uses dev seed 1001.
All proofs in this change are static.
SEEDGUARD (#10010) must import `HELD_OUT_SEEDS` from this module.

Reopen only with new material evidence or an explicit author decision; never
reuse an observed evaluation band for a fresh release claim.
