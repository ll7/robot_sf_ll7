# 0.0.8 decisions

## D-049 — Fresh sealed evaluation seeds

- **Date:** 2026-09-30
- **Question:** Which seeds can support a fresh 0.0.8 evaluation after fixes were
  developed and checked on outcomes from the 0.0.7 evaluation band?
- **Choice:** Use the fresh sealed `EVAL_SEEDS_0_0_8` tuple in
  `robot_sf/benchmark/seed_bands.py`, transported by
  `configs/benchmarks/seed_sets_0_0_8.yaml`. Development seeds remain 1001..1030.
  No planner or environment step on either held-out band is permitted **anywhere,
  except the sealed campaign (including its own spawn preflight) at the freeze commit**.
  The sealed campaign includes the main 0.0.8 evaluation and the three-width
  doorway slice (`three_width_doorway_release_0_0_8_v1.yaml`), minted together
  and bound to the same freeze commit. Both require `source_sha` equal to HEAD
  at execution. Retired seeds 111..140 have no execution exception; historical
  pins permit STATIC validation of archived artifacts only. A failed sealed
  spawn preflight requires an author decision before fixing and rerunning it.
  The full-release stress pre-run gate requires dev seed 1001.
- **Reason:** Social force, socnav_sampling and hybrid v4 were corrected and
  checked using outcomes from the retired evaluation band. Reusing that band
  would not support a fresh release claim. Preserve 0.0.7 artifacts unchanged.
- **Decided by:** The author approved the fresh seed decision. The orchestrator
  issued the delegated scope, stress-seed, freeze-bound doorway slice, protocol
  and diff-checker rulings; those are separate from the author's seed decision.
- **Rejected:** Reusing 111..140; historical execution exemptions; sealed seeds
  in unnamed releases, development, or unfrozen slices.
- **Implemented in:** `robot_sf/benchmark/release_protocol.py` checks resolved
  seeds in every policy mode, refuses retired seeds in non-historical releases,
  and applies the reverse sealed-seed identity/source check. The shared guard in
  `robot_sf/benchmark/spawn_preflight.py` refuses retired execution even with
  historical pins and protects the release runner and standalone preflight.
  `scripts/validation/check_seed_holdout_diff.py` treats every value in
  `seed_set*.yaml` and `seed_list*.yaml` as a seed and requires the same explicit
  allowlist for held-out names as for sealed literals. The #9748 validator and
  tuning runner protect both bands. SEEDGUARD (#10010) must import
  `HELD_OUT_SEEDS` from `robot_sf/benchmark/seed_bands.py`.
- **Enforcing test:** `tests/benchmark/test_newseeds.py`,
  `tests/benchmark/test_sealed_execution_policy.py`, and
  `tests/validation/test_check_seed_holdout_diff.py`; all holdout witnesses are
  static or use recording stubs, with environment creation forbidden.

Derivation label: `robot_sf_ll7 release 0.0.8 evaluation seeds v1 (sealed 2026-09-30)`.
SHA-256: `166597da1e0e813d8a9cdc810f4b85db1407e9286c3113821423e17c50908dc0`.
Initialize `random.Random(int(hash, 16))`, sample 30 from `range(50000, 60000)`,
then sort. The private mint independently cross-checks the derivation and list.

**Supersession:** D-049 supersedes D-008's reservation of 111..140 for new
release evaluation and D-048's choice of those seeds for the 0.0.8 campaign.
Those entries originate in PR #10032, merge train 1. This supersession applies
when both changes land; retain their historical records and annotate them with
D-049. Their historical pins are for static archived-artifact validation only.

**Reopen:** Only new material evidence or an explicit author decision reopens
this ruling; never reuse an observed evaluation band for a fresh release claim.
