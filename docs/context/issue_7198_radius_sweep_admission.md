# Issue #7198 — Current-source Gate 2 radius-sweep admission

## Plain-language summary

This note documents the reproducible preparation path for issue #7198, the Gate 2
admission child of [#6642](https://github.com/ll7/robot_sf_ll7/issues/6642) and
[#6600](https://github.com/ll7/robot_sf_ll7/issues/6600). The packet checks that
the closed Gate 1 radius-binding receipt still matches the current source,
freezes one candidate commit and all declared inputs, runs zero-episode
preflights for the 0.5 m, 0.8 m, and 1.0 m arms, and records private queue and
route evidence.

The packet is an admission aid, not a benchmark result. It never submits
production Slurm work and cannot authorize submission by itself. A maintainer
must review the exact packet and explicitly authorize one bounded campaign.

## Contract

The tracked contract is
configs/benchmarks/issue_7198_radius_sweep_admission_v1.yaml. It freezes:

- issue identity #7198 -> campaign #6642 -> parent #6600;
- Gate 1 schema, issue/parent identity, francis2023_narrow_doorway, radii
  0.5/0.8/1.0 m, and five required binding surfaces per radius;
- the 14-planner roster, 48 scenario cells, paper_eval_s30 seeds 111-140,
  horizon 600, dt=0.1, and differential-drive kinematics;
- 20,160 expected rows per arm and 60,480 total row identities with dimensions
  (radius_arm, planner_key, scenario_name, seed);
- the complete-row or explicit fail-closed missingness ledger contract;
- the public preflight command and the required enforced_staged checkpoint mode
  for submit-safe remote model staging;
- the private-ops queue, routing, preflight, and submission entry points;
- the requested 40 CPU, 0 GPU, 155 GB resource envelope and durable output
  retention boundary.

The packet generator records the exact three-arm config SHA-256 values, the
resolved Gate 1 report SHA-256, input checksums, command lines, preflight logs,
and private-ops read-only captures under output/. Those generated files are
worktree-local and should be promoted to the durable campaign results location
only after an authorized run; raw episodes, videos, and checkpoints do not
belong in git.

## Reproduction

From a clean linked worktree at the candidate commit:

    TF_CPP_MIN_LOG_LEVEL=3 uv run python \
      scripts/benchmark/prepare_radius_sweep_admission_issue_7198.py \
      --out output/issue_7198_radius_sweep_admission

The command exits 0 only for ready_for_authorized_submission. It exits 2 for
blocked or any malformed required input. Inspect
output/issue_7198_radius_sweep_admission/packet.json; the first_blocker field is
the smallest next repair target and blockers preserves all reasons.

The generator also runs the existing manifest checker and current Gate 1
canary. It invokes the public camera-ready runner only with
--mode preflight, --skip-publication-bundle, and
--checkpoint-preflight-mode enforced_staged; this resolves and checksum-verifies
the five declared checkpoint dependencies per arm before a packet can become
submit-safe. It verifies that no episode JSONL or Parquet files were emitted.

Private-ops inspection is read-only. The generator captures the queue summary
and route evaluation scoped to public issue #6642, requires exactly one
submit-eligible queue row and no active ledger job in that scope, checks that the
private checkout is clean, and records whether the durable results URI is
configured. It never calls submit_and_record.sh.

## Verdict boundary

ready_for_authorized_submission means only that the packet predicates passed.
It is not permission to submit, not evidence that the radius treatment changes
planner behavior, and not a paper-facing claim. Any stale Gate 1 identity,
changed non-radius factor, missing input, incomplete row accounting, unsafe
checkpoint state, absent queue admission, unverified capacity, dirty private
checkout, or unavailable durable artifact route yields blocked.

The Gate 3 rank-stability analysis in issue #6643 remains unavailable until an
authorized Gate 2 campaign produces complete row identities or a fail-closed
missingness ledger.
