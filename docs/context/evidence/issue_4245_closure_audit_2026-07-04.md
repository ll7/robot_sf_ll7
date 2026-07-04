# Issue #4245 Closure Audit

Plain-language summary: merged PR #4257 delivered the standalone offline
pretraining checkpoint manifest and manifest-driven online fine-tuning lane
requested by issue #4245. This audit maps the acceptance criteria to merged
evidence and does not add a benchmark, comparison, or paper-facing claim.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4245>
- Merged implementation PR: <https://github.com/ll7/robot_sf_ll7/pull/4257>
- Merge date: 2026-07-03
- Audit date: 2026-07-04

## Acceptance Mapping

| Acceptance criterion from #4245 | Delivered evidence | Status |
| --- | --- | --- |
| `pretrain_offline_policy.py` exists and emits a checkpoint, normalizer sidecar, and `offline-policy-checkpoint-manifest.v1`. | PR #4257 added `scripts/training/pretrain_offline_policy.py` and `robot_sf/training/offline_pretraining_manifest.py`. The manifest builder records checkpoint and normalizer paths and SHA-256 values. | Met by PR #4257. |
| `offline_to_online_finetune.py` loads the offline manifest, verifies hashes, fine-tunes online, and emits a chained manifest. | PR #4257 added `scripts/training/offline_to_online_finetune.py` and `offline-to-online-finetune-manifest.v1` support. The fine-tune manifest records the parent manifest hash, loaded checkpoint hash, output checkpoint hash, inherited dataset identity, and online timesteps. | Met by PR #4257. |
| CPU smoke for both stages passes tiny budgets. | PR #4257 committed smoke configs `configs/training/offline_online_rl/issue_4245_pretrain_smoke.yaml` and `configs/training/offline_online_rl/issue_4245_finetune_smoke.yaml`, plus `docs/context/evidence/issue_4245_offline_pretrain_finetune_smoke/smoke_report.md` reporting both smoke stages passed. | Met by PR #4257; diagnostic provenance smoke only. |
| Compact evidence under `docs/context/evidence/` records the provenance chain without checkpoint bytes. | PR #4257 added `docs/context/evidence/issue_4245_offline_pretrain_finetune_smoke/README.md`, `pretrain_manifest_summary.json`, `finetune_manifest_summary.json`, `provenance_chain.json`, and `smoke_report.md`. The packet states checkpoint, normalizer, dataset, and full generated manifest bytes remain ignored under `output/`. | Met by PR #4257. |
| Tests cover manifest completeness, hash mismatch, environment-space mismatch, command-line-interface smoke, and evidence artifact safety. | PR #4257 added `tests/training/test_offline_pretraining_manifest.py`, `tests/training/test_pretrain_offline_policy.py`, and `tests/training/test_offline_to_online_finetune.py`. The tests exercise missing path and checksum rejection, environment fingerprint mismatch, offline pretrain CLI output, online fine-tune manifest chaining, and compact evidence boundaries. | Met by PR #4257. |
| No benchmark or performance claim, scratch comparison, paper/dissertation claim, or #4244 matrix inclusion is made inside #4245. | PR #4257 body and the committed smoke evidence explicitly mark the lane as diagnostic provenance integrity only and `eligible_for_claim=false`. This audit adds only a closure mapping and does not edit benchmark, paper, dissertation, or #4244 comparison surfaces. | Met by PR #4257 and preserved by this audit. |

## Local Verification

Audit-time validation was limited to the closure layer and focused checks:

```bash
uv run --extra training pytest \
  tests/training/test_offline_pretraining_manifest.py \
  tests/training/test_pretrain_offline_policy.py \
  tests/training/test_offline_to_online_finetune.py -q

python3 -m json.tool docs/context/evidence/issue_4245_offline_pretrain_finetune_smoke/pretrain_manifest_summary.json
python3 -m json.tool docs/context/evidence/issue_4245_offline_pretrain_finetune_smoke/finetune_manifest_summary.json
python3 -m json.tool docs/context/evidence/issue_4245_offline_pretrain_finetune_smoke/provenance_chain.json
```

The existing PR #4257 evidence packet records the smoke command results. This
audit did not rerun a full benchmark campaign, submit Slurm or GPU work, edit
paper or dissertation claims, release, merge, delete, or close the issue.

## Closure Boundary

Issue #4245 is criteria-complete as a diagnostic provenance lane. Downstream
comparison-matrix consumption remains under issue #4244 and is not a missing
Issue #4245 acceptance criterion.
