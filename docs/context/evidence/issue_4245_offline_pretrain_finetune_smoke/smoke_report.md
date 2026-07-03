# Issue #4245 Smoke Report

Result: passed.

- Offline pretraining wrote `offline-policy-checkpoint-manifest.v1`.
- Fine-tuning loaded the offline checkpoint manifest, verified hashes, matched environment
  fingerprints, and wrote `offline-to-online-finetune-manifest.v1`.
- Both manifests set `eligible_for_claim=false`.
- No scratch SAC comparison, benchmark campaign, Slurm/GPU submission, or paper/dissertation claim
  edit was run.
- Checkpoint, normalizer, smoke dataset, and full manifest bytes remain ignored under `output/`.
