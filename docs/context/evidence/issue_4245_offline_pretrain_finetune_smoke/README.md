# Issue #4245 Offline Pretrain to Online Fine-Tune Smoke Evidence

This packet records a CPU smoke of the standalone offline SAC pretraining checkpoint manifest
lane and manifest-driven online fine-tune lane.

Claim boundary: diagnostic provenance integrity only. This is not benchmark evidence, not a
scratch-SAC comparison, and not a paper or dissertation claim.

Committed files contain compact manifest summaries and SHA-256 values only. Model checkpoints,
normalizer sidecars, smoke datasets, and full generated manifests remain under ignored `output/`.

Commands:

```bash
uv run --extra training python scripts/training/pretrain_offline_policy.py \
  --config configs/training/offline_online_rl/issue_4245_pretrain_smoke.yaml \
  --output-dir output/issue_4245_offline_pretrain_smoke \
  --manifest-out output/issue_4245_offline_pretrain_smoke/offline_checkpoint_manifest.json

uv run --extra training python scripts/training/offline_to_online_finetune.py \
  --config configs/training/offline_online_rl/issue_4245_finetune_smoke.yaml \
  --pretrained-manifest output/issue_4245_offline_pretrain_smoke/offline_checkpoint_manifest.json \
  --output-dir output/issue_4245_offline_to_online_finetune_smoke \
  --manifest-out output/issue_4245_offline_to_online_finetune_smoke/finetune_manifest.json
```
