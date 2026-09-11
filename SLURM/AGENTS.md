# SLURM Agent Playbook

Durable cluster safeguards for jobs submitted under `SLURM/` or in related training campaigns. Pair
this with [AGENTS.md](../AGENTS.md), [SLURM/readme.md](readme.md), and
[docs/dev/slurm_resource_audit.md](../docs/dev/slurm_resource_audit.md).

Auxme-cluster-specific details (node names, partitions, QoS profiles, per-user job limits, and
host-packing policy) live in the optional private operations overlay. See
[SLURM/Auxme/README.md](Auxme/README.md) for the public overlay contract. Never copy secrets or
host-specific policy into the repository.

## Configuration And Submission

- Use explicit, versioned configuration. Pass required environment and config inputs explicitly; do
  not rely on wrapper defaults for promotion or long-horizon submissions.
- Public wrappers validate their required environment and config inputs and fail with an actionable,
  non-secret message, including when the private operations overlay is absent.
- Prefer the repository wrappers under `scripts/dev/` over raw `sbatch`. For long jobs, prefer
  `scripts/dev/sbatch_use_max_time.sh` unless fixed wall time is required.
- Keep artifact output rooted in the configured artifact root and confirm synchronization on exit.
  Set `#SBATCH --output=output/slurm/%j-<description>.out` so job logs sort chronologically and stay
  ignored via the root `output/` rule. Never write `.out` files to the repository root.

## Custody And Identity

- Record job ID, config path, commit SHA, seeds, and artifact root for every submission.
- Preserve logs and minimal reproduction evidence for failures.

## Failure Classification

- Distinguish infrastructure or transient failures (for example an allocation handshake failure)
  from model or algorithm failures. Classify infrastructure failures as such and follow the
  campaign's documented resubmission rule; do not treat them as model failure.
- Missing artifacts in the expected output path: verify the artifact root and cleanup/sync path
  before concluding a run produced no results; check `/tmp/<user>/<jobid>/results`-style roots.

## Evidence Boundary

- Do not claim performance improvement from short gate runs alone, and do not freeze one campaign's
  horizon stages as durable repository law; stage progression and baseline anchoring are campaign
  decisions recorded with the active campaign.
- Fail closed on unsupported claims: preserve logs and block promotion claims until finite,
  deterministic evaluation is restored.
- Promotion and paper-facing claims require the campaign's minimum evidence. Fallback or degraded
  evidence is never success evidence.

## Transient Campaign Guidance

Campaign-specific environment variables, wrappers, trackers, horizons, baselines, and progression
rules belong to the active campaign configuration and context record, not this file. Keep the
current owner and review condition here:

| Transient guidance | Owner | Review condition |
| --- | --- | --- |
| issue-791 wrapper env vars, WandB policy, horizon stages, resubmission rules | [`configs/training/`](../configs/training/) plus the active campaign's [context record via the context index](../docs/context/README.md) | Re-review when the issue-791 campaign closes or its wrapper contract changes |

## Insight Capture

Persist reusable findings that change durable practice before closing the task. Examples: reliable
load-recovery signatures, best `num_envs`/CPU pairings by host, stable eval-cadence ranges, and
proven mitigations. Do not create a documentation edit when a run produced no material, reusable
finding; record the negative result instead.
