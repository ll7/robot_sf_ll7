# SLURM Resource Audit

[← Back to Documentation Index](../README.md)

Use this workflow to verify whether a training job is actually using the CPUs,
memory, and GPUs it requested from Slurm. The goal is to catch misconfigured
jobs early and tighten resource requests based on evidence instead of guesswork.

## What to inspect

Audit each run from three angles:

1. Slurm allocation: what the scheduler granted.
2. W&B system metrics: what the process and GPU actually used over time.
3. Training topology: whether the job is vectorized and how many worker
   processes it launches.

For the current PPO training flow, the most relevant files are:

- `scripts/training/train_ppo.py`
- `SLURM/Auxme/auxme_gpu.sl`
- the active config under `configs/training/ppo/`

## 1. Confirm the Slurm allocation

Inspect the live job allocation:

```bash
scontrol show job <job_id> | tr ' ' '\n' | grep -E '^(JobId=|JobName=|NodeList=|NumCPUs=|CPUs/Task=|MinMemoryNode=|TresPerNode=)'
```

Inspect the node topology and current node-wide load:

```bash
scontrol show node <node_name> | tr ' ' '\n' | grep -E '^(CPUAlloc=|CPUTot=|ThreadsPerCore=|CoresPerSocket=|Sockets=|Gres=|AllocTRES=|CfgTRES=)'
```

Interpretation:

- `NumCPUs` and `CPUs/Task` tell you the scheduler reservation for your job.
- `MinMemoryNode` is the memory request actually enforced for the job.
- `AllocTRES` on the node shows how crowded the node is overall.

## 2. Query W&B system metrics correctly

The W&B API stores system metrics in the `system` stream, not in the default
history stream. Use `history(stream="system", ...)`.

Example:

```bash
/home/luttkule/git/robot_sf_ll7/.venv/bin/python - <<'PY'
import json
import math
import statistics
import wandb

run = wandb.Api().run('ll7/robot_sf/<run_id>')
keys = [
    'system.cpu',
    'system.memory_percent',
    'system.proc.memory.availableMB',
    'system.proc.memory.percent',
    'system.proc.memory.rssMB',
    'system.proc.cpu.threads',
    'system.gpu.0.gpu',
    'system.gpu.0.memory',
    'system.gpu.0.memoryAllocated',
    'system.gpu.0.powerWatts',
    'system.gpu.0.powerPercent',
    'system.gpu.0.smClock',
    'system.gpu.0.memoryClock',
    'system.gpu.0.temp',
]

rows = run.history(stream='system', pandas=False, samples=5000)
summary = {}
for key in keys:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    if values:
        summary[key] = {
            'count': len(values),
            'min': min(values),
            'mean': statistics.fmean(values),
            'max': max(values),
            'last': values[-1],
        }

print(json.dumps(summary, indent=2))
PY
```

Reliable fields for sizing requests:

- `system.proc.memory.rssMB`: process resident set size. Use peak as the main
  memory signal.
- `system.gpu.0.memoryAllocated`: allocated GPU memory in GiB. Use peak/current
  to size GPU memory needs.
- `system.gpu.0.gpu`: GPU utilization percent. Useful for identifying whether
  the learner is GPU-bound.
- `system.gpu.0.powerWatts` and `system.gpu.0.powerPercent`: good secondary
  signals for low/high GPU activity.

More ambiguous fields:

- `system.cpu`: useful for trend direction, but do not treat it as a precise
  "percent of reserved cores used" measure.
- `system.proc.cpu.threads`: thread count, not CPU saturation.

## 3. Pull the W&B metadata file

The metadata file is useful for verifying what the process saw at startup,
including Slurm environment variables.

```bash
/home/luttkule/git/robot_sf_ll7/.venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
import wandb

run = wandb.Api().run('ll7/robot_sf/<run_id>')
with tempfile.TemporaryDirectory() as tmp:
    path = Path(run.file('wandb-metadata.json').download(root=tmp, replace=True).name)
    data = json.loads(path.read_text())

payload = {
    'host': data.get('host'),
    'cpu_count': data.get('cpu_count'),
    'cpu_count_logical': data.get('cpu_count_logical'),
    'gpu': data.get('gpu'),
    'gpu_count': data.get('gpu_count'),
    'memory_total': data.get('memory', {}).get('total'),
    'slurm': data.get('slurm', {}),
}
print(json.dumps(payload, indent=2))
PY
```

This is especially helpful when checking whether a run really picked up the
intended `cpus_per_task`, `job_gpus`, `mem_per_node`, partition, QoS, and node.

## 4. Confirm the training topology

For PPO runs, inspect both the config and the trainer:

- `num_envs`
- `worker_mode`
- whether `SubprocVecEnv` or `DummyVecEnv` is constructed

In `scripts/training/train_ppo.py`:

- `worker_mode == "subproc"` creates `SubprocVecEnv(env_fns)`.
- `worker_mode == "dummy"` creates `DummyVecEnv(env_fns)`.

This matters because resource use is driven by the actual process topology, not
only by the Slurm request.

## How to interpret the PPO runs we inspected

For the recent BR-06 PPO jobs, the pattern was consistent:

- process RSS peaked around 12 GiB,
- GPU memory allocation sat around 11.5 GiB,
- average GPU utilization stayed low (roughly single-digit percent),
- GPU power stayed far below the A30 ceiling most of the time.

That implies:

- the 120 GiB memory request is conservative,
- the job is not strongly GPU-bound,
- the job benefits from multiple CPU processes, but not necessarily from 24
  fully reserved CPUs.

## Why not set `num_envs` close to 12 or 24?

Do not equate `num_envs` with `cpus-per-task` automatically.

Reasons:

- Each `SubprocVecEnv` worker is a separate Python process with IPC overhead.
- The main PPO learner process also needs CPU time for rollout aggregation,
  batching, policy updates, logging, checkpointing, and W&B sync.
- One environment process can use more than one OS thread indirectly through
  NumPy, Torch, TensorFlow-linked libraries, BLAS, or simulator internals.
- More envs increase rollout batch size per iteration. With SB3 PPO that changes
  training dynamics, not just throughput. More envs can make updates less
  frequent and increase policy lag.
- More envs increase memory pressure and process-management overhead.

For the inspected BR-06 v8 PPO config, the job used:

- `num_envs: 8`
- `worker_mode: subproc`
- `CPUs/Task: 24`

That means the job is definitely multi-core, but the natural parallelism is
closer to "8 env workers plus learner and overhead" than to "24 cores saturated
all the time".

Practical rule:

- Start from the number of truly independent env workers you want.
- Leave headroom for the learner and library threads.
- Benchmark wall-clock FPS and W&B system metrics when moving from 8 to 12 to
  16 envs.
- Do not assume that 24 envs is better just because 24 CPUs were reserved.

## A quick audit checklist

Use this before deciding to change Slurm requests:

1. Is the run actually vectorized (`SubprocVecEnv`) or single-process?
2. Does W&B show low GPU util and low GPU power most of the time?
3. Is peak RSS far below the Slurm memory request?
4. Is node-level `AllocTRES` showing that the CPU-heavy request blocks extra
   GPU jobs on 4-GPU nodes?
5. If you lower CPUs or memory, does `time/fps` stay roughly stable?

If the answers are yes, yes, yes, yes, and yes, the job is likely over-requesting
resources.