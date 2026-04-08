# DreamerV3 BR-08 Slurm Handoff

This handoff is for moving the DreamerV3 BR-08 SocNav+grid challenger setup to a Slurm
machine. It is intentionally a gate-first workflow: do not start with the long full profile
unless the gate run starts producing usable training/evaluation signals.

## Scope

- Branch: `578-br-08-dreamerv3-retraining-v2`
- Gate config: `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml`
- Full config: `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml`
- Evidence note: `docs/context/issue_578_608_609_dreamerv3_parity.md`
- Follow-up research issue for world-model pretraining: #782

This branch adds DreamerV3 scenario-matrix training/evaluation parity. It does not claim a
successful Dreamer checkpoint.

## Machine Setup

```bash
git fetch origin
git checkout 578-br-08-dreamerv3-retraining-v2
git pull --ff-only
uv sync --extra rllib
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore
```

Quick local sanity checks on the Slurm login node:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml \
  --dry-run

uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml \
  --dry-run
```

## Recommended Gate Run

Use the gate config first. It is CPU-only, offline W&B, and short enough to catch construction,
Ray, observation, and reward-contract failures before spending GPU time.

```bash
srun -p <partition> --cpus-per-task=8 --mem=32G --time=06:00:00 --pty bash
cd /path/to/robot_sf_ll7
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore

uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml \
  --log-level WARNING
```

Gate artifacts:

- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_gate_<timestamp>/result.jsonl`
- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_gate_<timestamp>/run_summary.json`
- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_gate_<timestamp>/checkpoints/`
- offline W&B files under `output/wandb/`

Gate interpretation:

- Continue only if the run completes without env/Ray failures and the training metrics are not
  obviously stalled or degenerate.
- The gate config has periodic evaluation configured but disabled. If you need early eval on the
  gate, temporarily enable `evaluation.enabled: true` in a throwaway local copy of the YAML rather
  than changing the committed gate profile mid-run.
- A successful gate is not promotion evidence; it only justifies trying the full profile.

## Full Slurm Run

Use the full profile only after the gate is healthy. The full profile uses online W&B and `auto`
resource resolution from Slurm/CUDA environment variables.

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH -p <partition>
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -J dreamer-br08-grid

set -euo pipefail
cd /path/to/robot_sf_ll7
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore

uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml \
  --log-level WARNING
EOF
```

Full-run artifacts:

- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_<timestamp>/result.jsonl`
- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_<timestamp>/run_summary.json`
- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_<timestamp>/evaluation/`
- `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_<timestamp>/checkpoints/`
- online W&B group: `br08-benchmark-socnav-grid-full`

## Monitoring

Scheduler/process:

```bash
squeue -u "$USER"
sstat -j <jobid>.batch --format=JobID,MaxRSS,AveCPU,Elapsed
ps -fu "$USER" | rg train_dreamerv3_rllib.py
```

Run directory:

```bash
RUN_DIR=output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_<timestamp>
tail -f "$RUN_DIR/result.jsonl"
cat "$RUN_DIR/run_summary.json"
ls -lah "$RUN_DIR/checkpoints"
ls -lah "$RUN_DIR/evaluation"
tail -n 5 "$RUN_DIR/evaluation/"*.jsonl
```

W&B:

- Gate profile: offline group `br08-benchmark-socnav-grid-gate`, sync later with
  `wandb sync output/wandb/wandb/offline-run-*`.
- Full profile: online group `br08-benchmark-socnav-grid-full`.
- Watch `reward_mean`, `timesteps_total`, `eval/success_rate`, `eval/collision_rate`, and
  `eval/timeout_rate`.

Stop conditions:

- Stop and inspect if `result.jsonl` stops updating for a prolonged period while the job remains
  allocated.
- Stop the full run if periodic evaluation remains near-zero success with high collision/timeout
  after multiple evaluation intervals.
- Do not launch repeated full runs without a new hypothesis; use #782 if the likely blocker is
  world-model representation/pretraining rather than launcher parity.

## After The Run

1. Record the W&B run URL and run directory in #578.
2. If the full run produces a plausible checkpoint, evaluate that checkpoint with the external
   policy-analysis/benchmark gate before making any paper-facing claim.
3. If the full run fails or stays poor, keep #578 as a negative result and move to #782 rather
   than scaling the same configuration again.
