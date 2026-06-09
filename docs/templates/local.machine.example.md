# Local Machine Context Template

Purpose: Machine-specific execution hints for coding agents in this repository.

How to use:
- Copy this file to `local.machine.md` (or `local.machine.<name>.md`).
- Keep values local; do not commit machine-specific files.
- Never store secrets, tokens, or private keys here.

## Machine Identity
- machine_name: <short machine name>
- role: <main-laptop|gpu-worker|slurm-login|other>
- os: <macos|linux|windows>

## Resource Limits
- cpu_test_workers_max: <int>
- cpu_build_workers_max: <int>
- memory_notes: <optional notes>

## Execution Policy
- allow_long_training_local: <true|false>
- require_tmux_for_long_jobs: <true|false>
- allow_gpu_jobs: <true|false>
- gpu_description: <e.g., rtx-3070>
- allow_slurm_submission: <true|false>
- allow_carla_docker_local: <true|false>
- carla_docker_notes: <Docker daemon, NVIDIA Container Toolkit, CARLA image, ports, and client
  dependency status; say "not checked" when unknown>

## Preferred Commands
- test_command: <repo command>
- long_job_prefix: <e.g., tmux new -d -s run_name -->
- slurm_submit_command: <optional command>
- private_ops_repo: <optional absolute path to robot_sf_ll7-private-ops>
- carla_preflight_command: <e.g., scripts/dev/check_carla_runtime.sh>
- carla_smoke_command: <e.g., scripts/dev/check_carla_runtime.sh --smoke>

## Do/Do Not
- do:
  - <short bullet>
  - In linked worktrees, symlink the main checkout's `local.machine.md` before expensive local or
    CARLA commands when the machine-specific file already exists there.
  - Use `uv sync --all-extras --group carla` only for worktrees that need the host-side CARLA
    Python client; keep routine `uv sync --all-extras` installs CARLA-free.
- do_not:
  - <short bullet>
  - Treat a present CARLA Docker image as live-replay proof without a recorded smoke or replay run.

## Last Updated
- yyyy-mm-dd: <what changed>
