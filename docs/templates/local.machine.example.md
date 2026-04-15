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

## Preferred Commands
- test_command: <repo command>
- long_job_prefix: <e.g., tmux new -d -s run_name -->
- slurm_submit_command: <optional command>

## Do/Do Not
- do:
  - <short bullet>
- do_not:
  - <short bullet>

## Last Updated
- yyyy-mm-dd: <what changed>