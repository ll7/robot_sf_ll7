# Helpful Slurm Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `squeue -u $USER` | Show all your jobs (running, pending, etc.) | `squeue -u $USER` |
| `squeue` | Show all jobs on the cluster | `squeue` |
| `squeue -j <jobid>` | Show info about a specific job | `squeue -j 12345` |
| `sacct -u $USER` | Show accounting info (finished + running jobs) | `sacct -u $USER` |
| `scontrol show job <jobid>` | Detailed info about a job | `scontrol show job 12345` |
| `scancel <jobid>` | Cancel a job | `scancel 12345` |
| `scancel -u $USER` | Cancel all your jobs | `scancel -u $USER` |
| `sbatch <script.sh>` | Submit a batch job | `sbatch myjob.sh` |
| `srun <command>` | Run a command interactively under Slurm | `srun --pty bash` |
| `sinfo` | Show partition/node status | `sinfo` |
| `squeue -t R` | Show only running jobs | `squeue -t R -u $USER` |
| `squeue -t PD` | Show only pending jobs | `squeue -t PD -u $USER` |
| `watch -n 5 squeue -u $USER` | Live monitor of your jobs every 5s | `watch -n 5 squeue -u $USER` |

| Command | Purpose | Notes |
|---------|---------|-------|
| scontrol show config | Show full Slurm configuration (parameters, limits, partitions). | |
| sinfo -l | Detailed partition and node states (CPU, memory, availability). | |
| scontrol show partition | List all partitions with settings and limits. | |
| scontrol show node <nodename> | Show CPU, memory, features, gres, state for a node. | |
| sinfo -N -l | Detailed node list with states. | |
| scontrol show job <jobid> | Full job details (resources requested, allocated, env vars). | |
| sstat -j <jobid> | Live statistics (CPU, memory, I/O) for running jobs. | |
| sacct -j <jobid> --format=All | Complete accounting info for jobs (historical and current). | |
| env \| grep SLURM_ | Show all Slurm-provided environment variables in your session. | |
| printenv SLURM_JOB_ID | Inspect individual Slurm environment variables. | |