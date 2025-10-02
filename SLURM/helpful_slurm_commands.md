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