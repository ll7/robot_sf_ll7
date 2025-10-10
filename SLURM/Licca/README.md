# Robot SF on LiCCA

This guide condenses the LiCCA cluster documentation for Robot SF users. It focuses on
module management, filesystem layout, conda-based environments (the cluster does not
support `uv`), and Slurm job patterns for CPU and GPU workloads.

## Cluster quick facts
- Login node `licca-li-01` (aka `licca001`) and AMD EPYC compute nodes with 128 cores.
- Partitions: `epyc` (general CPU), `epyc-mem` (4 TiB RAM), `epyc-gpu` (3× A100 80GB),
  `epyc-gpu-sxm` (4× A100-SXM 80GB), `xeon-gpu` (H100-NVL), and `test` for short runs.
- Slurm defaults to one task and one CPU; always request explicit resources.

## Module hygiene
- Use `module purge` inside every job script to avoid leaking interactive modules.
- Discover software with `module avail`, `module spider <name>`, and load versions with
  `module load cuda/12.1` or similar.
- Documented best practice is to purge then load only what you need in each sbatch file.

## Python without `uv`
1. Load a conda-compatible manager: `module load miniforge` (or `micromamba`).
2. Create an environment in scratch (default paths are `/hpc/gpfs2/scratch/u/$USER/.conda`
   or `/hpc/gpfs2/scratch/u/$USER/micromamba`):
   ```bash
   conda create -n robot-sf python=3.11 pip
   # micromamba alternative: mm create -n robot-sf -c conda-forge python=3.11 pip
   ```
3. Activate the env (`conda activate robot-sf` / `mm activate robot-sf`).
4. Install Robot SF:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]
   ```
5. If you need CUDA wheels on the login node, set
   `CONDA_OVERRIDE_CUDA=12.8` before installing packages such as PyTorch.
6. **Do not run `conda init`**; per-cluster policy, rely on modules instead.
7. Cache heavy installs in scratch, not home; recreate environments instead of backing
   them up.

`SLURM/Licca/setup_conda_environment.sh` automates the steps above.

## Filesystem layout and staging
- Home: `/hpc/gpfs2/home/u/$USER` (backed up daily). Keep configs, checkpoints, sources.
- Scratch: `/hpc/gpfs2/scratch/u/$USER` (no backup). Use for conda envs, datasets,
  temporary outputs.
- Group directories live under `/hpc/gpfs2/home/g/<project>` and `/hpc/gpfs2/scratch/g/<project>`.
  Never change ACLs or run recursive `chmod` there.
- Node-local SSD: `/tmp` (800 GB per node). Copy inputs in, run, copy results back before
  exit—contents are wiped when the job finishes.
- RAM disk: `/dev/shm` (≈50 % of node RAM). Count RAM disk usage towards `--mem`, otherwise
  jobs are killed by the OOM handler.

## Job workflow
1. Prepare input data in group scratch or copy from home to the node-local `/tmp` in the
   job prologue.
2. Purge modules, load conda manager (and CUDA if needed), activate the `robot-sf`
   environment.
3. Export `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}` to match requested cores and avoid
   oversubscription from BLAS/NumPy defaults.
4. Launch work with `srun`—it propagates Slurm bindings automatically.
5. Archive logs from `slurm-<jobid>.out` and copy artifacts back before the job ends.

Useful monitoring commands: `sinfo` (partition/node state), `squeue -u $USER` (your jobs),
`scancel <jobid>` (cancel), `scontrol show job <jobid>` (debug pending reasons), and
`sacct -j <jobid>` (post-mortem accounting).

## CPU workloads
- Use `epyc` for standard runs and `epyc-mem` for memory-heavy evaluations.
- Specify at least `--ntasks=1`, `--cpus-per-task=<threads>`, and `--mem-per-cpu=<GiB>`.
- Always cap threading with `OMP_NUM_THREADS`; many scientific libraries spawn threads by
  default.
- Script template: `SLURM/Licca/robot_sf_cpu.sl` demonstrates a multi-threaded benchmark
  job that stages data to `/tmp`.

## GPU workloads
- Pick `epyc-gpu` for single/three A100 PCIe jobs, `epyc-gpu-sxm` for NVLink, `xeon-gpu`
  for H100 pairs, and request the minimum GPUs via `--gpus=a100:1` (or `h100:2`, etc.).
- Add `#SBATCH --gres-flags=enforce-binding` to keep CPU cores pinned to the assigned GPU.
- Keep CPU requests ≤32 cores per GPU to avoid cross-socket penalties and stranded GPUs.
- Load CUDA toolkits explicitly when needed (`module load cuda/12.1`).
- Consider enabling the NVIDIA Multi-Process Service via `module load cuda-mps` for
  workloads that underutilise a GPU; remember to `module unload cuda-mps` before exit.
- GPU template: `SLURM/Licca/robot_sf_gpu.sl` shows a single-GPU PPO training job.

## Example assets
- `setup_conda_environment.sh`: creates the `robot-sf` conda/micromamba env on scratch and
  installs project dependencies (with optional CUDA wheels).
- `robot_sf_cpu.sl`: CPU-oriented training/evaluation template with multi-thread control and
  `/tmp` staging.
- `robot_sf_gpu.sl`: GPU training template requesting one A100, enforce-binding, and optional
  CUDA MPS support hooks.

## Common pitfalls checklist
- Avoid `uv sync`; LiCCA supports conda/micromamba only.
- Never run `conda init` or edit ACLs in group directories.
- Remember to copy results off `/tmp` or `/dev/shm`; both are purged immediately when the
  job exits.
- Set `OMP_NUM_THREADS` in every sbatch script and request realistic CPU counts for GPU jobs.
- Install GPU packages from the login node using `CONDA_OVERRIDE_CUDA` to avoid missing
  wheels on compute nodes.
- Initialise `fast-pysf` (`git submodule update --init --recursive`) before packaging the
  project for submission.

