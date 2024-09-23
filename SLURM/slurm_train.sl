#!/usr/bin/env bash
  
#SBATCH --job-name=robot-sf
#SBATCH --partition=epyc-gpu
#SBATCH --time=10:00:00
 
# Request memory per CPU
#SBATCH --mem-per-cpu=2G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=64
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1
 
# Clear all interactively loaded modules
module purge
 
# Load a python package manager
module load cuda anaconda  # or micromamba or condaforge
 
# Activate a certain environment
conda activate conda_env
  
# set number of OpenMP threads (i.e. for numpy, etc...)
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# if you are adding your own level of parallelzation, you
# probably want to set OMP_NUM_THREADS=1 instead, in order
# to prevent the creation of too many threads (massive slowdown!)
 
# No need to pass number of tasks to srun
srun python3 slurm_PPO_robot_sf.py
