#!/usr/bin/env bash

#SBATCH --job-name=robot-sf
#SBATCH --partition=epyc-gpu-test
#SBATCH --time=2:00:00
 
# Request memory per CPU
#SBATCH --mem-per-cpu=2G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=64
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1

# Check if SLURM_EMAIL is set
if [ -z "$SLURM_EMAIL" ]; then
  echo "SLURM_EMAIL is not set. Please set it before running the script."
else
  # Add email notification
  #SBATCH --mail-user=$SLURM_EMAIL
  #SBATCH --mail-type=END,FAIL
  echo "SLURM_EMAIL is set to $SLURM_EMAIL"
fi


# # echo date and time
echo "Starting script at $(date)"

# # Create experiment description
echo "Run experiment with OMP_NUM_THREADS=1 because multithreading is in sb3"

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
export OMP_NUM_THREADS=1
 
# No need to pass number of tasks to srun
srun python3 log_gpu_cpu_usage.py

# echo date and time
# echo "Ending script at $(date)"
