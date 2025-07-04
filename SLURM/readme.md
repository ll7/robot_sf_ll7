# SLURM

```bash
sbatch slurm_train.sl
```

## Track CPU and GPU usage

Activate the conda environment:

```bash
conda activate conda_env
```

Install the required packages in the conda environment:

```bash
# Option 1: Using conda (recommended for conda environments)
conda install psutil gputil

# Option 2: Using pip within conda environment
pip install psutil gputil

# Option 3: Using uv (requires uv to be installed in the conda environment)
pip install uv  # install uv first if needed
uv pip install psutil gputil
```

Modify the `slurm_train.sl` file to run the training with util callback:

```bash
python log_gpu_cpu_usage.py
```

Run the script:

```bash
sbatch slurm_train.sl
```

To view the CPU and GPU usage results, follow these steps:

1. The tracking results are saved in a log file. The exact location of the log file will depend on the configuration in your `slurm_train.sl` and `log_gpu_cpu_usage.py` script. Please refer to the script to find the specific directory or file name where the results are stored.

2. The results should be saved in a tensorboard log file. You can view the results by running the following command:

```bash
tensorboard --logdir=<path_to_log_directory>
```

Make sure to adjust the instructions based on your specific setup and requirements.


