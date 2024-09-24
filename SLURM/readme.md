# SLURM

```bash
sbatch slurm_train.sl
```

## Track CPU and GPU usage

Activate the conda environment:

```bash
conda activate conda_env
```

Manually install the packages:

```bash
pip install psutil gputil
```

Modify the `slurm_train.sl` file to run the training with util callback:

```bash
python log_gpu_cpu_usage.py
```

Run the script:

```bash
sbatch slurm_train.sl
```

